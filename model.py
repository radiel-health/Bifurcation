"""
GNN model architecture for wall shear stress prediction (3D point cloud).

Two-stream architecture:
1. Flow Encoder: MLP([Re] → context_dim)
2. Geometry Encoder: GCN(node_features → hidden_dim)
3. FiLM Modulation: Context modulates geometry via γ, β
4. Task Head: GAT + MLP → WSS predictions (3 components: x, y, z)

Key design choices:
- KNN topology: Each node connects to k-nearest neighbors in 3D space
- Message passing: 3 GCN layers = 6-hop neighborhood context
- FiLM fusion: Flow context modulates geometry features multiplicatively
- Log1p normalization: Handles WSS values spanning orders of magnitude
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GPSConv
import torchbnn as bnn
from torch_geometric.utils import degree

class GeometryEncoder(nn.Module):
    """
    Process boundary mesh with graph convolutions.
    
    Architecture: 3 GCN layers with residual connections
    
    Args:
        input_dim: Node feature dimension (4 for 3D: x, y, z, p)
        hidden_dim: Hidden dimension
        num_layers: Number of GCN layers (default: 3)
        dropout: Dropout rate
    """
    
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=3, dropout=0.1, heads=4):
        super().__init__()
        
        # Ensure hidden_dim is divisible by heads to maintain consistent dimensions
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        # Calculate the dimension per head
        head_dim = hidden_dim // heads 
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GCN layers (GAT with multi-head attention)
        self.convs = nn.ModuleList([
            GATConv(
                in_channels=hidden_dim, 
                out_channels=head_dim, # Output of each head
                heads=heads,           # Number of attention heads
                concat=True,           # Concat heads to get back to hidden_dim
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization stays the same because concat=True 
        # brings the total output back to hidden_dim
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, edge_index):
        """
        Args:
            x: [num_nodes, input_dim] node features
            edge_index: [2, num_edges] edge connectivity
            
        Returns:
            h: [num_nodes, hidden_dim] geometry embeddings
        """
        # Project to hidden dimension
        h = self.input_proj(x)
        
        # Apply GCN layers with residual connections
        for i in range(self.num_layers):
            h_in = h
            
            # Graph convolution
            h = self.convs[i](h, edge_index)
            
            # Normalization + activation
            h = self.norms[i](h)
            h = F.relu(h)
            
            # Dropout
            if self.training:
                h = F.dropout(h, p=self.dropout)
            
            # Residual connection (after first layer)
            if i > 0:
                h = h + h_in
        
        return h

class TaskHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=2, 
                 num_layers=5, dropout=0.3, monte_carlo_sims=100, 
                 output_range=False, heads=2):
        super().__init__()
        
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        self.monte_carlo_sims = monte_carlo_sims
        self.output_range = output_range
        
        self.feature_align = nn.Linear(input_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            local_conv = GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // heads, 
                heads=heads,
                concat=True,
                dropout=dropout
            )
            
            self.convs.append(GPSConv(
                channels=hidden_dim,
                conv=local_conv,
                heads=heads,
                dropout=dropout,
                attn_type='multihead' 
            ))
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim // 2, out_features=output_dim)
        )

    # NEW: The Differentiable PDE Solver
    def relax_pde(self, y_raw, edge_index, alpha=0.15, num_iters=2):
        row, col = edge_index
        deg = degree(col, y_raw.size(0), dtype=y_raw.dtype)
        deg_inv = 1.0 / deg
        deg_inv[deg_inv == float('inf')] = 0.0
        
        y = y_raw.clone() 
        for _ in range(num_iters):
            neighbor_sum = torch.zeros_like(y)
            neighbor_sum.index_add_(0, col, y[row]) 
            neighbor_mean = neighbor_sum * deg_inv.unsqueeze(-1)
            y = y + alpha * (neighbor_mean - y)
        return y

    # UPDATED: Accept normals and apply physics
    def forward(self, h, edge_index, batch, normals):
        h = self.feature_align(h)
        
        for i in range(self.num_layers):
            h_in = h
            h = self.convs[i](h, edge_index, batch)
            h = self.norms[i](h)
            h = F.relu(h)
            if self.training:
                h = F.dropout(h, p=self.dropout)
            if i > 0:
                h = h + h_in
        
        # 1. BNN outputs raw, noisy sample fields
        y_preds_raw = [self.mlp(h) for _ in range(self.monte_carlo_sims)]
        
        y_preds_physical = []
        for y_raw in y_preds_raw:
            # 2. Relax each sample (Smooth)
            y_smoothed = self.relax_pde(y_raw, edge_index, alpha=0.15, num_iters=2)
            
            # 3. Tangency Projection (Flatten to wall)
            dot_product = (y_smoothed * normals).sum(dim=1, keepdim=True) 
            y_projected = y_smoothed - (dot_product * normals)
            
            y_preds_physical.append(y_projected)
            
        stacked_preds = torch.stack(y_preds_physical)
        
        if self.output_range:
            return torch.quantile(stacked_preds, torch.tensor([0.025, 0.975], device=h.device), dim=0)
        else:
            return torch.mean(stacked_preds, dim=0)


    def forward(self, data):
        x = data.x                    
        edge_index = data.edge_index  
        batch = data.batch            
        normals = data.normals        # NEW: Extract normals from batch
        flow_params = data.flow_params 
        
        context = self.flow_encoder(flow_params) 
        h_geom = self.geom_encoder(x, edge_index) 
        h_fused = self.film(h_geom, context, batch)  
        
        # NEW: Pass normals to the TaskHead
        y_pred = self.task_head(h_fused, edge_index, batch, normals)  
        
        return y_pred

class WSSPredictor(nn.Module):
    def __init__(
        self,
        node_feature_dim=3, # e.g., 3 coords + 3 normals + 1 Re = 7
        hidden_dim=64,
        output_dim=3,
        num_geom_layers=3,
        num_task_layers=2,
        task_hidden_dim=128,
        dropout=0.1,
        output_range=False
    ):
        super().__init__()
        
        # 1. Geometry Encoder (Now handles Re organically)
        self.geom_encoder = GeometryEncoder(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_geom_layers,
            dropout=dropout
        )
        
        # 2. Task Head (With the physics relaxation we added!)
        self.task_head = TaskHead(
            input_dim=hidden_dim,
            hidden_dim=task_hidden_dim,
            output_dim=output_dim,
            num_layers=num_task_layers,
            dropout=dropout,
            output_range=output_range
        )
    
    def forward(self, data):
        # 1. Extract data
        x = data.x                    # [num_nodes, 3] (coordinates)
        normals = data.normals        # [num_nodes, 3]
        flow_params = data.flow_params # [batch_size, 1] (Just Re now)
        edge_index = data.edge_index  
        batch = data.batch            
        
        # 2. Broadcast Re to every node in the batch!
        # This maps the batch-level Re to the individual nodes
        node_re = flow_params[batch]  # [num_nodes, 1]
        
        # 3. Concatenate everything into a single rich feature vector
        # Features: [x, y, z, nx, ny, nz, Re] (dimension = 7)
        combined_x = torch.cat([x, normals, node_re], dim=-1)
        
        # 4. Straight through the network
        h_geom = self.geom_encoder(combined_x, edge_index) 
        y_pred = self.task_head(h_geom, edge_index, batch, normals) 
        
        return y_pred
    
    def predict(self, data, denormalize_fn=None):
        """
        Inference with optional denormalization.
        
        Args:
            data: PyG Batch object
            denormalize_fn: Function to convert normalized predictions back to original scale
            
        Returns:
            y_pred: [num_nodes, output_dim] predictions (denormalized if fn provided)
        """
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(data)
            
            if denormalize_fn is not None:
                y_pred = denormalize_fn(y_pred)
            
            return y_pred

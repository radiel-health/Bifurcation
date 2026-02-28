"""
GNN model architecture for wall shear stress prediction (3D point cloud).
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
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=4, # <-- Now defaults to 4
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
            # This now outputs 4 features (3 for direction, 1 for magnitude)
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
        
        # 1. BNN outputs raw fields (Shape: [num_nodes, 4])
        y_preds_raw = [self.mlp(h) for _ in range(self.monte_carlo_sims)]
        
        y_preds_physical = []
        for y_raw in y_preds_raw:
            # A. Slice the components
            v_raw = y_raw[:, :3]        # [num_nodes, 3] The 3D Direction
            m_raw = y_raw[:, 3:4]       # [num_nodes, 1] The Scalar Magnitude
            
            # B. Ensure predicted magnitude is mathematically strictly positive
            m_pred = F.softplus(m_raw) 
            
            # C. Relax the direction vector (Smooth it)
            v_smoothed = self.relax_pde(v_raw, edge_index, alpha=0.15, num_iters=2)
            
            # D. Tangency Projection (Flatten to wall)
            dot_product = (v_smoothed * normals).sum(dim=1, keepdim=True) 
            v_projected = v_smoothed - (dot_product * normals)
            
            # E. Calculate current magnitude of the projected vector
            v_proj_mag = torch.norm(v_projected, dim=1, keepdim=True)
            
            # F. MAGNITUDE RESCALING
            # Multiply vector by (Predicted Mag / Current Mag)
            # We add 1e-8 to the denominator to prevent division by zero
            v_final = v_projected * (m_pred / (v_proj_mag + 1e-8))
            
            y_preds_physical.append(v_final)
            
        # The output is safely back to [num_nodes, 3]!
        stacked_preds = torch.stack(y_preds_physical)
        
        if self.output_range:
            return torch.quantile(stacked_preds, torch.tensor([0.025, 0.975], device=h.device), dim=0)
        else:
            return torch.mean(stacked_preds, dim=0)


class WSSPredictor(nn.Module):
    def __init__(
        self,
        node_feature_dim=7, 
        hidden_dim=64,
        output_dim=3, # Target dim from config (3)
        num_geom_layers=3,
        num_task_layers=2,
        task_hidden_dim=128,
        dropout=0.1,
        output_range=False
    ):
        super().__init__()
        
        self.geom_encoder = GeometryEncoder(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_geom_layers,
            dropout=dropout
        )
        
        self.task_head = TaskHead(
            input_dim=hidden_dim,
            hidden_dim=task_hidden_dim,
            # HARDCODED to 4 here to enable the auxiliary scalar task
            output_dim=4, 
            num_layers=num_task_layers,
            dropout=dropout,
            output_range=output_range
        )
    
    def forward(self, data):
        x = data.x                    
        normals = data.normals        
        flow_params = data.flow_params 
        edge_index = data.edge_index  
        batch = data.batch            
        
        node_re = flow_params[batch]  
        combined_x = torch.cat([x, normals, node_re], dim=-1)
        
        h_geom = self.geom_encoder(combined_x, edge_index) 
        y_pred = self.task_head(h_geom, edge_index, batch, normals) 
        
        return y_pred
    
    def predict(self, data, denormalize_fn=None):
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(data)
            if denormalize_fn is not None:
                y_pred = denormalize_fn(y_pred)
            return y_pred

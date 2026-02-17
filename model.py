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
from torch_geometric.nn import GCNConv, GATConv, GPSConv
from torch_geometric.data import Data, Batch
import torchbnn as bnn
from math import ceil


class FlowEncoder(nn.Module):
    """
    Encode flow parameters [Re] into context vector.
    
    Architecture: 2-layer MLP with ReLU
    
    Args:
        input_dim: 1 (Re only)
        hidden_dim: Hidden layer size
        output_dim: Context vector dimension
    """
    
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=64):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Batch norm for stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, flow_params):
        """
        Args:
            flow_params: [batch_size, 3] tensor
            
        Returns:
            context: [batch_size, output_dim] tensor
        """
        x = F.relu(self.bn1(self.fc1(flow_params)))
        x = self.fc2(x)
        return x


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


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    
    Modulates geometry features using flow context:
        h_out = γ(context) ⊙ h_geom + β(context)
    
    where γ and β are learned affine transformation parameters.
    
    Args:
        context_dim: Flow context vector dimension
        feature_dim: Geometry feature dimension
    """
    
    def __init__(self, context_dim=64, feature_dim=64):
        super().__init__()
        
        # Networks to produce γ (scale) and β (shift)
        self.gamma_net = nn.Linear(context_dim, feature_dim)
        self.beta_net = nn.Linear(context_dim, feature_dim)
    
    def forward(self, h_geom, context, batch):
        """
        Args:
            h_geom: [num_nodes, feature_dim] geometry features
            context: [batch_size, context_dim] flow context
            batch: [num_nodes] batch assignment for each node
            
        Returns:
            h_fused: [num_nodes, feature_dim] modulated features
        """
        # Generate per-node modulation parameters
        # context[batch] broadcasts context to each node in its graph
        gamma = self.gamma_net(context[batch])  # [num_nodes, feature_dim]
        beta = self.beta_net(context[batch])    # [num_nodes, feature_dim]
        
        # Apply affine transformation
        h_fused = gamma * h_geom + beta
        
        return h_fused


class TaskHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=2, 
                 num_layers=5, dropout=0.3, monte_carlo_sims=100, 
                 output_range=False, heads=2):
        super().__init__()
        
        # Ensure dimensionality consistency for multi-head attention
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        self.monte_carlo_sims = monte_carlo_sims
        self.output_range = output_range
        
        # NEW: Initial projection to align FiLM output (input_dim) 
        # with the internal GPS processing dimension (hidden_dim)
        self.feature_align = nn.Linear(input_dim, hidden_dim)
        
        # GPS layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            # All layers now operate at hidden_dim (128)
            # This allows GPSConv internal residuals (h = h + x) to match shapes
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
        
        # Layer norms 
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # MLP head for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim // 2, out_features=output_dim)
        )

    def forward(self, h, edge_index, batch):
        """
        Args:
            h: Node features from FiLM layer [num_nodes, 64]
            edge_index: Ring topology edges
            batch: Batch assignment vector
        """
        # 1. Project input to hidden_dim (64 -> 128)
        # This prevents the "Size mismatch" RuntimeError in GPSConv
        h = self.feature_align(h)
        
        for i in range(self.num_layers):
            h_in = h
            
            # GPSConv performs internal local + global message passing
            h = self.convs[i](h, edge_index, batch)
            h = self.norms[i](h)
            h = F.relu(h)
            
            if self.training:
                h = F.dropout(h, p=self.dropout)
            
            # Residual connection (now shapes match at 128)
            if i > 0:
                h = h + h_in
        
        # Monte Carlo sampling via Bayesian MLP
        y_preds = [self.mlp(h) for _ in range(self.monte_carlo_sims)]
        stacked_preds = torch.stack(y_preds)
        
        if self.output_range:
            return torch.quantile(stacked_preds, torch.tensor([0.025, 0.975], device=h.device), dim=0)
        else:
            return torch.mean(stacked_preds, dim=0)

class WSSPredictor(nn.Module):
    """
    Complete model: Two-stream GNN with FiLM modulation for WSS prediction.
    
    Architecture:
        1. Flow Encoder: [Re, Lx, Ly] → context vector
        2. Geometry Encoder: Node features + edges → geometry embeddings
        3. FiLM Fusion: Context modulates geometry
        4. Task Head: Fused features → WSS predictions
    
    Args:
        node_feature_dim: Dimension of node features (10)
        flow_param_dim: Dimension of flow parameters (3)
        hidden_dim: Hidden dimension for GNN layers
        context_dim: Flow context dimension
        output_dim: Output dimension (2 for x,y WSS components)
        num_geom_layers: Number of geometry encoder layers
        num_task_layers: Number of task head layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        node_feature_dim=4,
        flow_param_dim=1,
        hidden_dim=64,
        context_dim=64,
        output_dim=3,
        num_geom_layers=3,
        num_task_layers=2,
        task_hidden_dim=128,
        dropout=0.1
    ):
        super().__init__()
        
        # Component 1: Flow encoder
        self.flow_encoder = FlowEncoder(
            input_dim=flow_param_dim,
            hidden_dim=hidden_dim,
            output_dim=context_dim
        )
        
        # Component 2: Geometry encoder
        self.geom_encoder = GeometryEncoder(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_geom_layers,
            dropout=dropout
        )
        
        # Component 3: FiLM modulation
        self.film = FiLMLayer(
            context_dim=context_dim,
            feature_dim=hidden_dim
        )
        
        # Component 4: Task head
        self.task_head = TaskHead(
            input_dim=hidden_dim,
            hidden_dim=task_hidden_dim,
            output_dim=output_dim,
            num_layers=num_task_layers,
            dropout=dropout
        )
    
    def forward(self, data):
        """
        Forward pass through entire model.
        
        Args:
            data: PyG Batch object containing:
                - x: [num_nodes, node_feature_dim] node features
                - edge_index: [2, num_edges] edge connectivity
                - flow_params: [batch_size, 1] flow parameters (Re)
                - batch: [num_nodes] batch assignment
                
        Returns:
            y_pred: [num_nodes, output_dim] WSS predictions
        """
        # Extract data
        x = data.x                    # [num_nodes, 4]
        edge_index = data.edge_index  # [2, num_edges]
        batch = data.batch            # [num_nodes]
        
        # Flow parameters from data.flow_params
        flow_params = data.flow_params  # [batch_size, 1]
        
        # 1. Encode flow context
        context = self.flow_encoder(flow_params)  # [batch_size, context_dim]
        
        # 2. Encode geometry
        h_geom = self.geom_encoder(x, edge_index)  # [num_nodes, hidden_dim]
        
        # 3. Fuse via FiLM modulation
        h_fused = self.film(h_geom, context, batch)  # [num_nodes, hidden_dim]
        
        # 4. Predict WSS
        y_pred = self.task_head(h_fused, edge_index, batch)  # [num_nodes, 3]
        
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


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model):
    """
    Print model architecture summary.
    
    Args:
        model: WSSPredictor instance
    """
    print("=" * 80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 80)
    print()
    
    # Count parameters per component
    flow_params = count_parameters(model.flow_encoder)
    geom_params = count_parameters(model.geom_encoder)
    film_params = count_parameters(model.film)
    task_params = count_parameters(model.task_head)
    total_params = count_parameters(model)
    
    print("Component Parameters:")
    print(f"  Flow Encoder:     {flow_params:,}")
    print(f"  Geometry Encoder: {geom_params:,}")
    print(f"  FiLM Layer:       {film_params:,}")
    print(f"  Task Head:        {task_params:,}")
    print(f"  {'─' * 40}")
    print(f"  Total:            {total_params:,}")
    print()
    
    # Model size in MB
    param_size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per float32
    print(f"Model Size: {param_size_mb:.2f} MB")
    print()
    
    print("Architecture:")
    print(f"  Flow Encoder: [3] → [64] → [64]")
    print(f"  Geometry Encoder: [10] → [64] (3 GCN layers)")
    print(f"  FiLM Modulation: context[64] ⊙ geometry[64]")
    print(f"  Task Head: [64] → [128] → [2] (2 GAT + MLP)")
    print()
    
    print("=" * 80)


if __name__ == "__main__":
    """Test model creation and forward pass."""
    
    print("Testing WSSPredictor model...\n")
    
    # Create model
    model = WSSPredictor(
        node_feature_dim=10,
        flow_param_dim=3,
        hidden_dim=64,
        context_dim=64,
        output_dim=2,
        num_geom_layers=3,
        num_task_layers=2,
        task_hidden_dim=128,
        dropout=0.1
    )
    
    # Print summary
    get_model_summary(model)
    
    # Create dummy batch (2 graphs)
    print("\nTesting forward pass with dummy data...")
    
    # Graph 1: 100 nodes
    x1 = torch.randn(100, 10)
    edge_index1 = torch.randint(0, 100, (2, 200))
    
    # Graph 2: 150 nodes
    x2 = torch.randn(150, 10)
    edge_index2 = torch.randint(0, 150, (2, 300))
    
    # Flow parameters
    flow_params = torch.tensor([
        [1000.0, 1.0, 1.0],  # Re=1000, Lx=1, Ly=1
        [2000.0, 2.0, 1.0],  # Re=2000, Lx=2, Ly=1
    ])
    
    # Create PyG batch
    from torch_geometric.data import Data, Batch

    # for single graph inference, just pass in just one data object
    
    data1 = Data(x=x1, edge_index=edge_index1, flow_params=flow_params[0:1])
    data2 = Data(x=x2, edge_index=edge_index2, flow_params=flow_params[1:2])
    batch = Batch.from_data_list([data1, data2])
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        y_pred = model(batch)
    
    print(f" Forward pass successful!")
    num_nodes = batch.x.shape[0]
    print(f"   Input: {num_nodes} nodes (100 + 150)")
    print(f"   Output shape: {y_pred.shape}")
    print(f"   Output range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
    print()
    print("Model is ready for training! ")

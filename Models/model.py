"""
GNN model architecture for bifurcation WSS prediction.

Hybrid architecture combining:
- AVFlow Gen 3: EdgeUNetAggregator + 8×GCNConv stack
- LidDrivenCavity: FiLM conditioning on flow parameters

Architecture:
1. FlowEncoder: [Re, angle] → context vector (64D)
2. EdgeUNetAggregator: GraphUNet on line graph → per-node edge context (16D)
3. GCN Stack: 8 layers, bare GCNConv with ReLU (512D hidden)
4. FiLM Modulation: context modulates GCN output via γ, β
5. Output Head: Single Linear(512 → 3) for [wss_x, wss_y, wss_z]

This faithfully replicates AVFlow Gen 3 model with minimal FiLM modification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.nn import GraphUNet
from torch_geometric.data import Data, Batch
from typing import Optional


def build_line_graph(edge_index, num_nodes, device='cpu'):
    """
    Build line graph from edge index.
    
    In a line graph, edges become nodes, and two line-graph nodes are connected
    if their corresponding edges share a vertex in the original graph.
    
    Args:
        edge_index: [2, num_edges] original graph edges
        num_nodes: number of nodes in original graph
        device: torch device
        
    Returns:
        line_graph_edge_index: [2, num_line_edges] line graph connectivity
    """
    # edge_index: [2, E] where E is number of directed edges
    # Each column [u, v] represents edge u→v
    
    # Build adjacency: for each edge, find other edges sharing an endpoint
    num_edges = edge_index.size(1)
    
    # Create edge-to-edge adjacency
    # Two edges (u1,v1) and (u2,v2) are adjacent if they share a vertex
    # i.e., {u1,v1} ∩ {u2,v2} ≠ ∅
    
    src_list = []
    dst_list = []
    
    # For each edge i
    for i in range(num_edges):
        u_i, v_i = edge_index[0, i].item(), edge_index[1, i].item()
        
        # Find all edges j that share a vertex with edge i
        for j in range(num_edges):
            if i == j:
                continue
            u_j, v_j = edge_index[0, j].item(), edge_index[1, j].item()
            
            # Check if edges share a vertex
            if u_i == u_j or u_i == v_j or v_i == u_j or v_i == v_j:
                src_list.append(i)
                dst_list.append(j)
    
    if len(src_list) == 0:
        # No connections (isolated edges) - return empty line graph
        return torch.empty((2, 0), dtype=torch.long, device=device)
    
    line_edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
    return line_edge_index


class EdgeUNetAggregator(MessagePassing):
    """
    Aggregate edge features to nodes using GraphUNet on line graph.
    
    Process:
    1. Convert original graph to line graph (edges → nodes)
    2. Run GraphUNet on line graph to process edge features hierarchically
    3. Project processed edge features to target dimension
    4. Aggregate edge features to nodes via message passing
    
    This is the exact implementation from AVFlow Gen 3 (tawss_unet notebooks).
    
    Args:
        edge_channels: Input edge feature dimension (5)
        out_node_channels: Output per-node aggregated dimension (16)
        unet_hidden: GraphUNet hidden dimension (128)
        unet_depth: GraphUNet depth (4)
        pool_ratio: GraphUNet pooling ratio (0.5)
        aggr: Message aggregation method ('mean')
    """
    
    def __init__(
        self,
        edge_channels=5,
        out_node_channels=16,
        unet_hidden=128,
        unet_depth=4,
        pool_ratio=0.5,
        aggr='mean'
    ):
        super().__init__(aggr=aggr)
        
        # GraphUNet operates on line graph
        self.edge_unet = GraphUNet(
            in_channels=edge_channels,
            hidden_channels=unet_hidden,
            out_channels=edge_channels,  # Keep same dim, project later
            depth=unet_depth,
            pool_ratios=pool_ratio
        )
        
        # Project from edge_channels → out_node_channels
        self.project = nn.Linear(edge_channels, out_node_channels)
    
    def forward(self, edge_index, edge_attr, num_nodes):
        """
        Forward pass.
        
        Args:
            edge_index: [2, num_edges] original graph edges
            edge_attr: [num_edges, edge_channels] edge features
            num_nodes: number of nodes in original graph
            
        Returns:
            aggregated_node_features: [num_nodes, out_node_channels]
        """
        device = edge_index.device
        
        # Build line graph
        lg_edge_index = build_line_graph(edge_index, num_nodes, device)
        
        # Run GraphUNet on line graph (edges as nodes)
        edge_feat = self.edge_unet(edge_attr, lg_edge_index)
        
        # Project to output dimension
        edge_feat = self.project(edge_feat)
        
        # Aggregate to nodes via message passing
        return self.propagate(edge_index, size=(num_nodes, num_nodes), edge_attr=edge_feat)
    
    def message(self, edge_attr):
        """Message function: just pass edge features."""
        return edge_attr


class FlowEncoder(nn.Module):
    """
    Encode flow parameters [Re, angle] into context vector.
    
    Architecture: 2-layer MLP with BatchNorm + ReLU
    Adapted from LidDrivenCavity FlowEncoder.
    
    Args:
        input_dim: 2 (Re_normalized, angle_radians)
        hidden_dim: Hidden layer size (64)
        output_dim: Context vector dimension (64)
    """
    
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=64):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, flow_params):
        """
        Args:
            flow_params: [batch_size, 2] tensor
            
        Returns:
            context: [batch_size, output_dim] tensor
        """
        x = F.relu(self.bn1(self.fc1(flow_params)))
        x = self.fc2(x)
        return x


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    
    Modulates geometry features using flow context:
        h_out = γ(context) ⊙ h_geom + β(context)
    
    From LidDrivenCavity pipeline.
    
    Args:
        context_dim: Flow context vector dimension (64)
        feature_dim: Geometry feature dimension (512)
    """
    
    def __init__(self, context_dim=64, feature_dim=512):
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
        gamma = self.gamma_net(context[batch])  # [num_nodes, feature_dim]
        beta = self.beta_net(context[batch])    # [num_nodes, feature_dim]
        
        # Apply affine transformation
        h_fused = gamma * h_geom + beta
        
        return h_fused


class BifurcationWSSPredictor(nn.Module):
    """
    Complete model for bifurcation WSS prediction.
    
    Hybrid architecture:
    - AVFlow Gen 3 backbone: EdgeUNetAggregator + 8×GCNConv + Linear head
    - LidDrivenCavity FiLM: Flow context modulation
    
    Flow:
        flow_params [Re, angle] → FlowEncoder → context [64D]
                                                         ↓ FiLM
        node_features [4D] + edge_context [16D] → 8×GCNConv → h_geom [512D]
                                                         ↓
                                h_fused = γ·h_geom + β  [512D]
                                                         ↓
                                       Linear(512 → 3) → [wss_x, wss_y, wss_z]
    
    Args:
        original_node_feat_dim: Node feature dimension (4: x,y,z,degree)
        edge_channels: Edge feature dimension (5)
        aggregated_edge_feat_dim: Edge aggregator output dimension (16)
        hidden_gcn_dim: GCN hidden dimension (512)
        out_channels: Output dimension (3: wss_x, wss_y, wss_z)
        num_gcn_layers: Number of GCN layers (8)
        context_dim: Flow context dimension (64)
    """
    
    def __init__(
        self,
        original_node_feat_dim=4,
        edge_channels=5,
        aggregated_edge_feat_dim=16,
        hidden_gcn_dim=512,
        out_channels=3,
        num_gcn_layers=8,
        context_dim=64,
        unet_hidden=128,
        unet_depth=4,
        unet_pool_ratio=0.5
    ):
        super().__init__()
        
        # Flow encoder
        self.flow_encoder = FlowEncoder(
            input_dim=2,  # [Re_norm, angle_rad]
            hidden_dim=64,
            output_dim=context_dim
        )
        
        # Edge aggregator (GraphUNet on line graph)
        self.edge_aggregator = EdgeUNetAggregator(
            edge_channels=edge_channels,
            out_node_channels=aggregated_edge_feat_dim,
            unet_hidden=unet_hidden,
            unet_depth=unet_depth,
            pool_ratio=unet_pool_ratio,
            aggr='mean'
        )
        
        # GCN stack (8 layers, bare GCNConv + ReLU, no residuals/norms)
        self.convs = nn.ModuleList()
        
        # First layer: [node_feat + edge_context] → hidden
        gcn_input_dim = original_node_feat_dim + aggregated_edge_feat_dim
        self.convs.append(GCNConv(gcn_input_dim, hidden_gcn_dim))
        
        # Remaining layers: hidden → hidden
        for _ in range(num_gcn_layers - 1):
            self.convs.append(GCNConv(hidden_gcn_dim, hidden_gcn_dim))
        
        # FiLM modulation
        self.film = FiLMLayer(
            context_dim=context_dim,
            feature_dim=hidden_gcn_dim
        )
        
        # Output head (single linear layer)
        self.lin = nn.Linear(hidden_gcn_dim, out_channels)
    
    def forward(self, data):
        """
        Forward pass through entire model.
        
        Args:
            data: PyG Batch object containing:
                - x: [num_nodes, 4] node features
                - edge_index: [2, num_edges] edge connectivity
                - edge_attr: [num_edges, 5] edge features
                - flow_params: [batch_size, 2] flow parameters [Re, angle]
                - batch: [num_nodes] batch assignment
                
        Returns:
            y_pred: [num_nodes, 3] WSS predictions [x, y, z]
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        num_nodes = data.num_nodes
        batch = data.batch
        
        # Reconstruct flow_params if stored separately
        if hasattr(data, 'flow_params'):
            # flow_params gets batched incorrectly by PyG (concatenated along dim=0)
            # Need to reshape to [batch_size, 2]
            batch_size = batch.max().item() + 1
            flow_params = data.flow_params.reshape(batch_size, -1)  # [batch_size, 2]
        else:
            # Build from individual attributes
            flow_params = torch.stack([
                data.re.float(),
                data.angle.float()
            ], dim=1)
        
        # 1. Encode flow context
        context = self.flow_encoder(flow_params)  # [batch_size, context_dim]
        
        # 2. Aggregate edge features to nodes
        edge_feats = self.edge_aggregator(edge_index, edge_attr, num_nodes)  # [num_nodes, 16]
        
        # 3. Concatenate node features with edge context
        x = torch.cat([x, edge_feats], dim=1)  # [num_nodes, 4+16=20]
        
        # 4. GCN stack (bare convolutions with ReLU, no residuals)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))  # [num_nodes, 512]
        
        h_geom = x  # Geometry encoding
        
        # 5. FiLM modulation with flow context
        h_fused = self.film(h_geom, context, batch)  # [num_nodes, 512]
        
        # 6. Output head
        y_pred = self.lin(h_fused)  # [num_nodes, 3]
        
        return y_pred
    
    def predict(self, data, denormalize_fn=None):
        """
        Inference with optional denormalization.
        
        Args:
            data: PyG Batch object
            denormalize_fn: Function to convert normalized predictions to physical units
            
        Returns:
            y_pred: [num_nodes, 3] predictions (denormalized if fn provided)
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
        model: BifurcationWSSPredictor instance
    """
    print("=" * 80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 80)
    print()
    
    # Count parameters per component
    flow_params = count_parameters(model.flow_encoder)
    edge_params = count_parameters(model.edge_aggregator)
    gcn_params = count_parameters(model.convs)
    film_params = count_parameters(model.film)
    head_params = count_parameters(model.lin)
    total_params = count_parameters(model)
    
    print("Component Parameters:")
    print(f"  Flow Encoder:         {flow_params:,}")
    print(f"  Edge Aggregator:      {edge_params:,}")
    print(f"  GCN Stack (8 layers): {gcn_params:,}")
    print(f"  FiLM Layer:           {film_params:,}")
    print(f"  Output Head:          {head_params:,}")
    print(f"  {'-' * 50}")
    print(f"  Total:                {total_params:,}")
    print()
    
    # Model size in MB
    param_size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per float32
    print(f"Model Size: {param_size_mb:.2f} MB")
    print()
    
    print("Architecture:")
    print(f"  Flow Encoder:      [2] → [64] → [64]")
    print(f"  Edge Aggregator:   GraphUNet(in=5, hidden=128, out=5, depth=4)")
    print(f"                     → Linear(5 → 16) → mean aggregation")
    print(f"  GCN Stack:         [4+16=20] → [512] (8 layers, ReLU only)")
    print(f"  FiLM Modulation:   context[64] ⊙ geometry[512]")
    print(f"  Output Head:       Linear(512 → 3)")
    print()
    
    print("=" * 80)


if __name__ == "__main__":
    """Test model creation and forward pass."""
    
    print("Testing BifurcationWSSPredictor model...\n")
    
    # Create model
    model = BifurcationWSSPredictor(
        original_node_feat_dim=4,
        edge_channels=5,
        aggregated_edge_feat_dim=16,
        hidden_gcn_dim=512,
        out_channels=3,
        num_gcn_layers=8,
        context_dim=64
    )
    
    # Print summary
    get_model_summary(model)
    
    # Create dummy batch (2 graphs)
    print("\nTesting forward pass with dummy 3D mesh data...")
    
    # Graph 1: 1000 nodes
    x1 = torch.randn(1000, 4)  # [x,y,z,degree]
    edge_index1 = torch.randint(0, 1000, (2, 3000))
    edge_attr1 = torch.randn(3000, 5)  # 5D edge features
    
    # Graph 2: 1500 nodes
    x2 = torch.randn(1500, 4)
    edge_index2 = torch.randint(0, 1500, (2, 4500))
    edge_attr2 = torch.randn(4500, 5)
    
    # Flow parameters [Re_normalized, angle_radians]
    flow_params1 = torch.tensor([[0.5, 0.524]])  # Re~1000, angle=30°
    flow_params2 = torch.tensor([[0.8, 0.785]])  # Re~1700, angle=45°
    
    # Create PyG batch
    data1 = Data(x=x1, edge_index=edge_index1, edge_attr=edge_attr1, flow_params=flow_params1)
    data2 = Data(x=x2, edge_index=edge_index2, edge_attr=edge_attr2, flow_params=flow_params2)
    batch = Batch.from_data_list([data1, data2])
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        y_pred = model(batch)
    
    print(f"✓ Forward pass successful!")
    print(f"  Input: {batch.num_nodes} nodes (1000 + 1500)")
    print(f"  Output shape: {y_pred.shape}")
    print(f"  Output range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
    print()
    print("Model is ready for training!")

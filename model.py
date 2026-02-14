"""
GNN model for bifurcation wall shear stress prediction.

Architecture (adapted from AVFlow Gen 3 + FiLM conditioning):

    flow_params [log10(Re), angle_rad]
        │
        ▼
    FlowEncoder ──► context  (64-D)
                         │
                         │ FiLM  γ·h + β
                         ▼
    node_feat (3) ──┐
                    ├─► concat (3+16=19) ──► 8× GCNConv+ReLU (512-D) ──► h_geom
    EdgeUNet(4) ────┘                                                       │
        ▲                                                                   ▼
    edge_attr (4)                                                   h_fused = FiLM(h_geom, ctx)
                                                                           │
                                                                    Linear(512→3) ──► [wss_x, wss_y, wss_z]

Uses PyG's LineGraph transform for efficient line-graph construction.

Usage:
    python -m Bifurcation.model          # smoke-test with dummy data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing, GraphUNet
from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data, Batch


# ============================================================================
# Line-graph utility
# ============================================================================

_line_graph_transform = LineGraph(force_directed=False)


def build_line_graph(edge_index: torch.Tensor, num_nodes: int,
                     device: torch.device = None) -> torch.Tensor:
    """
    Build line-graph edge_index using PyG's sparse LineGraph transform.

    In the line graph every *edge* of the original graph becomes a *node*,
    and two line-graph nodes are connected when their original edges share
    a vertex.
    """
    tmp = Data(edge_index=edge_index, num_nodes=num_nodes)
    tmp = _line_graph_transform(tmp)
    ei = tmp.edge_index
    return ei.to(device) if device is not None else ei


# ============================================================================
# EdgeUNetAggregator  (AVFlow Gen 3)
# ============================================================================

class EdgeUNetAggregator(MessagePassing):
    """
    Process edge features with a Graph-U-Net on the line graph, then
    aggregate back to nodes via message passing.

    Steps:
        1. Convert original graph → line graph (edges become nodes)
        2. Run GraphUNet on line-graph → hierarchically refined edge features
        3. Project to ``out_node_channels``
        4. Mean-aggregate edge features arriving at each node

    Args:
        edge_channels:      input edge-feature dimension (4)
        out_node_channels:  output per-node dimension      (16)
        unet_hidden:        GraphUNet hidden channels       (128)
        unet_depth:         GraphUNet depth                 (4)
        pool_ratio:         GraphUNet pool ratio            (0.5)
        aggr:               aggregation mode                ('mean')
    """

    def __init__(
        self,
        edge_channels: int = 4,
        out_node_channels: int = 16,
        unet_hidden: int = 128,
        unet_depth: int = 4,
        pool_ratio: float = 0.5,
        aggr: str = "mean",
    ):
        super().__init__(aggr=aggr)
        self.edge_unet = GraphUNet(
            in_channels=edge_channels,
            hidden_channels=unet_hidden,
            out_channels=edge_channels,
            depth=unet_depth,
            pool_ratios=pool_ratio,
        )
        self.project = nn.Linear(edge_channels, out_node_channels)

    def forward(self, edge_index, edge_attr, num_nodes):
        lg_edge_index = build_line_graph(edge_index, num_nodes,
                                         device=edge_index.device)
        edge_feat = self.edge_unet(edge_attr, lg_edge_index)
        edge_feat = self.project(edge_feat)
        return self.propagate(edge_index, size=(num_nodes, num_nodes),
                              edge_attr=edge_feat)

    def message(self, edge_attr):
        return edge_attr


# ============================================================================
# FlowEncoder
# ============================================================================

class FlowEncoder(nn.Module):
    """
    Encode flow parameters ``[log10(Re), angle_rad]`` into a context vector.

    Architecture: Linear → BN → ReLU → Linear
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64,
                 output_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, flow_params):
        """flow_params: [batch_size, 2]  →  context: [batch_size, output_dim]"""
        x = F.relu(self.ln1(self.fc1(flow_params)))
        return self.fc2(x)


# ============================================================================
# FiLM modulation
# ============================================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.

        h_out = γ(context) ⊙ h_geom + β(context)

    ``context`` is broadcast to every node via the ``batch`` index.
    """

    def __init__(self, context_dim: int = 64, feature_dim: int = 512):
        super().__init__()
        self.gamma_net = nn.Linear(context_dim, feature_dim)
        self.beta_net = nn.Linear(context_dim, feature_dim)

    def forward(self, h_geom, context, batch):
        """
        h_geom  : [num_nodes, feature_dim]
        context : [batch_size, context_dim]
        batch   : [num_nodes]  (graph-membership index)
        """
        gamma = self.gamma_net(context[batch])   # [num_nodes, feature_dim]
        beta = self.beta_net(context[batch])     # [num_nodes, feature_dim]
        return gamma * h_geom + beta


# ============================================================================
# Full model
# ============================================================================

class BifurcationWSSPredictor(nn.Module):
    """
    Complete GNN for predicting 3-D wall shear stress on bifurcation
    vessel geometries.

    Combines:
      - **AVFlow Gen 3** backbone (EdgeUNetAggregator + deep GCN stack)
      - **FiLM** conditioning on flow parameters (Re, bifurcation angle)

    Args:
        node_feat_dim:            input node feature dim   (3: x,y,z)
        edge_channels:            input edge feature dim   (4)
        aggregated_edge_feat_dim: edge aggregator output   (16)
        hidden_gcn_dim:           GCN hidden width         (512)
        out_channels:             prediction dim           (3: wss_x,y,z)
        num_gcn_layers:           number of GCN layers     (8)
        context_dim:              FiLM context width       (64)
        flow_param_dim:           flow-encoder input       (2)
        unet_hidden:              GraphUNet hidden         (128)
        unet_depth:               GraphUNet depth          (4)
        unet_pool_ratio:          GraphUNet pool ratio     (0.5)
    """

    def __init__(
        self,
        node_feat_dim: int = 3,
        edge_channels: int = 4,
        aggregated_edge_feat_dim: int = 16,
        hidden_gcn_dim: int = 512,
        out_channels: int = 3,
        num_gcn_layers: int = 8,
        context_dim: int = 64,
        flow_param_dim: int = 2,
        unet_hidden: int = 128,
        unet_depth: int = 4,
        unet_pool_ratio: float = 0.5,
    ):
        super().__init__()

        # -- flow encoder --
        self.flow_encoder = FlowEncoder(
            input_dim=flow_param_dim,
            hidden_dim=64,
            output_dim=context_dim,
        )

        # -- edge aggregator --
        self.edge_aggregator = EdgeUNetAggregator(
            edge_channels=edge_channels,
            out_node_channels=aggregated_edge_feat_dim,
            unet_hidden=unet_hidden,
            unet_depth=unet_depth,
            pool_ratio=unet_pool_ratio,
            aggr="mean",
        )

        # -- GCN stack (8 layers, bare GCNConv + ReLU, no residuals) --
        gcn_input_dim = node_feat_dim + aggregated_edge_feat_dim
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(gcn_input_dim, hidden_gcn_dim))
        for _ in range(num_gcn_layers - 1):
            self.convs.append(GCNConv(hidden_gcn_dim, hidden_gcn_dim))

        # -- FiLM --
        self.film = FiLMLayer(context_dim=context_dim, feature_dim=hidden_gcn_dim)

        # -- output head --
        self.lin = nn.Linear(hidden_gcn_dim, out_channels)

    # --------------------------------------------------------------------- #

    def forward(self, data):
        """
        Args:
            data: PyG ``Batch`` (or single ``Data``) with fields
                ``x``, ``edge_index``, ``edge_attr``, ``re``, ``angle``,
                and ``batch``.

        Returns:
            y_pred: [num_nodes, 3]  predicted WSS vectors
        """
        x = data.x                       # [N, node_feat_dim]
        edge_index = data.edge_index     # [2, E]
        edge_attr = data.edge_attr       # [E, edge_channels]
        num_nodes = data.num_nodes
        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Build flow parameter tensor  [batch_size, 2]
        batch_size = int(batch.max().item()) + 1
        re_vals = data.re.view(batch_size)           # [B]
        angle_vals = data.angle.view(batch_size)      # [B]
        log_re = torch.log10(re_vals.clamp(min=1.0))  # log10(Re)
        angle_rad = angle_vals * (3.14159265358979 / 180.0)
        flow_params = torch.stack([log_re, angle_rad], dim=1)  # [B, 2]

        # 1. Flow context
        context = self.flow_encoder(flow_params)      # [B, context_dim]

        # 2. Edge aggregation
        edge_feats = self.edge_aggregator(edge_index, edge_attr, num_nodes)

        # 3. Concatenate node features + aggregated edge context
        x = torch.cat([x, edge_feats], dim=1)

        # 4. GCN stack
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        h_geom = x  # [N, hidden_gcn_dim]

        # 5. FiLM modulation
        h_fused = self.film(h_geom, context, batch)

        # 6. Output
        return self.lin(h_fused)                      # [N, 3]


# ============================================================================
# Utilities
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: BifurcationWSSPredictor):
    print("=" * 70)
    print("BIFURCATION WSS PREDICTOR — ARCHITECTURE SUMMARY")
    print("=" * 70)

    components = {
        "Flow Encoder":    model.flow_encoder,
        "Edge Aggregator": model.edge_aggregator,
        "GCN Stack":       model.convs,
        "FiLM Layer":      model.film,
        "Output Head":     model.lin,
    }
    total = count_parameters(model)
    for name, mod in components.items():
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        print(f"  {name:25s} {n:>12,} params")
    print(f"  {'─' * 40}")
    print(f"  {'TOTAL':25s} {total:>12,} params")
    print(f"  Model size ≈ {total * 4 / 1024**2:.1f} MB (float32)")
    print("=" * 70)


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    print("Smoke-testing BifurcationWSSPredictor …\n")

    model = BifurcationWSSPredictor()
    get_model_summary(model)

    # Dummy graph (500 nodes, 1500 edges – small for testing)
    N, E = 500, 1500
    x = torch.randn(N, 3)
    ei = torch.randint(0, N, (2, E))
    ea = torch.randn(E, 4)
    re = torch.tensor([500.0])
    angle = torch.tensor([45.0])

    data = Data(x=x, edge_index=ei, edge_attr=ea, re=re, angle=angle)

    model.eval()
    with torch.no_grad():
        y = model(data)

    print(f"\n  Input:  {N} nodes, {E} edges")
    print(f"  Output: {y.shape}  (expected [500, 3])")
    print(f"  Range:  [{y.min():.6f}, {y.max():.6f}]")
    print("\nOK ✓")

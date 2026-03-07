"""
GNN model for bifurcation wall shear stress prediction.

Architecture (GATConv + GPSConv + BayesianLinear):

    flow_params [log10(Re), angle_rad]
        │
        ▼
    FlowEncoder ──► context  (64-D)
                         │
                         │ FiLM  γ·h + β
                         ▼
    node_feat (4) ──► GeometryEncoder: 3× GATConv(edge_dim=4)+LayerNorm+ReLU+Residual (64-D) ──► h_geom
    edge_feat (4) ──►      [dist, dx, dy, dz] computed on-the-fly from coordinates              │
                                                                               h_fused = FiLM(h_geom, ctx)
                                                                                          │
                                                                         TaskHead: N× GPSConv+LayerNorm+ReLU+Residual (128-D)
                                                                                          │
                                                                              BayesLinear → 100× MC → [wss_x, wss_y, wss_z]

Usage:
    python -m Bifurcation.model          # smoke-test with dummy data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GPSConv
from torch_geometric.data import Data, Batch
import torchbnn as bnn


# ============================================================================
# FlowEncoder
# ============================================================================

class FlowEncoder(nn.Module):
    """
    Encode flow parameters into a context vector.

    Architecture: Linear → LayerNorm → ReLU → Linear

    Args:
        input_dim: Number of flow parameters (e.g. 1 for Re, 2 for Re+angle)
        hidden_dim: Hidden layer size
        output_dim: Context vector dimension
    """

    def __init__(self, input_dim=1, hidden_dim=64, output_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)

    def forward(self, flow_params):
        """flow_params: [batch_size, input_dim]  →  context: [batch_size, output_dim]"""
        x = F.relu(self.bn1(self.fc1(flow_params)))
        return self.fc2(x)


# ============================================================================
# GeometryEncoder
# ============================================================================

class GeometryEncoder(nn.Module):
    """
    Process boundary mesh with edge-conditioned multi-head graph attention.

    Architecture: input_proj → num_layers × (GATConv(edge_dim) + LayerNorm + ReLU + Residual)

    Edge features [dist, dx, dy, dz] are passed directly to each GATConv layer,
    which uses them to bias the attention coefficients. If edge_attr is not
    supplied to forward(), it is computed on-the-fly from x[:, :3] (coordinates).

    Args:
        input_dim:    Node feature dimension
        hidden_dim:   Hidden dimension (must be divisible by heads)
        edge_feat_dim: Raw edge feature dimension (4: dist, dx, dy, dz)
        num_layers:   Number of GATConv layers
        dropout:      Dropout rate
        heads:        Number of attention heads
    """

    def __init__(self, input_dim=4, hidden_dim=64, edge_feat_dim=4,
                 num_layers=3, dropout=0.1, heads=4):
        super().__init__()

        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")

        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_feat_dim = edge_feat_dim
        head_dim = hidden_dim // heads

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList([
            GATConv(
                in_channels=hidden_dim,
                out_channels=head_dim,
                heads=heads,
                concat=True,
                dropout=dropout,
                edge_dim=edge_feat_dim,   # pass edge features into attention
            )
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

    @staticmethod
    def _compute_edge_attr(x, edge_index):
        """Compute [dist, dx, dy, dz] from node coordinates and edge connectivity."""
        coords = x[:, :3]                          # [N, 3]  first 3 dims are x,y,z
        row, col = edge_index                      # src, dst
        diff = coords[col] - coords[row]           # [E, 3]
        dist = diff.norm(dim=1, keepdim=True)      # [E, 1]
        return torch.cat([dist, diff], dim=1)      # [E, 4]

    def forward(self, x, edge_index, edge_attr=None):
        """
        x:          [num_nodes, input_dim]
        edge_index: [2, num_edges]
        edge_attr:  [num_edges, edge_feat_dim] or None (computed on-the-fly)
        Returns h:  [num_nodes, hidden_dim]
        """
        if edge_attr is None:
            edge_attr = self._compute_edge_attr(x, edge_index)

        h = self.input_proj(x)

        for i in range(self.num_layers):
            h_in = h
            h = self.convs[i](h, edge_index, edge_attr=edge_attr)
            h = self.norms[i](h)
            h = F.relu(h)
            if self.training:
                h = F.dropout(h, p=self.dropout)
            if i > 0:
                h = h + h_in

        return h


# ============================================================================
# FiLM modulation
# ============================================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.

        h_out = γ(context) ⊙ h_geom + β(context)

    ``context`` is broadcast to every node via the ``batch`` index.
    """

    def __init__(self, context_dim=64, feature_dim=64):
        super().__init__()
        self.gamma_net = nn.Linear(context_dim, feature_dim)
        self.beta_net = nn.Linear(context_dim, feature_dim)

    def forward(self, h_geom, context, batch):
        """
        h_geom  : [num_nodes, feature_dim]
        context : [batch_size, context_dim]
        batch   : [num_nodes]  (graph-membership index)
        """
        gamma = self.gamma_net(context[batch])
        beta = self.beta_net(context[batch])
        return gamma * h_geom + beta


# ============================================================================
# TaskHead
# ============================================================================

class TaskHead(nn.Module):
    """
    GPS layers (local GATConv + global multi-head attention) with a
    Bayesian linear output head for uncertainty quantification.

    Monte Carlo sampling: N stochastic forward passes → mean or 95% CI.

    Args:
        input_dim: Dimension of incoming features (from FiLM)
        hidden_dim: Internal GPS processing dimension (must be divisible by heads)
        output_dim: Prediction targets (3 for wss_x/y/z)
        num_layers: Number of GPSConv layers
        dropout: Dropout rate
        monte_carlo_sims: Number of MC forward passes for BNN output
        output_range: If True return [0.025, 0.975] quantiles; else return mean
        heads: Number of attention heads for GPS and local GATConv
    """

    def __init__(self, input_dim=64, hidden_dim=128, output_dim=3,
                 num_layers=5, dropout=0.3, monte_carlo_sims=100,
                 output_range=False, heads=2):
        super().__init__()

        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")

        self.num_layers = num_layers
        self.dropout = dropout
        self.monte_carlo_sims = monte_carlo_sims
        self.output_range = output_range

        # Project FiLM output dim → GPS processing dim
        self.feature_align = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            local_conv = GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
            )
            self.convs.append(GPSConv(
                channels=hidden_dim,
                conv=local_conv,
                heads=heads,
                dropout=dropout,
                attn_type='multihead',
            ))

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            bnn.BayesLinear(
                prior_mu=0, prior_sigma=0.1,
                in_features=hidden_dim // 2,
                out_features=output_dim,
            ),
        )

    def forward(self, h, edge_index, batch):
        """
        h: [num_nodes, input_dim]  (from FiLM layer)
        edge_index: [2, num_edges]
        batch: [num_nodes]
        """
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

        y_preds = [self.mlp(h) for _ in range(self.monte_carlo_sims)]
        stacked = torch.stack(y_preds)   # [mc_sims, num_nodes, output_dim]

        if self.output_range:
            return torch.quantile(
                stacked,
                torch.tensor([0.025, 0.975], device=h.device),
                dim=0,
            )
        return torch.mean(stacked, dim=0)


# ============================================================================
# Full model
# ============================================================================

class WSSPredictor(nn.Module):
    """
    Complete model: FlowEncoder + GeometryEncoder (GATConv) + FiLM + TaskHead (GPSConv + BayesLinear).

    Args:
        node_feature_dim: Node feature dimension (3: x,y,z)
        flow_param_dim: Flow parameter dimension (2: log10(Re) + angle_rad)
        hidden_dim: GeometryEncoder hidden dimension
        context_dim: Flow context dimension
        output_dim: WSS output dimension (3: wss_x,y,z)
        num_geom_layers: GeometryEncoder GATConv layers
        num_task_layers: TaskHead GPSConv layers
        task_hidden_dim: TaskHead internal hidden dimension
        dropout: Dropout rate
        monte_carlo_sims: BNN MC samples per forward pass
    """

    def __init__(
        self,
        node_feature_dim=3,
        flow_param_dim=2,
        hidden_dim=64,
        context_dim=64,
        output_dim=3,
        num_geom_layers=3,
        num_task_layers=2,
        task_hidden_dim=128,
        dropout=0.1,
        monte_carlo_sims=100,
        # Backward-compat aliases used by train.py
        node_feat_dim=None,
        edge_feat_dim=4,      # raw edge feature dim: [dist, dx, dy, dz]
        out_channels=None,
        num_layers=None,
    ):
        super().__init__()

        # Resolve backward-compat aliases
        if node_feat_dim is not None:
            node_feature_dim = node_feat_dim
        if out_channels is not None:
            output_dim = out_channels
        if num_layers is not None:
            num_task_layers = num_layers

        self.flow_param_dim = flow_param_dim

        self.flow_encoder = FlowEncoder(
            input_dim=flow_param_dim,
            hidden_dim=hidden_dim,
            output_dim=context_dim,
        )

        self.geom_encoder = GeometryEncoder(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            edge_feat_dim=edge_feat_dim,
            num_layers=num_geom_layers,
            dropout=dropout,
        )

        self.film = FiLMLayer(
            context_dim=context_dim,
            feature_dim=hidden_dim,
        )

        self.task_head = TaskHead(
            input_dim=hidden_dim,
            hidden_dim=task_hidden_dim,
            output_dim=output_dim,
            num_layers=num_task_layers,
            dropout=dropout,
            monte_carlo_sims=monte_carlo_sims,
        )

    def forward(self, data):
        """
        data: PyG Batch/Data with fields x, edge_index, re, angle, batch.
        Returns y_pred: [num_nodes, output_dim] WSS predictions.
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Build flow parameter tensor.
        # Supports two data formats:
        #   - New: data.flow_params [B, 3] = [Re, angle_deg, mesh_level]
        #   - Old: data.re [B] + data.angle [B] (separate scalar fields)
        batch_size = int(batch.max().item()) + 1

        if hasattr(data, "flow_params") and data.flow_params is not None:
            fp = data.flow_params.view(batch_size, -1)      # [B, ≥2]
            log_re = torch.log10(fp[:, 0].clamp(min=1.0))
            angle_rad = fp[:, 1] * (3.14159265358979 / 180.0)
        else:
            re_vals = data.re.view(batch_size)
            log_re = torch.log10(re_vals.clamp(min=1.0))
            angle_vals = data.angle.view(batch_size)
            angle_rad = angle_vals * (3.14159265358979 / 180.0)

        if self.flow_param_dim >= 2:
            flow_params = torch.stack([log_re, angle_rad], dim=1)   # [B, 2]
        else:
            flow_params = log_re.unsqueeze(1)   # [B, 1]

        # Edge features: use stored edge_attr if available, else compute on-the-fly
        edge_attr = data.edge_attr if (hasattr(data, "edge_attr") and
                                       data.edge_attr is not None) else None

        context = self.flow_encoder(flow_params)                       # [B, context_dim]
        h_geom = self.geom_encoder(x, edge_index, edge_attr)           # [N, hidden_dim]
        h_fused = self.film(h_geom, context, batch)                    # [N, hidden_dim]
        return self.task_head(h_fused, edge_index, batch)              # [N, output_dim]

    def predict(self, data, denormalize_fn=None):
        """Inference with uncertainty (output_range=True on task_head) and optional denorm."""
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(data)
            if denormalize_fn is not None:
                y_pred = denormalize_fn(y_pred)
            return y_pred


# Backward-compatibility alias used by train.py
BifurcationWSSPredictor = WSSPredictor


# ============================================================================
# Utilities
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: WSSPredictor):
    print("=" * 70)
    print("BIFURCATION WSS PREDICTOR — ARCHITECTURE SUMMARY")
    print("=" * 70)

    components = {
        "Flow Encoder":     model.flow_encoder,
        "Geometry Encoder": model.geom_encoder,
        "FiLM Layer":       model.film,
        "Task Head":        model.task_head,
    }
    total = count_parameters(model)
    for name, mod in components.items():
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        print(f"  {name:25s} {n:>12,} params")
    print(f"  {'-' * 40}")
    print(f"  {'TOTAL':25s} {total:>12,} params")
    print(f"  Model size ~= {total * 4 / 1024**2:.1f} MB (float32)")
    print("=" * 70)


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    print("Smoke-testing WSSPredictor (GAT + GPS + BayesLinear) ...\n")

    model = WSSPredictor(
        node_feature_dim=4,
        flow_param_dim=2,
        hidden_dim=64,
        context_dim=64,
        output_dim=3,
        num_geom_layers=3,
        num_task_layers=2,
        task_hidden_dim=128,
        dropout=0.1,
        monte_carlo_sims=5,   # small for smoke test
    )
    get_model_summary(model)

    N, E = 500, 1500
    x = torch.randn(N, 4)                        # 4 node features
    ei = torch.randint(0, N, (2, E))
    re = torch.tensor([500.0])
    angle = torch.tensor([45.0])

    # edge_attr=None → computed on-the-fly from x[:, :3]
    data = Data(x=x, edge_index=ei, re=re, angle=angle)

    model.eval()
    with torch.no_grad():
        y = model(data)

    print(f"\n  Input:  {N} nodes, {E} edges")
    print(f"  Output: {y.shape}  (expected [{N}, 3])")
    print(f"  Range:  [{y.min():.6f}, {y.max():.6f}]")
    print("\nOK")

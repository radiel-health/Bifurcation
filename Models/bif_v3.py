"""
GNN model v3 for bifurcation wall shear stress prediction.

Architecture changes from v2 (bif_v2.py):
  - GINEConv  →  GATv2Conv  (attention-weighted message passing; 4 heads)
  - FlowEncoder: 2-layer 64-hidden → 3-layer 128-hidden (more Re-regime capacity)
  - node_feat_dim: 10 → 18  (adds 8 Laplacian Positional Encoding features)
  - Dropout(0.15) on conv outputs (replaces GINE-internal MLP dropout)
  - BayesianLinear output head and FiLM modulation unchanged

Usage:
    python -m Bifurcation.Models.bif_v3     # smoke test
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Optional Bayesian layer (torchbnn)
# ---------------------------------------------------------------------------

try:
    import torchbnn
    _BAYESIAN_AVAILABLE = True
except ImportError:
    _BAYESIAN_AVAILABLE = False

    import warnings
    warnings.warn(
        "torchbnn not installed — BayesianLinear replaced with plain Linear + MC-Dropout. "
        "Install with: pip install torchbnn",
        stacklevel=2,
    )


def _make_output_head(in_dim: int, out_dim: int, dropout_rate: float = 0.1):
    if _BAYESIAN_AVAILABLE:
        return torchbnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                    in_features=in_dim, out_features=out_dim)
    else:
        return nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_dim, out_dim),
        )


def _kl_loss(model: nn.Module) -> torch.Tensor:
    if not _BAYESIAN_AVAILABLE:
        return torch.tensor(0.0)
    bkl = torchbnn.BKLLoss(reduction="mean", last_layer_only=False)
    return bkl(model)


# ============================================================================
# Sub-modules
# ============================================================================

class FlowEncoderV3(nn.Module):
    """
    3-layer flow encoder for [log10(Re), angle_rad] → 64-D context.

    Deeper than v2 (2-layer) to better distinguish Re regimes:
      laminar (Re300-700) / transitional (Re800-1400) / turbulent (Re1500+).
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128,
                 output_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, flow_params):
        x = F.relu(self.ln1(self.fc1(flow_params)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


class GATv2ConvBlock(nn.Module):
    """
    One GATv2Conv + LayerNorm + Dropout, packaged for gradient checkpointing.

    Wrapping in a Module lets torch.utils.checkpoint.checkpoint recompute
    the forward pass during backward, trading compute for memory.
    On 40K-node graphs, 8 unchecked layers need ~14 GB; checkpointing
    keeps peak memory to ~4-5 GB.
    """

    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int,
                 dropout: float = 0.15):
        super().__init__()
        self.conv = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=head_dim,
            heads=num_heads,
            concat=True,
            edge_dim=hidden_dim,
            dropout=dropout,
            add_self_loops=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        x_out = self.conv(x, edge_index, edge_attr=edge_attr)
        x_out = self.norm(x_out)
        x_out = F.relu(x_out)
        return self.drop(x_out)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation:  h_out = γ(ctx) ⊙ h + β(ctx)"""

    def __init__(self, context_dim: int = 64, feature_dim: int = 384):
        super().__init__()
        self.gamma_net = nn.Linear(context_dim, feature_dim)
        self.beta_net  = nn.Linear(context_dim, feature_dim)

    def forward(self, h_geom, context, batch):
        gamma = self.gamma_net(context[batch])
        beta  = self.beta_net(context[batch])
        return gamma * h_geom + beta


# ============================================================================
# V3 model
# ============================================================================

class BifurcationWSSPredictorV3(nn.Module):
    """
    V3 GNN for predicting 3-D wall shear stress on bifurcation surfaces.

    V3 improvements over v2:
      - GATv2Conv: attention-based aggregation (each neighbor weighted by
        learned attention score that depends on node+edge features).
      - 3-layer FlowEncoder with 128-D hidden for better Re-regime separation.
      - 18-D node features: 10 physics features + 8 Laplacian PE features.

    Args:
        node_feat_dim:  input node feature dim  (18 for v3)
        edge_feat_dim:  input edge feature dim  (4)
        hidden_dim:     GNN hidden width        (384)
        num_heads:      GATv2Conv attention heads (4; head_dim = hidden_dim//4)
        out_channels:   prediction dim          (3)
        num_layers:     GATv2Conv layer count   (8)
        context_dim:    FiLM context width      (64)
        flow_param_dim: flow-encoder input      (2)
    """

    def __init__(
        self,
        node_feat_dim:  int = 18,
        edge_feat_dim:  int = 4,
        hidden_dim:     int = 384,
        num_heads:      int = 4,
        out_channels:   int = 3,
        num_layers:     int = 8,
        context_dim:    int = 64,
        flow_param_dim: int = 2,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        head_dim = hidden_dim // num_heads

        self.hidden_dim = hidden_dim

        # -- flow encoder (3-layer, 128-hidden) --
        self.flow_encoder = FlowEncoderV3(
            input_dim=flow_param_dim,
            hidden_dim=128,
            output_dim=context_dim,
        )

        # -- edge encoder --
        self.edge_encoder = nn.Linear(edge_feat_dim, hidden_dim)

        # -- input projection: node_feat_dim → hidden_dim --
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        # -- GATv2Conv stack (gradient-checkpointed blocks) --
        self.conv_blocks = nn.ModuleList([
            GATv2ConvBlock(hidden_dim, num_heads, head_dim, dropout=0.15)
            for _ in range(num_layers)
        ])

        # -- FiLM --
        self.film = FiLMLayer(context_dim=context_dim, feature_dim=hidden_dim)

        # -- stochastic output head --
        self.output_head = _make_output_head(hidden_dim, out_channels)

        self._mc_mode = False

    # --------------------------------------------------------------------- #

    def enable_mc_mode(self):
        self._mc_mode = True
        if not _BAYESIAN_AVAILABLE:
            self.output_head.train()

    def disable_mc_mode(self):
        self._mc_mode = False
        self.eval()

    # --------------------------------------------------------------------- #

    def forward(self, data):
        """
        Args:
            data: PyG Batch / Data with fields:
                  x, edge_index, edge_attr, re, angle, batch

        Returns:
            y_pred: [num_nodes, 3]
        """
        x          = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr
        batch      = (data.batch if hasattr(data, "batch") and data.batch is not None
                      else torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        batch_size = int(batch.max().item()) + 1
        re_vals    = data.re.view(batch_size)
        angle_vals = data.angle.view(batch_size)
        log_re     = torch.log10(re_vals.clamp(min=1.0))
        angle_rad  = angle_vals * (3.14159265358979 / 180.0)
        flow_params = torch.stack([log_re, angle_rad], dim=1)  # [B, 2]

        # 1. Flow context (3-layer encoder)
        context = self.flow_encoder(flow_params)        # [B, context_dim]

        # 2. Encode edge features
        edge_attr = self.edge_encoder(edge_attr)        # [E, hidden_dim]

        # 3. Project input features
        x = self.input_proj(x)                          # [N, hidden_dim]

        # 4. GATv2Conv stack with residuals (gradient checkpointing saves ~10 GB)
        for block in self.conv_blocks:
            if self.training:
                x_out = grad_checkpoint(
                    block, x, edge_index, edge_attr, use_reentrant=False
                )
            else:
                x_out = block(x, edge_index, edge_attr)
            x = x_out + x                              # residual

        h_geom = x  # [N, hidden_dim]

        # 5. FiLM modulation
        h_fused = self.film(h_geom, context, batch)    # [N, hidden_dim]

        # 6. Output
        return self.output_head(h_fused)                # [N, 3]

    # --------------------------------------------------------------------- #

    def kl_loss(self) -> torch.Tensor:
        return _kl_loss(self)


# ============================================================================
# Utilities
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: BifurcationWSSPredictorV3):
    print("=" * 70)
    print("BIFURCATION WSS PREDICTOR V3 — ARCHITECTURE SUMMARY")
    print("=" * 70)
    components = {
        "Flow Encoder (3-layer)": model.flow_encoder,
        "Input Projection":       model.input_proj,
        "Edge Encoder":           model.edge_encoder,
        "GATv2Conv Blocks":       model.conv_blocks,
        "FiLM Layer":             model.film,
        "Output Head":            model.output_head,
    }
    total = count_parameters(model)
    for name, mod in components.items():
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        print(f"  {name:30s} {n:>12,} params")
    print(f"  {'─' * 44}")
    print(f"  {'TOTAL':30s} {total:>12,} params")
    print(f"  Model size ≈ {total * 4 / 1024**2:.1f} MB (float32)")
    bayesian_str = "BayesianLinear" if _BAYESIAN_AVAILABLE else "Linear+MCDropout"
    print(f"  Output head type: {bayesian_str}")
    print("=" * 70)


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    print("Smoke-testing BifurcationWSSPredictorV3 …\n")

    model = BifurcationWSSPredictorV3()
    get_model_summary(model)

    N, E = 500, 1500
    x  = torch.randn(N, 18)      # 18 features (v3: 10 physics + 8 LPE)
    ei = torch.randint(0, N, (2, E))
    ea = torch.randn(E, 4)
    re    = torch.tensor([500.0])
    angle = torch.tensor([45.0])

    data = Data(x=x, edge_index=ei, edge_attr=ea, re=re, angle=angle)

    model.eval()
    with torch.no_grad():
        y = model(data)
    print(f"\n  Deterministic pass:  {y.shape}  range=[{y.min():.4f}, {y.max():.4f}]")

    # MC pass
    model.enable_mc_mode()
    preds = []
    for _ in range(20):
        with torch.no_grad():
            preds.append(model(data))
    preds = torch.stack(preds, dim=0)
    mean  = preds.mean(0)
    std   = preds.std(0)
    print(f"  MC mean (20 passes): {mean.shape}  std mean={std.mean():.6f}")
    print("\nOK ✓")

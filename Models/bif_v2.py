"""
GNN model v2 for bifurcation wall shear stress prediction.

Architecture changes from v1 (model.py):
  - node_feat_dim: 3  →  10  (physics-informed features)
  - hidden_dim:    256 →  384
  - num_layers:    6   →  8   (residual connections from layer 0)
  - Output head:   Linear  →  BayesianLinear (uncertainty quantification)
    Falls back to plain Linear with MC-Dropout if torchbnn is not installed.

Uncertainty inference:
  - Call model.enable_mc_mode() before running N forward passes
  - The stochasticity comes from the BayesianLinear weight sampling (or dropout)
  - Mean ± std across N passes gives per-node uncertainty estimates

Usage:
    python -m Bifurcation.Models.bif_v2     # smoke test
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
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
    """
    Create the stochastic output head.

    If torchbnn is available:  BayesianLinear (samples weights from posterior).
    Otherwise:                 Linear with Dropout (MC-Dropout approximation).
    """
    if _BAYESIAN_AVAILABLE:
        return torchbnn.BayesianLinear(in_dim, out_dim)
    else:
        return nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_dim, out_dim),
        )


def _kl_loss(model: nn.Module) -> torch.Tensor:
    """
    Compute KL divergence term for BayesianLinear layers.

    Returns zero if torchbnn is not available (plain Linear head).
    """
    if not _BAYESIAN_AVAILABLE:
        return torch.tensor(0.0)
    bkl = torchbnn.BKLLoss(reduction="mean", last_layer_only=False)
    return bkl(model)


# ============================================================================
# Sub-modules (reuse v1 design, parameterised for v2 dims)
# ============================================================================

class FlowEncoder(nn.Module):
    """Encode [log10(Re), angle_rad] → context vector (64-D)."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64,
                 output_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, flow_params):
        x = F.relu(self.ln1(self.fc1(flow_params)))
        return self.fc2(x)


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
# V2 model
# ============================================================================

class BifurcationWSSPredictorV2(nn.Module):
    """
    V2 GNN for predicting 3-D wall shear stress on bifurcation / carotid
    vessel surfaces.

    Improvements over v1 (BifurcationWSSPredictor):
      - 10-dimensional physics-informed node features instead of 3
      - Wider (384) and deeper (8 layers) GINEConv stack
      - Residual connections from layer 0 (requires input projection)
      - BayesianLinear output head for uncertainty quantification
      - enable_mc_mode() / disable_mc_mode() for inference

    Args:
        node_feat_dim:  input node feature dim  (10 for v2)
        edge_feat_dim:  input edge feature dim  (4: dist,dx,dy,dz)
        hidden_dim:     GNN hidden width        (384)
        out_channels:   prediction dim          (3: wss_x,y,z)
        num_layers:     GINE layer count        (8)
        context_dim:    FiLM context width      (64)
        flow_param_dim: flow-encoder input      (2)
    """

    def __init__(
        self,
        node_feat_dim: int = 10,
        edge_feat_dim: int = 4,
        hidden_dim:    int = 384,
        out_channels:  int = 3,
        num_layers:    int = 8,
        context_dim:   int = 64,
        flow_param_dim: int = 2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # -- flow encoder --
        self.flow_encoder = FlowEncoder(
            input_dim=flow_param_dim,
            hidden_dim=64,
            output_dim=context_dim,
        )

        # -- edge encoder --
        self.edge_encoder = nn.Linear(edge_feat_dim, hidden_dim)

        # -- input projection: node_feat_dim → hidden_dim  (enables residuals from layer 0) --
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        # -- GINE stack --
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=hidden_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # -- FiLM --
        self.film = FiLMLayer(context_dim=context_dim, feature_dim=hidden_dim)

        # -- stochastic output head --
        self.output_head = _make_output_head(hidden_dim, out_channels)

        # -- MC mode flag --
        self._mc_mode = False

    # --------------------------------------------------------------------- #

    def enable_mc_mode(self):
        """
        Enable stochastic forward pass for MC uncertainty estimation.

        For BayesianLinear: weights are sampled each forward call (always on).
        For MC-Dropout fallback: puts Dropout layers in train() mode.
        """
        self._mc_mode = True
        if not _BAYESIAN_AVAILABLE:
            # Enable dropout by setting train mode on the output head only
            self.output_head.train()

    def disable_mc_mode(self):
        """Disable MC stochasticity (deterministic mean prediction)."""
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

        # 1. Flow context
        context = self.flow_encoder(flow_params)        # [B, context_dim]

        # 2. Encode edge features
        edge_attr = self.edge_encoder(edge_attr)        # [E, hidden_dim]

        # 3. Project input features to hidden_dim (enables residuals from layer 0)
        x = self.input_proj(x)                          # [N, hidden_dim]

        # 4. GINE stack with residuals from every layer
        for conv, norm in zip(self.convs, self.norms):
            x_out = conv(x, edge_index, edge_attr)
            x_out = norm(x_out)
            x_out = F.relu(x_out)
            x = x_out + x                              # residual at every layer

        h_geom = x  # [N, hidden_dim]

        # 5. FiLM modulation
        h_fused = self.film(h_geom, context, batch)    # [N, hidden_dim]

        # 6. Output (stochastic if BayesianLinear / MC-Dropout)
        return self.output_head(h_fused)                # [N, 3]

    # --------------------------------------------------------------------- #

    def kl_loss(self) -> torch.Tensor:
        """KL divergence for BayesianLinear head (used in train_v2.py)."""
        return _kl_loss(self)


# ============================================================================
# Utilities
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: BifurcationWSSPredictorV2):
    print("=" * 70)
    print("BIFURCATION WSS PREDICTOR V2 — ARCHITECTURE SUMMARY")
    print("=" * 70)
    components = {
        "Flow Encoder":   model.flow_encoder,
        "Input Projection": model.input_proj,
        "Edge Encoder":   model.edge_encoder,
        "GINE Stack":     model.convs,
        "LayerNorms":     model.norms,
        "FiLM Layer":     model.film,
        "Output Head":    model.output_head,
    }
    total = count_parameters(model)
    for name, mod in components.items():
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        print(f"  {name:25s} {n:>12,} params")
    print(f"  {'─' * 40}")
    print(f"  {'TOTAL':25s} {total:>12,} params")
    print(f"  Model size ≈ {total * 4 / 1024**2:.1f} MB (float32)")
    bayesian_str = "BayesianLinear" if _BAYESIAN_AVAILABLE else "Linear+MCDropout"
    print(f"  Output head type: {bayesian_str}")
    print("=" * 70)


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    print("Smoke-testing BifurcationWSSPredictorV2 …\n")

    model = BifurcationWSSPredictorV2()
    get_model_summary(model)

    N, E = 500, 1500
    x  = torch.randn(N, 10)      # 10 features (v2)
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
    preds = torch.stack(preds, dim=0)          # [20, N, 3]
    mean  = preds.mean(0)
    std   = preds.std(0)
    print(f"  MC mean (20 passes): {mean.shape}  std mean={std.mean():.6f}")
    print("\nOK ✓")

"""
GNN model v4 for bifurcation wall shear stress prediction (pulsatile flow).

Architecture changes from v3 (bif_v3.py):
  - FlowEncoder input: 2-dim → 4-dim
      v3: [log10(Re), angle_rad]
      v4: [log10(Re), angle_rad, sin(2πφ), cos(2πφ)]
    where φ = cardiac phase ∈ [0, 1) (t mod 1.0 for 1 Hz waveform).
  - data.phase (shape [1]) added to input Data object.
  - All other components (GATv2Conv backbone, FiLM, BayesianLinear head)
    are unchanged from v3.

Usage:
    python -m Bifurcation.Models.bif_v4     # smoke test
"""

import math

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
# Sub-modules  (identical to v3)
# ============================================================================

class FlowEncoderV3(nn.Module):
    """
    3-layer flow encoder.

    V4: input_dim=4 by default — encodes
        [log10(Re), angle_rad, sin(2πφ), cos(2πφ)]
    so the model can predict WSS at any cardiac phase φ.
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 128,
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
# V4 model
# ============================================================================

class BifurcationWSSPredictorV4(nn.Module):
    """
    V4 GNN for predicting 3-D wall shear stress on bifurcation surfaces
    under pulsatile flow conditions.

    Extends V3 with cardiac phase conditioning:
      - flow_param_dim = 4: [log10(Re), angle_rad, sin(2πφ), cos(2πφ)]
      - data.phase (shape [1]) must be present on input Data objects.
      - All other architecture components are unchanged from V3.

    At inference, predict 20 phases per case and integrate to get:
      TAWSS = mean_t(|WSS(t)|)
      OSI   = 0.5 × (1 - |mean_t(WSS(t))| / TAWSS)

    Args:
        node_feat_dim:  input node feature dim  (10, physics features only)
        edge_feat_dim:  input edge feature dim  (4)
        hidden_dim:     GNN hidden width        (384)
        num_heads:      GATv2Conv heads         (4; head_dim = hidden_dim//4)
        out_channels:   prediction dim          (3)
        num_layers:     GATv2Conv layer count   (8)
        context_dim:    FiLM context width      (64)
        flow_param_dim: flow-encoder input      (4)
    """

    def __init__(
        self,
        node_feat_dim:  int = 10,
        edge_feat_dim:  int = 4,
        hidden_dim:     int = 384,
        num_heads:      int = 4,
        out_channels:   int = 3,
        num_layers:     int = 8,
        context_dim:    int = 64,
        flow_param_dim: int = 4,
        cluster_parts:  int | None = None,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        head_dim = hidden_dim // num_heads

        self.hidden_dim = hidden_dim

        # -- flow encoder (3-layer, 128-hidden, 4-dim input for v4) --
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
        self.cluster_parts = cluster_parts

    # --------------------------------------------------------------------- #

    def enable_mc_mode(self):
        self._mc_mode = True
        if not _BAYESIAN_AVAILABLE:
            self.output_head.train()

    def disable_mc_mode(self):
        self._mc_mode = False
        self.eval()

    # --------------------------------------------------------------------- #

    def _forward_clustered(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor,
        batch:      torch.Tensor,
        num_parts:  int,
    ) -> torch.Tensor:
        num_nodes  = x.size(0)
        batch_size = int(batch.max().item()) + 1
        h_out      = torch.zeros_like(x)

        for g in range(batch_size):
            g_mask  = batch == g
            g_nodes = g_mask.nonzero(as_tuple=True)[0]
            n_g     = g_nodes.size(0)

            g_local = torch.full(
                (num_nodes,), -1, dtype=torch.long, device=x.device
            )
            g_local[g_nodes] = torch.arange(n_g, device=x.device)

            src, dst  = edge_index
            g_emask   = g_mask[src] & g_mask[dst]
            g_ei      = torch.stack(
                [g_local[src[g_emask]], g_local[dst[g_emask]]], dim=0
            )
            g_ea      = edge_attr[g_emask]
            x_g       = x[g_nodes]

            perm   = torch.randperm(n_g, device=x.device)
            c_size = math.ceil(n_g / num_parts)
            h_g    = torch.zeros_like(x_g)

            for p in range(num_parts):
                c_local = perm[p * c_size : (p + 1) * c_size]
                n_c     = c_local.size(0)
                if n_c == 0:
                    continue

                c_remap = torch.full(
                    (n_g,), -1, dtype=torch.long, device=x.device
                )
                c_remap[c_local] = torch.arange(n_c, device=x.device)

                src_g, dst_g = g_ei
                intra        = (c_remap[src_g] >= 0) & (c_remap[dst_g] >= 0)
                c_ei         = torch.stack(
                    [c_remap[src_g[intra]], c_remap[dst_g[intra]]], dim=0
                )
                c_ea         = g_ea[intra]
                x_c          = x_g[c_local]

                for block in self.conv_blocks:
                    x_c_out = block(x_c, c_ei, c_ea)
                    x_c = x_c_out + x_c

                h_g[c_local] = x_c

            h_out[g_nodes] = h_g

        return h_out

    # --------------------------------------------------------------------- #

    def forward(self, data):
        """
        Args:
            data: PyG Batch / Data with fields:
                  x, edge_index, edge_attr, re, angle, phase, batch

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
        phase_vals = data.phase.view(batch_size)

        log_re    = torch.log10(re_vals.clamp(min=1.0))
        angle_rad = angle_vals * (math.pi / 180.0)
        sin_phi   = torch.sin(2.0 * math.pi * phase_vals)
        cos_phi   = torch.cos(2.0 * math.pi * phase_vals)

        flow_params = torch.stack([log_re, angle_rad, sin_phi, cos_phi], dim=1)  # [B, 4]

        # 1. Flow context
        context = self.flow_encoder(flow_params)        # [B, context_dim]

        # 2. Encode edge features
        edge_attr = self.edge_encoder(edge_attr)        # [E, hidden_dim]

        # 3. Project input features
        x = self.input_proj(x)                          # [N, hidden_dim]

        # 4. GATv2Conv stack with residuals
        if self.cluster_parts is not None:
            h_geom = self._forward_clustered(
                x, edge_index, edge_attr, batch, self.cluster_parts
            )
        else:
            for block in self.conv_blocks:
                x_out = block(x, edge_index, edge_attr)
                x = x_out + x
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


def get_model_summary(model: BifurcationWSSPredictorV4):
    print("=" * 70)
    print("BIFURCATION WSS PREDICTOR V4 (PULSATILE) — ARCHITECTURE SUMMARY")
    print("=" * 70)
    components = {
        "Flow Encoder (4-dim input)": model.flow_encoder,
        "Input Projection":           model.input_proj,
        "Edge Encoder":               model.edge_encoder,
        "GATv2Conv Blocks":           model.conv_blocks,
        "FiLM Layer":                 model.film,
        "Output Head":                model.output_head,
    }
    total = count_parameters(model)
    for name, mod in components.items():
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        print(f"  {name:32s} {n:>12,} params")
    print(f"  {'─' * 46}")
    print(f"  {'TOTAL':32s} {total:>12,} params")
    print(f"  Model size ≈ {total * 4 / 1024**2:.1f} MB (float32)")
    bayesian_str = "BayesianLinear" if _BAYESIAN_AVAILABLE else "Linear+MCDropout"
    print(f"  Output head type: {bayesian_str}")
    print(f"  Flow params: [log10(Re), angle_rad, sin(2πφ), cos(2πφ)]")
    print("=" * 70)


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    print("Smoke-testing BifurcationWSSPredictorV4 …\n")

    model = BifurcationWSSPredictorV4()
    get_model_summary(model)

    N, E = 500, 1500
    x     = torch.randn(N, 10)      # 10 physics features (LPE disabled)
    ei    = torch.randint(0, N, (2, E))
    ea    = torch.randn(E, 4)
    re    = torch.tensor([500.0])
    angle = torch.tensor([45.0])
    phase = torch.tensor([0.25])    # quarter-cycle (trough)

    data = Data(x=x, edge_index=ei, edge_attr=ea, re=re, angle=angle, phase=phase)

    model.eval()
    with torch.no_grad():
        y = model(data)
    print(f"\n  Deterministic pass:  {y.shape}  range=[{y.min():.4f}, {y.max():.4f}]")
    assert y.shape == (N, 3), f"Expected ({N}, 3), got {y.shape}"

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

    # Verify phase-conditioning produces different outputs at different phases
    model.disable_mc_mode()
    model.eval()
    results = {}
    for phi in [0.0, 0.25, 0.5, 0.75]:
        d = Data(x=x, edge_index=ei, edge_attr=ea, re=re, angle=angle,
                 phase=torch.tensor([phi]))
        with torch.no_grad():
            results[phi] = model(d).mean().item()
    print(f"\n  Phase-conditioned mean output:")
    for phi, val in results.items():
        print(f"    φ={phi:.2f}  →  mean WSS = {val:.4f}")
    assert len(set(round(v, 4) for v in results.values())) > 1, \
        "Phase conditioning has no effect — check forward()"

    print("\nOK ✓")

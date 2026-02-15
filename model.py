"""
GNN model for bifurcation wall shear stress prediction.

Architecture (GINEConv + FiLM conditioning):

    flow_params [log10(Re), angle_rad]
        │
        ▼
    FlowEncoder ──► context  (64-D)
                         │
                         │ FiLM  γ·h + β
                         ▼
    node_feat (3) ──► 6× GINEConv+LayerNorm+ReLU+Residual (256-D) ──► h_geom
        +                                                                   │
    edge_attr (4)                                                           ▼
                                                            h_fused = FiLM(h_geom, ctx)
                                                                           │
                                                                    Linear(256→3) ──► [wss_x, wss_y, wss_z]

GINEConv natively uses edge features at every layer. No line graph needed.

Usage:
    python -m Bifurcation.model          # smoke-test with dummy data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.data import Data, Batch


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
      - **GINEConv** backbone (edge-conditioned message passing at every layer)
      - **FiLM** conditioning on flow parameters (Re, bifurcation angle)
      - **Residual connections** + LayerNorm for training stability

    Args:
        node_feat_dim:      input node feature dim   (3: x,y,z)
        edge_feat_dim:      input edge feature dim   (4: dist,dx,dy,dz)
        hidden_dim:         GNN hidden width         (256)
        out_channels:       prediction dim           (3: wss_x,y,z)
        num_layers:         number of GINE layers    (6)
        context_dim:        FiLM context width       (64)
        flow_param_dim:     flow-encoder input       (2)
    """

    def __init__(
        self,
        node_feat_dim: int = 3,
        edge_feat_dim: int = 4,
        hidden_dim: int = 256,
        out_channels: int = 3,
        num_layers: int = 6,
        context_dim: int = 64,
        flow_param_dim: int = 2,
    ):
        super().__init__()

        # -- flow encoder --
        self.flow_encoder = FlowEncoder(
            input_dim=flow_param_dim,
            hidden_dim=64,
            output_dim=context_dim,
        )

        # -- edge encoder: project 4-D edge features to hidden_dim --
        self.edge_encoder = nn.Linear(edge_feat_dim, hidden_dim)

        # -- GINE stack with residual connections + LayerNorm --
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer: node_feat_dim → hidden_dim
        mlp_in = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs.append(GINEConv(nn=mlp_in, edge_dim=hidden_dim, train_eps=True))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Remaining layers: hidden_dim → hidden_dim with residuals
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=hidden_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # -- FiLM --
        self.film = FiLMLayer(context_dim=context_dim, feature_dim=hidden_dim)

        # -- output head --
        self.lin = nn.Linear(hidden_dim, out_channels)

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
        edge_attr = data.edge_attr       # [E, edge_feat_dim]
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

        # 2. Encode edge features once
        edge_attr = self.edge_encoder(edge_attr)      # [E, hidden_dim]

        # 3. GINE stack with residuals
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_out = conv(x, edge_index, edge_attr)
            x_out = norm(x_out)
            x_out = F.relu(x_out)
            # Add residual connection (skip first layer since dimensions change)
            if i > 0:
                x_out = x_out + x
            x = x_out

        h_geom = x  # [N, hidden_dim]

        # 4. FiLM modulation
        h_fused = self.film(h_geom, context, batch)

        # 5. Output
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
        "Edge Encoder":    model.edge_encoder,
        "GINE Stack":      model.convs,
        "LayerNorms":      model.norms,
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

"""
Configuration for Bifurcation WSS prediction v3.

Key changes from config_v2.py (v2 → v3):
  - node_feat_dim: 10 → 18  (adds 8 Laplacian Positional Encoding features)
  - lpe_k = 8               (k eigenvectors of normalised graph Laplacian)
  - GATv2Conv replaces GINEConv  (attention-based aggregation)
  - num_heads = 4           (GATv2Conv multi-head attention)
  - FlowEncoder: 2-layer → 3-layer, hidden 64 → 128 (more Re-regime capacity)
  - Separate output directories (Models_v3/, ProcessedData_v3/)
"""

from pathlib import Path
import math


class ConfigV3:
    # =========================================================================
    # PATHS
    # =========================================================================

    project_root = Path(__file__).parent

    data_root = project_root.parent / "Data" / "Bifurcation" / "results" / "openFoam(1e-5)"

    processed_data_dir = project_root / "ProcessedData_v3"
    models_dir         = project_root / "Models_v3"
    predictions_dir    = project_root / "predictions_v3"
    figures_dir        = project_root / "figures_v3"
    results_dir        = project_root / "results_v3"

    # =========================================================================
    # DATA DESCRIPTION  (same operating range as v2)
    # =========================================================================

    geometry_folders = [
        "bifurcation_angle30_500_ascii",
        "bifurcation_angle30_750_ascii",
        "bifurcation_angle30_1000_ascii",
        "bifurcation_angle45_500_ascii",
        "bifurcation_angle45_750_ascii",
        "bifurcation_angle45_1000_ascii",
        "bifurcation_angle60_500_ascii",
        "bifurcation_angle60_750_ascii",
        "bifurcation_angle60_1000_ascii",
    ]

    re_min    = 300   # Re100/200 excluded (viscous-dominated, OOD)
    re_max    = 2100
    re_step   = 100
    re_values = list(range(re_min, re_max + 1, re_step))  # 19 values

    mesh_re        = "Re100"
    angles         = [30, 45, 60]
    mesh_levels    = [500, 750, 1000]
    num_geometries = 9
    num_re         = 19
    total_samples  = 171  # 9 × 19

    # Re brackets for stratified splitting
    re_brackets = [(300, 700), (800, 1400), (1500, 2100)]

    # =========================================================================
    # FEATURE DIMENSIONS  (v3)
    # =========================================================================

    # V3 node features (18 = 10 physics + 8 LPE):
    #   0-2:  x, y, z             (face centroid coordinates)
    #   3:    dist_to_junction
    #   4:    arc_length
    #   5:    branch_depth
    #   6-8:  nx, ny, nz
    #   9:    local_curvature
    #   10-17: |λ_1|...|λ_8|     (absolute Laplacian eigenvectors)
    lpe_k         = 8
    node_feat_dim = 18   # 10 physics + lpe_k

    # Edge / flow / output dims unchanged
    edge_feat_dim  = 4
    flow_param_dim = 2
    output_dim     = 3

    # =========================================================================
    # MODEL HYPERPARAMETERS
    # =========================================================================

    hidden_dim  = 384
    num_layers  = 8
    context_dim = 64
    num_heads   = 4    # GATv2Conv attention heads; head_dim = hidden_dim // num_heads = 96

    # =========================================================================
    # TRAINING HYPERPARAMETERS
    # =========================================================================

    batch_size    = 1
    learning_rate = 1e-3
    weight_decay  = 1e-5
    num_epochs    = 500
    grad_clip     = 1.0

    scheduler_factor   = 0.5
    scheduler_patience = 10
    scheduler_min_lr   = 1e-6

    early_stop_patience = 50

    kl_weight = 0.025

    # Uniform component weights — Y is clinically irrelevant
    component_loss_weights = [1.0, 1.0, 1.0]

    use_amp = True

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    use_log_transform = True

    # =========================================================================
    # DATA SPLITTING
    # =========================================================================

    split_mode = "re_angle_stratified"

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def parse_geometry_folder(folder_name: str):
        """'bifurcation_angle45_750_ascii' → (45, 750)"""
        parts = folder_name.replace("bifurcation_angle", "").replace("_ascii", "")
        tokens = parts.split("_")
        return int(tokens[0]), int(tokens[1])

    @classmethod
    def get_geometry_path(cls, folder_name: str) -> Path:
        return cls.data_root / folder_name

    @classmethod
    def get_mesh_path(cls, folder_name: str) -> Path:
        return cls.data_root / folder_name / cls.mesh_re / "constant" / "polyMesh"

    @classmethod
    def angle_to_radians(cls, angle_deg: int) -> float:
        return angle_deg * math.pi / 180.0

    @classmethod
    def create_directories(cls):
        for d in [cls.processed_data_dir, cls.models_dir,
                  cls.predictions_dir, cls.figures_dir, cls.results_dir]:
            d.mkdir(parents=True, exist_ok=True)


config_v3 = ConfigV3()

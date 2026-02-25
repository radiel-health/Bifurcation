"""
Configuration for Bifurcation WSS prediction v2.

Key changes from config.py (v1):
  - node_feat_dim: 3  →  10  (physics-informed features)
  - hidden_dim:    256 →  384 (more capacity for richer input)
  - num_layers:    6   →  8
  - num_epochs:    100 →  500 (longer budget; early stopping guards)
  - early_stop_patience: 15 → 50
  - scheduler_patience:   5 → 10
  - Bayesian KL weight added (kl_weight = 0.025)
  - AMP training enabled
  - Separate output directories (Models_v2/, ProcessedData_v2/)
"""

from pathlib import Path
import math


class ConfigV2:
    # =========================================================================
    # PATHS  (v2 uses separate dirs so v1 artifacts are untouched)
    # =========================================================================

    project_root = Path(__file__).parent

    # Raw OpenFOAM data (same source as v1)
    data_root = project_root.parent / "Data" / "Bifurcation" / "results" / "openFoam(1e-5)"

    # v2-specific outputs
    processed_data_dir = project_root / "ProcessedData_v2"
    models_dir         = project_root / "Models_v2"
    predictions_dir    = project_root / "predictions_v2"
    figures_dir        = project_root / "figures_v2"
    results_dir        = project_root / "results_v2"

    # =========================================================================
    # DATA DESCRIPTION  (same as v1)
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

    re_min    = 100
    re_max    = 2100
    re_step   = 100
    re_values = list(range(re_min, re_max + 1, re_step))  # 21 values

    mesh_re       = "Re100"
    angles        = [30, 45, 60]
    mesh_levels   = [500, 750, 1000]
    num_geometries = 9
    num_re         = 21
    total_samples  = 189

    # =========================================================================
    # FEATURE DIMENSIONS  (updated for v2)
    # =========================================================================

    # V2 node features (10):
    #   0-2: x, y, z   (face centroid coordinates, raw — normalised by dataset)
    #   3:   dist_to_junction  (Euclidean distance to estimated bifurcation centre)
    #   4:   arc_length        (normalised 0→1 along assigned branch)
    #   5:   branch_depth      (0 = parent/inlet branch, 1 = daughter branch)
    #   6-8: nx, ny, nz        (unit surface normal computed from face vertices)
    #   9:   local_curvature   (mean angle between face normal and neighbour normals)
    node_feat_dim = 10

    # Edge features unchanged: [euclidean_distance, dx, dy, dz]
    edge_feat_dim = 4

    # Flow context: [log10(Re), angle_radians]  (same as v1, extensible for carotid)
    flow_param_dim = 2

    # Prediction target: WSS vector (wss_x, wss_y, wss_z)
    output_dim = 3

    # =========================================================================
    # MODEL HYPERPARAMETERS
    # =========================================================================

    hidden_dim   = 384   # ↑ from 256 — wider for richer node features
    num_layers   = 8     # ↑ from 6
    context_dim  = 64    # unchanged

    # =========================================================================
    # TRAINING HYPERPARAMETERS
    # =========================================================================

    batch_size   = 1     # Large graphs (~35-43K nodes) require batch_size=1
    learning_rate = 1e-3
    weight_decay  = 1e-5
    num_epochs    = 500  # ↑ from 100 (early stopping will trigger before this)
    grad_clip     = 1.0

    # Scheduler
    scheduler_factor   = 0.5
    scheduler_patience = 10   # ↑ from 5
    scheduler_min_lr   = 1e-6

    # Early stopping
    early_stop_patience = 50  # ↑ from 15

    # Bayesian uncertainty (KL regularisation on BayesianLinear output head)
    kl_weight = 0.025

    # Component loss weights [x, y, z]
    # Slightly upweight Y to combat its lower R² in v1
    component_loss_weights = [1.0, 2.0, 1.0]

    # AMP (Automatic Mixed Precision)  — set False on CPU-only machines
    use_amp = True

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    use_log_transform = True   # sign-preserving log1p for WSS targets (same as v1)

    # =========================================================================
    # DATA SPLITTING
    # =========================================================================

    # "re_angle_stratified": ensures each angle × Re-bracket combo appears in test
    # Re brackets: low=[100-700], mid=[800-1400], high=[1500-2100]
    split_mode = "re_angle_stratified"
    re_brackets = [(100, 700), (800, 1400), (1500, 2100)]

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


config_v2 = ConfigV2()

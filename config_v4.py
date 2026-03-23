"""
Configuration for Bifurcation WSS prediction v4 (pulsatile).

Key changes from config_v3.py (v3 → v4):
  - data_root: steady-state openFoam(1e-5) → pulsatile openFoam_pulsatile
  - data_root_steady: added (for polyMesh shared with steady-state cases)
  - re_values: Re300-2100 (19) → Re100-2100 (21, includes Re100/200)
  - re_brackets: adds (100, 200) bracket
  - num_timesteps = 20  (t=3.1,...,5.0, last 2 pulsatile cycles)
  - timestep_values: list of 20 float timesteps
  - flow_param_dim: 2 → 4  ([log10(Re), angle_rad, sin(2πφ), cos(2πφ)])
  - total_samples: 171 → 3780 (9 geo × 21 Re × 20 timesteps)
  - Separate output dirs (Models_v4/, ProcessedData_v4/)
"""

from pathlib import Path
import math


class ConfigV4:
    # =========================================================================
    # PATHS
    # =========================================================================

    project_root = Path(__file__).parent

    # Auto-detect results dir: local monorepo uses "Data/" (capital D),
    # EC2 standalone repo uses "data/" (lowercase d).
    def _find_results_dir(self) -> Path:
        candidates = [
            self.project_root.parent / "Data" / "Bifurcation" / "results",   # local monorepo
            self.project_root.parent / "data" / "Bifurcation" / "results",   # EC2 standalone
            Path.home() / "data" / "Bifurcation" / "results",                # EC2 home-relative
        ]
        # Prefer the candidate that actually has pulsatile data
        for p in candidates:
            if (p / "openFoam_pulsatile").exists():
                return p
        # Fall back to any existing results dir
        for p in candidates:
            if p.exists():
                return p
        return candidates[0]  # last resort; will surface FileNotFoundError later

    @property
    def data_root(self) -> Path:
        """Pulsatile simulation outputs."""
        return self._find_results_dir() / "openFoam_pulsatile"

    @property
    def data_root_steady(self) -> Path:
        """Steady-state cases — polyMesh shared with pulsatile runs."""
        return self._find_results_dir() / "openFoam(1e-5)"

    processed_data_dir = project_root / "ProcessedData_v4"
    models_dir         = project_root / "Models_v4"
    predictions_dir    = project_root / "predictions_v4"
    figures_dir        = project_root / "figures_v4"
    results_dir        = project_root / "results_v4"

    # =========================================================================
    # DATA DESCRIPTION
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

    # All 21 Re values — Re100/200 included (pulsatile instantaneous Re spans 50%–150%×Re_mean)
    re_min    = 100
    re_max    = 2100
    re_step   = 100
    re_values = list(range(re_min, re_max + 1, re_step))  # 21 values

    mesh_re        = "Re100"   # polyMesh source (shared across all Re for each geometry)
    angles         = [30, 45, 60]
    mesh_levels    = [500, 750, 1000]
    num_geometries = 9
    num_re         = 21
    num_timesteps  = 20        # last 2 pulsatile cycles: t=3.1, 3.2, ..., 5.0
    total_samples  = 9 * 21 * 20  # 3780

    # Pulsatile time snapshots (purgeWrite 20, writeInterval 0.1, t=3.0 to 5.0)
    timestep_values = [round(3.1 + 0.1 * i, 1) for i in range(20)]  # [3.1, 3.2, ..., 5.0]

    # Re brackets for stratified splitting (adds Re100/200 bracket vs v3)
    re_brackets = [(100, 200), (300, 700), (800, 1400), (1500, 2100)]

    # =========================================================================
    # FEATURE DIMENSIONS  (v4 — same physics features as v3)
    # =========================================================================

    # Node features (10 physics, LPE disabled — breaks rotation augmentation):
    #   0-2:  x, y, z             (face centroid coordinates)
    #   3:    dist_to_junction
    #   4:    arc_length
    #   5:    branch_depth
    #   6-8:  nx, ny, nz
    #   9:    local_curvature
    use_lpe       = False
    lpe_k         = 8
    node_feat_dim = 10 + (lpe_k if use_lpe else 0)  # 10

    edge_feat_dim  = 4
    flow_param_dim = 4   # [log10(Re), angle_rad, sin(2πφ), cos(2πφ)]
    output_dim     = 3

    # =========================================================================
    # MODEL HYPERPARAMETERS  (unchanged from v3)
    # =========================================================================

    hidden_dim  = 384
    num_layers  = 8
    context_dim = 64
    num_heads   = 4

    # =========================================================================
    # TRAINING HYPERPARAMETERS  (unchanged from v3)
    # =========================================================================

    batch_size    = 8
    learning_rate = 1e-3
    weight_decay  = 1e-5
    num_epochs    = 500
    grad_clip     = 1.0

    scheduler_factor   = 0.5
    scheduler_patience = 10
    scheduler_min_lr   = 1e-6

    early_stop_patience = 50

    kl_weight = 0.025

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

    def get_geometry_path(self, folder_name: str) -> Path:
        """Path to pulsatile geometry folder."""
        return self.data_root / folder_name

    def get_steady_geometry_path(self, folder_name: str) -> Path:
        """Path to steady-state geometry folder (for polyMesh)."""
        return self.data_root_steady / folder_name

    def get_mesh_path(self, folder_name: str) -> Path:
        """Path to polyMesh (from steady-state cases — shared mesh)."""
        return self.data_root_steady / folder_name / self.mesh_re / "constant" / "polyMesh"

    @staticmethod
    def angle_to_radians(angle_deg: int) -> float:
        return angle_deg * math.pi / 180.0

    @staticmethod
    def phase_from_time(t: float) -> float:
        """t → φ ∈ [0, 1)  (phase within cardiac cycle)"""
        return t % 1.0

    def create_directories(self):
        for d in [self.processed_data_dir, self.models_dir,
                  self.predictions_dir, self.figures_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)


config_v4 = ConfigV4()

"""
Central configuration for bifurcation WSS prediction pipeline.

All data paths, model hyperparameters, training settings, and
normalization options are defined here.
"""

from pathlib import Path
import math


class Config:
    # =========================================================================
    # PATHS
    # =========================================================================

    # Project root  (Bifurcation/ lives here)
    project_root = Path(__file__).parent

    # Raw OpenFOAM data
    data_root = project_root.parent / "Data" / "Bifurcation" / "results" / "openFoam(1e-5)"

    # Outputs
    processed_data_dir = project_root / "ProcessedData"
    models_dir = project_root / "Models"
    predictions_dir = project_root / "predictions"
    figures_dir = project_root / "figures"

    # =========================================================================
    # DATA DESCRIPTION
    # =========================================================================

    # All geometry folders (angle x mesh-element count)
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

    # Reynolds numbers (100 – 2100, step 100)
    re_min = 100
    re_max = 2100
    re_step = 100
    re_values = list(range(re_min, re_max + 1, re_step))  # 21 values

    # Mesh is stored only in Re100 for each geometry
    mesh_re = "Re100"

    # Angles present in the dataset (degrees)
    angles = [30, 45, 60]

    # Mesh element levels
    mesh_levels = [500, 750, 1000]

    # Total dataset size
    num_geometries = len(geometry_folders)          # 9
    num_re = len(re_values)                         # 21
    total_samples = num_geometries * num_re          # 189

    # =========================================================================
    # FEATURE DIMENSIONS
    # =========================================================================

    # Node features: face-centre coordinates + padding (x, y, z, 0)
    node_feat_dim = 4

    # Edge features: [dist, dx, dy, dz] — computed on-the-fly if not in stored data
    edge_feat_dim = 4

    # Flow context: [log10(Re), angle_radians]
    flow_param_dim = 2

    # Prediction target: WSS vector (wss_x, wss_y, wss_z)
    output_dim = 3

    # =========================================================================
    # MODEL HYPERPARAMETERS
    # =========================================================================

    # GINE stack
    hidden_dim = 256           # Hidden dimension for GINEConv layers
    num_layers = 6             # Number of GINEConv layers

    # Flow encoder / FiLM
    context_dim = 64

    # =========================================================================
    # TRAINING HYPERPARAMETERS
    # =========================================================================

    batch_size = 1            # Large graphs (~35-43K nodes) ⇒ one graph per step
    learning_rate = 1e-3
    weight_decay = 1e-5
    num_epochs = 100
    grad_clip = 1.0

    # Scheduler
    scheduler_factor = 0.5
    scheduler_patience = 5
    scheduler_min_lr = 1e-6

    # Early stopping
    early_stop_patience = 15

    # Loss
    magnitude_loss_weight = 0.1   # λ for ||pred|| vs ||target|| penalty

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    use_log_transform = True   # sign-preserving log1p for WSS targets

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    # Junction detection - apex only (bifurcation point)
    curvature_percentile = 85           # Threshold for high-curvature detection
    min_junction_cluster_size = 50      # Minimum cell count per junction
    min_junction_surface_area = 1e-6    # Filter out tiny regions
    max_junction_aspect_ratio = 3.0     # Avoid elongated streaks
    use_spatial_clustering = True       # Use DBSCAN for robust clustering
    dbscan_eps = 0.003                  # Spatial proximity threshold

    # Visualization settings
    junction_highlight_color = "red"
    junction_zoom_factor = 1.5
    annotation_font_size = 12

    # Export options
    include_point_data = True           # Interpolate WSS to vertices for glyphs
    include_curvature_field = True      # Export curvature diagnostic

    # Optional paths
    junction_points_file = None         # Override with JSON path

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def parse_geometry_folder(folder_name: str):
        """
        Extract angle and mesh level from folder name.

        Example: 'bifurcation_angle45_750_ascii' → (45, 750)
        """
        parts = folder_name.replace("bifurcation_angle", "").replace("_ascii", "")
        # parts is e.g. '45_750'
        tokens = parts.split("_")
        angle = int(tokens[0])
        mesh_level = int(tokens[1])
        return angle, mesh_level

    @classmethod
    def get_geometry_path(cls, folder_name: str) -> Path:
        return cls.data_root / folder_name

    @classmethod
    def get_mesh_path(cls, folder_name: str) -> Path:
        """Return path to polyMesh directory for a geometry."""
        return cls.data_root / folder_name / cls.mesh_re / "constant" / "polyMesh"

    @classmethod
    def angle_to_radians(cls, angle_deg: int) -> float:
        return angle_deg * math.pi / 180.0

    @classmethod
    def normalize_re(cls, re: float) -> float:
        """Normalise Reynolds number to log10 scale."""
        return math.log10(re)

    @classmethod
    def create_directories(cls):
        """Create all output directories."""
        for d in [cls.processed_data_dir, cls.models_dir,
                  cls.predictions_dir, cls.figures_dir]:
            d.mkdir(parents=True, exist_ok=True)


config = Config()

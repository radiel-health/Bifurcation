"""
Configuration file for bifurcation WSS prediction model

Adapted from LidDrivenCavity model for bifurcation geometry
"""
from pathlib import Path


class Config:
    """
    Configuration for bifurcation WSS prediction model
    """

    # Root directories
    project_root = Path(__file__).parent
    
    # Data paths - use project ProcessedData directory
    data_root = project_root  # Changed to use local project root
    processed_data_dir = project_root / "ProcessedData"
    
    # Output directories
    checkpoint_dir = project_root / "Models"
    results_dir = project_root / "results"
    figures_dir = project_root / "figures"
    analysis_dir = project_root / "analysis_results"
    
    # Create directories
    for dir_path in [processed_data_dir, checkpoint_dir, results_dir, 
                     figures_dir, analysis_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # ========================================================================
    # BIFURCATION GEOMETRY SPECIFICATIONS
    # ========================================================================
    
    # Available bifurcation angles (degrees)
    bifurcation_angles = [30, 45, 60]
    
    # Available mesh refinements (number of elements)
    mesh_refinements = [500, 750, 1000]
    
    # Reynolds numbers (from simulation results)
    re_min = 100
    re_max = 2100
    re_step = 100
    re_values = list(range(re_min, re_max + 1, re_step))  # [100, 200, ..., 2100]
    
    # Case naming pattern
    case_pattern = "bifurcation_angle{angle}_{mesh}_ascii"
    
    # Total number of cases
    num_angles = len(bifurcation_angles)
    num_meshes = len(mesh_refinements)
    num_re_values = len(re_values)
    
    # Approximate domain size (from geometry analysis)
    # These are approximations - actual coordinates normalized per-case
    domain_sizes = {
        "Lx": 0.94,   # X extent
        "Ly": 0.15,   # Y extent (vessel diameter)
        "Lz": 1.31,   # Z extent (flow direction)
    }
    
    # ========================================================================
    # BIFURCATION REGION DEFINITIONS
    # ========================================================================
    
    # Z-coordinate thresholds for region classification (relative to z_range)
    region_thresholds = {
        'inlet_end': 0.25,      # Bottom 25% is inlet
        'critical_start': 0.35,  # Middle 35-60% is critical region
        'critical_end': 0.60,
        'outlet_start': 0.65,    # Top 65%+ are outlets
    }
    
    # Region labels
    regions = ['inlet', 'critical', 'outlet_left', 'outlet_right', 'other']
    
    # ========================================================================
    # CSV DATA FORMAT
    # ========================================================================
    
    # Column names in wall_wss.csv files
    csv_columns = {
        'coords': ['x', 'y', 'z'],
        'pressure': ['p'],
        'wss_components': ['wss_x', 'wss_y', 'wss_z'],
        'wss_magnitude': ['wss_mag']
    }
    
    # ========================================================================
    # FEATURE ENGINEERING
    # ========================================================================
    
    # Number of input features per node
    # Proposed features (14 total):
    # 0-2: x, y, z (normalized coordinates)
    # 3: Reynolds number (normalized)
    # 4: Bifurcation angle (normalized: 30/60, 45/60, 60/60)
    # 5: Mesh refinement (normalized: 500/1000, 750/1000, 1000/1000)
    # 6-9: Region one-hot encoding (inlet, critical, outlet_left, outlet_right)
    # 10: Distance to apex (normalized)
    # 11: Radial distance from centerline (normalized)
    # 12: Angle around centerline (radians, -pi to pi)
    # 13: Local curvature indicator (0 for straight inlet, 1 for curved critical/outlets)
    
    num_features = 14
    
    # Target: WSS magnitude (1 value per node)
    num_targets = 1
    
    # ========================================================================
    # MODEL ARCHITECTURE
    # ========================================================================
    
    # Graph neural network architecture
    model_type = "GraphSAGE"  # or "GCN", "GAT"
    
    # Hidden dimensions
    hidden_dim = 128
    num_layers = 4
    dropout = 0.1
    
    # Activation function
    activation = "relu"  # or "elu", "gelu"
    
    # ========================================================================
    # TRAINING SETTINGS
    # ========================================================================
    
    # Data splits
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    
    # Training hyperparameters
    batch_size = 8
    learning_rate = 1e-4
    weight_decay = 1e-5
    num_epochs = 200
    patience = 20  # Early stopping patience
    
    # Loss function
    loss_fn = "mse"  # or "mae", "huber"
    
    # Optimizer
    optimizer_name = "adam"  # or "adamw", "sgd"
    
    # Learning rate scheduler
    use_scheduler = True
    scheduler_type = "reduce_on_plateau"  # or "cosine", "step"
    scheduler_patience = 10
    scheduler_factor = 0.5
    
    # ========================================================================
    # PREPROCESSING SETTINGS
    # ========================================================================
    
    # Graph construction
    k_neighbors = 8  # Number of nearest neighbors for graph edges
    
    # Normalization
    normalize_features = True
    normalize_targets = True
    normalization_method = "standard"  # or "minmax"
    
    # Data augmentation (optional)
    use_augmentation = False
    augmentation_types = []  # ["rotation", "scaling", "noise"]
    
    # ========================================================================
    # EVALUATION SETTINGS
    # ========================================================================
    
    # Metrics to track
    metrics = ["mse", "mae", "r2", "relative_error"]
    
    # Regions to evaluate separately
    evaluate_by_region = True
    
    # Critical region weight (if we want to emphasize critical region in loss)
    critical_region_weight = 1.0  # Set > 1.0 to emphasize
    
    # ========================================================================
    # COMPUTATIONAL SETTINGS
    # ========================================================================
    
    # Device
    use_cuda = True  # Set to False to use CPU
    
    # Random seed for reproducibility
    seed = 42
    
    # Number of workers for data loading
    num_workers = 4
    
    # Mixed precision training
    use_amp = True
    
    # Gradient clipping
    clip_grad_norm = 1.0
    
    # ========================================================================
    # LOGGING AND CHECKPOINTING
    # ========================================================================
    
    # Logging
    log_interval = 10  # Log every N batches
    val_interval = 1  # Validate every N epochs
    
    # Checkpointing
    save_best_only = True
    checkpoint_metric = "val_loss"  # or "val_mae", "val_r2"
    
    # Tensorboard
    use_tensorboard = True
    tensorboard_dir = project_root / "runs"
    
    # ========================================================================
    # VISUALIZATION SETTINGS
    # ========================================================================
    
    # Plot settings
    dpi = 150
    figure_format = "png"
    
    # 3D visualization
    plot_3d = True
    plot_2d_projections = True
    
    # Color maps
    wss_colormap = "hot"
    error_colormap = "viridis"


# Create global config instance
config = Config()

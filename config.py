# Purpose: Store all settings in one place so you don't have hardcoded values everywhere
# What it needs:

# Paths to raw CSVs and where to save processed graphs
# Domain sizes for each aspect ratio (Lx, Ly values for 1×1, 2×1, 1×2)
# Model hyperparameters (hidden dimensions, number of layers)
# Training settings (batch size, learning rate, number of epochs)
# Split strategy (which Re values are train vs test)

# Why: So you can change settings without digging through code

"""
Configuration file for lid-driven cavity ML surrogate model.
All settings, paths, and hyperparameters in one place.
"""
from pathlib import Path


class Config:
    """
    Central configuration for the entire project.
    Change settings here instead of editing code in multiple files.
    """

    # Root project directory
    project_root = Path(__file__).parent  # LidDrivenCavity directory
    repo_root = project_root.parent  # LidDrivenHolder directory
    
    # Data directories
    data_root = repo_root / "Data"  # Raw CFD simulation data
    raw_data_dir = data_root  # CSV files are in results-* subdirectories
    processed_data_dir = project_root / "ProcessedData"  # Saved PyG graphs
    
    # Output directories
    checkpoint_dir = project_root / "Models"  # Saved models (existing dir)
    results_dir = project_root / "results"         # Evaluation results
    figures_dir = project_root / "figures"         # Plots and visualizations
    
    # Raw data subdirectories (actual folder names from Data/)
    aspect_ratio_folders = {
        "1x1": "results-1x1-aspect",
        "1x2": "results-1x2-aspect",
        "2x1": "results-2x1-aspect",
    }
    
    # =========================================================================
    # DOMAIN SPECIFICATIONS - Physics/geometry knowledge
    # =========================================================================
    
    # Available aspect ratios
    aspect_ratios = ["1x1", "2x1", "1x2"]
    
    # Domain dimensions (Lx, Ly) for each aspect ratio
    # Update these if your actual domain sizes differ!
    domain_sizes = {
        "1x1": {"Lx": 1.0, "Ly": 1.0},
        "2x1": {"Lx": 2.0, "Ly": 1.0},
        "1x2": {"Lx": 1.0, "Ly": 2.0},
    }
    
    # Reynolds numbers available in dataset
    # Re from 100 to 3250 in steps of 50 (64 values total)
    # IMPORTANT: Skip Re=100,150 for 1x2 aspect ratio (WSS zeros issue)
    re_min = 100
    re_max = 3250
    re_step = 50
    re_values = list(range(re_min, re_max + 1, re_step))  # [100, 150, 200, ..., 3250]
    
    # Problematic cases to exclude
    exclude_cases = {
        "1x2": [100, 150],  # Stationary walls have 99% WSS zeros at low Re
    }
    
    # Total number of cases (for verification)
    num_aspect_ratios = len(aspect_ratios)
    num_re_values = len(re_values)  # 64 Re values
    total_cases_theoretical = num_aspect_ratios * num_re_values  # 192
    total_cases_usable = total_cases_theoretical - 2  # 190 (excluding 1x2 Re100, Re150)
    
    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    
    # CSV column names from Fluent export
    csv_columns = {
        'coords': ['x', 'y'],
        'wss_components': ['wss_mag', 'wss_x', 'wss_y'],
        'additional': ['p', 'velocity-magnitude', 'x-velocity', 'y-velocity']
    }
    
    # Number of features per node
    # Current features (10 total):
    # 0: x_normalized (x / Lx)
    # 1: y_normalized (y / Ly)
    # 2: on_top (binary)
    # 3: on_bottom (binary)
    # 4: on_left (binary)
    # 5: on_right (binary)
    # 6: arc_length (0 to 1, surface coordinate)
    # 7: distance_to_nearest_corner (normalized)
    # 8: wall_type (0=stationary, 1=moving)
    # 9: local_curvature (0 for flat walls)
    node_feature_dim = 10
    
    # Flow parameters dimension: [Re, Lx, Ly]
    flow_param_dim = 3
    
    # Target outputs (WSS components)
    # Predict both components; magnitude can be derived
    target_dim = 2  # [x_wss, y_wss]
    # Or predict all 3: target_dim = 3  # [wss_mag, x_wss, y_wss]
    predict_components_only = True  # If True, only predict x/y and derive magnitude
    
    # Tolerance for wall classification (as fraction of domain size)
    wall_tolerance = 0.001  # 0.1% of domain size
    
    # File naming patterns
    moving_wall_pattern = "moving_wall_full_Re{re}.csv"
    stationary_wall_pattern = "stat_walls_full_Re{re}.csv"  # 1x1 uses this
    stationary_wall_pattern_alt = "stationary_walls_full_Re{re}.csv"  # 1x2, 2x1 use this
    
    # =========================================================================
    # MODEL ARCHITECTURE
    # =========================================================================
    
    # Flow encoder (processes [Re, Lx, Ly])
    context_dim = 64  # Output dimension of flow context encoder
    
    # Geometry encoder (processes boundary mesh)
    hidden_dim = 64  # Hidden dimension for GNN layers
    num_geom_layers = 3  # Number of GCN layers in geometry encoder
    
    # Task head (predicts WSS)
    task_hidden_dim = 128  # Hidden dimension in task head
    num_task_layers = 2  # Number of GCN layers in task head
    
    # Fusion method
    fusion_type = "film"  # Options: "film" or "concat"
    
    # Regularization
    dropout_rate = 0.0  # Dropout between GNN layers (0.0 = no dropout)
    
    # =========================================================================
    # DATA FILTERING
    # =========================================================================
    
    # Top wall filtering
    # If True, removes top (moving lid) wall from training/validation/testing
    # Improves model performance on stationary walls (bottom/left/right)
    filter_top_wall = True  # Set to True to train on 3 walls only
    
    # =========================================================================
    # TRAINING HYPERPARAMETERS
    # =========================================================================
    
    # Optimization
    batch_size = 8  # Automatically increased to 11 when filter_top_wall=True (compensates for ~25% fewer nodes)
    learning_rate = 1e-3
    weight_decay = 1e-5  # L2 regularization
    num_epochs = 500
    
    # Learning rate scheduler
    use_scheduler = True
    scheduler_type = "plateau"  # ReduceLROnPlateau
    scheduler_patience = 10  # Epochs with no improvement before reducing LR
    scheduler_factor = 0.5  # Factor to reduce LR by
    scheduler_min_lr = 1e-6  # Minimum learning rate
    
    # Early stopping
    use_early_stopping = True
    early_stop_patience = 50  # Epochs with no improvement before stopping
    
    # Gradient clipping (prevents exploding gradients)
    use_grad_clip = True
    grad_clip_value = 1.0
    
    # =========================================================================
    # DATA NORMALIZATION
    # =========================================================================
    
    # WSS normalization strategy
    # Options: "log" (log-transform then z-score), "standard" (z-score only), "robust" (median/IQR)
    wss_normalization = "log"
    
    # Epsilon for log transform (prevents log(0))
    # Use log1p(wss) = log(1 + wss) which handles zeros naturally
    log_epsilon = 1e-15  # Small epsilon for any remaining numerical issues
    use_log1p = True  # Recommended: log(1+x) instead of log(x+epsilon)
    
    # =========================================================================
    # TRAIN/VAL/TEST SPLIT STRATEGY
    # =========================================================================
    
    # Split strategy
    # Options: 
    #   "re_interp" - Hold out specific Re values to test interpolation
    #   "re_extrap" - Hold out high Re to test extrapolation
    #   "ar_transfer" - Hold out entire aspect ratio for transfer learning
    #   "random" - Random split across all data
    split_strategy = "re_interp"
    
    # For "re_interp": Test on Re values ending in 50 (odd multiples of 50)
    # This tests interpolation between training Re values
    test_re_values = [re for re in re_values if re % 100 == 50]
    # test_re_values = [150, 250, 350, 450, ..., 3150]
    
    # For "re_extrap": Test on high Re values
    # test_re_values = [re for re in re_values if re >= 2500]
    
    # For "ar_transfer": Test on one aspect ratio
    # test_aspect_ratios = ["1x2"]
    
    # Validation split (from training data)
    val_split = 0.15  # 15% of training data for validation
    
    # =========================================================================
    # CHECKPOINTING & LOGGING
    # =========================================================================
    
    # Model checkpointing
    save_every_n_epochs = 20  # Save checkpoint every N epochs
    save_best_only = True  # Only save when validation improves
    
    # Logging
    log_interval = 10  # Print metrics every N batches during training
    
    # Weights & Biases (optional cloud logging)
    use_wandb = False  # Set to True if you want to use W&B
    wandb_project = "lid-driven-cavity-ml"
    wandb_entity = None  # Your W&B username (if using)
    
    # =========================================================================
    # PHYSICS-INFORMED LOSSES (Optional - can enable later)
    # =========================================================================
    
    # Boundary condition loss (penalize non-zero WSS on stationary walls)
    use_bc_loss = False
    lambda_bc = 0.1
    
    # Smoothness loss (penalize large gradients between neighbors)
    use_smoothness_loss = False
    lambda_smooth = 0.01
    
    # Symmetry loss (for 1x1 cavity at low Re)
    use_symmetry_loss = False
    lambda_symmetry = 0.05
    symmetry_re_threshold = 400  # Only enforce below this Re
    
    # =========================================================================
    # COMPUTATIONAL SETTINGS
    # =========================================================================
    
    # Device
    device = "cuda"  # Options: "cuda", "cpu", "mps" (for Mac M1/M2)
    
    # Number of workers for data loading
    num_workers = 0  # Set to 0 to avoid multiprocessing issues, increase if data loading is slow
    
    # Mixed precision training (faster on newer GPUs)
    use_amp = False  # Automatic Mixed Precision
    
    # Random seed for reproducibility
    random_seed = 42
    
    # =========================================================================
    # EVALUATION SETTINGS
    # =========================================================================
    
    # Metrics to compute
    metrics = ["mae", "rmse", "mape", "r2"]
    
    # Number of cases to visualize in evaluation
    num_visualization_cases = 10
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def create_directories(self):
        """Create all necessary directories if they don't exist"""
        dirs_to_create = [
            self.processed_data_dir,
            self.checkpoint_dir,
            self.results_dir,
            self.figures_dir,
        ]
        
        # Create processed data subdirectories for each aspect ratio
        for ar in self.aspect_ratios:
            dirs_to_create.append(self.processed_data_dir / ar)
        
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_device(self):
        """Get torch device (cuda/cpu/mps)"""
        import torch
        
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif self.device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def get_batch_size(self):
        """Get effective batch size (adjusted for top wall filtering)"""
        # Keep batch size the same - the increased batch size was causing
        # incomplete batches with only 1 sample (BatchNorm fails)
        return self.batch_size
    
    def get_csv_path(self, aspect_ratio, re_value, wall_type="moving"):
        """
        Get path to CSV file for a specific case.
        
        Args:
            aspect_ratio: "1x1", "1x2", or "2x1"
            re_value: Reynolds number (100-3250)
            wall_type: "moving" or "stationary"
        
        Returns:
            Path object to CSV file
        """
        folder_name = self.aspect_ratio_folders[aspect_ratio]
        re_dir = f"Re{re_value}"
        
        if wall_type == "moving":
            filename = self.moving_wall_pattern.format(re=re_value)
        else:
            # Try both naming conventions
            filename = self.stationary_wall_pattern.format(re=re_value)
        
        return self.data_root / folder_name / re_dir / filename
    
    def should_exclude_case(self, aspect_ratio, re_value):
        """Check if a case should be excluded due to data quality issues"""
        if aspect_ratio in self.exclude_cases:
            return re_value in self.exclude_cases[aspect_ratio]
        return False
    
    def print_summary(self):
        """Print configuration summary"""
        print("=" * 70)
        print("CONFIGURATION SUMMARY")
        print("=" * 70)
        print(f"\nDATA:")
        print(f"  Total cases (theoretical): {self.total_cases_theoretical}")
        print(f"  Usable cases: {self.total_cases_usable}")
        print(f"  Aspect ratios: {self.aspect_ratios}")
        print(f"  Re range: {self.re_min} to {self.re_max} (step {self.re_step}, {len(self.re_values)} values)")
        print(f"  Excluded: 1x2 Re100,150 (WSS zeros)")
        print(f"  Data location: {self.data_root}")
        print(f"  Split strategy: {self.split_strategy}")
        print(f"  Test Re values: {len(self.test_re_values)} cases")
        
        print(f"\nMODEL:")
        print(f"  Node features: {self.node_feature_dim}")
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Geom layers: {self.num_geom_layers}")
        print(f"  Task layers: {self.num_task_layers}")
        print(f"  Fusion: {self.fusion_type}")
        
        print(f"\nTRAINING:")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Device: {self.device}")
        
        print("=" * 70)
    
    def __repr__(self):
        """String representation"""
        return (
            f"Config(\n"
            f"  cases={self.total_cases}, "
            f"  hidden_dim={self.hidden_dim}, "
            f"  batch_size={self.batch_size}, "
            f"  lr={self.learning_rate}\n"
            f")"
        )


# =========================================================================
# CREATE GLOBAL CONFIG INSTANCE
# =========================================================================

# This is the object you'll import in other files
config = Config()

# Create directories when config is first imported
config.create_directories()


# =========================================================================
# USAGE IN OTHER FILES
# =========================================================================

# In process_data.py, model.py, train.py, etc.:
#
#   from config import config
#
#   csv_path = config.raw_data_dir / "1x1" / "Re100.csv"
#   model = Model(config)
#   optimizer = Adam(model.parameters(), lr=config.learning_rate)
#


if __name__ == "__main__":
    # Test: print configuration summary
    config.print_summary()
    
    print(f"\nTest Re values ({len(config.test_re_values)}):")
    print(config.test_re_values[:10], "...")
    
    print(f"\nTrain Re values ({len(config.re_values) - len(config.test_re_values)}):")
    train_re = [re for re in config.re_values if re not in config.test_re_values]
    print(train_re[:10], "...")
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
    processed_data_dir = project_root / "ProcessedData"  # Saved PyG graphs (3D)
    input_data_dir = project_root / "Data" / "openFoam(1e-3)"  # Raw CFD input data
    
    # Google Drive download URL for data zip
    google_drive_zip_url = "https://drive.google.com/uc?export=download&id=1h66tx2eT9lZk1wWf3IeEpBBMs7xq-Wth"
    
    # Output directories
    checkpoint_dir = project_root / "Models"  # Saved models (existing dir)
    results_dir = project_root / "results"         # Evaluation results
    figures_dir = project_root / "figures"         # Plots and visualizations
    
    # =========================================================================
    # DOMAIN SPECIFICATIONS - Physics/geometry knowledge
    # =========================================================================
    
    # 3D point cloud data - no aspect ratios
    re_min = 100
    re_max = 3250
    re_step = 50
    re_values = list(range(re_min, re_max + 1, re_step))
    
    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    
    # Number of features per node (3D point cloud)
    # Current features (4 total):
    # 0: x coordinate
    # 1: y coordinate
    # 2: z coordinate
    # 3: pressure (p)
    node_feature_dim = 4
    
    # Flow parameters dimension: [Re, angle, child_size]
    flow_param_dim = 3
    
    # Target outputs (WSS components)
    target_dim = 3  # [wss_x, wss_y, wss_z]
    
    # =========================================================================
    # MODEL ARCHITECTURE
    # =========================================================================
    
    # Flow encoder (processes [Re])
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
    # TRAINING HYPERPARAMETERS
    # =========================================================================
    
    # Optimization
    batch_size = 8
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
    
    # For "ar_transfer": Hold out entire aspect ratio for transfer learning
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
        """Get batch size"""
        return self.batch_size
    
    def print_summary(self):
        """Print configuration summary"""
        print("=" * 70)
        print("CONFIGURATION SUMMARY")
        print("=" * 70)
        print(f"\nDATA:")
        print(f"  Re range: {self.re_min} to {self.re_max} (step {self.re_step}, {len(self.re_values)} values)")
        print(f"  Data location: {self.processed_data_dir}")
        
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
            f"  re_values={len(self.re_values)}, "
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

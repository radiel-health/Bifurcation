"""
Configuration module for Bifurcation WSS Prediction Pipeline.

Central configuration hub for all hyperparameters, paths, and domain specifications.
Follows the clean config pattern from LidDrivenCavity project.
"""

from pathlib import Path
import torch
import json
from typing import Dict, List, Tuple, Optional


class Config:
    """
    Central configuration for bifurcation WSS prediction.
    
    Manages:
    - File paths and directory structure
    - Domain parameters (angles, Reynolds numbers, mesh levels)
    - Model architecture hyperparameters
    - Training configuration
    - Data preprocessing settings
    """
    
    def __init__(self):
        # ============================================================================
        # PATHS
        # ============================================================================
        self.project_root = Path(__file__).parent
        self.data_root = self.project_root.parent / "Data" / "Bifurcation"
        self.results_root = self.data_root / "results"
        
        # Project directories
        self.processed_data_dir = self.project_root / "ProcessedData"
        self.checkpoint_dir = self.project_root / "Models"
        self.results_dir = self.project_root / "results"
        self.figures_dir = self.project_root / "figures"
        
        # ============================================================================
        # DOMAIN PARAMETERS
        # ============================================================================
        # Bifurcation angles (degrees)
        self.angles = [30, 45, 60]
        
        # Mesh refinement levels
        self.mesh_levels = {
            "base": "",              # bifurcation_angle{A}/
            "750": "_750",           # bifurcation_angle{A}_750/
            "1000": "_1000"          # bifurcation_angle{A}_1000/
        }
        
        # Reynolds number range
        self.re_min = 100
        self.re_max = 2100
        self.re_step = 100
        self.reynolds_numbers = list(range(self.re_min, self.re_max + 1, self.re_step))
        
        # Known incomplete cases (exclude from data catalog)
        self.incomplete_cases = [
            # angle30_1000 missing Re1800-2100
            (30, "1000", 1800),
            (30, "1000", 1900),
            (30, "1000", 2000),
            (30, "1000", 2100),
        ]
        
        # ============================================================================
        # FEATURE SPECIFICATION
        # ============================================================================
        # Node features: [x_norm, y_norm, z_norm, degree] (4D)
        self.node_feature_dim = 4
        
        # Edge features: [dihedral_angle, min_inner_angle, max_inner_angle, 
        #                  min_edge_ratio, max_edge_ratio] (5D)
        self.edge_feature_dim = 5
        
        # Flow conditioning parameters: [Re_normalized, angle_radians] (2D)
        self.flow_param_dim = 2
        
        # Target: [wss_x, wss_y, wss_z] (3D)
        self.output_dim = 3
        
        # ============================================================================
        # MODEL ARCHITECTURE (AVFlow Gen 3 + FiLM)
        # ============================================================================
        # Edge aggregator (GraphUNet)
        self.edge_channels = 5
        self.aggregated_edge_feat_dim = 16  # Corrected from plan (was 32)
        self.unet_hidden = 128
        self.unet_depth = 4
        self.unet_pool_ratio = 0.5
        
        # GCN stack
        self.hidden_gcn_dim = 512
        self.num_gcn_layers = 8
        
        # FiLM conditioning
        self.context_dim = 64
        
        # Flow encoder
        self.flow_hidden_dim = 64
        
        # Output head (single linear layer)
        self.out_channels = 3  # [wss_x, wss_y, wss_z]
        
        # ============================================================================
        # TRAINING CONFIGURATION
        # ============================================================================
        self.batch_size = 2  # Small due to ~17K nodes per graph
        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.num_epochs = 300
        
        # Scheduler (ReduceLROnPlateau)
        self.scheduler_mode = 'min'
        self.scheduler_factor = 0.5
        self.scheduler_patience = 10
        self.min_lr = 1e-6
        
        # Early stopping
        self.early_stop_patience = 30
        self.early_stop_tol = 1e-4
        
        # Gradient clipping
        self.grad_clip = 1.0
        
        # Loss function
        self.loss_fn = 'mse'  # Options: 'mse', 'huber'
        
        # LOO-CV ensemble
        self.n_folds = 3
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # ============================================================================
        # DATA PREPROCESSING
        # ============================================================================
        # Normalization strategy
        self.normalize_features = True  # Z-score on node/edge features
        self.normalize_targets = True   # Sign-preserving log1p + z-score on WSS
        
        # Sign-preserving log transform for WSS
        self.use_log_transform = True
        
        # ============================================================================
        # SPLIT STRATEGIES
        # ============================================================================
        self.split_strategy = "re_interp"  # Options: "re_interp", "angle_transfer", "random"
        
        # Train/val/test split ratios (for random split)
        self.train_ratio = 0.70
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
        # Held-out Re values for interpolation testing (odd hundreds)
        self.holdout_re_values = [300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100]
        
        # Held-out angle for transfer learning test
        self.holdout_angle = 45  # Train on 30°+60°, test on 45°
        
        # ============================================================================
        # CALIBRATION
        # ============================================================================
        self.calibration_n_outer_splits = 25
        self.calibration_n_inner_splits = 7
        self.calibration_lambda_min = 0.0
        self.calibration_lambda_max = 2.0
        self.calibration_lambda_steps = 120
        
        # ============================================================================
        # VISUALIZATION
        # ============================================================================
        self.vis_cmap = "viridis"
        self.vis_figsize = (12, 8)
        self.vis_dpi = 150
        
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def create_directories(self):
        """Create all necessary directories if they don't exist."""
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def get_device(self):
        """Get torch device with fallback."""
        return torch.device(self.device)
    
    def get_csv_path(self, angle: int, mesh_level: str, re: int) -> Optional[Path]:
        """
        Get path to wall_data CSV file for a given case.
        
        Args:
            angle: Bifurcation angle (30, 45, 60)
            mesh_level: "base", "750", or "1000"
            re: Reynolds number
            
        Returns:
            Path to CSV file, or None if case is incomplete
        """
        if (angle, mesh_level, re) in self.incomplete_cases:
            return None
        
        suffix = self.mesh_levels[mesh_level]
        result_dir = self.results_root / f"bifurcation_angle{angle}{suffix}" / f"Re{re}"
        csv_path = result_dir / f"wall_data_Re{re}.csv"
        
        if csv_path.exists():
            return csv_path
        return None
    
    def get_mesh_path(self, angle: int, mesh_level: str) -> Optional[Path]:
        """
        Get path to .msh file for a given angle and mesh level.
        
        Args:
            angle: Bifurcation angle (30, 45, 60)
            mesh_level: "base", "750", or "1000"
            
        Returns:
            Path to .msh file
        """
        suffix = self.mesh_levels[mesh_level]
        if suffix == "":
            mesh_path = self.data_root / f"bifurcation_angle{angle}.msh"
        else:
            mesh_path = self.data_root / f"bifurcation_angle{angle}{suffix}.msh"
        
        return mesh_path if mesh_path.exists() else None
    
    def should_exclude_case(self, angle: int, mesh_level: str, re: int) -> bool:
        """Check if a case should be excluded (incomplete data)."""
        return (angle, mesh_level, re) in self.incomplete_cases
    
    def get_available_cases(self) -> List[Tuple[int, str, int]]:
        """
        Get list of all available (angle, mesh_level, re) combinations.
        
        Returns:
            List of tuples: [(angle, mesh_level, re), ...]
        """
        cases = []
        for angle in self.angles:
            for mesh_level_name in self.mesh_levels.keys():
                for re in self.reynolds_numbers:
                    if not self.should_exclude_case(angle, mesh_level_name, re):
                        csv_path = self.get_csv_path(angle, mesh_level_name, re)
                        if csv_path is not None:
                            cases.append((angle, mesh_level_name, re))
        return cases
    
    def print_summary(self):
        """Print configuration summary."""
        print("=" * 80)
        print("BIFURCATION WSS PREDICTION - CONFIGURATION SUMMARY")
        print("=" * 80)
        print()
        print("Paths:")
        print(f"  Project root:      {self.project_root}")
        print(f"  Data root:         {self.data_root}")
        print(f"  Processed data:    {self.processed_data_dir}")
        print(f"  Checkpoints:       {self.checkpoint_dir}")
        print()
        print("Domain:")
        print(f"  Angles:            {self.angles}°")
        print(f"  Mesh levels:       {list(self.mesh_levels.keys())}")
        print(f"  Reynolds range:    {self.re_min} - {self.re_max} (step {self.re_step})")
        print(f"  Total Re values:   {len(self.reynolds_numbers)}")
        print()
        print("Model Architecture:")
        print(f"  Node features:     {self.node_feature_dim}D")
        print(f"  Edge features:     {self.edge_feature_dim}D → {self.aggregated_edge_feat_dim}D (via GraphUNet)")
        print(f"  GCN stack:         {self.num_gcn_layers} layers × {self.hidden_gcn_dim}D")
        print(f"  FiLM context:      {self.context_dim}D")
        print(f"  Output:            {self.out_channels}D [wss_x, wss_y, wss_z]")
        print()
        print("Training:")
        print(f"  Batch size:        {self.batch_size}")
        print(f"  Learning rate:     {self.lr}")
        print(f"  Epochs:            {self.num_epochs}")
        print(f"  Device:            {self.device}")
        print(f"  Ensemble folds:    {self.n_folds}")
        print()
        print("Data:")
        available = self.get_available_cases()
        print(f"  Available cases:   {len(available)}")
        by_angle = {}
        for angle, mesh_level, re in available:
            key = f"{angle}° ({mesh_level})"
            by_angle[key] = by_angle.get(key, 0) + 1
        for key, count in sorted(by_angle.items()):
            print(f"    {key:20s} {count} cases")
        print()
        print("=" * 80)


# Global configuration instance
config = Config()


if __name__ == "__main__":
    """Test configuration and data catalog."""
    config.print_summary()
    
    # Test path resolution
    print("\nTesting path resolution...")
    test_case = (30, "750", 100)
    csv_path = config.get_csv_path(*test_case)
    mesh_path = config.get_mesh_path(test_case[0], test_case[1])
    
    print(f"  Test case: angle={test_case[0]}°, mesh={test_case[1]}, Re={test_case[2]}")
    print(f"  CSV path: {csv_path}")
    print(f"  CSV exists: {csv_path.exists() if csv_path else 'N/A'}")
    print(f"  Mesh path: {mesh_path}")
    print(f"  Mesh exists: {mesh_path.exists() if mesh_path else 'N/A'}")

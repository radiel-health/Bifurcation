"""
Dataset loader for preprocessed PyG graphs

Loads wall shear stress prediction graphs with:
- Train/val/test splits
- Normalization statistics
- Batching support
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import json
from config import Config

def filter_edge_index(edge_index: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    """Filter edges and remap node indices after removing nodes.
    
    When nodes are removed from a graph, edges must be updated:
    1. Remove edges that connect to deleted nodes
    2. Remap remaining node indices to be contiguous (0, 1, 2, ...)
    
    Args:
        edge_index: [2, num_edges] edge connectivity tensor
        keep_mask: [num_nodes] boolean mask indicating which nodes to keep
        
    Returns:
        Filtered and remapped edge_index tensor
        
    Example:
        Original: nodes=[0,1,2,3,4], edges=[[0,1,2,3],[1,2,3,4]]
        Remove node 2: keep_mask=[T,T,F,T,T]
        Result: nodes=[0,1,2,3], edges=[[0,1,2],[1,2,3]] (indices remapped)
    """
    # Step 1: Keep only edges where both endpoints are in the keep_mask
    valid_edges = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
    edge_index_filtered = edge_index[:, valid_edges]
    
    # Step 2: Remap node indices to be contiguous
    # Create mapping from old indices to new indices
    # Example: keep_mask=[T,T,F,T,T] -> old_to_new=[0,1,-1,2,3]
    old_to_new = torch.cumsum(keep_mask.long(), dim=0) - 1
    edge_index_remapped = old_to_new[edge_index_filtered]
    
    return edge_index_remapped
class WSSDataset(Dataset):
    """
    Dataset for loading preprocessed wall shear stress graphs
    
    Features:
    - Loads from ProcessedData/ directory
    - Computes normalization statistics on training set
    - Supports train/val/test splits
    - Handles multiple aspect ratios
    """
    
    def __init__(
        self, 
        root: Optional[str] = None,
        split: str = 'train',
        normalize: bool = True,
        normalize_features: bool = False,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        filter_top_wall: bool = False,
    ):
        """
        Initialize dataset
        
        Args:
            root: Root directory containing ProcessedData/ (defaults to LidDrivenCavity/)
            split: 'train', 'val', or 'test'
            normalize: Whether to normalize features and targets
            split_ratios: (train, val, test) split ratios, must sum to 1.0
            seed: Random seed for reproducible splits
            filter_top_wall: If True, removes top (moving lid) wall nodes from graphs
        """
        self.split = split
        self.normalize = normalize
        self.normalize_features = normalize_features
        self.split_ratios = split_ratios
        self.seed = seed
        self.filter_top_wall = filter_top_wall
        
        # Get root directory
        if root is None:
            config = Config()
            root = config.repo_root / "LidDrivenCavity"
        else:
            root = Path(root)
        
        self.data_dir = root / "ProcessedData"
        
        # Use separate stats file for filtered data
        if self.filter_top_wall:
            self.stats_file = self.data_dir / "normalization_stats_no_top.json"
        else:
            self.stats_file = self.data_dir / "normalization_stats.json"
        
        # Validate
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}"
        assert abs(sum(split_ratios) - 1.0) < 1e-6, f"Split ratios must sum to 1.0"
        
        # Get all graph files from subdirectories (1x1/, 1x2/, 2x1/)
        self.all_files = []
        for ar_dir in ['1x1', '1x2', '2x1']:
            ar_path = self.data_dir / ar_dir
            if ar_path.exists():
                self.all_files.extend(sorted(list(ar_path.glob("Re*.pt"))))
        
        # Sort by aspect ratio and Re number
        self.all_files = sorted(self.all_files)
        
        print(f"\nFound {len(self.all_files)} preprocessed graphs")
        
        # Create splits
        self._create_splits()
        
        # Load normalization stats or compute them
        if normalize:
            self._load_or_compute_stats()
    
    def _create_splits(self):
        """Create train/val/test splits with stratification by aspect ratio"""
        # Group files by aspect ratio
        ar_groups = {'1x1': [], '1x2': [], '2x1': []}
        for f in self.all_files:
            # Get aspect ratio from parent directory name
            ar = f.parent.name
            if ar in ar_groups:
                ar_groups[ar].append(f)
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        
        train_files, val_files, test_files = [], [], []
        
        # Split each aspect ratio group separately (stratified split)
        for ar, files in ar_groups.items():
            n = len(files)
            indices = np.random.permutation(n)
            
            n_train = int(n * self.split_ratios[0])
            n_val = int(n * self.split_ratios[1])
            
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]
            
            train_files.extend([files[i] for i in train_idx])
            val_files.extend([files[i] for i in val_idx])
            test_files.extend([files[i] for i in test_idx])
        
        # Assign to this split
        if self.split == 'train':
            self.files = train_files
        elif self.split == 'val':
            self.files = val_files
        else:
            self.files = test_files
        
        print(f"\nSplit: {self.split}")
        print(f"  Total graphs: {len(self.files)}")
        print(f"  1x1: {sum(1 for f in self.files if '1x1' in str(f.parent))}")
        print(f"  1x2: {sum(1 for f in self.files if '1x2' in str(f.parent))}")
        print(f"  2x1: {sum(1 for f in self.files if '2x1' in str(f.parent))}")
    
    def _load_or_compute_stats(self):
        """Load normalization stats from file or compute from training set"""
        if self.stats_file.exists():
            # Load existing stats
            with open(self.stats_file, 'r') as f:
                stats = json.load(f)
            
            self.feature_mean = torch.tensor(stats['feature_mean'])
            self.feature_std = torch.tensor(stats['feature_std'])
            self.target_mean = torch.tensor(stats['target_mean'])
            self.target_std = torch.tensor(stats['target_std'])
            self.flow_mean = torch.tensor(stats['flow_mean'])
            self.flow_std = torch.tensor(stats['flow_std'])
            
            print(f"\nLoaded normalization stats from {self.stats_file}")
        
        elif self.split == 'train':
            # Compute stats from training set
            print("\nComputing normalization statistics from training set...")
            self._compute_stats()
        
        else:
            raise FileNotFoundError(
                f"Normalization stats not found at {self.stats_file}. "
                "Please run with split='train' first to compute statistics."
            )
    
    def _compute_stats(self):
        """Compute normalization statistics from training data"""
        # Collect all features, targets, and flow params
        all_features = []
        all_targets = []
        all_flow_params = []
        
        for file_path in self.files:
            data = torch.load(file_path, weights_only=False)
            
            # Filter top wall if enabled
            if self.filter_top_wall:
                on_top = data.x[:, 2].bool()  # Feature index 2 is 'on_top' flag
                keep_mask = ~on_top
                all_features.append(data.x[keep_mask])
                all_targets.append(data.y[keep_mask])
            else:
                all_features.append(data.x)
                all_targets.append(data.y)
            
            all_flow_params.append(data.flow_params.unsqueeze(0))
        
        # Concatenate
        all_features = torch.cat(all_features, dim=0)  # [total_nodes, 10]
        all_targets = torch.cat(all_targets, dim=0)    # [total_nodes, 2]
        all_flow_params = torch.cat(all_flow_params, dim=0)  # [n_graphs, 3]
        
        # Compute statistics
        # Features: mean/std per dimension
        self.feature_mean = all_features.mean(dim=0)
        self.feature_std = all_features.std(dim=0)
        # Avoid division by zero for constant features
        self.feature_std[self.feature_std < 1e-8] = 1.0

        sign = torch.sign(all_targets)
        log_mag = torch.log1p(torch.abs(all_targets))
        signed_log = sign * log_mag 
        
        # Targets: use log1p transform for WSS (handles zeros and large range)
        # Store stats for log-transformed values
      #  log_targets = torch.log1p(torch.abs(all_targets))
        self.target_mean = signed_log.mean(dim=0)
        self.target_std = signed_log.std(dim=0)
        self.target_std[self.target_std < 1e-8] = 1.0
        
        # Flow params: normalize Re, Lx, Ly
        self.flow_mean = all_flow_params.mean(dim=0)
        self.flow_std = all_flow_params.std(dim=0)
        self.flow_std[self.flow_std < 1e-8] = 1.0
        
        # Save to file
        stats = {
            'feature_mean': self.feature_mean.tolist(),
            'feature_std': self.feature_std.tolist(),
            'target_mean': self.target_mean.tolist(),
            'target_std': self.target_std.tolist(),
            'flow_mean': self.flow_mean.tolist(),
            'flow_std': self.flow_std.tolist(),
        }
        
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved normalization stats to {self.stats_file}")
        print(f"\nNormalization statistics:")
        print(f"  Feature mean: {self.feature_mean}")
        print(f"  Feature std: {self.feature_std}")
        print(f"  Target (log) mean: {self.target_mean}")
        print(f"  Target (log) std: {self.target_std}")
        print(f"  Flow mean: {self.flow_mean}")
        print(f"  Flow std: {self.flow_std}")
    
    def __len__(self):
        """Number of graphs in this split"""
        return len(self.files)
    
    def __getitem__(self, idx):
        """
        Load and optionally normalize a graph
        
        Returns:
            Data object with normalized features if normalize=True
        """
        # Load graph
        data = torch.load(self.files[idx], weights_only=False)
        
        # Filter top wall nodes if enabled
        if self.filter_top_wall:
            on_top = data.x[:, 2].bool()  # Feature index 2 is 'on_top' flag
            keep_mask = ~on_top  # Keep all nodes except top wall
            
            # Apply mask to node features, targets, and positions
            data.x = data.x[keep_mask]
            data.y = data.y[keep_mask]
            data.pos = data.pos[keep_mask]
            
            # Filter and remap edges
            data.edge_index = filter_edge_index(data.edge_index, keep_mask)
            
            # Update node count
            data.num_nodes = keep_mask.sum().item()

        if self.normalize_features:
            # normalizing the node features
            data.x = (data.x - self.feature_mean) / self.feature_std
        
        if self.normalize:            
            # Normalize targets using log1p transform
            # Sign is preserved, magnitude is log-transformed
            sign = torch.sign(data.y)
            log_mag = torch.log1p(torch.abs(data.y))

            signed_log = sign * log_mag

            data.y = (signed_log - self.target_mean) / self.target_std
            
            # Normalize flow parameters
            data.flow_params = (data.flow_params - self.flow_mean) / self.flow_std
        
        return data
    
    def denormalize_targets(self, normalized_targets: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized predictions back to original WSS scale
        
        Args:
            normalized_targets: [n_nodes, 2] tensor of normalized predictions
            
        Returns:
            Original-scale WSS predictions [n_nodes, 2]
        """
        if not self.normalize:
            return normalized_targets
    
        signed_log = normalized_targets * self.target_std + self.target_mean
        
        # Reverse normalization
        sign = torch.sign(signed_log)
        log_mag = torch.abs(signed_log)
        
        # Reverse log1p transform
        mag = torch.expm1(log_mag)
        
        return sign * mag


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    normalize: bool = True,
    normalize_features: bool = False,
    filter_top_wall: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to get train/val/test dataloaders
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        split_ratios: (train, val, test) split ratios
        seed: Random seed for splits
        normalize: Whether to normalize data
        filter_top_wall: If True, removes top wall nodes from all graphs
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = WSSDataset(split='train', normalize=normalize, normalize_features=normalize_features, 
                               split_ratios=split_ratios, seed=seed, filter_top_wall=filter_top_wall)
    val_dataset = WSSDataset(split='val', normalize=normalize, normalize_features=normalize_features, 
                             split_ratios=split_ratios, seed=seed, filter_top_wall=filter_top_wall)
    test_dataset = WSSDataset(split='test', normalize=normalize, normalize_features=normalize_features, 
                              split_ratios=split_ratios, seed=seed, filter_top_wall=filter_top_wall)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test dataset loading
    print("="*80)
    print("Testing WSSDataset")
    print("="*80)
    
    # Create datasets
    train_ds = WSSDataset(split='train', normalize=True, seed=42)
    val_ds = WSSDataset(split='val', normalize=True, seed=42)
    test_ds = WSSDataset(split='test', normalize=True, seed=42)
    
    print(f"\n{'='*80}")
    print("Dataset sizes:")
    print(f"  Train: {len(train_ds)}")
    print(f"  Val: {len(val_ds)}")
    print(f"  Test: {len(test_ds)}")
    
    # Test loading a sample
    print(f"\n{'='*80}")
    print("Sample from training set:")
    sample = train_ds[0]
    print(f"  Nodes: {sample.x.shape[0]}")
    print(f"  Features: {sample.x.shape[1]}")
    print(f"  Edges: {sample.edge_index.shape[1]}")
    print(f"  Targets: {sample.y.shape}")
    print(f"  Flow params: {sample.flow_params}")
    print(f"  Metadata: Re={sample.re}, AR={sample.ar}, Lx={sample.lx}, Ly={sample.ly}")
    
    # Test denormalization
    print(f"\n{'='*80}")
    print("Testing denormalization:")
    denorm_targets = train_ds.denormalize_targets(sample.y)
    print(f"  Normalized WSS range: [{sample.y.min():.6f}, {sample.y.max():.6f}]")
    print(f"  Denormalized WSS range: [{denorm_targets.min():.6e}, {denorm_targets.max():.6e}]")
    
    # Test dataloaders
    print(f"\n{'='*80}")
    print("Testing dataloaders:")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8, seed=42)
    
    batch = next(iter(train_loader))
    print(f"  Batch size: {batch.num_graphs}")
    print(f"  Total nodes in batch: {batch.x.shape[0]}")
    print(f"  Total edges in batch: {batch.edge_index.shape[1]}")
    print(f"  Flow params shape: {batch.flow_params.shape}")
    
    print(f"\n{'='*80}")
    print("✅ All tests passed!")
    print("="*80)

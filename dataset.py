"""
Dataset loader for 3D point cloud WSS prediction graphs

Loads preprocessed PyG graphs from ProcessedData/3D/ with:
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


class WSSDataset(Dataset):
    """
    Dataset for loading 3D point cloud wall shear stress graphs
    
    Features:
    - Loads from ProcessedData/3D/ directory
    - Computes normalization statistics on training set
    - Supports train/val/test splits
    """
    
    def __init__(
        self, 
        root: Optional[str] = None,
        split: str = 'train',
        normalize: bool = True,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
    ):
        """
        Initialize dataset
        
        Args:
            root: Root directory containing ProcessedData/3D/ (defaults to project root)
            split: 'train', 'val', or 'test'
            normalize: Whether to normalize features and targets
            split_ratios: (train, val, test) split ratios, must sum to 1.0
            seed: Random seed for reproducible splits
        """
        self.split = split
        self.normalize = normalize
        self.split_ratios = split_ratios
        self.seed = seed
        
        # Get root directory
        if root is None:
            config = Config()
            root = config.repo_root / "Bifurcation"
        else:
            root = Path(root)
        
        self.data_dir = root / "ProcessedData" / "3D"
        
        self.stats_file = self.data_dir / "normalization_stats.json"
        
        # Validate
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}"
        assert abs(sum(split_ratios) - 1.0) < 1e-6, f"Split ratios must sum to 1.0"
        
        # Get all .pt files
        self.all_files = sorted(list(self.data_dir.glob("*.pt")))
        
        print(f"\nFound {len(self.all_files)} preprocessed graphs in {self.data_dir}")
        
        # Create splits
        self._create_splits()
        
        # Load normalization stats or compute them
        if normalize:
            self._load_or_compute_stats()
    
    def _create_splits(self):
        """Create random train/val/test splits"""
        np.random.seed(self.seed)
        
        n = len(self.all_files)
        indices = np.random.permutation(n)
        
        n_train = int(n * self.split_ratios[0])
        n_val = int(n * self.split_ratios[1])
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        if self.split == 'train':
            self.files = [self.all_files[i] for i in train_idx]
        elif self.split == 'val':
            self.files = [self.all_files[i] for i in val_idx]
        else:
            self.files = [self.all_files[i] for i in test_idx]
        
        print(f"\nSplit: {self.split}")
        print(f"  Total graphs: {len(self.files)}")
    
    def _load_or_compute_stats(self):
        """Load normalization stats from file or compute from training set"""
        if self.stats_file.exists():
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
            print("\nComputing normalization statistics from training set...")
            self._compute_stats()
        
        else:
            raise FileNotFoundError(
                f"Normalization stats not found at {self.stats_file}. "
                "Please run with split='train' first to compute statistics."
            )
    
    def _compute_stats(self):
        """Compute normalization statistics from training data"""
        all_features = []
        all_targets = []
        all_flow_params = []
        
        for file_path in self.files:
            data = torch.load(file_path, weights_only=False)
            all_features.append(data.x)
            all_targets.append(data.y)
            all_flow_params.append(data.flow_params.unsqueeze(0))
        
        all_features = torch.cat(all_features, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_flow_params = torch.cat(all_flow_params, dim=0)
        
        self.feature_mean = all_features.mean(dim=0)
        self.feature_std = all_features.std(dim=0)
        self.feature_std[self.feature_std < 1e-8] = 1.0

        sign = torch.sign(all_targets)
        log_mag = torch.log1p(torch.abs(all_targets))
        signed_log = sign * log_mag 
        
        self.target_mean = signed_log.mean(dim=0)
        self.target_std = signed_log.std(dim=0)
        self.target_std[self.target_std < 1e-8] = 1.0
        
        self.flow_mean = all_flow_params.mean(dim=0)
        self.flow_std = all_flow_params.std(dim=0)
        self.flow_std[self.flow_std < 1e-8] = 1.0
        
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
        return len(self.files)
    
    def __getitem__(self, idx):
        """Load and optionally normalize a graph"""
        data = torch.load(self.files[idx], weights_only=False)
        
        if self.normalize:
            # Normalize features
            data.x = (data.x - self.feature_mean) / self.feature_std
            
            # Normalize targets using log1p transform
            sign = torch.sign(data.y)
            log_mag = torch.log1p(torch.abs(data.y))
            signed_log = sign * log_mag
            data.y = (signed_log - self.target_mean) / self.target_std
            
            # Normalize flow parameters
            data.flow_params = (data.flow_params - self.flow_mean) / self.flow_std
        
        return data
    
    def denormalize_targets(self, normalized_targets: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions back to original WSS scale"""
        if not self.normalize:
            return normalized_targets
    
        signed_log = normalized_targets * self.target_std + self.target_mean
        
        sign = torch.sign(signed_log)
        log_mag = torch.abs(signed_log)
        
        mag = torch.expm1(log_mag)
        
        return sign * mag


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get train/val/test dataloaders"""
    train_dataset = WSSDataset(split='train', normalize=normalize, 
                               split_ratios=split_ratios, seed=seed)
    val_dataset = WSSDataset(split='val', normalize=normalize, 
                            split_ratios=split_ratios, seed=seed)
    test_dataset = WSSDataset(split='test', normalize=normalize, 
                              split_ratios=split_ratios, seed=seed)
    
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
    print("="*80)
    print("Testing WSSDataset (3D)")
    print("="*80)
    
    train_ds = WSSDataset(split='train', normalize=True, seed=42)
    val_ds = WSSDataset(split='val', normalize=True, seed=42)
    test_ds = WSSDataset(split='test', normalize=True, seed=42)
    
    print(f"\n{'='*80}")
    print("Dataset sizes:")
    print(f"  Train: {len(train_ds)}")
    print(f"  Val: {len(val_ds)}")
    print(f"  Test: {len(test_ds)}")
    
    print(f"\n{'='*80}")
    print("Sample from training set:")
    sample = train_ds[0]
    print(f"  Nodes: {sample.x.shape[0]}")
    print(f"  Features: {sample.x.shape[1]}")
    print(f"  Edges: {sample.edge_index.shape[1]}")
    print(f"  Targets: {sample.y.shape}")
    print(f"  Flow params: {sample.flow_params}")
    
    print(f"\n{'='*80}")
    print("Testing denormalization:")
    denorm_targets = train_ds.denormalize_targets(sample.y)
    print(f"  Normalized WSS range: [{sample.y.min():.6f}, {sample.y.max():.6f}]")
    print(f"  Denormalized WSS range: [{denorm_targets.min():.6e}, {denorm_targets.max():.6e}]")
    
    print(f"\n{'='*80}")
    print("Testing dataloaders:")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8, seed=42)
    
    batch = next(iter(train_loader))
    print(f"  Batch size: {batch.num_graphs}")
    print(f"  Total nodes in batch: {batch.x.shape[0]}")
    print(f"  Total edges in batch: {batch.edge_index.shape[1]}")
    print(f"  Flow params shape: {batch.flow_params.shape}")
    
    print(f"\n{'='*80}")
    print("All tests passed!")
    print("="*80)

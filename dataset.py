"""
Dataset module for bifurcation WSS prediction.

Handles:
- Loading preprocessed PyG graphs
- Normalization (z-score on features, sign-preserving log1p + z-score on targets)
- Train/val/test splitting with multiple strategies
- DataLoader construction

Adapted from LidDrivenCavity dataset.py with 3D mesh considerations.
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pathlib import Path
import json
import numpy as np
from typing import Tuple, List, Optional
from config import config


class BifurcationWSSDataset(Dataset):
    """
    Dataset for bifurcation WSS prediction.
    
    Loads preprocessed .pt graphs, applies normalization, and manages splits.
    
    Args:
        split: 'train', 'val', or 'test'
        strategy: Split strategy ('re_interp', 'angle_transfer', 'random')
        normalize: Whether to normalize features and targets
        transform: Optional transform to apply
    """
    
    def __init__(
        self,
        split='train',
        strategy='re_interp',
        normalize=True,
        transform=None
    ):
        self.split = split
        self.strategy = strategy
        self.normalize = normalize
        self.transform = transform
        
        # Load all graph files
        self.graph_files = sorted(config.processed_data_dir.rglob("*.pt"))
        
        if len(self.graph_files) == 0:
            raise ValueError(
                f"No preprocessed graphs found in {config.processed_data_dir}. "
                f"Run preprocessing first: python -m Preprocessing.pre_process"
            )
        
        # Parse metadata from file paths
        self.metadata = []
        for path in self.graph_files:
            # Path format: ProcessedData/angle{A}_mesh{M}/Re{R}.pt
            mesh_dir = path.parent.name  # e.g., "angle30_mesh750"
            re_file = path.stem  # e.g., "Re100"
            
            angle = int(mesh_dir.split('_')[0].replace('angle', ''))
            mesh_level = mesh_dir.split('_')[1].replace('mesh', '')
            re = int(re_file.replace('Re', ''))
            
            self.metadata.append({
                'path': path,
                'angle': angle,
                'mesh_level': mesh_level,
                're': re
            })
        
        # Apply split strategy
        self.indices = self._get_split_indices()
        
        # Load normalization stats (if normalizing)
        if self.normalize:
            self._load_normalization_stats()
    
    def _get_split_indices(self) -> List[int]:
        """
        Get indices for current split based on strategy.
        
        Returns:
            indices: List of indices for this split
        """
        if self.strategy == 're_interp':
            return self._split_re_interp()
        elif self.strategy == 'angle_transfer':
            return self._split_angle_transfer()
        elif self.strategy == 'random':
            return self._split_random()
        else:
            raise ValueError(f"Unknown split strategy: {self.strategy}")
    
    def _split_re_interp(self) -> List[int]:
        """
        Split by Reynolds number for interpolation testing.
        
        Holdout: odd multiples of 100 (300, 500, 700, ...)
        Train/Val: even multiples and extremes (100, 200, 400, ...)
        """
        holdout_re = set(config.holdout_re_values)
        
        train_val_indices = []
        test_indices = []
        
        for i, meta in enumerate(self.metadata):
            if meta['re'] in holdout_re:
                test_indices.append(i)
            else:
                train_val_indices.append(i)
        
        # Split train_val further (85% train, 15% val)
        np.random.seed(42)
        np.random.shuffle(train_val_indices)
        n_val = int(len(train_val_indices) * 0.15)
        val_indices = train_val_indices[:n_val]
        train_indices = train_val_indices[n_val:]
        
        if self.split == 'train':
            return train_indices
        elif self.split == 'val':
            return val_indices
        else:  # test
            return test_indices
    
    def _split_angle_transfer(self) -> List[int]:
        """
        Split by angle for transfer learning testing.
        
        Holdout one angle for test, use others for train/val.
        """
        holdout_angle = config.holdout_angle
        
        train_val_indices = []
        test_indices = []
        
        for i, meta in enumerate(self.metadata):
            if meta['angle'] == holdout_angle:
                test_indices.append(i)
            else:
                train_val_indices.append(i)
        
        # Split train_val
        np.random.seed(42)
        np.random.shuffle(train_val_indices)
        n_val = int(len(train_val_indices) * 0.15)
        val_indices = train_val_indices[:n_val]
        train_indices = train_val_indices[n_val:]
        
        if self.split == 'train':
            return train_indices
        elif self.split == 'val':
            return val_indices
        else:  # test
            return test_indices
    
    def _split_random(self) -> List[int]:
        """
        Random stratified split (stratified by angle).
        """
        # Group by angle
        angle_groups = {}
        for i, meta in enumerate(self.metadata):
            angle = meta['angle']
            if angle not in angle_groups:
                angle_groups[angle] = []
            angle_groups[angle].append(i)
        
        # Shuffle each group and split
        train_indices = []
        val_indices = []
        test_indices = []
        
        np.random.seed(42)
        for angle, indices in angle_groups.items():
            np.random.shuffle(indices)
            n = len(indices)
            n_train = int(n * config.train_ratio)
            n_val = int(n * config.val_ratio)
            
            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train + n_val])
            test_indices.extend(indices[n_train + n_val:])
        
        if self.split == 'train':
            return train_indices
        elif self.split == 'val':
            return val_indices
        else:  # test
            return test_indices
    
    def _load_normalization_stats(self):
        """Load normalization statistics from JSON."""
        stats_path = config.processed_data_dir / "normalization_stats.json"
        
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Normalization stats not found: {stats_path}. "
                f"Run preprocessing first."
            )
        
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        self.feature_mean = torch.tensor(stats['feature_mean'])
        self.feature_std = torch.tensor(stats['feature_std'])
        self.edge_attr_mean = torch.tensor(stats['edge_attr_mean'])
        self.edge_attr_std = torch.tensor(stats['edge_attr_std'])
        self.target_mean = torch.tensor(stats['target_mean'])
        self.target_std = torch.tensor(stats['target_std'])
        self.use_log_transform = stats['use_log_transform']
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Get a single graph.
        
        Args:
            idx: Index in the split
            
        Returns:
            data: PyG Data object (normalized if self.normalize=True)
        """
        # Map to global index
        global_idx = self.indices[idx]
        path = self.metadata[global_idx]['path']
        
        # Load graph
        data = torch.load(path, weights_only=False)
        
        # Apply normalization
        if self.normalize:
            data = self._normalize(data)
        
        # Apply optional transform
        if self.transform is not None:
            data = self.transform(data)
        
        return data
    
    def _normalize(self, data):
        """
        Normalize node features, edge features, and targets.
        
        Args:
            data: PyG Data object
            
        Returns:
            data: Normalized PyG Data object
        """
        # Z-score normalize node features
        data.x = (data.x - self.feature_mean) / (self.feature_std + 1e-8)
        
        # Z-score normalize edge features
        data.edge_attr = (data.edge_attr - self.edge_attr_mean) / (self.edge_attr_std + 1e-8)
        
        # Normalize targets
        if self.use_log_transform:
            # Sign-preserving log1p transform
            sign = torch.sign(data.y)
            log_y = sign * torch.log1p(torch.abs(data.y))
            # Then z-score
            data.y = (log_y - self.target_mean) / (self.target_std + 1e-8)
        else:
            # Direct z-score
            data.y = (data.y - self.target_mean) / (self.target_std + 1e-8)
        
        return data
    
    def denormalize_targets(self, y_normalized):
        """
        Denormalize targets back to physical units.
        
        Args:
            y_normalized: [N, 3] normalized targets
            
        Returns:
            y_physical: [N, 3] targets in Pa
        """
        # Reverse z-score
        y = y_normalized * self.target_std + self.target_mean
        
        # Reverse log1p transform if used
        if self.use_log_transform:
            sign = torch.sign(y)
            y = sign * (torch.exp(torch.abs(y)) - 1)
        
        return y


def get_dataloaders(
    strategy='re_interp',
    batch_size=None,
    num_workers=0,
    pin_memory=True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        strategy: Split strategy ('re_interp', 'angle_transfer', 'random')
        batch_size: Batch size (uses config.batch_size if None)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        train_loader, val_loader, test_loader
    """
    if batch_size is None:
        batch_size = config.batch_size
    
    # Create datasets
    train_dataset = BifurcationWSSDataset(split='train', strategy=strategy)
    val_dataset = BifurcationWSSDataset(split='val', strategy=strategy)
    test_dataset = BifurcationWSSDataset(split='test', strategy=strategy)
    
    print(f"Dataset sizes ({strategy} split):")
    print(f"  Train: {len(train_dataset)} graphs")
    print(f"  Val:   {len(val_dataset)} graphs")
    print(f"  Test:  {len(test_dataset)} graphs")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """Test dataset loading."""
    print("Testing BifurcationWSSDataset...\n")
    
    # Test loading
    try:
        dataset = BifurcationWSSDataset(split='train', strategy='re_interp')
        print(f"✓ Loaded train dataset: {len(dataset)} graphs")
        
        # Test getitem
        data = dataset[0]
        print(f"\nSample graph:")
        print(f"  Nodes: {data.x.shape[0]}")
        print(f"  Edges: {data.edge_index.shape[1]}")
        print(f"  Node features: {data.x.shape}")
        print(f"  Edge features: {data.edge_attr.shape}")
        print(f"  Targets: {data.y.shape}")
        print(f"  Flow params: {data.flow_params}")
        
        # Test dataloaders
        print("\nTesting dataloaders...")
        train_loader, val_loader, test_loader = get_dataloaders(batch_size=2)
        
        batch = next(iter(train_loader))
        print(f"\nSample batch:")
        print(f"  Batch nodes: {batch.x.shape[0]}")
        print(f"  Batch edges: {batch.edge_index.shape[1]}")
        print(f"  Batch size: {batch.num_graphs}")
        
        print("\n✓ All tests passed!")
        
    except FileNotFoundError as e:
        print(f"✗ {e}")
        print("Run preprocessing first: python -m Preprocessing.pre_process")

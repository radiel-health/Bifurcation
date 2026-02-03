"""
Dataset loader for bifurcation WSS prediction.

Loads preprocessed PyTorch Geometric graphs from ProcessedData/
and creates train/val/test splits with normalization.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

from config import Config


class WSSDataset(Dataset):
    """
    Dataset for bifurcation wall shear stress prediction.
    
    Loads preprocessed .pt files containing PyG graphs with:
    - x: node features [N, 14]
    - edge_index: graph connectivity [2, E]
    - y: WSS magnitude targets [N, 1]
    - pos: 3D coordinates [N, 3]
    - region: region labels [N]
    - wss_components: [N, 3] (wss_x, wss_y, wss_z)
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        normalize: bool = True,
        normalization_stats: Dict = None,
        transform=None,
        pre_transform=None,
    ):
        """
        Args:
            root: Root directory containing ProcessedData/
            split: 'train', 'val', or 'test'
            normalize: Whether to normalize features and targets
            normalization_stats: Pre-computed stats (for val/test)
            transform: Optional transform to apply
            pre_transform: Optional pre-transform
        """
        self.split = split
        self.normalize = normalize
        self.normalization_stats = normalization_stats
        
        super().__init__(root, transform, pre_transform)
        
        # Load all preprocessed files
        self._processed_dir = Path(root) / 'ProcessedData'
        self.file_paths = self._collect_files()
        
        # Create stratified split by bifurcation angle
        self._indices = self._create_split()
        
        # Compute normalization statistics from training set
        if self.normalize and self.normalization_stats is None:
            if self.split == 'train':
                self.normalization_stats = self._compute_normalization_stats()
            else:
                raise ValueError("normalization_stats must be provided for val/test splits")
    
    def _collect_files(self) -> List[Path]:
        """Collect all .pt files from ProcessedData directory."""
        files = []
        for angle_dir in self._processed_dir.iterdir():
            if angle_dir.is_dir() and angle_dir.name.startswith('angle'):
                for pt_file in angle_dir.glob('*.pt'):
                    files.append(pt_file)
        
        files.sort()  # Ensure consistent ordering
        print(f"Found {len(files)} preprocessed cases")
        return files
    
    def _create_split(self) -> List[int]:
        """Create stratified train/val/test split by angle."""
        # Group files by angle for stratification
        angle_groups = {'angle30': [], 'angle45': [], 'angle60': []}
        for idx, file_path in enumerate(self.file_paths):
            angle = file_path.parent.name
            if angle in angle_groups:
                angle_groups[angle].append(idx)
        
        # Remove empty groups
        angle_groups = {k: v for k, v in angle_groups.items() if len(v) > 0}
        
        # Split each angle group with same proportions
        train_indices, val_indices, test_indices = [], [], []
        
        for angle, indices in angle_groups.items():
            if len(indices) < 3:
                # If too few samples, put all in training
                train_indices.extend(indices)
                continue
                
            # 70% train, 15% val, 15% test
            train_idx, temp_idx = train_test_split(
                indices, test_size=0.3, random_state=42
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, random_state=42
            )
            
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)
            test_indices.extend(test_idx)
        
        # Sort for consistency
        train_indices.sort()
        val_indices.sort()
        test_indices.sort()
        
        split_map = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        
        indices = split_map[self.split]
        print(f"{self.split} split: {len(indices)} cases")
        
        return indices
    
    def _compute_normalization_stats(self) -> Dict:
        """Compute mean and std for features and targets from training set."""
        print("Computing normalization statistics from training set...")
        
        feature_list = []
        target_list = []
        
        # Collect all features and targets from training set
        for idx in self._indices:
            data = torch.load(self.file_paths[idx], weights_only=False)
            feature_list.append(data.x)
            target_list.append(data.y)
        
        # Stack and compute statistics
        all_features = torch.cat(feature_list, dim=0)  # [N_total, 14]
        all_targets = torch.cat(target_list, dim=0)    # [N_total, 1]
        
        stats = {
            'feature_mean': all_features.mean(dim=0),   # [14]
            'feature_std': all_features.std(dim=0),     # [14]
            'target_mean': all_targets.mean(),
            'target_std': all_targets.std(),
        }
        
        # Avoid division by zero
        stats['feature_std'][stats['feature_std'] < 1e-6] = 1.0
        if stats['target_std'] < 1e-6:
            stats['target_std'] = torch.tensor(1.0)
        
        print(f"Feature mean: {stats['feature_mean'][:5]}...")  # First 5
        print(f"Feature std: {stats['feature_std'][:5]}...")
        print(f"Target mean: {stats['target_mean']:.6f}")
        print(f"Target std: {stats['target_std']:.6f}")
        
        return stats
    
    def len(self) -> int:
        """Number of samples in this split."""
        return len(self._indices)
    
    def indices(self):
        """Return indices for this split (required by PyG Dataset)."""
        return range(len(self._indices))
    
    def get(self, idx: int) -> Data:
        """Load and return a single graph."""
        # idx is already the index into _indices (PyG handles this)
        # Map split index to global file index
        file_idx = self._indices[idx]
        file_path = self.file_paths[file_idx]
        
        # Load preprocessed graph
        data = torch.load(file_path, weights_only=False)
        
        # Apply normalization if enabled
        if self.normalize:
            data = self._normalize_data(data)
        
        return data
    
    def _normalize_data(self, data: Data) -> Data:
        """Normalize features and targets using pre-computed statistics."""
        stats = self.normalization_stats
        
        # Normalize features: (x - mean) / std
        data.x = (data.x - stats['feature_mean']) / stats['feature_std']
        
        # Normalize targets
        data.y = (data.y - stats['target_mean']) / stats['target_std']
        
        return data
    
    def denormalize_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Denormalize model predictions back to original scale.
        
        Args:
            predictions: Normalized predictions [N, 1] or [N]
            
        Returns:
            Denormalized predictions in original WSS units
        """
        if not self.normalize:
            return predictions
        
        stats = self.normalization_stats
        return predictions * stats['target_std'] + stats['target_mean']


def create_dataloaders(
    config: Config,
    batch_size: int = 1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Configuration object
        batch_size: Batch size (default 1 for full graphs)
        num_workers: Number of data loading workers
        
    Returns:
        train_loader, val_loader, test_loader, normalization_stats
    """
    # Create training dataset and compute normalization stats
    train_dataset = WSSDataset(
        root=config.data_root,
        split='train',
        normalize=config.normalize_features,
    )
    
    # Get normalization stats from training set
    norm_stats = train_dataset.normalization_stats
    
    # Create val and test datasets with same normalization
    val_dataset = WSSDataset(
        root=config.data_root,
        split='val',
        normalize=config.normalize_features,
        normalization_stats=norm_stats,
    )
    
    test_dataset = WSSDataset(
        root=config.data_root,
        split='test',
        normalize=config.normalize_features,
        normalization_stats=norm_stats,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} cases")
    print(f"  Val:   {len(val_dataset)} cases")
    print(f"  Test:  {len(test_dataset)} cases")
    print(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} cases")
    
    return train_loader, val_loader, test_loader, norm_stats


if __name__ == '__main__':
    """Test dataset loading."""
    from config import Config
    
    print("Testing WSSDataset...")
    config = Config()
    
    # Test dataset creation
    train_dataset = WSSDataset(
        root=config.data_root,
        split='train',
        normalize=True,
    )
    
    print(f"\nTrain dataset: {len(train_dataset)} samples")
    
    # Test loading a sample
    sample = train_dataset[0]
    print(f"\nSample graph:")
    print(f"  Nodes: {sample.num_nodes}")
    print(f"  Edges: {sample.num_edges}")
    print(f"  Features shape: {sample.x.shape}")
    print(f"  Target shape: {sample.y.shape}")
    print(f"  Position shape: {sample.pos.shape}")
    print(f"  Case: {sample.case_name}")
    
    # Test dataloaders
    print("\n" + "="*60)
    print("Testing dataloaders...")
    train_loader, val_loader, test_loader, norm_stats = create_dataloaders(
        config, batch_size=2
    )
    
    # Test batch loading
    batch = next(iter(train_loader))
    print(f"\nBatch info:")
    print(f"  Batch size: {batch.num_graphs}")
    print(f"  Total nodes: {batch.num_nodes}")
    print(f"  Total edges: {batch.num_edges}")
    print(f"  Features: {batch.x.shape}")
    print(f"  Targets: {batch.y.shape}")
    
    print("\nDataset test complete!")

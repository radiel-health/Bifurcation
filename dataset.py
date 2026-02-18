"""
Dataset loader for 3D point cloud WSS prediction graphs

Loads preprocessed PyG graphs from ProcessedData/3D/ with:
- Train/val/test splits
- Normalization statistics
- Batching support
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from config import Config
from utils.stats import (
    load_or_compute_stats,
    normalize_data,
    denormalize_targets,
    NormalizationStats,
)
from typing import final, override


@final
class WSSDataset(Dataset[Data]):
    """
    Dataset for loading 3D point cloud wall shear stress graphs

    Features:
    - Loads from ProcessedData/3D/ directory
    - Computes normalization statistics on training set
    - Supports train/val/test splits
    """

    def __init__(
        self,
        root: str | None = None,
        split: str = "train",
        normalize: bool = True,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
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
        super().__init__()
        self.split = split
        self.normalize = normalize
        self.split_ratios = split_ratios
        self.seed = seed

        # Get root directory
        rootPath: Path | None = (
            Config().repo_root / "Bifurcation" if root is None else Path(root)
        )
        self.data_dir = rootPath / "ProcessedData" / "3D"

        self.stats_file = self.data_dir / "normalization_stats.json"

        # Validate
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        assert abs(sum(split_ratios) - 1.0) < 1e-6, f"Split ratios must sum to 1.0"

        # Get all .pt files
        self.all_files = sorted(list(self.data_dir.glob("*.pt")))

        print(f"\nFound {len(self.all_files)} preprocessed graphs in {self.data_dir}")

        # Create splits
        self._create_splits()

        # Load normalization stats or compute them
        if self.normalize:
            self.normalization_stats: NormalizationStats = load_or_compute_stats(
                self.files, self.data_dir, self.split
            )

    def _create_splits(self):
        """Create random train/val/test splits"""
        np.random.seed(self.seed)

        n = len(self.all_files)
        indices = np.random.permutation(n)

        n_train = int(n * self.split_ratios[0])
        n_val = int(n * self.split_ratios[1])

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        match self.split:
            case "train":
                self.files = [self.all_files[i] for i in train_idx]
            case "val":
                self.files = [self.all_files[i] for i in val_idx]
            case _:
                self.files = [self.all_files[i] for i in test_idx]

        print(f"\nSplit: {self.split}")
        print(f"  Total graphs: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    @override
    def __getitem__(self, idx: int):
        """Load and optionally normalize a graph"""
        data = torch.load(self.files[idx], weights_only=False)

        if self.normalize:
            assert self.normalization_stats is not None
            data = normalize_data(
                data=data, normalization_stats=self.normalization_stats
            )

        return data

    def denormalize_targets(self, normalized_targets: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions back to original WSS scale"""
        assert (
            self.normalize and self.normalization_stats is not None
        ), "Cannot denormalize un-normalized targets"
        return denormalize_targets(
            normalized_targets,
            self.normalization_stats["target_mean"],
            self.normalization_stats["target_std"],
        )


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    normalize: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Get train/val/test dataloaders"""
    train_dataset = WSSDataset(
        split="train", normalize=normalize, split_ratios=split_ratios, seed=seed
    )
    val_dataset = WSSDataset(
        split="val", normalize=normalize, split_ratios=split_ratios, seed=seed
    )
    test_dataset = WSSDataset(
        split="test", normalize=normalize, split_ratios=split_ratios, seed=seed
    )

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

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("=" * 80)
    print("Testing WSSDataset (3D)")
    print("=" * 80)

    train_ds = WSSDataset(split="train", normalize=True, seed=42)
    val_ds = WSSDataset(split="val", normalize=True, seed=42)
    test_ds = WSSDataset(split="test", normalize=True, seed=42)

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
    print(
        f"  Denormalized WSS range: [{denorm_targets.min():.6e}, {denorm_targets.max():.6e}]"
    )

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
    print("=" * 80)

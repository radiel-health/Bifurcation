import json
from pathlib import Path
from typing import TypedDict
import torch
from torch_geometric.data import Data


class NormalizationStats(TypedDict):
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor
    flow_mean: torch.Tensor
    flow_std: torch.Tensor


def compute_stats(file_paths: list[Path]) -> NormalizationStats:
    """
    Compute normalization statistics from training data.

    Args:
        file_paths: list of paths to .pt graph files

    Returns:
        Dictionary with mean/std for features, targets, and flow params
    """
    all_features = []
    all_targets = []
    all_flow_params = []

    for file_path in file_paths:
        data = torch.load(file_path, weights_only=False)
        all_features.append(data.x)
        all_targets.append(data.y)
        all_flow_params.append(data.flow_params.unsqueeze(0))

    all_features = torch.cat(all_features, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_flow_params = torch.cat(all_flow_params, dim=0)

    feature_mean = all_features.mean(dim=0)
    feature_std = all_features.std(dim=0)
    feature_std[feature_std < 1e-8] = 1.0

    sign = torch.sign(all_targets)
    log_mag = torch.log1p(torch.abs(all_targets))
    signed_log = sign * log_mag

    target_mean = signed_log.mean(dim=0)
    target_std = signed_log.std(dim=0)
    target_std[target_std < 1e-8] = 1.0

    flow_mean = all_flow_params.mean(dim=0)
    flow_std = all_flow_params.std(dim=0)
    flow_std[flow_std < 1e-8] = 1.0

    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "flow_mean": flow_mean,
        "flow_std": flow_std,
    }


def save_stats(stats: NormalizationStats, stats_file: Path):
    """Save normalization stats to JSON file."""
    stats_dict = {
        "feature_mean": stats["feature_mean"].tolist(),
        "feature_std": stats["feature_std"].tolist(),
        "target_mean": stats["target_mean"].tolist(),
        "target_std": stats["target_std"].tolist(),
        "flow_mean": stats["flow_mean"].tolist(),
        "flow_std": stats["flow_std"].tolist(),
    }

    with open(stats_file, "w") as f:
        json.dump(stats_dict, f, indent=2)


def load_stats(stats_file: Path) -> NormalizationStats:
    """Load normalization stats from JSON file."""
    with open(stats_file, "r") as f:
        stats = json.load(f)

    return {
        "feature_mean": torch.tensor(stats["feature_mean"]),
        "feature_std": torch.tensor(stats["feature_std"]),
        "target_mean": torch.tensor(stats["target_mean"]),
        "target_std": torch.tensor(stats["target_std"]),
        "flow_mean": torch.tensor(stats["flow_mean"]),
        "flow_std": torch.tensor(stats["flow_std"]),
    }


def load_or_compute_stats(
    files: list[Path], data_dir: Path, split: str
) -> NormalizationStats:
    """
    Load normalization stats from file or compute from training set.

    Args:
        files: list of file paths for current split
        data_dir: Directory containing the data
        split: 'train', 'val', or 'test'
        normalize: Whether normalization is enabled

    Returns:
        Dictionary with feature_mean, feature_std, target_mean, target_std, flow_mean, flow_std
    """
    stats_file = data_dir / "normalization_stats.json"

    if stats_file.exists():
        print(f"\nLoaded normalization stats from {stats_file}")
        return load_stats(stats_file)

    elif split == "train":
        print("\nComputing normalization statistics from training set...")
        stats = compute_stats(files)
        save_stats(stats, stats_file)
        print(f"Saved normalization stats to {stats_file}")
        print(f"\nNormalization statistics:")
        print(f"  Feature mean: {stats['feature_mean']}")
        print(f"  Feature std: {stats['feature_std']}")
        print(f"  Target (log) mean: {stats['target_mean']}")
        print(f"  Target (log) std: {stats['target_std']}")
        print(f"  Flow mean: {stats['flow_mean']}")
        print(f"  Flow std: {stats['flow_std']}")
        return stats

    else:
        raise FileNotFoundError(
            f"Normalization stats not found at {stats_file}. "
            "Please run with split='train' first to compute statistics."
        )


def normalize_data(
    data: Data,
    normalization_stats: NormalizationStats,
) -> Data:
    """Normalize features, targets, and flow parameters using precomputed stats."""
    feature_mean = normalization_stats["feature_mean"]
    feature_std = normalization_stats["feature_std"]
    target_mean = normalization_stats["target_mean"]
    target_std = normalization_stats["target_std"]
    flow_mean = normalization_stats["flow_mean"]
    flow_std = normalization_stats["flow_std"]
    data.x = (data.x - feature_mean) / feature_std

    sign = torch.sign(data.y)
    log_mag = torch.log1p(torch.abs(data.y))
    signed_log = sign * log_mag
    data.y = (signed_log - target_mean) / target_std

    data.flow_params = (data.flow_params - flow_mean) / flow_std

    return data


def denormalize_targets(
    normalized_targets: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    """Convert normalized predictions back to original WSS scale."""
    signed_log = normalized_targets * target_std + target_mean

    sign = torch.sign(signed_log)
    log_mag = torch.abs(signed_log)

    mag = torch.expm1(log_mag)

    return sign * mag

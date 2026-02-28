import json
from pathlib import Path
from typing import TypedDict
import torch
from torch_geometric.data import Data


class NormalizationStats(TypedDict):
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    target_scale: torch.Tensor  # NEW: Single scalar for vector-safe scaling
    flow_mean: torch.Tensor
    flow_std: torch.Tensor


def compute_stats(file_paths: list[Path]) -> NormalizationStats:
    """
    Compute normalization statistics from training data.
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

    # 1. Coordinate Stats (Isotropic Scaling)
    # Translate by 3D mean, but scale uniformly using a single global scalar!
    feature_mean = all_features.mean(dim=0)
    feature_std = all_features.std() # Single scalar prevents geometric warping
    if feature_std < 1e-8:
        feature_std = torch.tensor(1.0)

    # 2. Target WSS Stats (Vector-Safe Scaling)
    # Scale by 99th percentile magnitude to preserve vector direction
    mags = torch.norm(all_targets, dim=1)
    target_scale = torch.quantile(mags, 0.99)
    if target_scale < 1e-8:
        target_scale = torch.tensor(1.0)

    # 3. Flow Stats (Re)
    flow_mean = all_flow_params.mean(dim=0)
    flow_std = all_flow_params.std(dim=0)
    flow_std[flow_std < 1e-8] = 1.0

    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_scale": target_scale,
        "flow_mean": flow_mean,
        "flow_std": flow_std,
    }


def save_stats(stats: NormalizationStats, stats_file: Path):
    """Save normalization stats to JSON file."""
    stats_dict = {
        "feature_mean": stats["feature_mean"].tolist(),
        "feature_std": float(stats["feature_std"]),
        "target_scale": float(stats["target_scale"]),
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
        "target_scale": torch.tensor(stats["target_scale"]),
        "flow_mean": torch.tensor(stats["flow_mean"]),
        "flow_std": torch.tensor(stats["flow_std"]),
    }


def load_or_compute_stats(
    files: list[Path], data_dir: Path, split: str
) -> NormalizationStats:
    """Load normalization stats from file or compute from training set."""
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
        print(f"  Feature std (global scale): {stats['feature_std']:.6f}")
        print(f"  Target scale (99th %ile Mag): {stats['target_scale']:.6f}")
        print(f"  Flow mean: {stats['flow_mean']}")
        print(f"  Flow std: {stats['flow_std']}")
        return stats

    else:
        raise FileNotFoundError(
            f"Normalization stats not found at {stats_file}."
        )


def normalize_data(
    data: Data,
    normalization_stats: NormalizationStats,
) -> Data:
    """Normalize features, targets, and flow parameters using precomputed stats."""
    # Centered and uniformly scaled coordinates
    data.x = (data.x - normalization_stats["feature_mean"]) / normalization_stats["feature_std"]

    # Pure scalar multiplication preserves exact 3D vector directions!
    data.y = data.y / normalization_stats["target_scale"]

    data.flow_params = (data.flow_params - normalization_stats["flow_mean"]) / normalization_stats["flow_std"]

    return data


def denormalize_targets(
    normalized_targets: torch.Tensor,
    target_scale: torch.Tensor,
) -> torch.Tensor:
    """Convert normalized predictions back to original WSS scale."""
    return normalized_targets * target_scale

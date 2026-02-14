"""
Evaluation utilities for bifurcation WSS predictions.

Computes per-component and magnitude metrics (MAE, RMSE, R²),
broken down by geometry and Reynolds number.

Usage:
    # Evaluate a checkpoint on its test split
    python -m Bifurcation.evaluate --model Models/best_model.pt --mode random

    # Evaluate a LOOCV fold on the held-out geometry
    python -m Bifurcation.evaluate \\
        --model Models/best_model_bifurcation_angle45_750_ascii.pt \\
        --mode loocv-geo --holdout bifurcation_angle45_750_ascii
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from Bifurcation.config import config
from Bifurcation.dataset import (
    BifurcationWSSDataset,
    denormalize_wss,
    get_split_paths,
    compute_normalization_stats,
)
from Bifurcation.model import BifurcationWSSPredictor


# ============================================================================
# Metrics
# ============================================================================

def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def r_squared(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def relative_error(pred: np.ndarray, true: np.ndarray, eps: float = 1e-10) -> float:
    return float(np.mean(np.abs(pred - true) / (np.abs(true) + eps)))


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """
    Compute a full suite of metrics on (N, 3) WSS arrays.

    Returns dict with per-component and magnitude metrics.
    """
    results = {}

    # Per-component
    for i, label in enumerate(["wss_x", "wss_y", "wss_z"]):
        p, t = pred[:, i], true[:, i]
        results[f"{label}_mae"] = mae(p, t)
        results[f"{label}_rmse"] = rmse(p, t)
        results[f"{label}_r2"] = r_squared(p, t)
        results[f"{label}_rel_err"] = relative_error(p, t)

    # Magnitude
    mag_pred = np.linalg.norm(pred, axis=1)
    mag_true = np.linalg.norm(true, axis=1)
    results["mag_mae"] = mae(mag_pred, mag_true)
    results["mag_rmse"] = rmse(mag_pred, mag_true)
    results["mag_r2"] = r_squared(mag_pred, mag_true)
    results["mag_rel_err"] = relative_error(mag_pred, mag_true)

    return results


# ============================================================================
# Full evaluation
# ============================================================================

@torch.no_grad()
def evaluate_loader(
    model: BifurcationWSSPredictor,
    loader: DataLoader,
    norm_stats: dict,
    device: torch.device,
) -> tuple[dict, list]:
    """
    Run inference on a DataLoader and collect predictions + metrics.

    Returns
    -------
    agg_metrics : dict   – aggregated metrics over all samples
    per_sample  : list   – [{geo, re, metrics, pred, true}, …]
    """
    model.eval()
    all_pred, all_true = [], []
    per_sample = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        y_norm = model(batch)

        # De-normalise
        y_pred = denormalize_wss(y_norm, norm_stats).cpu().numpy()
        y_true_norm = batch.y.cpu()
        y_true = denormalize_wss(y_true_norm, norm_stats).numpy()

        all_pred.append(y_pred)
        all_true.append(y_true)

        # Per-sample breakdown (batch_size=1 assumed)
        geo = batch.geo_name[0] if isinstance(batch.geo_name, list) else batch.geo_name
        re_name = batch.re_name[0] if isinstance(batch.re_name, list) else batch.re_name
        sample_metrics = compute_metrics(y_pred, y_true)
        per_sample.append({
            "geo": geo,
            "re": re_name,
            "metrics": sample_metrics,
        })

    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    agg_metrics = compute_metrics(all_pred, all_true)

    return agg_metrics, per_sample


def evaluate_checkpoint(
    ckpt_path: str,
    mode: str = "random",
    holdout_geo: str | None = None,
    device: torch.device | None = None,
) -> tuple[dict, list]:
    """High-level entry: load checkpoint → evaluate on test split."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    model = BifurcationWSSPredictor(
        node_feat_dim=cfg.get("node_feat_dim", config.node_feat_dim),
        edge_channels=cfg.get("edge_feat_dim", config.edge_feat_dim),
        aggregated_edge_feat_dim=cfg.get("aggregated_edge_feat_dim", config.aggregated_edge_feat_dim),
        hidden_gcn_dim=cfg.get("hidden_gcn_dim", config.hidden_gcn_dim),
        out_channels=cfg.get("output_dim", config.output_dim),
        num_gcn_layers=cfg.get("num_gcn_layers", config.num_gcn_layers),
        context_dim=cfg.get("context_dim", config.context_dim),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    norm_stats = ckpt.get("norm_stats")
    if norm_stats is None:
        from Bifurcation.dataset import load_normalization_stats
        norm_stats = load_normalization_stats()

    # Get test split
    _train_p, _val_p, test_p = get_split_paths(
        mode, holdout_geo=holdout_geo or cfg.get("holdout_geo")
    )
    test_ds = BifurcationWSSDataset(test_p, norm_stats)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    agg, per_sample = evaluate_loader(model, test_loader, norm_stats, device)
    return agg, per_sample


# ============================================================================
# Pretty-print helpers
# ============================================================================

def print_metrics(metrics: dict, title: str = ""):
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    for key in ["wss_x", "wss_y", "wss_z", "mag"]:
        print(f"  {key:8s}  MAE={metrics[f'{key}_mae']:.6e}  "
              f"RMSE={metrics[f'{key}_rmse']:.6e}  "
              f"R²={metrics[f'{key}_r2']:.4f}  "
              f"RelErr={metrics[f'{key}_rel_err']:.4f}")


def print_per_geometry(per_sample: list):
    """Aggregate per-sample results by geometry."""
    geo_groups: dict[str, list] = defaultdict(list)
    for s in per_sample:
        geo_groups[s["geo"]].append(s["metrics"])

    print(f"\n{'='*60}")
    print("  Per-Geometry Magnitude RMSE & R²")
    print(f"{'='*60}")
    for geo in sorted(geo_groups):
        rmses = [m["mag_rmse"] for m in geo_groups[geo]]
        r2s = [m["mag_r2"] for m in geo_groups[geo]]
        print(f"  {geo:45s}  RMSE={np.mean(rmses):.6e}  R²={np.mean(r2s):.4f}")


def print_per_re(per_sample: list):
    """Aggregate per-sample results by Reynolds number."""
    re_groups: dict[str, list] = defaultdict(list)
    for s in per_sample:
        re_groups[s["re"]].append(s["metrics"])

    print(f"\n{'='*60}")
    print("  Per-Re Magnitude RMSE & R²")
    print(f"{'='*60}")
    for re_name in sorted(re_groups, key=lambda x: int(x.replace("Re", ""))):
        rmses = [m["mag_rmse"] for m in re_groups[re_name]]
        r2s = [m["mag_r2"] for m in re_groups[re_name]]
        print(f"  {re_name:10s}  RMSE={np.mean(rmses):.6e}  R²={np.mean(r2s):.4f}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate bifurcation WSS model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", choices=["random", "loocv-geo"], default="random")
    parser.add_argument("--holdout", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save", type=str, default=None,
                        help="Save results JSON to this path")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agg, per_sample = evaluate_checkpoint(
        args.model, mode=args.mode, holdout_geo=args.holdout, device=device
    )

    print_metrics(agg, title="Aggregate Test Metrics")
    print_per_geometry(per_sample)
    print_per_re(per_sample)

    if args.save:
        out = {"aggregate": agg, "per_sample": [
            {"geo": s["geo"], "re": s["re"], **s["metrics"]} for s in per_sample
        ]}
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved results → {args.save}")

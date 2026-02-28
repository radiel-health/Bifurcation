"""
Evaluation script v2 for bifurcation WSS predictions.

Improvements over evaluate.py (v1):
  - Per-component metrics (MAE, RMSE, R²) with separate per-branch breakdown
  - Per-Re sweep: R² vs Re plot (identifies problematic Reynolds regimes)
  - Calibration check: does model uncertainty correlate with actual error?
  - Comparison mode: side-by-side v1 vs v2 metrics (--compare)
  - Results saved to results_v2/

Usage:
    # Evaluate v2 checkpoint on test split
    python -m Bifurcation.evaluate_v2 --model Models_v2/best_model_v2.pt

    # Side-by-side comparison with v1
    python -m Bifurcation.evaluate_v2 \\
        --model Models_v2/best_model_v2.pt \\
        --compare Models/best_model.pt

    # LOOCV fold evaluation
    python -m Bifurcation.evaluate_v2 \\
        --model Models_v2/best_model_v2_bifurcation_angle45_750_ascii.pt \\
        --mode loocv-geo --holdout bifurcation_angle45_750_ascii

    # Save results JSON + plots
    python -m Bifurcation.evaluate_v2 --model Models_v2/best_model_v2.pt --save --plot
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from Bifurcation.config_v2 import config_v2
from Bifurcation.dataset_v2 import (
    BifurcationWSSDatasetV2,
    denormalize_wss_v2,
    get_split_paths_v2,
    compute_normalization_stats_v2,
)
from Bifurcation.Models.bif_v2 import BifurcationWSSPredictorV2

# V1 imports (for comparison mode)
from Bifurcation.config import config as config_v1
from Bifurcation.dataset import (
    BifurcationWSSDataset,
    denormalize_wss,
    get_split_paths,
    compute_normalization_stats,
)
from Bifurcation.model import BifurcationWSSPredictor


# ============================================================================
# Metrics  (same as v1 for consistency)
# ============================================================================

def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def r_squared(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def relative_error(pred: np.ndarray, true: np.ndarray, floor_percentile: float = 10.0) -> float:
    """
    Median relative error with a percentile floor on the denominator.

    Nodes where |true| < 10th-percentile of |true| across the field get their
    denominator floored to that percentile value, preventing near-zero WSS nodes
    (e.g. symmetry-plane nodes in WSS_X) from blowing up the metric.
    """
    abs_true = np.abs(true)
    floor    = float(np.percentile(abs_true, floor_percentile))
    denom    = np.maximum(abs_true, floor)
    return float(np.median(np.abs(pred - true) / denom))


def nrmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Normalized RMSE = RMSE / std(true). Scale-free, unaffected by near-zero values."""
    std_true = float(np.std(true))
    return float(np.sqrt(np.mean((pred - true) ** 2)) / (std_true + 1e-30))


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> Dict:
    results = {}
    for i, label in enumerate(["wss_x", "wss_y", "wss_z"]):
        p, t = pred[:, i], true[:, i]
        results[f"{label}_mae"]     = mae(p, t)
        results[f"{label}_rmse"]    = rmse(p, t)
        results[f"{label}_r2"]      = r_squared(p, t)
        results[f"{label}_rel_err"] = relative_error(p, t)
        results[f"{label}_nrmse"]   = nrmse(p, t)

    mag_pred = np.linalg.norm(pred, axis=1)
    mag_true = np.linalg.norm(true, axis=1)
    results["mag_mae"]     = mae(mag_pred, mag_true)
    results["mag_rmse"]    = rmse(mag_pred, mag_true)
    results["mag_r2"]      = r_squared(mag_pred, mag_true)
    results["mag_rel_err"] = relative_error(mag_pred, mag_true)
    results["mag_nrmse"]   = nrmse(mag_pred, mag_true)

    return results


# ============================================================================
# Full evaluation over a DataLoader
# ============================================================================

@torch.no_grad()
def evaluate_loader_v2(
    model:      BifurcationWSSPredictorV2,
    loader:     DataLoader,
    norm_stats: Dict,
    device:     torch.device,
) -> Tuple[Dict, List[Dict]]:
    """
    Run inference and collect per-sample results.

    Returns
    -------
    agg_metrics : dict
    per_sample  : list of { geo, re, angle, branch_depth_mean, metrics }
    """
    model.eval()
    all_pred, all_true = [], []
    per_sample = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch  = batch.to(device)
        y_norm = model(batch)

        y_pred = denormalize_wss_v2(y_norm,    norm_stats).cpu().numpy()
        y_true = denormalize_wss_v2(batch.y,   norm_stats).cpu().numpy()

        all_pred.append(y_pred)
        all_true.append(y_true)

        geo    = batch.geo_name[0] if isinstance(batch.geo_name, list) else batch.geo_name
        re_nm  = batch.re_name[0]  if isinstance(batch.re_name,  list) else batch.re_name
        angle_deg, _ = config_v2.parse_geometry_folder(geo)

        # Branch depth feature (index 5) from the normalised x
        # Denormalise just the branch_depth column
        x_mean5 = norm_stats["x_mean"][5]
        x_std5  = norm_stats["x_std"][5]
        bd_norm = batch.x[:, 5].cpu().numpy()
        branch_depth = bd_norm * x_std5 + x_mean5  # ≈ 0 or 1

        sample_metrics = compute_metrics(y_pred, y_true)
        per_sample.append({
            "geo":               geo,
            "re":                re_nm,
            "angle":             angle_deg,
            "branch_depth_mean": float(branch_depth.mean()),
            "metrics":           sample_metrics,
        })

    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    return compute_metrics(all_pred, all_true), per_sample, all_pred, all_true


# ============================================================================
# Calibration check
# ============================================================================

def calibration_check(
    model:      BifurcationWSSPredictorV2,
    loader:     DataLoader,
    norm_stats: Dict,
    device:     torch.device,
    mc_samples: int = 30,
) -> Dict:
    """
    Check if model uncertainty (std across MC passes) correlates with error.

    A well-calibrated model should have higher std where its mean prediction
    is wrong. Computes Spearman correlation between per-node |error| and std.
    """
    from scipy.stats import spearmanr

    model.enable_mc_mode()
    all_errors, all_stds = [], []

    for batch in tqdm(loader, desc="Calibration", leave=False):
        batch = batch.to(device)
        preds = []
        with torch.no_grad():
            for _ in range(mc_samples):
                y_norm = model(batch)
                y_phys = denormalize_wss_v2(y_norm, norm_stats)
                preds.append(y_phys.cpu())

        preds_t  = torch.stack(preds, 0)         # [S, N, 3]
        mean_p   = preds_t.mean(0).numpy()       # [N, 3]
        std_p    = preds_t.std(0).numpy()        # [N, 3]

        y_true   = denormalize_wss_v2(batch.y, norm_stats).cpu().numpy()
        error    = np.abs(mean_p - y_true)

        all_errors.append(error.mean(axis=1))    # per-node mean-component error
        all_stds.append(std_p.mean(axis=1))      # per-node mean-component std

    model.disable_mc_mode()

    all_errors = np.concatenate(all_errors)
    all_stds   = np.concatenate(all_stds)

    rho, pval = spearmanr(all_errors, all_stds)
    return {"spearman_rho": float(rho), "p_value": float(pval),
            "interpretation": "good" if rho > 0.3 else "poor"}


# ============================================================================
# High-level entry point
# ============================================================================

def evaluate_checkpoint_v2(
    ckpt_path:   str,
    mode:        str                     = "re_angle_stratified",
    holdout_geo: Optional[str]           = None,
    device:      Optional[torch.device]  = None,
    run_calibration: bool                = False,
) -> Tuple[Dict, List[Dict]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg    = ckpt.get("config", {})

    model = BifurcationWSSPredictorV2(
        node_feat_dim  = cfg.get("node_feat_dim",  config_v2.node_feat_dim),
        edge_feat_dim  = cfg.get("edge_feat_dim",  config_v2.edge_feat_dim),
        hidden_dim     = cfg.get("hidden_dim",     config_v2.hidden_dim),
        out_channels   = cfg.get("output_dim",     config_v2.output_dim),
        num_layers     = cfg.get("num_layers",     config_v2.num_layers),
        context_dim    = cfg.get("context_dim",    config_v2.context_dim),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    norm_stats = ckpt.get("norm_stats") or compute_normalization_stats_v2()
    _train_p, _val_p, test_p = get_split_paths_v2(
        mode, holdout_geo=holdout_geo or cfg.get("holdout_geo")
    )
    test_ds     = BifurcationWSSDatasetV2(test_p, norm_stats)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    agg, per_sample, all_pred, all_true = evaluate_loader_v2(model, test_loader, norm_stats, device)

    if run_calibration:
        calib = calibration_check(model, test_loader, norm_stats, device)
        agg["calibration"] = calib

    return agg, per_sample, all_pred, all_true


# ============================================================================
# Pretty-print helpers
# ============================================================================

def print_metrics(metrics: Dict, title: str = ""):
    if title:
        print(f"\n{'='*75}")
        print(f"  {title}")
        print(f"{'='*75}")
    for key in ["wss_x", "wss_y", "wss_z", "mag"]:
        print(f"  {key:8s}  "
              f"MAE={metrics[f'{key}_mae']:.4e}  "
              f"RMSE={metrics[f'{key}_rmse']:.4e}  "
              f"R²={metrics[f'{key}_r2']:.4f}  "
              f"MedRelErr={metrics[f'{key}_rel_err']:.4f}  "
              f"NRMSE={metrics[f'{key}_nrmse']:.4f}")


def print_per_geometry(per_sample: List[Dict]):
    geo_groups: Dict[str, List] = defaultdict(list)
    for s in per_sample:
        geo_groups[s["geo"]].append(s["metrics"])

    print(f"\n{'='*65}")
    print("  Per-Geometry:  mag RMSE  |  X R²  Y R²  Z R²")
    print(f"{'='*65}")
    for geo in sorted(geo_groups):
        ms   = geo_groups[geo]
        rmse = np.mean([m["mag_rmse"]    for m in ms])
        rx   = np.mean([m["wss_x_r2"]   for m in ms])
        ry   = np.mean([m["wss_y_r2"]   for m in ms])
        rz   = np.mean([m["wss_z_r2"]   for m in ms])
        print(f"  {geo:45s}  {rmse:.4e}  | {rx:.3f} {ry:.3f} {rz:.3f}")


def print_per_re(per_sample: List[Dict]):
    re_groups: Dict[str, List] = defaultdict(list)
    for s in per_sample:
        re_groups[s["re"]].append(s["metrics"])

    print(f"\n{'='*65}")
    print("  Per-Re:  mag R²  |  X R²  Y R²  Z R²")
    print(f"{'='*65}")
    for re_nm in sorted(re_groups, key=lambda x: int(x.replace("Re", ""))):
        ms  = re_groups[re_nm]
        rmg = np.mean([m["mag_r2"]     for m in ms])
        rx  = np.mean([m["wss_x_r2"]  for m in ms])
        ry  = np.mean([m["wss_y_r2"]  for m in ms])
        rz  = np.mean([m["wss_z_r2"]  for m in ms])
        print(f"  {re_nm:10s}  mag={rmg:.3f}  | {rx:.3f} {ry:.3f} {rz:.3f}")


def print_per_branch(per_sample: List[Dict]):
    """
    Break down metrics by parent (branch_depth ≈ 0) vs daughter (≈ 1).
    """
    parent_metrics, daughter_metrics = [], []
    for s in per_sample:
        if s["branch_depth_mean"] < 0.5:
            parent_metrics.append(s["metrics"])
        else:
            daughter_metrics.append(s["metrics"])

    print(f"\n{'='*65}")
    print("  Per-Branch:  mag R²  |  X R²  Y R²  Z R²")
    print(f"{'='*65}")
    for label, group in [("Parent  (depth=0)", parent_metrics),
                          ("Daughter(depth=1)", daughter_metrics)]:
        if group:
            rmg = np.mean([m["mag_r2"]     for m in group])
            rx  = np.mean([m["wss_x_r2"]  for m in group])
            ry  = np.mean([m["wss_y_r2"]  for m in group])
            rz  = np.mean([m["wss_z_r2"]  for m in group])
            print(f"  {label}: n={len(group):3d}  mag={rmg:.3f}  "
                  f"| {rx:.3f} {ry:.3f} {rz:.3f}")


def compare_v1_v2(
    v2_ckpt: str,
    v1_ckpt: str,
    device: Optional[torch.device] = None,
):
    """Side-by-side v1 vs v2 aggregate metrics on their respective test splits."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*65)
    print("  V1 vs V2 COMPARISON")
    print("="*65)

    # --- V1 ---
    ckpt1 = torch.load(v1_ckpt, map_location="cpu", weights_only=False)
    cfg1  = ckpt1.get("config", {})
    m1 = BifurcationWSSPredictor(
        node_feat_dim = cfg1.get("node_feat_dim", config_v1.node_feat_dim),
        edge_feat_dim = cfg1.get("edge_feat_dim", config_v1.edge_feat_dim),
        hidden_dim    = cfg1.get("hidden_dim",    config_v1.hidden_dim),
        out_channels  = cfg1.get("output_dim",    config_v1.output_dim),
        num_layers    = cfg1.get("num_layers",    config_v1.num_layers),
        context_dim   = cfg1.get("context_dim",   config_v1.context_dim),
    )
    m1.load_state_dict(ckpt1["model_state"])
    m1.to(device).eval()

    ns1 = ckpt1.get("norm_stats") or compute_normalization_stats()
    _tr1, _va1, te1 = get_split_paths("random")
    test_loader1    = DataLoader(
        BifurcationWSSDataset(te1, ns1), batch_size=1, shuffle=False
    )

    all_pred1, all_true1 = [], []
    with torch.no_grad():
        for b in tqdm(test_loader1, desc="V1 eval", leave=False):
            b = b.to(device)
            y = m1(b)
            all_pred1.append(denormalize_wss(y, ns1).cpu().numpy())
            all_true1.append(denormalize_wss(b.y, ns1).cpu().numpy())
    agg1 = compute_metrics(
        np.concatenate(all_pred1), np.concatenate(all_true1)
    )

    # --- V2 ---
    agg2, _, _, _ = evaluate_checkpoint_v2(v2_ckpt, device=device)

    # Print comparison
    print(f"\n  {'Metric':20s}  {'V1':>12s}  {'V2':>12s}  {'Delta':>12s}")
    print(f"  {'-'*60}")
    for key in ["wss_x_r2", "wss_y_r2", "wss_z_r2", "mag_r2",
                "wss_x_mae", "wss_y_mae", "wss_z_mae", "mag_mae"]:
        v1_val = agg1.get(key, float("nan"))
        v2_val = agg2.get(key, float("nan"))
        delta  = v2_val - v1_val
        sym    = "^" if delta > 0 else "v"
        print(f"  {key:20s}  {v1_val:>12.4f}  {v2_val:>12.4f}  {delta:>+11.4f} {sym}")


# ============================================================================
# Optional plots
# ============================================================================

def _plot_r2_vs_re(per_sample: List[Dict], out_path: Path):
    """R² vs Re with ±1σ shaded band across geometries."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping R² vs Re plot")
        return

    re_groups: Dict[int, List] = defaultdict(list)
    for s in per_sample:
        re_groups[int(s["re"].replace("Re", ""))].append(s["metrics"])

    res = sorted(re_groups.keys())

    def stats(key):
        means = np.array([np.mean([m[key] for m in re_groups[r]]) for r in res])
        stds  = np.array([np.std( [m[key] for m in re_groups[r]]) for r in res])
        return means, stds

    fig, ax = plt.subplots(figsize=(12, 6))
    for key, label, color, ls in [
        ("mag_r2",   "Magnitude", "black",   "-"),
        ("wss_x_r2", "WSS_X",    "#2196F3", "--"),
        ("wss_y_r2", "WSS_Y",    "#F44336", "--"),
        ("wss_z_r2", "WSS_Z",    "#4CAF50", "--"),
    ]:
        m, s = stats(key)
        ax.plot(res, m, color=color, linestyle=ls, linewidth=2,
                marker="o", markersize=4, label=label)
        ax.fill_between(res, m - s, m + s, alpha=0.12, color=color)

    ax.axhline(0,   color="gray", linestyle=":",  linewidth=0.8)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.5, alpha=0.4,
               label="R²=0.9 target")
    ax.set_xlabel("Reynolds Number", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("V2 Model: R² vs Reynolds Number  (shaded = ±1σ across geometries)",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.6, 1.05)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"R² vs Re plot          → {out_path}")


def _plot_scatter_pred_vs_true(pred: np.ndarray, true: np.ndarray, out_dir: Path):
    """2×2 predicted-vs-true scatter plots for X, Y, Z, and magnitude."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    mag_pred = np.linalg.norm(pred, axis=1)
    mag_true = np.linalg.norm(true, axis=1)

    panels = [
        ("WSS_X",      pred[:, 0], true[:, 0], "#2196F3"),
        ("WSS_Y",      pred[:, 1], true[:, 1], "#F44336"),
        ("WSS_Z",      pred[:, 2], true[:, 2], "#4CAF50"),
        ("Magnitude",  mag_pred,   mag_true,   "#9C27B0"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (label, p, t, color) in zip(axes.flat, panels):
        # Thin out points for speed: max 40k
        idx = np.random.choice(len(p), min(len(p), 40_000), replace=False)
        ax.scatter(t[idx], p[idx], s=1, alpha=0.15, color=color, rasterized=True)
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lim, lim, "k--", linewidth=1, label="y = x")
        r2_val = r_squared(p, t)
        ax.set_title(f"{label}   R²={r2_val:.4f}", fontsize=12)
        ax.set_xlabel("CFD (true)", fontsize=10)
        ax.set_ylabel("GNN (predicted)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle("V2 Model — Predicted vs True WSS  (test set, all nodes)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / "scatter_pred_vs_true_v2.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Scatter plot           → {out_path}")


def _plot_r2_heatmap(per_sample: List[Dict], out_path: Path):
    """Heatmap of magnitude R² across bifurcation angle × Reynolds number."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return

    angles  = sorted(set(s["angle"] for s in per_sample))
    re_vals = sorted(set(int(s["re"].replace("Re", "")) for s in per_sample))

    grid = np.full((len(angles), len(re_vals)), np.nan)
    for s in per_sample:
        ai = angles.index(s["angle"])
        ri = re_vals.index(int(s["re"].replace("Re", "")))
        grid[ai, ri] = s["metrics"]["mag_r2"]

    fig, ax = plt.subplots(figsize=(max(14, len(re_vals) * 0.9), 4))
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(re_vals)))
    ax.set_xticklabels([str(r) for r in re_vals], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(angles)))
    ax.set_yticklabels([f"{a}°" for a in angles], fontsize=11)
    ax.set_xlabel("Reynolds Number", fontsize=12)
    ax.set_ylabel("Bifurcation Angle", fontsize=12)
    ax.set_title("V2 Model — Magnitude R²  (green=good, red=poor, grey=not in test set)",
                 fontsize=12)

    for i in range(len(angles)):
        for j in range(len(re_vals)):
            val = grid[i, j]
            if not np.isnan(val):
                txt_color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, color=txt_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="R²", shrink=0.8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"R² heatmap             → {out_path}")


def _plot_error_distributions(pred: np.ndarray, true: np.ndarray, out_path: Path):
    """Per-component relative error histograms (clipped at 3× for readability)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    mag_pred = np.linalg.norm(pred, axis=1)
    mag_true = np.linalg.norm(true, axis=1)
    CLIP = 3.0

    panels = [
        ("WSS_X",     pred[:, 0], true[:, 0], "#2196F3"),
        ("WSS_Y",     pred[:, 1], true[:, 1], "#F44336"),
        ("WSS_Z",     pred[:, 2], true[:, 2], "#4CAF50"),
        ("Magnitude", mag_pred,   mag_true,   "#9C27B0"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    for ax, (label, p, t, color) in zip(axes, panels):
        abs_t   = np.abs(t)
        floor   = float(np.percentile(abs_t, 10))
        denom   = np.maximum(abs_t, floor)
        rel_err = np.abs(p - t) / denom
        clipped = np.clip(rel_err, 0, CLIP)
        pct_clipped = 100.0 * (rel_err > CLIP).mean()

        ax.hist(clipped, bins=60, color=color, alpha=0.75, edgecolor="white",
                linewidth=0.3)
        med = float(np.median(rel_err))
        ax.axvline(med, color="black", linestyle="--", linewidth=1.5,
                   label=f"Median={med:.2f}")
        ax.set_title(f"{label}\nR²={r_squared(p, t):.3f}", fontsize=11)
        ax.set_xlabel(f"Relative error  ({pct_clipped:.1f}% > {CLIP}×)", fontsize=9)
        ax.set_ylabel("Node count", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"V2 Model — Relative Error Distribution  (clipped at {CLIP}×)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Error distributions    → {out_path}")


def _plot_nrmse_bars(metrics: Dict, out_path: Path):
    """
    Grouped bar chart: NRMSE for WSS_X, WSS_Y, WSS_Z, and Magnitude.
    Also overlays the R² for context (secondary axis).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    components  = ["WSS_X", "WSS_Y", "WSS_Z", "Magnitude"]
    keys_nrmse  = ["wss_x_nrmse", "wss_y_nrmse", "wss_z_nrmse", "mag_nrmse"]
    keys_r2     = ["wss_x_r2",    "wss_y_r2",    "wss_z_r2",    "mag_r2"]
    colors      = ["#2196F3", "#F44336", "#4CAF50", "#9C27B0"]

    nrmse_vals  = [metrics[k] for k in keys_nrmse]
    r2_vals     = [metrics[k] for k in keys_r2]

    x = np.arange(len(components))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    bars = ax1.bar(x, nrmse_vals, color=colors, alpha=0.8, width=0.5, zorder=2)
    ax1.set_ylabel("NRMSE  (lower = better)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(components, fontsize=12)
    ax1.set_ylim(0, max(nrmse_vals) * 1.35)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.8,
                label="NRMSE=1  (baseline — predicting mean)")
    ax1.grid(True, axis="y", alpha=0.25, zorder=0)

    # Annotate NRMSE value on each bar
    for bar, val in zip(bars, nrmse_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Secondary axis: R²
    ax2 = ax1.twinx()
    ax2.plot(x, r2_vals, color="black", marker="D", markersize=7,
             linewidth=1.5, linestyle=":", label="R²", zorder=3)
    ax2.set_ylabel("R²  (higher = better)", fontsize=12)
    ax2.set_ylim(-0.3, 1.15)
    ax2.axhline(0.9, color="darkgray", linestyle=":", linewidth=0.6, alpha=0.5)

    # Combine legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=10, loc="upper right")

    ax1.set_title("V2 Model — NRMSE and R² by Component  (test set)", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"NRMSE bar chart        → {out_path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate bifurcation WSS model v2")
    parser.add_argument("--model",   type=str, required=True,
                        help="Path to v2 checkpoint (best_model_v2.pt)")
    parser.add_argument("--mode",    choices=["re_angle_stratified", "random", "loocv-geo"],
                        default="re_angle_stratified")
    parser.add_argument("--holdout", type=str, default=None)
    parser.add_argument("--device",  type=str, default=None)
    parser.add_argument("--save",    action="store_true",
                        help="Save results JSON to results_v2/")
    parser.add_argument("--plot",    action="store_true",
                        help="Save R² vs Re plot")
    parser.add_argument("--calibration", action="store_true",
                        help="Run MC calibration check (slow)")
    parser.add_argument("--compare", type=str, default=None,
                        metavar="V1_CKPT",
                        help="Path to v1 checkpoint for side-by-side comparison")
    args = parser.parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    agg, per_sample, all_pred, all_true = evaluate_checkpoint_v2(
        ckpt_path        = args.model,
        mode             = args.mode,
        holdout_geo      = args.holdout,
        device           = device,
        run_calibration  = args.calibration,
    )

    print_metrics(agg,     title="V2 Aggregate Test Metrics")
    print_per_geometry(per_sample)
    print_per_re(per_sample)
    print_per_branch(per_sample)

    if "calibration" in agg:
        calib = agg["calibration"]
        print(f"\n  Calibration: Spearman ρ = {calib['spearman_rho']:.4f} "
              f"(p={calib['p_value']:.3e})  → {calib['interpretation']}")

    if args.compare:
        compare_v1_v2(args.model, args.compare, device)

    if args.save:
        config_v2.results_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "aggregate":  agg,
            "per_sample": [
                {"geo": s["geo"], "re": s["re"], "angle": s["angle"],
                 "branch_depth_mean": s["branch_depth_mean"],
                 **s["metrics"]}
                for s in per_sample
            ],
        }
        results_path = config_v2.results_dir / "evaluation_v2.json"
        with open(results_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults → {results_path}")

    if args.plot:
        config_v2.results_dir.mkdir(parents=True, exist_ok=True)
        _plot_r2_vs_re(per_sample,           config_v2.results_dir / "r2_vs_re_v2.png")
        _plot_scatter_pred_vs_true(all_pred, all_true, config_v2.results_dir)
        _plot_r2_heatmap(per_sample,         config_v2.results_dir / "r2_heatmap_v2.png")
        _plot_error_distributions(all_pred,  all_true, config_v2.results_dir / "error_dist_v2.png")
        _plot_nrmse_bars(agg,                config_v2.results_dir / "nrmse_bars_v2.png")

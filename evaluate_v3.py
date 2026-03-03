"""
Evaluation script v3 for bifurcation WSS predictions.

Differences from evaluate_v2.py:
  - Uses BifurcationWSSPredictorV3 (GATv2Conv, num_heads param)
  - Supports --data-dir: evaluate on any directory of *.pt files
    (bifurcation ProcessedData_v3/ OR carotid ProcessedData_carotid/)
  - --mode test uses bifurcation test split (re_angle_stratified)
  - Handles carotid re_name="systolic" gracefully in per-Re table

Usage:
    # Evaluate v3 checkpoint on bifurcation test split
    python -m Bifurcation.evaluate_v3 --model Bifurcation/Models_v3/best_model_v3.pt

    # Zero-shot evaluation on carotid .pt files
    python -m Bifurcation.evaluate_v3 \\
        --model Bifurcation/Models_v3/best_model_v3.pt \\
        --data-dir Bifurcation/ProcessedData_carotid

    # Save results + plots
    python -m Bifurcation.evaluate_v3 \\
        --model Bifurcation/Models_v3/best_model_v3.pt \\
        --save --plot
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

from Bifurcation.config_v3 import config_v3
from Bifurcation.dataset_v3 import (
    BifurcationWSSDatasetV3,
    denormalize_wss_v3,
    get_split_paths_v3,
    compute_normalization_stats_v3,
)
from Bifurcation.Models.bif_v3 import BifurcationWSSPredictorV3


# ============================================================================
# Metrics  (identical to v2 for consistency)
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
    abs_true = np.abs(true)
    floor    = float(np.percentile(abs_true, floor_percentile))
    denom    = np.maximum(abs_true, floor)
    return float(np.median(np.abs(pred - true) / denom))


def nrmse(pred: np.ndarray, true: np.ndarray) -> float:
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
# Model loading
# ============================================================================

def load_model_v3(
    ckpt_path: Path,
    device: torch.device,
) -> Tuple["BifurcationWSSPredictorV3", Dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    model = BifurcationWSSPredictorV3(
        node_feat_dim  = cfg.get("node_feat_dim",  config_v3.node_feat_dim),
        edge_feat_dim  = cfg.get("edge_feat_dim",  config_v3.edge_feat_dim),
        hidden_dim     = cfg.get("hidden_dim",     config_v3.hidden_dim),
        num_heads      = cfg.get("num_heads",      config_v3.num_heads),
        out_channels   = cfg.get("output_dim",     config_v3.output_dim),
        num_layers     = cfg.get("num_layers",     config_v3.num_layers),
        context_dim    = cfg.get("context_dim",    config_v3.context_dim),
        flow_param_dim = cfg.get("flow_param_dim", config_v3.flow_param_dim),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    norm_stats = ckpt.get("norm_stats") or {}
    return model, norm_stats


# ============================================================================
# Data path helpers
# ============================================================================

# Confirmed no-WSS cases from quality check — avoids loading all files to filter
_NO_WSS = {
    'carotid_case_k_001_left', 'carotid_case_k_002_left', 'carotid_case_k_002_right',
    'carotid_case_k_007_right', 'carotid_case_w_007_left', 'carotid_case_w_007_right',
    'carotid_case_w_008_left', 'carotid_case_w_008_right', 'carotid_case_w_030_right',
    'carotid_case_w_040_right',
}

def get_data_paths_from_dir(data_dir: Path) -> List[Path]:
    """
    Collect all labeled .pt files from a directory.
      - <data_dir>/*/systolic.pt  (carotid)
      - <data_dir>/*/Re*.pt       (bifurcation)
    """
    paths  = [p for p in sorted(data_dir.glob("*/systolic.pt"))
              if p.parent.name not in _NO_WSS]
    paths += sorted(data_dir.glob("*/Re*.pt"))
    return paths


# ============================================================================
# Full evaluation over a DataLoader
# ============================================================================

@torch.no_grad()
def evaluate_loader_v3(
    model:      BifurcationWSSPredictorV3,
    loader:     DataLoader,
    norm_stats: Dict,
    device:     torch.device,
) -> Tuple[Dict, List[Dict], np.ndarray, np.ndarray]:
    """
    Run inference and collect per-sample results.

    Returns
    -------
    agg_metrics, per_sample, all_pred, all_true
    """
    model.eval()
    all_pred, all_true = [], []
    per_sample = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch  = batch.to(device)
        y_norm = model(batch)

        y_pred = denormalize_wss_v3(y_norm,  norm_stats).cpu().numpy()
        y_true = denormalize_wss_v3(batch.y, norm_stats).cpu().numpy()

        all_pred.append(y_pred)
        all_true.append(y_true)

        geo     = batch.geo_name[0] if isinstance(batch.geo_name, list) else batch.geo_name
        re_nm   = batch.re_name[0]  if isinstance(batch.re_name,  list) else batch.re_name
        re_val  = float(batch.re.view(1)[0])
        ang_val = float(batch.angle.view(1)[0])  # degrees, as stored

        # Branch depth from normalised x feature (index 5)
        x_mean5 = norm_stats.get("x_mean", [0]*10)[5]
        x_std5  = norm_stats.get("x_std",  [1]*10)[5]
        bd_norm  = batch.x[:, 5].cpu().numpy()
        branch_depth = bd_norm * x_std5 + x_mean5  # ≈ 0 or 1

        sample_metrics = compute_metrics(y_pred, y_true)
        per_sample.append({
            "geo":               geo,
            "re":                re_nm,
            "re_val":            re_val,
            "angle":             ang_val,
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
    model:      BifurcationWSSPredictorV3,
    loader:     DataLoader,
    norm_stats: Dict,
    device:     torch.device,
    mc_samples: int = 30,
) -> Dict:
    from scipy.stats import spearmanr

    model.enable_mc_mode()
    all_errors, all_stds = [], []

    for batch in tqdm(loader, desc="Calibration", leave=False):
        batch = batch.to(device)
        preds = []
        with torch.no_grad():
            for _ in range(mc_samples):
                y_norm = model(batch)
                y_phys = denormalize_wss_v3(y_norm, norm_stats)
                preds.append(y_phys.cpu())

        preds_t  = torch.stack(preds, 0)
        mean_p   = preds_t.mean(0).numpy()
        std_p    = preds_t.std(0).numpy()

        y_true   = denormalize_wss_v3(batch.y, norm_stats).cpu().numpy()
        error    = np.abs(mean_p - y_true)

        all_errors.append(error.mean(axis=1))
        all_stds.append(std_p.mean(axis=1))

    model.disable_mc_mode()

    all_errors = np.concatenate(all_errors)
    all_stds   = np.concatenate(all_stds)

    rho, pval = spearmanr(all_errors, all_stds)
    return {"spearman_rho": float(rho), "p_value": float(pval),
            "interpretation": "good" if rho > 0.3 else "poor"}


# ============================================================================
# High-level entry points
# ============================================================================

def evaluate_checkpoint_v3(
    ckpt_path:       str,
    mode:            str                    = "re_angle_stratified",
    holdout_geo:     Optional[str]          = None,
    data_dir:        Optional[Path]         = None,
    device:          Optional[torch.device] = None,
    run_calibration: bool                   = False,
) -> Tuple[Dict, List[Dict], np.ndarray, np.ndarray]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, norm_stats = load_model_v3(Path(ckpt_path), device)
    model.eval()

    if data_dir is not None:
        # Evaluate on an arbitrary directory of .pt files
        print(f"\nData source: {data_dir}")
        paths = get_data_paths_from_dir(Path(data_dir))
        print(f"Found {len(paths)} labeled .pt files")
    else:
        # Bifurcation test split
        _train_p, _val_p, paths = get_split_paths_v3(
            mode, holdout_geo=holdout_geo
        )
        print(f"\nBifurcation test split ({mode}): {len(paths)} cases")

    ds     = BifurcationWSSDatasetV3(paths, norm_stats, augment=False, use_lpe=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    agg, per_sample, all_pred, all_true = evaluate_loader_v3(
        model, loader, norm_stats, device
    )

    if run_calibration:
        calib = calibration_check(model, loader, norm_stats, device)
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
        r    = np.mean([m["mag_rmse"]    for m in ms])
        rx   = np.mean([m["wss_x_r2"]   for m in ms])
        ry   = np.mean([m["wss_y_r2"]   for m in ms])
        rz   = np.mean([m["wss_z_r2"]   for m in ms])
        print(f"  {geo:50s}  {r:.4e}  | {rx:.3f} {ry:.3f} {rz:.3f}")


def print_per_re(per_sample: List[Dict]):
    re_groups: Dict[float, List] = defaultdict(list)
    for s in per_sample:
        re_groups[s["re_val"]].append(s["metrics"])

    print(f"\n{'='*65}")
    print("  Per-Re:  mag R²  |  X R²  Y R²  Z R²")
    print(f"{'='*65}")
    for re_v in sorted(re_groups.keys()):
        ms  = re_groups[re_v]
        rmg = np.mean([m["mag_r2"]    for m in ms])
        rx  = np.mean([m["wss_x_r2"] for m in ms])
        ry  = np.mean([m["wss_y_r2"] for m in ms])
        rz  = np.mean([m["wss_z_r2"] for m in ms])
        print(f"  Re={re_v:6.0f}  n={len(ms):3d}  mag={rmg:.3f}  | {rx:.3f} {ry:.3f} {rz:.3f}")


def print_per_branch(per_sample: List[Dict]):
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
            rmg = np.mean([m["mag_r2"]    for m in group])
            rx  = np.mean([m["wss_x_r2"] for m in group])
            ry  = np.mean([m["wss_y_r2"] for m in group])
            rz  = np.mean([m["wss_z_r2"] for m in group])
            print(f"  {label}: n={len(group):3d}  mag={rmg:.3f}  "
                  f"| {rx:.3f} {ry:.3f} {rz:.3f}")


# ============================================================================
# Optional plots
# ============================================================================

def _plot_r2_vs_re(per_sample: List[Dict], out_path: Path, title_prefix: str = "V3"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping R² vs Re plot")
        return

    re_groups: Dict[float, List] = defaultdict(list)
    for s in per_sample:
        re_groups[s["re_val"]].append(s["metrics"])

    res = sorted(re_groups.keys())
    if len(res) < 2:
        print("  Skipping R² vs Re plot (only one Re value)")
        return

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
    ax.set_title(f"{title_prefix} Model: R² vs Reynolds Number  (shaded = ±1σ)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.6, 1.05)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"R² vs Re plot          → {out_path}")


def _plot_scatter_pred_vs_true(pred: np.ndarray, true: np.ndarray, out_dir: Path,
                                title_prefix: str = "V3"):
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

    fig.suptitle(f"{title_prefix} Model — Predicted vs True WSS  (test set, all nodes)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / f"scatter_pred_vs_true_v3.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Scatter plot           → {out_path}")


def _plot_error_distributions(pred: np.ndarray, true: np.ndarray, out_path: Path,
                               title_prefix: str = "V3"):
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

        ax.hist(clipped, bins=60, color=color, alpha=0.75, edgecolor="white", linewidth=0.3)
        med = float(np.median(rel_err))
        ax.axvline(med, color="black", linestyle="--", linewidth=1.5,
                   label=f"Median={med:.2f}")
        ax.set_title(f"{label}\nR²={r_squared(p, t):.3f}", fontsize=11)
        ax.set_xlabel(f"Relative error  ({pct_clipped:.1f}% > {CLIP}×)", fontsize=9)
        ax.set_ylabel("Node count", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"{title_prefix} Model — Relative Error Distribution  (clipped at {CLIP}×)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Error distributions    → {out_path}")


def _plot_nrmse_bars(metrics: Dict, out_path: Path, title_prefix: str = "V3"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    components = ["WSS_X", "WSS_Y", "WSS_Z", "Magnitude"]
    keys_nrmse = ["wss_x_nrmse", "wss_y_nrmse", "wss_z_nrmse", "mag_nrmse"]
    keys_r2    = ["wss_x_r2",    "wss_y_r2",    "wss_z_r2",    "mag_r2"]
    colors     = ["#2196F3", "#F44336", "#4CAF50", "#9C27B0"]

    nrmse_vals = [metrics[k] for k in keys_nrmse]
    r2_vals    = [metrics[k] for k in keys_r2]

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

    for bar, val in zip(bars, nrmse_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(x, r2_vals, color="black", marker="D", markersize=7,
             linewidth=1.5, linestyle=":", label="R²", zorder=3)
    ax2.set_ylabel("R²  (higher = better)", fontsize=12)
    ax2.set_ylim(-0.3, 1.15)
    ax2.axhline(0.9, color="darkgray", linestyle=":", linewidth=0.6, alpha=0.5)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=10, loc="upper right")

    ax1.set_title(f"{title_prefix} Model — NRMSE and R² by Component  (test set)",
                  fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"NRMSE bar chart        → {out_path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate bifurcation/carotid WSS model v3")
    parser.add_argument("--model",    type=str, required=True,
                        help="Path to v3 checkpoint (best_model_v3.pt)")
    parser.add_argument("--mode",     choices=["re_angle_stratified", "random", "loocv-geo"],
                        default="re_angle_stratified",
                        help="Bifurcation test split mode (ignored if --data-dir is set)")
    parser.add_argument("--holdout",  type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Evaluate on all labeled .pt files in this directory "
                             "(carotid or bifurcation ProcessedData). "
                             "If not set, uses bifurcation test split.")
    parser.add_argument("--device",   type=str, default=None)
    parser.add_argument("--save",     action="store_true",
                        help="Save results JSON to results_v3/ (or results_v3_carotid/)")
    parser.add_argument("--plot",     action="store_true",
                        help="Save metric plots")
    parser.add_argument("--calibration", action="store_true",
                        help="Run MC calibration check (slow)")
    args = parser.parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    data_dir = Path(args.data_dir) if args.data_dir else None

    agg, per_sample, all_pred, all_true = evaluate_checkpoint_v3(
        ckpt_path        = args.model,
        mode             = args.mode,
        holdout_geo      = args.holdout,
        data_dir         = data_dir,
        device           = device,
        run_calibration  = args.calibration,
    )

    # Title prefix depends on whether this is carotid or bifurcation
    is_carotid   = data_dir is not None and "carotid" in str(data_dir).lower()
    title_prefix = "V3 (carotid zero-shot)" if is_carotid else "V3"

    print_metrics(agg,     title=f"{title_prefix} Aggregate Metrics")
    print_per_geometry(per_sample)
    print_per_re(per_sample)
    print_per_branch(per_sample)

    if "calibration" in agg:
        calib = agg["calibration"]
        print(f"\n  Calibration: Spearman ρ = {calib['spearman_rho']:.4f} "
              f"(p={calib['p_value']:.3e})  → {calib['interpretation']}")

    # Output directory
    if is_carotid:
        out_dir = config_v3.results_dir.parent / "results_v3_carotid"
    else:
        out_dir = config_v3.results_dir

    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "carotid" if is_carotid else "bifurcation"
        out = {
            "aggregate":  agg,
            "per_sample": [
                {"geo": s["geo"], "re": s["re"], "re_val": s["re_val"],
                 "angle": s["angle"], "branch_depth_mean": s["branch_depth_mean"],
                 **s["metrics"]}
                for s in per_sample
            ],
        }
        results_path = out_dir / f"evaluation_v3_{tag}.json"
        with open(results_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults → {results_path}")

    if args.plot:
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_carotid" if is_carotid else ""
        _plot_r2_vs_re(per_sample, out_dir / f"r2_vs_re_v3{suffix}.png", title_prefix)
        _plot_scatter_pred_vs_true(all_pred, all_true, out_dir, title_prefix)
        _plot_error_distributions(all_pred, all_true,
                                   out_dir / f"error_dist_v3{suffix}.png", title_prefix)
        _plot_nrmse_bars(agg, out_dir / f"nrmse_bars_v3{suffix}.png", title_prefix)

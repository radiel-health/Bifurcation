"""
Training script v2 for bifurcation WSS prediction.

Improvements over train.py (v1):
  - Automatic Mixed Precision (AMP) — faster on CUDA, same on CPU
  - Per-component weighted MSE loss (configurable, default upweights Y)
  - Bayesian KL loss from BayesianLinear output head
  - re_angle_stratified data split (fairer evaluation than random)
  - Longer training budget (500 epochs) with higher early-stop patience (50)

Usage:
    # Standard run (re_angle_stratified split)
    python -m Bifurcation.train_v2

    # Leave-one-geometry-out cross-validation
    python -m Bifurcation.train_v2 --mode loocv-geo

    # Single LOOCV fold
    python -m Bifurcation.train_v2 --mode loocv-geo --holdout bifurcation_angle45_750_ascii

    # Build v2 ProcessedData cache first, then train
    python -m Bifurcation.train_v2 --process
"""

import argparse
import json
import time
from pathlib import Path

import torch
torch.set_num_threads(16)
torch.set_num_interop_threads(2)
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from Bifurcation.config_v2 import config_v2
from Bifurcation.dataset_v2 import (
    BifurcationWSSDatasetV2,
    build_all_processed_v2,
    compute_normalization_stats_v2,
    get_split_paths_v2,
    save_normalization_stats_v2,
)
from Bifurcation.Models.bif_v2 import BifurcationWSSPredictorV2, count_parameters, get_model_summary


# ============================================================================
# Loss
# ============================================================================

def compute_loss_v2(
    y_pred:    torch.Tensor,
    y_true:    torch.Tensor,
    component_weights: torch.Tensor,
    model:     BifurcationWSSPredictorV2,
    kl_weight: float = 0.0,
) -> torch.Tensor:
    """
    Weighted component MSE  +  optional Bayesian KL term.

    loss = sum_i(w_i * MSE_i(pred[:,i], true[:,i]))  +  kl_weight * KL(model)

    component_weights [3] are normalised to sum=1, so the total MSE is
    comparable in scale to plain MSE.
    """
    # Per-component MSE then weighted sum
    sq_err     = (y_pred - y_true) ** 2              # [N, 3]
    per_comp   = sq_err.mean(dim=0)                   # [3]
    weighted   = (component_weights * per_comp).sum() # scalar

    if kl_weight > 0:
        kl = model.kl_loss()
        return weighted + kl_weight * kl

    return weighted


# ============================================================================
# Train / validate one epoch
# ============================================================================

def train_epoch(
    model, loader, optimizer, scaler, device,
    component_weights, grad_clip, kl_weight, use_amp,
):
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc="Training", leave=False)

    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                y_pred = model(batch)
                loss   = compute_loss_v2(
                    y_pred, batch.y, component_weights, model, kl_weight
                )
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            y_pred = model(batch)
            loss   = compute_loss_v2(
                y_pred, batch.y, component_weights, model, kl_weight
            )
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        n          += batch.num_graphs
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return total_loss / max(n, 1)


@torch.no_grad()
def validate_epoch(model, loader, device, component_weights, kl_weight):
    model.eval()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc="Validation", leave=False)

    for batch in pbar:
        batch  = batch.to(device)
        y_pred = model(batch)
        loss   = compute_loss_v2(
            y_pred, batch.y, component_weights, model, kl_weight
        )
        total_loss += loss.item() * batch.num_graphs
        n          += batch.num_graphs
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return total_loss / max(n, 1)


# ============================================================================
# Full training run
# ============================================================================

def train_v2(
    mode:        str             = "re_angle_stratified",
    holdout_geo: str | None      = None,
    epochs:      int | None      = None,
    device:      torch.device | None = None,
    tag:         str             = "",
):
    """
    Full training loop for the v2 model.

    Returns: (best_val_loss, model_path)
    """
    config_v2.create_directories()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs or config_v2.num_epochs
    use_amp = config_v2.use_amp and device.type == "cuda"

    print(f"\n{'='*70}")
    print(f"Bifurcation WSS Predictor V2  |  mode={mode}  "
          f"holdout={holdout_geo}  epochs={epochs}")
    if torch.cuda.is_available():
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"{'='*70}\n")

    # ---- data ----
    train_p, val_p, test_p = get_split_paths_v2(mode, holdout_geo)
    norm_stats = compute_normalization_stats_v2(train_p)
    save_normalization_stats_v2(norm_stats)

    train_ds = BifurcationWSSDatasetV2(train_p, norm_stats)
    val_ds   = BifurcationWSSDatasetV2(val_p,   norm_stats)

    train_loader = DataLoader(train_ds, batch_size=config_v2.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config_v2.batch_size, shuffle=False)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_p)}")

    # ---- component loss weights (normalised to sum=1) ----
    raw_w = torch.tensor(config_v2.component_loss_weights, dtype=torch.float32)
    component_weights = (raw_w / raw_w.sum()).to(device)
    print(f"Component loss weights (normalised): {component_weights.tolist()}")

    # ---- model ----
    model = BifurcationWSSPredictorV2(
        node_feat_dim  = config_v2.node_feat_dim,
        edge_feat_dim  = config_v2.edge_feat_dim,
        hidden_dim     = config_v2.hidden_dim,
        out_channels   = config_v2.output_dim,
        num_layers     = config_v2.num_layers,
        context_dim    = config_v2.context_dim,
        flow_param_dim = config_v2.flow_param_dim,
    ).to(device)
    get_model_summary(model)

    # ---- optimiser / AMP scaler / scheduler ----
    optimizer = optim.Adam(
        model.parameters(),
        lr           = config_v2.learning_rate,
        weight_decay = config_v2.weight_decay,
    )
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor   = config_v2.scheduler_factor,
        patience = config_v2.scheduler_patience,
        min_lr   = config_v2.scheduler_min_lr,
    )

    # ---- checkpoint ----
    ckpt_name = f"best_model_v2{'_' + tag if tag else ''}.pt"
    ckpt_path = config_v2.models_dir / ckpt_name

    # ---- training loop ----
    best_val  = float("inf")
    no_improve = 0
    history   = {"train_loss": [], "val_loss": [], "lr": []}

    print(f"\n{'='*70}")
    print("Starting v2 training loop …")
    print(f"{'='*70}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, device,
            component_weights = component_weights,
            grad_clip         = config_v2.grad_clip,
            kl_weight         = config_v2.kl_weight,
            use_amp           = use_amp,
        )
        val_loss = validate_epoch(
            model, val_loader, device,
            component_weights = component_weights,
            kl_weight         = config_v2.kl_weight,
        )
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr_now)

        dt       = time.time() - t0
        improved = val_loss < best_val
        marker   = " *" if improved else ""

        gpu_mem = ""
        if torch.cuda.is_available():
            mem_alloc   = torch.cuda.memory_allocated(device) / 1024**3
            mem_reserve = torch.cuda.memory_reserved(device)  / 1024**3
            gpu_mem     = f"  GPU: {mem_alloc:.2f}/{mem_reserve:.2f}GB"

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"lr={lr_now:.2e}  {dt:.1f}s{marker}{gpu_mem}"
        )

        if improved:
            best_val   = val_loss
            no_improve = 0
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "opt_state":   optimizer.state_dict(),
                    "val_loss":    val_loss,
                    "norm_stats":  norm_stats,
                    "config": {
                        "model_version":   "v2",
                        "mode":            mode,
                        "holdout_geo":     holdout_geo,
                        "node_feat_dim":   config_v2.node_feat_dim,
                        "edge_feat_dim":   config_v2.edge_feat_dim,
                        "hidden_dim":      config_v2.hidden_dim,
                        "num_layers":      config_v2.num_layers,
                        "context_dim":     config_v2.context_dim,
                        "output_dim":      config_v2.output_dim,
                        "flow_param_dim":  config_v2.flow_param_dim,
                    },
                },
                ckpt_path,
            )
        else:
            no_improve += 1
            if no_improve >= config_v2.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(patience={config_v2.early_stop_patience})")
                break

    # ---- save history ----
    hist_path = config_v2.models_dir / f"training_log_v2{'_' + tag if tag else ''}.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Checkpoint:    {ckpt_path}")
    print(f"History:       {hist_path}")

    return best_val, ckpt_path


# ============================================================================
# LOOCV driver
# ============================================================================

def run_loocv_v2(epochs=None, device=None):
    results = {}
    for geo in config_v2.geometry_folders:
        print(f"\n{'#'*70}")
        print(f"# FOLD: holdout = {geo}")
        print(f"{'#'*70}")
        val_loss, ckpt = train_v2(
            mode        = "loocv-geo",
            holdout_geo = geo,
            epochs      = epochs,
            device      = device,
            tag         = geo,
        )
        results[geo] = {"val_loss": val_loss, "ckpt": str(ckpt)}

    print(f"\n{'='*70}")
    print("LOOCV V2 RESULTS")
    print(f"{'='*70}")
    for geo, info in results.items():
        print(f"  {geo:45s}  val_loss={info['val_loss']:.6f}")
    mean_loss = sum(r["val_loss"] for r in results.values()) / len(results)
    print(f"\n  Mean val loss: {mean_loss:.6f}")

    summary_path = config_v2.models_dir / "loocv_v2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {summary_path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bifurcation WSS model v2")
    parser.add_argument("--mode", choices=["re_angle_stratified", "random", "loocv-geo"],
                        default="re_angle_stratified")
    parser.add_argument("--holdout", type=str, default=None)
    parser.add_argument("--epochs",  type=int, default=None)
    parser.add_argument("--device",  type=str, default=None)
    parser.add_argument("--process", action="store_true",
                        help="Build v2 ProcessedData cache before training")
    parser.add_argument("--no-amp",  action="store_true",
                        help="Disable AMP even on CUDA (for debugging)")
    args = parser.parse_args()

    if args.no_amp:
        config_v2.use_amp = False

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if args.process:
        build_all_processed_v2()

    if args.mode == "loocv-geo" and args.holdout is None:
        run_loocv_v2(epochs=args.epochs, device=device)
    else:
        train_v2(
            mode        = args.mode,
            holdout_geo = args.holdout,
            epochs      = args.epochs,
            device      = device,
        )

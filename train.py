"""
Training script for bifurcation WSS prediction.

Supports two modes:
  --mode random      80/10/10 random split (quick iteration)
  --mode loocv-geo   Leave-one-geometry-out CV (9 folds, generalization test)

Usage:
    python -m Bifurcation.train --mode random --epochs 50
    python -m Bifurcation.train --mode loocv-geo
    python -m Bifurcation.train --mode loocv-geo --holdout bifurcation_angle45_750_ascii
"""

import argparse
import json
import time
from pathlib import Path

import torch
torch.set_num_threads(16)  # Use 16 cores
torch.set_num_interop_threads(2)
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from Bifurcation.config import config
from Bifurcation.dataset import (
    BifurcationWSSDataset,
    build_all_processed,
    compute_normalization_stats,
    get_dataloaders,
    get_split_paths,
    save_normalization_stats,
)
from Bifurcation.model import BifurcationWSSPredictor, count_parameters, get_model_summary


# ============================================================================
# Loss
# ============================================================================

def compute_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                 magnitude_weight: float = 0.0) -> torch.Tensor:
    """
    MSE on normalised WSS components  +  optional magnitude penalty.

    loss = MSE(pred, target) + λ · MSE(||pred||, ||target||)
    """
    mse = nn.functional.mse_loss(y_pred, y_true)

    if magnitude_weight > 0:
        mag_pred = y_pred.norm(dim=1)
        mag_true = y_true.norm(dim=1)
        mag_loss = nn.functional.mse_loss(mag_pred, mag_true)
        return mse + magnitude_weight * mag_loss

    return mse


# ============================================================================
# Train / validate one epoch
# ============================================================================

def train_epoch(model, loader, optimizer, device, grad_clip=None,
                mag_weight=0.0):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        y_pred = model(batch)
        loss = compute_loss(y_pred, batch.y, mag_weight)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / max(n, 1)


@torch.no_grad()
def validate_epoch(model, loader, device, mag_weight=0.0):
    model.eval()
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        y_pred = model(batch)
        loss = compute_loss(y_pred, batch.y, mag_weight)
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / max(n, 1)


# ============================================================================
# Full training run
# ============================================================================

def train(
    mode: str = "random",
    holdout_geo: str | None = None,
    epochs: int | None = None,
    device: torch.device | None = None,
    tag: str = "",
):
    """
    Run a full training loop and save the best checkpoint.

    Returns: (best_val_loss, model_path)
    """
    config.create_directories()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs or config.num_epochs

    print(f"\n{'='*70}")
    print(f"Training  |  mode={mode}  holdout={holdout_geo}  epochs={epochs}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    # ---- data ----
    train_p, val_p, test_p = get_split_paths(mode, holdout_geo)
    norm_stats = compute_normalization_stats(train_p)
    save_normalization_stats(norm_stats)

    train_ds = BifurcationWSSDataset(train_p, norm_stats)
    val_ds = BifurcationWSSDataset(val_p, norm_stats)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}  |  Test samples: {len(test_p)}")

    # ---- model ----
    model = BifurcationWSSPredictor(
        node_feat_dim=config.node_feat_dim,
        edge_feat_dim=config.edge_feat_dim,
        hidden_dim=config.hidden_dim,
        out_channels=config.output_dim,
        num_layers=config.num_layers,
        context_dim=config.context_dim,
        flow_param_dim=config.flow_param_dim,
    ).to(device)
    get_model_summary(model)

    # ---- optimiser / scheduler ----
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )

    # ---- checkpoint path ----
    ckpt_name = f"best_model{'_' + tag if tag else ''}.pt"
    ckpt_path = config.models_dir / ckpt_name

    # ---- training loop ----
    best_val = float("inf")
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            grad_clip=config.grad_clip,
            mag_weight=config.magnitude_loss_weight,
        )
        val_loss = validate_epoch(
            model, val_loader, device,
            mag_weight=config.magnitude_loss_weight,
        )
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr_now)

        dt = time.time() - t0
        improved = val_loss < best_val
        marker = " *" if improved else ""

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"lr={lr_now:.2e}  {dt:.1f}s{marker}"
        )

        if improved:
            best_val = val_loss
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "norm_stats": norm_stats,
                    "config": {
                        "mode": mode,
                        "holdout_geo": holdout_geo,
                        "node_feat_dim": config.node_feat_dim,
                        "edge_feat_dim": config.edge_feat_dim,
                        "hidden_dim": config.hidden_dim,
                        "num_layers": config.num_layers,
                        "context_dim": config.context_dim,
                        "output_dim": config.output_dim,
                    },
                },
                ckpt_path,
            )
        else:
            no_improve += 1
            if no_improve >= config.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={config.early_stop_patience})")
                break

    # ---- save history ----
    hist_path = config.models_dir / f"training_log{'_' + tag if tag else ''}.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Checkpoint:    {ckpt_path}")
    print(f"History:       {hist_path}")

    return best_val, ckpt_path


# ============================================================================
# LOOCV driver
# ============================================================================

def run_loocv(epochs=None, device=None):
    """Run leave-one-geometry-out cross-validation (9 folds)."""
    results = {}
    for geo in config.geometry_folders:
        print(f"\n{'#'*70}")
        print(f"# FOLD: holdout = {geo}")
        print(f"{'#'*70}")
        val_loss, ckpt = train(
            mode="loocv-geo",
            holdout_geo=geo,
            epochs=epochs,
            device=device,
            tag=geo,
        )
        results[geo] = {"val_loss": val_loss, "ckpt": str(ckpt)}

    print(f"\n{'='*70}")
    print("LOOCV RESULTS")
    print(f"{'='*70}")
    for geo, info in results.items():
        print(f"  {geo:45s}  val_loss={info['val_loss']:.6f}")
    mean_loss = sum(r["val_loss"] for r in results.values()) / len(results)
    print(f"\n  Mean val loss: {mean_loss:.6f}")

    # Save summary
    summary_path = config.models_dir / "loocv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {summary_path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bifurcation WSS model")
    parser.add_argument("--mode", choices=["random", "loocv-geo"], default="random")
    parser.add_argument("--holdout", type=str, default=None,
                        help="Geometry to hold out (loocv-geo, single fold)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--process", action="store_true",
                        help="Build processed .pt files before training")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.process:
        build_all_processed()

    if args.mode == "loocv-geo" and args.holdout is None:
        run_loocv(epochs=args.epochs, device=device)
    else:
        train(
            mode=args.mode,
            holdout_geo=args.holdout,
            epochs=args.epochs,
            device=device,
        )

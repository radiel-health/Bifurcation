"""
Training script v3 for bifurcation WSS prediction.

V3 changes over train_v2.py:
  - Uses BifurcationWSSPredictorV3 (GATv2Conv + 3-layer FlowEncoder)
  - Uses BifurcationWSSDatasetV3   (18-dim node features with LPE)
  - Uses config_v3 / Models_v3 / ProcessedData_v3

Usage:
    # Build v3 cache then train
    python -m Bifurcation.train_v3 --process

    # Train (assuming ProcessedData_v3 already built)
    python -m Bifurcation.train_v3

    # LOOCV
    python -m Bifurcation.train_v3 --mode loocv-geo
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

from Bifurcation.config_v3 import config_v3
from Bifurcation.dataset_v3 import (
    BifurcationWSSDatasetV3,
    build_all_processed_v3,
    compute_normalization_stats_v3,
    get_split_paths_v3,
    save_normalization_stats_v3,
)
from Bifurcation.Models.bif_v3 import (
    BifurcationWSSPredictorV3,
    count_parameters,
    get_model_summary,
)


# ============================================================================
# Loss
# ============================================================================

def compute_loss_v3(
    y_pred:            torch.Tensor,
    y_true:            torch.Tensor,
    component_weights: torch.Tensor,
    model:             BifurcationWSSPredictorV3,
    kl_weight:         float = 0.0,
) -> torch.Tensor:
    sq_err   = (y_pred - y_true) ** 2
    per_comp = sq_err.mean(dim=0)
    weighted = (component_weights * per_comp).sum()

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
                loss   = compute_loss_v3(
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
            loss   = compute_loss_v3(
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
        loss   = compute_loss_v3(
            y_pred, batch.y, component_weights, model, kl_weight
        )
        total_loss += loss.item() * batch.num_graphs
        n          += batch.num_graphs
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return total_loss / max(n, 1)


# ============================================================================
# Full training run
# ============================================================================

def train_v3(
    mode:        str              = "re_angle_stratified",
    holdout_geo: str | None       = None,
    epochs:      int | None       = None,
    device:      torch.device | None = None,
    tag:         str              = "",
):
    config_v3.create_directories()
    device  = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs  = epochs or config_v3.num_epochs
    use_amp = config_v3.use_amp and device.type == "cuda"

    print(f"\n{'='*70}")
    print(f"Bifurcation WSS Predictor V3  |  mode={mode}  "
          f"holdout={holdout_geo}  epochs={epochs}")
    if torch.cuda.is_available():
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"{'='*70}\n")

    # ---- data ----
    train_p, val_p, test_p = get_split_paths_v3(mode, holdout_geo)
    norm_stats = compute_normalization_stats_v3(train_p)
    save_normalization_stats_v3(norm_stats)

    train_ds = BifurcationWSSDatasetV3(train_p, norm_stats)
    val_ds   = BifurcationWSSDatasetV3(val_p,   norm_stats)

    train_loader = DataLoader(train_ds, batch_size=config_v3.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config_v3.batch_size, shuffle=False)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_p)}")

    # ---- component loss weights ----
    raw_w = torch.tensor(config_v3.component_loss_weights, dtype=torch.float32)
    component_weights = (raw_w / raw_w.sum()).to(device)
    print(f"Component loss weights (normalised): {component_weights.tolist()}")

    # ---- model ----
    model = BifurcationWSSPredictorV3(
        node_feat_dim  = config_v3.node_feat_dim,
        edge_feat_dim  = config_v3.edge_feat_dim,
        hidden_dim     = config_v3.hidden_dim,
        num_heads      = config_v3.num_heads,
        out_channels   = config_v3.output_dim,
        num_layers     = config_v3.num_layers,
        context_dim    = config_v3.context_dim,
        flow_param_dim = config_v3.flow_param_dim,
    ).to(device)
    get_model_summary(model)

    # ---- optimiser / scaler / scheduler ----
    optimizer = optim.Adam(
        model.parameters(),
        lr           = config_v3.learning_rate,
        weight_decay = config_v3.weight_decay,
    )
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor   = config_v3.scheduler_factor,
        patience = config_v3.scheduler_patience,
        min_lr   = config_v3.scheduler_min_lr,
    )

    # ---- checkpoint ----
    ckpt_name = f"best_model_v3{'_' + tag if tag else ''}.pt"
    ckpt_path = config_v3.models_dir / ckpt_name

    # ---- training loop ----
    best_val   = float("inf")
    no_improve = 0
    history    = {"train_loss": [], "val_loss": [], "lr": []}

    print(f"\n{'='*70}")
    print("Starting v3 training loop …")
    print(f"{'='*70}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, device,
            component_weights = component_weights,
            grad_clip         = config_v3.grad_clip,
            kl_weight         = config_v3.kl_weight,
            use_amp           = use_amp,
        )
        val_loss = validate_epoch(
            model, val_loader, device,
            component_weights = component_weights,
            kl_weight         = config_v3.kl_weight,
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
                        "model_version":   "v3",
                        "mode":            mode,
                        "holdout_geo":     holdout_geo,
                        "node_feat_dim":   config_v3.node_feat_dim,
                        "edge_feat_dim":   config_v3.edge_feat_dim,
                        "hidden_dim":      config_v3.hidden_dim,
                        "num_heads":       config_v3.num_heads,
                        "num_layers":      config_v3.num_layers,
                        "context_dim":     config_v3.context_dim,
                        "output_dim":      config_v3.output_dim,
                        "flow_param_dim":  config_v3.flow_param_dim,
                        "lpe_k":           config_v3.lpe_k,
                    },
                },
                ckpt_path,
            )
        else:
            no_improve += 1
            if no_improve >= config_v3.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(patience={config_v3.early_stop_patience})")
                break

    # ---- save history ----
    hist_path = config_v3.models_dir / f"training_log_v3{'_' + tag if tag else ''}.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Checkpoint:    {ckpt_path}")
    print(f"History:       {hist_path}")

    return best_val, ckpt_path


# ============================================================================
# LOOCV driver
# ============================================================================

def run_loocv_v3(epochs=None, device=None):
    results = {}
    for geo in config_v3.geometry_folders:
        print(f"\n{'#'*70}")
        print(f"# FOLD: holdout = {geo}")
        print(f"{'#'*70}")
        val_loss, ckpt = train_v3(
            mode        = "loocv-geo",
            holdout_geo = geo,
            epochs      = epochs,
            device      = device,
            tag         = geo,
        )
        results[geo] = {"val_loss": val_loss, "ckpt": str(ckpt)}

    print(f"\n{'='*70}")
    print("LOOCV V3 RESULTS")
    print(f"{'='*70}")
    for geo, info in results.items():
        print(f"  {geo:45s}  val_loss={info['val_loss']:.6f}")
    mean_loss = sum(r["val_loss"] for r in results.values()) / len(results)
    print(f"\n  Mean val loss: {mean_loss:.6f}")

    summary_path = config_v3.models_dir / "loocv_v3_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {summary_path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bifurcation WSS model v3")
    parser.add_argument("--mode", choices=["re_angle_stratified", "random", "loocv-geo"],
                        default="re_angle_stratified")
    parser.add_argument("--holdout", type=str, default=None)
    parser.add_argument("--epochs",  type=int, default=None)
    parser.add_argument("--device",  type=str, default=None)
    parser.add_argument("--process", action="store_true",
                        help="Build v3 ProcessedData cache before training")
    parser.add_argument("--no-amp",  action="store_true")
    args = parser.parse_args()

    if args.no_amp:
        config_v3.use_amp = False

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if args.process:
        build_all_processed_v3()

    if args.mode == "loocv-geo" and args.holdout is None:
        run_loocv_v3(epochs=args.epochs, device=device)
    else:
        train_v3(
            mode        = args.mode,
            holdout_geo = args.holdout,
            epochs      = args.epochs,
            device      = device,
        )

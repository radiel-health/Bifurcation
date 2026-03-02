"""
Mixed training: fine-tune V3 model on bifurcation + carotid data.

Strategy:
  1. Load bifurcation train split (ProcessedData_v3) from EC2
  2. Load carotid .pt files (ProcessedData_carotid), split by patient 80/20
  3. Compute combined normalisation stats over all training data
  4. Fine-tune from pretrained best_model_v3.pt at LR=1e-4
  5. Validate on bifurcation val split (carotid test held out for morning)
  6. Save best model → Models_v3/best_model_mixed.pt
  7. Save carotid test patient IDs → results_v3/carotid_test_ids.json

NOTE on memory:
  Carotid meshes: ~15K nodes, ~1.7M edges per case (very dense).
  T4 (12 GB) should handle with gradient checkpointing (already in bif_v3.py).
  If OOM, set --max-carotid-train to limit carotid training cases.

Usage:
    # On EC2 (after uploading ProcessedData_carotid):
    python -m Bifurcation.train_v3_mixed \\
        --pretrained Bifurcation/Models_v3/best_model_v3.pt

    # Limit carotid cases if OOM:
    python -m Bifurcation.train_v3_mixed \\
        --pretrained Bifurcation/Models_v3/best_model_v3.pt \\
        --max-carotid-train 40
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
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
    compute_normalization_stats_v3,
    save_normalization_stats_v3,
    get_split_paths_v3,
)
from Bifurcation.Models.bif_v3 import (
    BifurcationWSSPredictorV3,
    count_parameters,
    get_model_summary,
)
from Bifurcation.train_v3 import compute_loss_v3, train_epoch, validate_epoch


# ============================================================================
# Carotid data helpers
# ============================================================================

CAROTID_DIR = config_v3.project_root.parent / "Bifurcation" / "ProcessedData_carotid"


def load_carotid_paths(carotid_dir: Path) -> List[Path]:
    """Return paths to all carotid systolic.pt files that have WSS labels."""
    paths = []
    for p in sorted(carotid_dir.glob("*/systolic.pt")):
        d = torch.load(p, weights_only=False)
        if d.y is not None and d.y.shape[0] > 0:
            paths.append(p)
    return paths


def split_carotid_by_patient(
    paths: List[Path],
    train_ratio: float = 0.80,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """
    Split carotid paths into train/test by patient ID to avoid data leakage.

    Patient ID extracted as the first 4 underscore-joined tokens of the directory name.
    e.g. 'carotid_case_k_001_left' → patient 'carotid_case_k_001'
    Both left and right carotids from the same patient go to the same split.
    """
    patients: dict = defaultdict(list)
    for p in paths:
        patient_id = "_".join(p.parent.name.split("_")[:4])  # carotid_case_k_001
        patients[patient_id].append(p)

    patient_ids = sorted(patients.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(patient_ids)

    n_train    = int(train_ratio * len(patient_ids))
    train_pids = set(patient_ids[:n_train])
    test_pids  = set(patient_ids[n_train:])

    train_paths = [p for pid in sorted(train_pids) for p in patients[pid]]
    test_paths  = [p for pid in sorted(test_pids)  for p in patients[pid]]

    return train_paths, test_paths, sorted(train_pids), sorted(test_pids)


# ============================================================================
# Main training function
# ============================================================================

def train_v3_mixed(
    pretrained_ckpt: Optional[str]           = None,
    carotid_dir:     Optional[Path]          = None,
    epochs:          Optional[int]           = None,
    device:          Optional[torch.device]  = None,
    max_carotid_train: Optional[int]         = None,
    lr:              float                   = 1e-4,
):
    config_v3.create_directories()
    device  = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs  = epochs or config_v3.num_epochs
    use_amp = config_v3.use_amp and device.type == "cuda"
    carotid_dir = carotid_dir or CAROTID_DIR

    print(f"\n{'='*70}")
    print("Bifurcation + Carotid Mixed Training V3")
    if torch.cuda.is_available():
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"Device: {device}")
    print(f"AMP: {use_amp}  |  Fine-tune LR: {lr}")
    print(f"{'='*70}\n")

    # ── bifurcation data ─────────────────────────────────────────────────────
    bif_train, bif_val, bif_test = get_split_paths_v3()
    print(f"Bifurcation  train={len(bif_train)}  val={len(bif_val)}  test={len(bif_test)}")

    # ── carotid data ─────────────────────────────────────────────────────────
    carotid_all                             = load_carotid_paths(carotid_dir)
    car_train, car_test, train_pids, test_pids = split_carotid_by_patient(carotid_all)

    if max_carotid_train and len(car_train) > max_carotid_train:
        print(f"Limiting carotid train to {max_carotid_train} cases "
              f"(from {len(car_train)})")
        car_train = car_train[:max_carotid_train]

    print(f"Carotid      train={len(car_train)}  test={len(car_test)}")
    print(f"  Train patients: {len(train_pids)}  |  Test patients: {len(test_pids)}")

    # Save test patient IDs for morning evaluation
    config_v3.results_dir.mkdir(parents=True, exist_ok=True)
    test_ids_path = config_v3.results_dir / "carotid_test_ids.json"
    with open(test_ids_path, "w") as f:
        json.dump({"test_patient_ids": test_pids,
                   "test_paths": [str(p) for p in car_test]}, f, indent=2)
    print(f"  Test IDs saved → {test_ids_path}")

    # ── combined norm stats ───────────────────────────────────────────────────
    combined_train = bif_train + car_train
    print(f"\nComputing combined normalisation stats over {len(combined_train)} samples …")
    combined_stats = compute_normalization_stats_v3(combined_train)
    stats_path     = config_v3.models_dir / "normalization_stats_mixed.json"
    save_normalization_stats_v3(combined_stats, path=stats_path)
    print(f"Saved combined stats → {stats_path}")

    # ── datasets / loaders ───────────────────────────────────────────────────
    train_ds = BifurcationWSSDatasetV3(
        combined_train, combined_stats, augment=True,  use_lpe=False
    )
    val_ds = BifurcationWSSDatasetV3(
        bif_val,        combined_stats, augment=False, use_lpe=False
    )

    train_loader = DataLoader(train_ds, batch_size=config_v3.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config_v3.batch_size, shuffle=False)

    print(f"\nTrain loader: {len(train_ds)} samples  |  "
          f"Val loader: {len(val_ds)} samples")

    # ── component loss weights ────────────────────────────────────────────────
    raw_w             = torch.tensor(config_v3.component_loss_weights, dtype=torch.float32)
    component_weights = (raw_w / raw_w.sum()).to(device)
    print(f"Component loss weights: {component_weights.tolist()}")

    # ── model ─────────────────────────────────────────────────────────────────
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

    # ── load pretrained weights ───────────────────────────────────────────────
    if pretrained_ckpt:
        print(f"\nLoading pretrained weights from {pretrained_ckpt}")
        ckpt = torch.load(pretrained_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded (epoch {ckpt.get('epoch', '?')}, "
              f"val_loss {ckpt.get('val_loss', '?'):.6f})")
    else:
        print("\nNo pretrained weights — training from scratch")

    get_model_summary(model)

    # ── optimiser / scheduler ────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=config_v3.weight_decay)
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor   = config_v3.scheduler_factor,
        patience = config_v3.scheduler_patience,
        min_lr   = config_v3.scheduler_min_lr,
    )

    # ── checkpoint path ───────────────────────────────────────────────────────
    ckpt_path = config_v3.models_dir / "best_model_mixed.pt"

    # ── training loop ─────────────────────────────────────────────────────────
    best_val   = float("inf")
    no_improve = 0
    history    = {"train_loss": [], "val_loss": [], "lr": []}

    print(f"\n{'='*70}")
    print("Starting mixed training loop …")
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
                    "norm_stats":  combined_stats,
                    "config": {
                        "model_version":  "v3_mixed",
                        "node_feat_dim":  config_v3.node_feat_dim,
                        "edge_feat_dim":  config_v3.edge_feat_dim,
                        "hidden_dim":     config_v3.hidden_dim,
                        "num_heads":      config_v3.num_heads,
                        "num_layers":     config_v3.num_layers,
                        "context_dim":    config_v3.context_dim,
                        "output_dim":     config_v3.output_dim,
                        "flow_param_dim": config_v3.flow_param_dim,
                        "pretrained":     pretrained_ckpt,
                        "bif_train":      len(bif_train),
                        "car_train":      len(car_train),
                        "car_test":       len(car_test),
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

    # ── save history ──────────────────────────────────────────────────────────
    hist_path = config_v3.models_dir / "training_log_mixed.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Checkpoint:    {ckpt_path}")
    print(f"History:       {hist_path}")
    print(f"Test IDs:      {test_ids_path}")

    return best_val, ckpt_path


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune V3 bifurcation model on mixed bifurcation + carotid data"
    )
    parser.add_argument("--pretrained",      type=str, default=None,
                        help="Path to pretrained v3 checkpoint (best_model_v3.pt)")
    parser.add_argument("--carotid-dir",     type=str, default=None,
                        help="Path to ProcessedData_carotid/ (default: auto-detected)")
    parser.add_argument("--epochs",          type=int, default=None)
    parser.add_argument("--device",          type=str, default=None)
    parser.add_argument("--lr",              type=float, default=1e-4,
                        help="Fine-tune learning rate (default: 1e-4, 10x lower than train_v3)")
    parser.add_argument("--max-carotid-train", type=int, default=None,
                        help="Limit carotid training cases (useful for OOM on T4)")
    parser.add_argument("--no-amp",          action="store_true")
    args = parser.parse_args()

    if args.no_amp:
        config_v3.use_amp = False

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    carotid_dir = Path(args.carotid_dir) if args.carotid_dir else None

    train_v3_mixed(
        pretrained_ckpt    = args.pretrained,
        carotid_dir        = carotid_dir,
        epochs             = args.epochs,
        device             = device,
        max_carotid_train  = args.max_carotid_train,
        lr                 = args.lr,
    )

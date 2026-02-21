#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP C — TRAINING
Train a node-level regression GNN on PyTorch Geometric Data objects produced by your preprocessing.

Dataset:
  Re50.pt, Re75.pt, ... each is a Data object with:
    data.pos        [N,3]
    data.edge_index [2,E]
    data.x          [N,F]
    data.re         float (scalar condition)
    data.y          [N,3]  (WSS vector target per node)

Output:
  best_model.pt  (checkpoint with weights + normalization stats)

IMPORTANT:
- This training code avoids torch_scatter/torch_sparse by using a custom message passing layer
  implemented with pure PyTorch index_add_.
- Works with PyTorch Geometric DataLoader batching.

Author: (you + ChatGPT)
"""

from __future__ import annotations

import os
import math
import glob
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# ======================================================================================
# USER SETTINGS (EDIT THESE ONCE)
# ======================================================================================

DATA_DIR   = Path(r"C:\\Users\\radie\\Desktop\\trainingprocess\\ProcessedData\\coronary_extracted_vessel")
OUT_DIR    = Path(r"C:\\Users\\radie\\Desktop\\trainingprocess\\TRAIN_OUT")
CKPT_NAME  = "best_model.pt"

# Train/Val/Test split over the 18 graphs (later: do geometry-level splits)
SPLIT_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

# Training hyperparameters
EPOCHS         = 400
BATCH_SIZE     = 2          # graphs per batch (18 graphs total)
LR             = 1e-3
WEIGHT_DECAY   = 1e-5
GRAD_CLIP_NORM = 1.0

# Early stopping
PATIENCE       = 40         # stop if no val improvement for this many epochs
MIN_DELTA      = 1e-6

# Model config
HIDDEN_DIM   = 128
NUM_LAYERS   = 6
DROPOUT      = 0.10

# Condition handling
# Append Re (normalized) as an extra node feature channel
APPEND_RE_TO_NODE_FEATURES = True

# Target normalization (recommended)
NORMALIZE_Y = True

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================================================


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_graph_files(data_dir: Path) -> List[Path]:
    files = sorted([Path(p) for p in glob.glob(str(data_dir / "Re*.pt"))])
    if not files:
        raise FileNotFoundError(f"No Re*.pt files found in: {data_dir}")
    return files


def load_graphs(files: List[Path]) -> List[Data]:
    graphs = []
    for p in files:
        d = torch.load(p, map_location="cpu", weights_only=False)
        if not isinstance(d, Data):
            raise TypeError(f"{p.name} did not 
                            load as torch_geometric.data.Data")
        # Minimal sanity:
        if not hasattr(d, "y"):
            raise ValueError(f"{p.name} missing data.y (labels)")
        if not hasattr(d, "re"):
            raise ValueError(f"{p.name} missing data.re (condition)")
        graphs.append(d)
    return graphs


def split_indices(n: int, train_frac: float, val_frac: float, test_frac: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_train = int(round(train_frac * n))
    n_val   = int(round(val_frac * n))
    # ensure total = n
    n_train = min(n_train, n)
    n_val   = min(n_val, n - n_train)
    n_test  = n - n_train - n_val
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]
    assert len(test_idx) == n_test
    return train_idx, val_idx, test_idx


def compute_re_stats(graphs: List[Data]) -> Tuple[float, float]:
    re_vals = np.array([float(g.re) for g in graphs], dtype=np.float64)
    mu = float(re_vals.mean())
    sd = float(re_vals.std() + 1e-12)
    return mu, sd


def compute_y_stats(graphs: List[Data]) -> Tuple[torch.Tensor, torch.Tensor]:
    # Concatenate all y across graphs
    ys = torch.cat([g.y.reshape(-1, g.y.shape[-1]).float() for g in graphs], dim=0)
    mu = ys.mean(dim=0)
    sd = ys.std(dim=0) + 1e-12
    return mu, sd


def attach_re_feature(data: Data, re_mu: float, re_sd: float) -> Data:
    """Append normalized Re as a node feature channel."""
    re_norm = (float(data.re) - re_mu) / re_sd
    re_feat = torch.full((data.num_nodes, 1), float(re_norm), dtype=torch.float32)
    if data.x is None:
        data.x = re_feat
    else:
        data.x = torch.cat([data.x.float(), re_feat], dim=1)
    return data


def normalize_y(data: Data, y_mu: torch.Tensor, y_sd: torch.Tensor) -> Data:
    data.y = (data.y.float() - y_mu) / y_sd
    return data


# -------------------------
# Pure-PyTorch Message Passing Layer (no torch_scatter needed)
# -------------------------

class GraphMP(nn.Module):
    """
    Simple message passing:
      h' = W_self h + W_nei * AGG(h_nei)

    AGG = mean over neighbors using index_add_ + degree.
    Works with batched PyG graphs (edge_index already offset).
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_nei  = nn.Linear(in_dim, out_dim)
        self.dropout  = dropout
        self.norm     = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # edge_index: [2, E], directed edges
        row, col = edge_index[0], edge_index[1]  # messages: col -> row

        # Aggregate neighbor features into each row node
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col])

        deg = torch.zeros((x.size(0), 1), device=x.device, dtype=x.dtype)
        ones = torch.ones((row.numel(), 1), device=x.device, dtype=x.dtype)
        deg.index_add_(0, row, ones)
        agg = agg / (deg + 1e-12)  # mean

        out = self.lin_self(x) + self.lin_nei(agg)
        out = self.norm(out)
        out = F.silu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class WSSNet(nn.Module):
    """
    Node-level regression network:
      input: data.x (optionally includes Re)
      output: pred y_hat [num_nodes, 3]
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 3, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.layers = nn.ModuleList([
            GraphMP(hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)
        ])

        self.dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x.float()
        ei = data.edge_index.long()
        h = self.enc(x)

        for layer in self.layers:
            h_new = layer(h, ei)
            h = h + h_new  # residual

        y_hat = self.dec(h)
        return y_hat


# -------------------------
# Training / Eval loops
# -------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total_loss = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.float()
        loss = F.mse_loss(pred, y, reduction="sum")
        total_loss += float(loss.item())
        total_nodes += int(y.numel())  # counts scalar elements

    return total_loss / max(total_nodes, 1)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str, grad_clip: float) -> float:
    model.train()
    total_loss = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        y = batch.y.float()
        loss = F.mse_loss(pred, y, reduction="mean")
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # track weighted by node count
        total_loss += float(loss.item()) * int(y.numel())
        total_nodes += int(y.numel())

    return total_loss / max(total_nodes, 1)


# -------------------------
# Main
# -------------------------

def main():
    set_seed(SPLIT_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("STEP C — TRAINING: Re*.pt graphs -> best_model.pt (learned weights)")
    print("=" * 96)
    print(f"DATA_DIR   : {DATA_DIR}")
    print(f"OUT_DIR    : {OUT_DIR}")
    print(f"DEVICE     : {DEVICE}")
    print(f"EPOCHS     : {EPOCHS}")
    print(f"BATCH_SIZE : {BATCH_SIZE}")
    print()

    files = list_graph_files(DATA_DIR)
    graphs = load_graphs(files)
    n = len(graphs)
    print(f"[OK] Loaded {n} graphs:")
    print("     " + ", ".join([p.name for p in files]))
    print()

    # Split
    train_idx, val_idx, test_idx = split_indices(n, TRAIN_FRAC, VAL_FRAC, TEST_FRAC, SPLIT_SEED)
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs   = [graphs[i] for i in val_idx]
    test_graphs  = [graphs[i] for i in test_idx]

    print(f"Split: train={len(train_graphs)} val={len(val_graphs)} test={len(test_graphs)}")
    print(f"Train idx: {sorted(train_idx)}")
    print(f"Val idx  : {sorted(val_idx)}")
    print(f"Test idx : {sorted(test_idx)}")
    print()

    # Condition stats (Re)
    re_mu, re_sd = compute_re_stats(train_graphs)
    print(f"Re stats (train): mean={re_mu:.4f}, std={re_sd:.4f}")

    # Y normalization stats
    if NORMALIZE_Y:
        y_mu, y_sd = compute_y_stats(train_graphs)
        print(f"Y stats (train): mean={y_mu.tolist()}, std={y_sd.tolist()}")
    else:
        y_mu = torch.zeros((train_graphs[0].y.shape[-1],), dtype=torch.float32)
        y_sd = torch.ones((train_graphs[0].y.shape[-1],), dtype=torch.float32)

    # Prepare graphs: append Re feature, normalize y (train/val/test consistently)
    def prep_list(gs: List[Data]) -> List[Data]:
        out = []
        for g in gs:
            g2 = g.clone()
            if APPEND_RE_TO_NODE_FEATURES:
                g2 = attach_re_feature(g2, re_mu, re_sd)
            if NORMALIZE_Y:
                g2 = normalize_y(g2, y_mu, y_sd)
            out.append(g2)
        return out

    train_graphs = prep_list(train_graphs)
    val_graphs   = prep_list(val_graphs)
    test_graphs  = prep_list(test_graphs)

    # Infer input feature dim
    in_dim = int(train_graphs[0].x.shape[1])
    out_dim = int(train_graphs[0].y.shape[1])

    print(f"\nModel IO: in_dim={in_dim}  -> out_dim={out_dim}")
    print()

    # DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=BATCH_SIZE, shuffle=False) if len(val_graphs) else None
    test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False) if len(test_graphs) else None

    # Model
    model = WSSNet(
        in_dim=in_dim,
        hidden_dim=HIDDEN_DIM,
        out_dim=out_dim,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Training loop with early stopping
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, GRAD_CLIP_NORM)

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, DEVICE)
        else:
            val_loss = train_loss

        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})

        improved = (best_val - val_loss) > MIN_DELTA
        if improved:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0

            ckpt = {
                "model_state": model.state_dict(),
                "in_dim": in_dim,
                "out_dim": out_dim,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "append_re_to_x": APPEND_RE_TO_NODE_FEATURES,
                "normalize_y": NORMALIZE_Y,
                "re_mu": re_mu,
                "re_sd": re_sd,
                "y_mu": y_mu.cpu(),
                "y_sd": y_sd.cpu(),
                "best_epoch": best_epoch,
                "best_val_mse": best_val,
                "data_dir": str(DATA_DIR),
                "files": [p.name for p in files],
                "split": {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx},
            }
            torch.save(ckpt, OUT_DIR / CKPT_NAME)

        else:
            bad_epochs += 1

        if epoch % 10 == 0 or epoch == 1:
            tag = " *" if improved else ""
            print(f"Epoch {epoch:4d} | train_mse={train_loss:.6e} | val_mse={val_loss:.6e}{tag}")

        if bad_epochs >= PATIENCE:
            print(f"\n[EARLY STOP] No improvement for {PATIENCE} epochs. Best epoch={best_epoch} val={best_val:.6e}")
            break

    # Final evaluation on test (denormalize-aware metric)
    # Note: we trained in normalized y-space if NORMALIZE_Y=True.
    # For a more interpretable metric in physical units, we can compute MSE in original units too.
    if test_loader is not None and len(test_graphs) > 0:
        # Load best checkpoint back in
        ckpt = torch.load(OUT_DIR / CKPT_NAME, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # Compute normalized MSE
        test_mse_norm = evaluate(model, test_loader, DEVICE)

        # Compute physical MSE (denorm)
        y_mu_t = ckpt["y_mu"].to(DEVICE).view(1, -1)
        y_sd_t = ckpt["y_sd"].to(DEVICE).view(1, -1)

        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                pred = model(batch)
                y = batch.y.float()
                # denormalize
                pred_phys = pred * y_sd_t + y_mu_t
                y_phys = y * y_sd_t + y_mu_t
                mse = F.mse_loss(pred_phys, y_phys, reduction="sum")
                total += float(mse.item())
                count += int(y.numel())
        test_mse_phys = total / max(count, 1)

        print("\n" + "=" * 96)
        print("TEST EVAL (using best checkpoint)")
        print("=" * 96)
        print(f"Test MSE (normalized space): {test_mse_norm:.6e}")
        print(f"Test MSE (physical units)  : {test_mse_phys:.6e}")
    else:
        print("\n(No test set to evaluate — split produced empty test.)")

    # Save training history json
    hist_path = OUT_DIR / "train_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\n[OK] Saved training history: {hist_path}")
    print(f"[OK] Best checkpoint saved: {OUT_DIR / CKPT_NAME}")
    print("\nDONE")


if __name__ == "__main__":
    main()
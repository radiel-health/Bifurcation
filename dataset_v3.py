"""
Dataset module v3 for bifurcation WSS prediction.

New in v3 compared to dataset_v2.py:
  - compute_laplacian_pe(): 8 Laplacian Positional Encoding features
      Uses absolute eigenvectors of the normalised graph Laplacian.
      |λ_1|...|λ_8| appended to the 10 physics features → 18-dim node features.
  - sample_to_pyg_v3(): builds 18-dim node feature graph
  - build_all_processed_v3(): caches to ProcessedData_v3/
  - All v2 physics features (dist_to_junction, arc_length, branch_depth,
    normals, curvature) are preserved unchanged.

Usage:
    python -m Bifurcation.dataset_v3 --process   # build v3 cache (~1-2 h first run)
    python -m Bifurcation.dataset_v3 --check     # quick sanity check
    python -m Bifurcation.dataset_v3 --stats     # compute & save norm stats
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from Bifurcation.config_v3 import config_v3
from Bifurcation.dataset import (
    parse_openfoam_vector_field,
    parse_openfoam_faces,
    parse_boundary,
    load_sample,
)
# Reuse all physics feature helpers from v2
from Bifurcation.dataset_v2 import (
    compute_physics_features_v2,
)


# ============================================================================
# Rotation augmentation
# ============================================================================

def rotate_z(data: Data, theta_deg: float) -> Data:
    """
    Rotate the mesh around the Z-axis (inlet flow axis) by theta_deg degrees.

    Physically valid: azimuthal symmetry means rotated geometry has identical
    WSS magnitude distribution — only the X/Y components transform consistently.

    Tensors rotated: xyz coords, surface normals, edge dx/dy/dz, WSS target.
    Scalars left unchanged: dist_to_junction, arc_length, branch_depth,
                             curvature, edge distance, LPE eigenvectors.
    """
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    R = torch.tensor([
        [cos_t, -sin_t, 0.0],
        [sin_t,  cos_t, 0.0],
        [0.0,    0.0,   1.0],
    ], dtype=data.x.dtype)
    data = data.clone()
    data.x[:, 0:3]         = data.x[:, 0:3]         @ R.t()   # xyz coords
    data.x[:, 6:9]         = data.x[:, 6:9]         @ R.t()   # nx, ny, nz normals
    data.edge_attr[:, 1:4] = data.edge_attr[:, 1:4] @ R.t()   # dx, dy, dz
    data.y                 = data.y                  @ R.t()   # WSS [wx, wy, wz]
    return data


# ============================================================================
# Laplacian Positional Encoding
# ============================================================================

def compute_laplacian_pe(
    edge_index: np.ndarray,
    num_nodes:  int,
    k:          int = 8,
) -> np.ndarray:
    """
    Compute k Laplacian eigenvectors as positional encodings.

    Uses the normalised Laplacian L = I - D^{-1/2} A D^{-1/2}.
    Returns absolute values to resolve the sign ambiguity inherent in
    eigenvectors (|λ_1|...|λ_k|), shape (num_nodes, k).

    Falls back to zeros on any error (graph too small, scipy unavailable, etc).

    Parameters
    ----------
    edge_index : (2, E) int array — directed edge list (will be symmetrised)
    num_nodes  : N
    k          : number of eigenvectors to return
    """
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import eigsh

        src, dst = edge_index[0], edge_index[1]

        # Symmetrise: treat graph as undirected
        all_src = np.concatenate([src, dst])
        all_dst = np.concatenate([dst, src])

        # Degree
        degree = np.bincount(all_src, minlength=num_nodes).astype(np.float64)
        degree = np.maximum(degree, 1.0)  # avoid div-by-zero for isolated nodes
        deg_inv_sqrt = 1.0 / np.sqrt(degree)

        # Off-diagonal entries of normalised Laplacian:  -1/sqrt(d_i * d_j)
        off_data = -deg_inv_sqrt[all_src] * deg_inv_sqrt[all_dst]

        # Build sparse matrix: off-diagonal + diagonal
        rows = np.concatenate([all_src, np.arange(num_nodes)])
        cols = np.concatenate([all_dst, np.arange(num_nodes)])
        data = np.concatenate([off_data, np.ones(num_nodes, dtype=np.float64)])

        L = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        L = L.tocsr()
        # Symmetrise numerically (floating-point safety)
        L = (L + L.T) * 0.5

        # Request k+1 eigenvectors (first is trivial, eigenvalue ≈ 0)
        actual_k = min(k + 1, num_nodes - 2)
        if actual_k < 2:
            return np.zeros((num_nodes, k), dtype=np.float32)

        eigenvalues, eigenvectors = eigsh(
            L, k=actual_k, which="SM", tol=1e-4, maxiter=3000
        )

        # Sort by ascending eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, idx]

        # Skip the trivial eigenvector (index 0, eigenvalue ≈ 0)
        nontrivial = eigenvectors[:, 1 : k + 1]   # (N, up_to_k)

        # Absolute value for sign stability
        lpe = np.abs(nontrivial).astype(np.float32)

        # Pad to exactly k columns if the graph had fewer non-trivial eigenvectors
        if lpe.shape[1] < k:
            pad = np.zeros((num_nodes, k - lpe.shape[1]), dtype=np.float32)
            lpe = np.concatenate([lpe, pad], axis=1)

        return lpe[:, :k]

    except Exception:
        return np.zeros((num_nodes, k), dtype=np.float32)


# ============================================================================
# V3 graph builder
# ============================================================================

def sample_to_pyg_v3(geometry_folder: str, re_case: str) -> Data:
    """
    Build a single PyG Data object with 18-dim node features (v3).

    Features: 10 physics features (same as v2) + 8 Laplacian PE features.
    """
    geo_path = config_v3.get_geometry_path(geometry_folder)
    angle, _mesh_level = config_v3.parse_geometry_folder(geometry_folder)

    (wall_centres, wss, edge_index, edge_attr,
     re_num, boundary, surface_mesh_data) = load_sample(
        str(geo_path), re_case, mesh_re=config_v3.mesh_re,
        return_surface_mesh=True,
    )
    all_points, wall_face_vertices, _wall_face_indices = surface_mesh_data

    poly_path = geo_path / config_v3.mesh_re / "constant" / "polyMesh"
    all_faces = parse_openfoam_faces(str(poly_path / "faces"))

    # 10 physics features (reuse v2 computation)
    physics_feats = compute_physics_features_v2(
        wall_centres       = wall_centres,
        all_points         = all_points,
        wall_face_vertices = wall_face_vertices,
        boundary           = boundary,
        all_faces          = all_faces,
        edge_index         = edge_index,
    )  # (N, 10)

    # 8 Laplacian PE features
    lpe_feats = compute_laplacian_pe(
        edge_index = edge_index,
        num_nodes  = len(wall_centres),
        k          = config_v3.lpe_k,
    )  # (N, 8)

    # Concatenate → 18-dim
    x = np.concatenate([physics_feats, lpe_feats], axis=1)  # (N, 18)

    return Data(
        x          = torch.tensor(x,            dtype=torch.float32),
        pos        = torch.tensor(wall_centres,  dtype=torch.float32),
        y          = torch.tensor(wss,           dtype=torch.float32),
        edge_index = torch.tensor(edge_index,    dtype=torch.long),
        edge_attr  = torch.tensor(edge_attr,     dtype=torch.float32),
        re         = torch.tensor([re_num],      dtype=torch.float32),
        angle      = torch.tensor([angle],       dtype=torch.float32),
        geo_name   = geometry_folder,
        re_name    = re_case,
    )


# ============================================================================
# Build v3 processed dataset cache
# ============================================================================

def build_all_processed_v3(force: bool = False):
    """
    Build and cache all (geometry, Re) graphs with v3 features (18-dim).

    Output: ProcessedData_v3/<geometry>/Re<N>.pt

    Note: LPE computation (scipy eigsh on ~40K nodes) adds ~30-60 s per graph.
    Expect ~1-2 h for all 171 samples on first run.
    """
    config_v3.create_directories()
    built, skipped, failed = 0, 0, 0
    failed_cases = []

    pbar = tqdm(total=config_v3.total_samples, desc="Building v3 dataset (LPE)")
    for geo in config_v3.geometry_folders:
        out_dir = config_v3.processed_data_dir / geo
        out_dir.mkdir(parents=True, exist_ok=True)

        for re_val in config_v3.re_values:
            re_case  = f"Re{re_val}"
            out_path = out_dir / f"{re_case}.pt"

            if out_path.exists() and not force:
                skipped += 1
                pbar.update(1)
                continue

            try:
                data = sample_to_pyg_v3(geo, re_case)
                torch.save(data, out_path)
                built += 1
            except Exception as e:
                failed += 1
                failed_cases.append(f"{geo}/{re_case}: {e}")
            pbar.update(1)
    pbar.close()

    print(f"\nBuilt {built} | Skipped (cached) {skipped} | Failed {failed}")
    if failed_cases:
        for msg in failed_cases[:10]:
            print(f"  FAIL: {msg}")


# ============================================================================
# Normalization helpers
# ============================================================================

def compute_normalization_stats_v3(
    graph_paths: Optional[List[Path]] = None,
) -> Dict:
    if graph_paths is None:
        graph_paths = sorted(config_v3.processed_data_dir.rglob("*.pt"))

    all_x, all_edge, all_y = [], [], []
    for p in graph_paths:
        d = torch.load(p, weights_only=False)
        all_x.append(d.x)
        all_edge.append(d.edge_attr)
        all_y.append(d.y)

    all_x    = torch.cat(all_x,    0)
    all_edge = torch.cat(all_edge, 0)
    all_y    = torch.cat(all_y,    0)

    x_mean, x_std = all_x.mean(0), all_x.std(0)
    e_mean, e_std = all_edge.mean(0), all_edge.std(0)

    if config_v3.use_log_transform:
        sign  = torch.sign(all_y)
        log_y = sign * torch.log1p(torch.abs(all_y))
        y_mean, y_std = log_y.mean(0), log_y.std(0)
    else:
        y_mean, y_std = all_y.mean(0), all_y.std(0)

    x_std = torch.clamp(x_std, min=1e-8)
    e_std = torch.clamp(e_std, min=1e-8)
    y_std = torch.clamp(y_std, min=1e-8)

    return {
        "x_mean":            x_mean.tolist(),
        "x_std":             x_std.tolist(),
        "edge_mean":         e_mean.tolist(),
        "edge_std":          e_std.tolist(),
        "y_mean":            y_mean.tolist(),
        "y_std":             y_std.tolist(),
        "use_log_transform": config_v3.use_log_transform,
    }


def save_normalization_stats_v3(stats: Dict, path: Optional[Path] = None):
    path = path or (config_v3.processed_data_dir / "normalization_stats_v3.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved v3 normalization stats → {path}")


def load_normalization_stats_v3(path: Optional[Path] = None) -> Dict:
    path = path or (config_v3.processed_data_dir / "normalization_stats_v3.json")
    with open(path) as f:
        return json.load(f)


# ============================================================================
# De-normalisation (for inference)
# ============================================================================

def denormalize_wss_v3(y_norm: torch.Tensor, stats: Dict) -> torch.Tensor:
    y_mean = torch.tensor(stats["y_mean"], dtype=y_norm.dtype, device=y_norm.device)
    y_std  = torch.tensor(stats["y_std"],  dtype=y_norm.dtype, device=y_norm.device)

    y = y_norm * y_std + y_mean

    if stats.get("use_log_transform", False):
        sign = torch.sign(y)
        y    = sign * torch.expm1(torch.abs(y))

    return y


# ============================================================================
# Dataset class
# ============================================================================

class BifurcationWSSDatasetV3(Dataset):
    """
    PyTorch Dataset loading v3 .pt graph files (18-dim node features).
    Applies z-score normalisation on the fly (same logic as v2).
    """

    def __init__(
        self,
        graph_paths: List[Path],
        norm_stats:  Optional[Dict] = None,
        augment:     bool = False,
        aug_angles:  tuple = (-15.0, -7.5, 7.5, 15.0),
        use_lpe:     bool = True,
    ):
        super().__init__()
        self.graph_paths = list(graph_paths)
        self.norm_stats  = norm_stats
        self.augment     = augment
        self.aug_angles  = list(aug_angles)
        self.use_lpe     = use_lpe

        if norm_stats is not None:
            self._x_mean = torch.tensor(norm_stats["x_mean"],    dtype=torch.float32)
            self._x_std  = torch.tensor(norm_stats["x_std"],     dtype=torch.float32)
            self._e_mean = torch.tensor(norm_stats["edge_mean"],  dtype=torch.float32)
            self._e_std  = torch.tensor(norm_stats["edge_std"],   dtype=torch.float32)
            self._y_mean = torch.tensor(norm_stats["y_mean"],     dtype=torch.float32)
            self._y_std  = torch.tensor(norm_stats["y_std"],      dtype=torch.float32)
            if not self.use_lpe:
                self._x_mean = self._x_mean[:10]
                self._x_std  = self._x_std[:10]

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx) -> Data:
        data = torch.load(self.graph_paths[idx], weights_only=False)

        if self.augment:
            theta = float(np.random.choice([0.0] + self.aug_angles))
            if theta != 0.0:
                data = rotate_z(data, theta)

        if not self.use_lpe:
            data.x = data.x[:, :10].clone()

        if self.norm_stats is not None:
            data.x         = (data.x         - self._x_mean) / self._x_std
            data.edge_attr = (data.edge_attr  - self._e_mean) / self._e_std

            if self.norm_stats.get("use_log_transform", False):
                sign   = torch.sign(data.y)
                data.y = sign * torch.log1p(torch.abs(data.y))
            data.y = (data.y - self._y_mean) / self._y_std

        return data


# ============================================================================
# Stratified splitting  (same logic as v2; reads from ProcessedData_v3)
# ============================================================================

from collections import defaultdict


def _geometry_from_path(p: Path) -> str:
    return p.parent.name


def _re_from_path(p: Path) -> int:
    return int(p.stem.replace("Re", ""))


def _re_bracket(re: int, brackets: List[Tuple[int, int]]) -> int:
    for i, (lo, hi) in enumerate(brackets):
        if lo <= re <= hi:
            return i
    return len(brackets) - 1


def get_split_paths_v3(
    mode:        str = "re_angle_stratified",
    holdout_geo: Optional[str] = None,
    seed:        int = 42,
    train_ratio: float = 0.75,
    val_ratio:   float = 0.10,
) -> Tuple[List[Path], List[Path], List[Path]]:
    all_paths = sorted(config_v3.processed_data_dir.rglob("*.pt"))
    if not all_paths:
        raise FileNotFoundError(
            "No v3 .pt files found — run: python -m Bifurcation.dataset_v3 --process"
        )

    rng = np.random.RandomState(seed)

    if mode == "loocv-geo":
        assert holdout_geo is not None
        test_paths  = [p for p in all_paths if _geometry_from_path(p) == holdout_geo]
        remaining   = [p for p in all_paths if _geometry_from_path(p) != holdout_geo]
        rem_geos    = sorted(set(_geometry_from_path(p) for p in remaining))
        val_geo     = rng.choice(rem_geos)
        val_paths   = [p for p in remaining if _geometry_from_path(p) == val_geo]
        train_paths = [p for p in remaining if _geometry_from_path(p) != val_geo]
        return train_paths, val_paths, test_paths

    elif mode == "random":
        geo_groups: Dict[str, List[Path]] = defaultdict(list)
        for p in all_paths:
            geo_groups[_geometry_from_path(p)].append(p)
        train_paths, val_paths, test_paths = [], [], []
        for _geo, paths in geo_groups.items():
            paths = sorted(paths)
            rng.shuffle(paths)
            n       = len(paths)
            n_train = int(n * train_ratio)
            n_val   = int(n * val_ratio)
            train_paths.extend(paths[:n_train])
            val_paths.extend(paths[n_train: n_train + n_val])
            test_paths.extend(paths[n_train + n_val:])
        return train_paths, val_paths, test_paths

    elif mode == "re_angle_stratified":
        brackets = config_v3.re_brackets
        cell_groups: Dict[Tuple, List[Path]] = defaultdict(list)
        for p in all_paths:
            geo     = _geometry_from_path(p)
            re      = _re_from_path(p)
            angle, _ = config_v3.parse_geometry_folder(geo)
            bracket  = _re_bracket(re, brackets)
            cell_groups[(angle, bracket)].append(p)

        test_paths, val_paths, train_paths = [], [], []
        for _cell, paths in cell_groups.items():
            paths = sorted(paths)
            rng.shuffle(paths)
            n      = len(paths)
            n_test = max(1, int(n * (1.0 - train_ratio - val_ratio)))
            n_val  = max(1, int(n * val_ratio))
            n_test = min(n_test, n - n_val - 1)
            test_paths.extend(paths[:n_test])
            val_paths.extend(paths[n_test: n_test + n_val])
            train_paths.extend(paths[n_test + n_val:])
        return train_paths, val_paths, test_paths

    else:
        raise ValueError(f"Unknown split mode: {mode}")


def get_dataloaders_v3(
    mode:        str = "re_angle_stratified",
    holdout_geo: Optional[str] = None,
    seed:        int = 42,
    norm_stats:  Optional[Dict] = None,
    batch_size:  Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_p, val_p, test_p = get_split_paths_v3(mode, holdout_geo, seed)

    if norm_stats is None:
        norm_stats = compute_normalization_stats_v3(train_p)
        save_normalization_stats_v3(norm_stats)

    bs = batch_size or config_v3.batch_size

    train_ds = BifurcationWSSDatasetV3(train_p, norm_stats)
    val_ds   = BifurcationWSSDatasetV3(val_p,   norm_stats)
    test_ds  = BifurcationWSSDatasetV3(test_p,  norm_stats)

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(val_ds,   batch_size=bs, shuffle=False),
        DataLoader(test_ds,  batch_size=bs, shuffle=False),
    )


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bifurcation v3 dataset utilities")
    parser.add_argument("--process", action="store_true",
                        help="Build all v3 .pt graphs (physics + LPE features)")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if cached")
    parser.add_argument("--check", action="store_true",
                        help="Parse one sample and print v3 feature summary")
    parser.add_argument("--stats", action="store_true",
                        help="Compute & save v3 normalisation stats")
    args = parser.parse_args()

    if args.process:
        build_all_processed_v3(force=args.force)
        stats = compute_normalization_stats_v3()
        save_normalization_stats_v3(stats)

    elif args.check:
        geo     = "bifurcation_angle45_750_ascii"
        re_case = "Re500"
        print(f"Parsing (v3) {geo} / {re_case} ...")
        data = sample_to_pyg_v3(geo, re_case)
        print(f"  Nodes:           {data.x.shape[0]}")
        print(f"  Node features:   {data.x.shape}  (expected [N, 18])")
        print(f"  Edges:           {data.edge_index.shape[1]}")
        print(f"  WSS targets:     {data.y.shape}")
        print(f"  Re={data.re.item():.0f}  angle={data.angle.item():.0f}°")

        labels = ["x", "y", "z", "dist_junc", "arc_len", "branch_depth",
                  "nx", "ny", "nz", "curvature",
                  "lpe_0", "lpe_1", "lpe_2", "lpe_3",
                  "lpe_4", "lpe_5", "lpe_6", "lpe_7"]
        for i, lbl in enumerate(labels):
            col = data.x[:, i]
            print(f"  feat[{i:2d}] {lbl:15s}: min={col.min():.4f}  max={col.max():.4f}  "
                  f"mean={col.mean():.4f}")
        print("OK")

    elif args.stats:
        stats = compute_normalization_stats_v3()
        save_normalization_stats_v3(stats)
        print(json.dumps(stats, indent=2))

    else:
        parser.print_help()

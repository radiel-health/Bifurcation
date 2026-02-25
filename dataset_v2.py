"""
Dataset module v2 for bifurcation WSS prediction.

New in v2 compared to dataset.py:
  - compute_physics_features_v2(): expands node features 3 → 10
      0-2: normalised x,y,z
      3:   dist_to_junction
      4:   arc_length along branch (0→1)
      5:   branch_depth (0=parent, 1=daughter)
      6-8: surface normal (nx, ny, nz)
      9:   local curvature (mean neighbour normal deviation)
  - sample_to_pyg_v2(): builds enhanced PyG graph
  - build_all_processed_v2(): caches to ProcessedData_v2/
  - get_split_paths_v2(): re_angle_stratified split
  - BifurcationWSSDatasetV2: loads from ProcessedData_v2/

Physics feature computation is purely geometric (numpy, no VTK dependency).
It uses the inlet/outlet boundary patches already present in the OpenFOAM mesh
to determine branch topology — this approach generalises to carotid arteries
as long as boundary patches with type ≠ "wall" are present.

Usage:
    python -m Bifurcation.dataset_v2 --process   # build v2 cache
    python -m Bifurcation.dataset_v2 --check     # quick sanity check
    python -m Bifurcation.dataset_v2 --stats     # compute & save norm stats
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from Bifurcation.config_v2 import config_v2
from Bifurcation.dataset import (
    parse_openfoam_vector_field,
    parse_openfoam_faces,
    parse_boundary,
    load_sample,
)


# ============================================================================
# Physics feature helpers (pure numpy, no VTK)
# ============================================================================

def _compute_face_normals(
    all_points: np.ndarray,
    wall_face_vertices: List[List[int]],
) -> np.ndarray:
    """
    Compute unit outward normal for each wall face via cross product.

    For faces with ≥3 vertices, uses the first three to define the plane.

    Returns
    -------
    normals : (N, 3)  unit normals
    """
    normals = np.zeros((len(wall_face_vertices), 3), dtype=np.float32)
    for i, verts in enumerate(wall_face_vertices):
        pts = all_points[verts]
        if len(pts) >= 3:
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            n  = np.cross(v1, v2).astype(np.float32)
            norm = np.linalg.norm(n)
            normals[i] = n / norm if norm > 1e-10 else np.array([0., 0., 1.], dtype=np.float32)
        else:
            normals[i] = np.array([0., 0., 1.], dtype=np.float32)
    return normals


def _compute_local_curvature(
    normals: np.ndarray,
    edge_index: np.ndarray,
) -> np.ndarray:
    """
    Approximate per-face curvature as mean angular deviation from neighbour normals.

    Returns
    -------
    curvature : (N,)  in radians, ∈ [0, π]
    """
    n_nodes = len(normals)
    curvature_sum   = np.zeros(n_nodes, dtype=np.float32)
    neighbour_count = np.zeros(n_nodes, dtype=np.float32)

    src, dst = edge_index[0], edge_index[1]
    cos_angles = np.clip(
        np.einsum("ij,ij->i", normals[src], normals[dst]),
        -1.0, 1.0,
    )
    angles = np.arccos(cos_angles).astype(np.float32)

    np.add.at(curvature_sum,   src, angles)
    np.add.at(neighbour_count, src, 1.0)

    mask = neighbour_count > 0
    curvature = np.zeros(n_nodes, dtype=np.float32)
    curvature[mask] = curvature_sum[mask] / neighbour_count[mask]
    return curvature


def _estimate_branch_topology(
    wall_centres: np.ndarray,
    all_points:   np.ndarray,
    all_faces:    List[List[int]],
    boundary:     Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate per-face branch topology from boundary patch geometry.

    Strategy (topology-aware, carotid-generalisable):
      1. Identify non-wall patches (inlet/outlet) from boundary dict.
      2. Compute each patch's face centroid.
      3. Sort by Z coordinate: lowest = inlet / parent branch.
         (Works for parametric bifurcations running along Z.
          For carotid data the same logic holds if vessel runs along Z.)
      4. Assign each wall face to its nearest patch centroid.
         → nearest==inlet → branch_depth 0 (parent)
         → otherwise     → branch_depth 1 (daughter)
      5. Estimate junction centre as geometric mean of patch centroids.
      6. Compute per-face arc length (0→1) projected along branch axis.

    Falls back to bounding-box heuristics when non-wall patches are missing.

    Returns
    -------
    dist_to_junction : (N,)
    branch_depth     : (N,)   float32, 0 or 1
    arc_length       : (N,)   float32, ∈ [0, 1]
    """
    N = len(wall_centres)
    dist_to_junction = np.zeros(N, dtype=np.float32)
    branch_depth     = np.zeros(N, dtype=np.float32)
    arc_length       = np.full(N, 0.5, dtype=np.float32)  # default: midpoint

    # Collect non-wall patch centroids
    non_wall_patches = {k: v for k, v in boundary.items() if v["type"] != "wall"}
    patch_centroids = []
    for _name, info in non_wall_patches.items():
        face_ctrs = []
        for i in range(info["nFaces"]):
            fi = info["startFace"] + i
            if fi < len(all_faces):
                pts = all_points[all_faces[fi]]
                face_ctrs.append(pts.mean(axis=0))
        if face_ctrs:
            patch_centroids.append(np.mean(face_ctrs, axis=0).astype(np.float32))

    if len(patch_centroids) < 2:
        # Fallback: use bounding-box-based approximation
        bbox_min = wall_centres.min(axis=0)
        bbox_max = wall_centres.max(axis=0)
        junction_center = (bbox_min + bbox_max) / 2.0
        dist_to_junction = np.linalg.norm(wall_centres - junction_center, axis=1).astype(np.float32)
        # Arc length along longest axis
        axis_idx = np.argmax(bbox_max - bbox_min)
        projections = wall_centres[:, axis_idx]
        proj_min, proj_max = projections.min(), projections.max()
        if proj_max > proj_min:
            arc_length = ((projections - proj_min) / (proj_max - proj_min)).astype(np.float32)
        return dist_to_junction, branch_depth, arc_length

    # Sort patch centroids by Z (ascending): inlet is lowest Z
    patch_centroids.sort(key=lambda c: c[2])
    inlet_center   = patch_centroids[0]
    outlet_centers = patch_centroids[1:]   # 1 or 2 outlets

    # Junction estimate: geometric mean of all patch centroids
    junction_center = np.mean(patch_centroids, axis=0).astype(np.float32)
    dist_to_junction = np.linalg.norm(
        wall_centres - junction_center, axis=1
    ).astype(np.float32)

    # Assign each wall face to nearest patch centroid
    all_anchors = [inlet_center] + outlet_centers
    dists_to_anchors = np.array(
        [np.linalg.norm(wall_centres - c, axis=1) for c in all_anchors]
    )                                                        # (n_patches, N)
    nearest_patch = np.argmin(dists_to_anchors, axis=0)     # (N,)

    branch_depth = (nearest_patch > 0).astype(np.float32)

    # Arc length per branch (projection onto branch axis)
    # Parent branch: inlet_center → junction_center
    parent_mask = nearest_patch == 0
    if parent_mask.any():
        axis     = junction_center - inlet_center
        axis_len = np.linalg.norm(axis)
        if axis_len > 1e-10:
            axis_unit   = axis / axis_len
            proj        = (wall_centres[parent_mask] - inlet_center) @ axis_unit
            pmin, pmax  = proj.min(), proj.max()
            if pmax > pmin:
                arc_length[parent_mask] = ((proj - pmin) / (pmax - pmin)).astype(np.float32)

    # Daughter branches: junction_center → outlet_center
    for out_idx, out_center in enumerate(outlet_centers):
        mask = nearest_patch == (out_idx + 1)
        if mask.any():
            axis     = out_center - junction_center
            axis_len = np.linalg.norm(axis)
            if axis_len > 1e-10:
                axis_unit  = axis / axis_len
                proj       = (wall_centres[mask] - junction_center) @ axis_unit
                pmin, pmax = proj.min(), proj.max()
                if pmax > pmin:
                    arc_length[mask] = ((proj - pmin) / (pmax - pmin)).astype(np.float32)

    return dist_to_junction, branch_depth, arc_length


def compute_physics_features_v2(
    wall_centres:      np.ndarray,       # (N, 3)  face centroids
    all_points:        np.ndarray,       # (P, 3)  all mesh vertices
    wall_face_vertices: List[List[int]], # vertex lists per wall face
    boundary:          Dict,             # boundary patch dict
    all_faces:         List[List[int]], # all faces (global index)
    edge_index:        np.ndarray,       # (2, E)
) -> np.ndarray:
    """
    Assemble 10 physics-informed node features per wall face.

    Feature layout:
      0-2: x, y, z             (face centroid, raw physical units)
      3:   dist_to_junction     (Euclidean; normalised later by dataset)
      4:   arc_length           (0→1 along branch axis)
      5:   branch_depth         (0=parent/inlet, 1=daughter)
      6-8: nx, ny, nz           (unit surface normal)
      9:   local_curvature      (mean neighbour normal deviation, radians)

    Returns
    -------
    features : (N, 10)  float32
    """
    N = len(wall_centres)
    features = np.zeros((N, 10), dtype=np.float32)

    # Coordinates (raw — z-scored by BifurcationWSSDatasetV2 at load time)
    features[:, 0:3] = wall_centres.astype(np.float32)

    # Branch topology
    dist_junc, b_depth, arc_len = _estimate_branch_topology(
        wall_centres, all_points, all_faces, boundary
    )
    features[:, 3] = dist_junc
    features[:, 4] = arc_len
    features[:, 5] = b_depth

    # Surface normals
    normals = _compute_face_normals(all_points, wall_face_vertices)
    features[:, 6:9] = normals

    # Local curvature
    features[:, 9] = _compute_local_curvature(normals, edge_index)

    return features


# ============================================================================
# V2 graph builder (extends dataset.py)
# ============================================================================

def sample_to_pyg_v2(geometry_folder: str, re_case: str) -> Data:
    """
    Build a single PyG Data object with 10-dim physics features (v2).

    Loads raw OpenFOAM mesh, computes physics features, returns Data.
    """
    geo_path = config_v2.get_geometry_path(geometry_folder)
    angle, _mesh_level = config_v2.parse_geometry_folder(geometry_folder)

    # Load basic data including surface mesh
    (wall_centres, wss, edge_index, edge_attr,
     re_num, boundary, surface_mesh_data) = load_sample(
        str(geo_path), re_case, mesh_re=config_v2.mesh_re,
        return_surface_mesh=True,
    )
    all_points, wall_face_vertices, _wall_face_indices = surface_mesh_data

    # Load all faces for non-wall patch centroid computation
    poly_path = geo_path / config_v2.mesh_re / "constant" / "polyMesh"
    all_faces = parse_openfoam_faces(str(poly_path / "faces"))

    # Compute 10 physics-informed node features
    x = compute_physics_features_v2(
        wall_centres    = wall_centres,
        all_points      = all_points,
        wall_face_vertices = wall_face_vertices,
        boundary        = boundary,
        all_faces       = all_faces,
        edge_index      = edge_index,
    )

    return Data(
        x          = torch.tensor(x,            dtype=torch.float32),  # (N, 10)
        pos        = torch.tensor(wall_centres,  dtype=torch.float32),  # (N, 3)  for viz
        y          = torch.tensor(wss,           dtype=torch.float32),  # (N, 3)
        edge_index = torch.tensor(edge_index,    dtype=torch.long),
        edge_attr  = torch.tensor(edge_attr,     dtype=torch.float32),
        re         = torch.tensor([re_num],      dtype=torch.float32),
        angle      = torch.tensor([angle],       dtype=torch.float32),
        geo_name   = geometry_folder,
        re_name    = re_case,
    )


# ============================================================================
# Build v2 processed dataset cache
# ============================================================================

def build_all_processed_v2(force: bool = False):
    """
    Build and cache all (geometry, Re) graphs with v2 physics features.

    Output: ProcessedData_v2/<geometry>/Re<N>.pt
    """
    config_v2.create_directories()
    built, skipped, failed = 0, 0, 0
    failed_cases = []

    pbar = tqdm(total=config_v2.total_samples, desc="Building v2 dataset")
    for geo in config_v2.geometry_folders:
        out_dir = config_v2.processed_data_dir / geo
        out_dir.mkdir(parents=True, exist_ok=True)

        for re_val in config_v2.re_values:
            re_case   = f"Re{re_val}"
            out_path  = out_dir / f"{re_case}.pt"

            if out_path.exists() and not force:
                skipped += 1
                pbar.update(1)
                continue

            try:
                data = sample_to_pyg_v2(geo, re_case)
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

def compute_normalization_stats_v2(
    graph_paths: Optional[List[Path]] = None,
) -> Dict:
    """
    Compute mean/std for the 10-dim node features, 4-dim edge features,
    and 3-dim WSS targets (sign-preserving log1p on targets).
    """
    if graph_paths is None:
        graph_paths = sorted(config_v2.processed_data_dir.rglob("*.pt"))

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

    if config_v2.use_log_transform:
        sign  = torch.sign(all_y)
        log_y = sign * torch.log1p(torch.abs(all_y))
        y_mean, y_std = log_y.mean(0), log_y.std(0)
    else:
        y_mean, y_std = all_y.mean(0), all_y.std(0)

    # Clamp stds away from zero
    x_std = torch.clamp(x_std, min=1e-8)
    e_std = torch.clamp(e_std, min=1e-8)
    y_std = torch.clamp(y_std, min=1e-8)

    return {
        "x_mean":           x_mean.tolist(),
        "x_std":            x_std.tolist(),
        "edge_mean":        e_mean.tolist(),
        "edge_std":         e_std.tolist(),
        "y_mean":           y_mean.tolist(),
        "y_std":            y_std.tolist(),
        "use_log_transform": config_v2.use_log_transform,
    }


def save_normalization_stats_v2(stats: Dict, path: Optional[Path] = None):
    path = path or (config_v2.processed_data_dir / "normalization_stats_v2.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved v2 normalization stats → {path}")


def load_normalization_stats_v2(path: Optional[Path] = None) -> Dict:
    path = path or (config_v2.processed_data_dir / "normalization_stats_v2.json")
    with open(path) as f:
        return json.load(f)


# ============================================================================
# De-normalisation (for inference)
# ============================================================================

def denormalize_wss_v2(y_norm: torch.Tensor, stats: Dict) -> torch.Tensor:
    """Reverse sign-preserving log1p + z-score normalisation."""
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

class BifurcationWSSDatasetV2(Dataset):
    """
    PyTorch Dataset loading v2 .pt graph files.

    Applies z-score normalisation on the fly:
      - node features (10-dim): per-dimension mean/std
      - edge features (4-dim):  per-dimension mean/std
      - WSS targets  (3-dim):   sign-preserving log1p then per-dim z-score
    """

    def __init__(
        self,
        graph_paths: List[Path],
        norm_stats:  Optional[Dict] = None,
    ):
        super().__init__()
        self.graph_paths = list(graph_paths)
        self.norm_stats  = norm_stats

        if norm_stats is not None:
            self._x_mean = torch.tensor(norm_stats["x_mean"],    dtype=torch.float32)
            self._x_std  = torch.tensor(norm_stats["x_std"],     dtype=torch.float32)
            self._e_mean = torch.tensor(norm_stats["edge_mean"],  dtype=torch.float32)
            self._e_std  = torch.tensor(norm_stats["edge_std"],   dtype=torch.float32)
            self._y_mean = torch.tensor(norm_stats["y_mean"],     dtype=torch.float32)
            self._y_std  = torch.tensor(norm_stats["y_std"],      dtype=torch.float32)

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx) -> Data:
        data = torch.load(self.graph_paths[idx], weights_only=False)

        if self.norm_stats is not None:
            data.x        = (data.x        - self._x_mean) / self._x_std
            data.edge_attr = (data.edge_attr - self._e_mean) / self._e_std

            if self.norm_stats.get("use_log_transform", False):
                sign   = torch.sign(data.y)
                data.y = sign * torch.log1p(torch.abs(data.y))
            data.y = (data.y - self._y_mean) / self._y_std

        return data


# ============================================================================
# Stratified splitting
# ============================================================================

def _geometry_from_path(p: Path) -> str:
    return p.parent.name


def _re_from_path(p: Path) -> int:
    return int(p.stem.replace("Re", ""))


def _re_bracket(re: int, brackets: List[Tuple[int, int]]) -> int:
    for i, (lo, hi) in enumerate(brackets):
        if lo <= re <= hi:
            return i
    return len(brackets) - 1


def get_split_paths_v2(
    mode: str = "re_angle_stratified",
    holdout_geo: Optional[str] = None,
    seed: int = 42,
    train_ratio: float = 0.75,
    val_ratio:   float = 0.10,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Return (train, val, test) path lists.

    Modes
    -----
    re_angle_stratified (default):
        Ensures every (angle, Re-bracket) combination has at least one test
        sample.  Avoids the v1 problem of entire low-Re regimes in test.
    loocv-geo:
        Hold out all samples from one geometry (same as v1).
    random:
        Stratified random 80/10/10 by geometry (same as v1).
    """
    all_paths = sorted(config_v2.processed_data_dir.rglob("*.pt"))
    if not all_paths:
        raise FileNotFoundError(
            "No v2 .pt files found — run: python -m Bifurcation.dataset_v2 --process"
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
        # Group by (angle, Re-bracket) cell
        # For each cell, hold out ~20% as test (at least 1 per cell)
        brackets = config_v2.re_brackets

        cell_groups: Dict[Tuple, List[Path]] = defaultdict(list)
        for p in all_paths:
            geo   = _geometry_from_path(p)
            re    = _re_from_path(p)
            angle, _ = config_v2.parse_geometry_folder(geo)
            bracket  = _re_bracket(re, brackets)
            cell_groups[(angle, bracket)].append(p)

        test_paths, val_paths, train_paths = [], [], []

        for _cell, paths in cell_groups.items():
            paths = sorted(paths)
            rng.shuffle(paths)
            n        = len(paths)
            n_test   = max(1, int(n * (1.0 - train_ratio - val_ratio)))
            n_val    = max(1, int(n * val_ratio))
            n_test   = min(n_test, n - n_val - 1)  # ensure at least 1 train

            test_paths.extend(paths[:n_test])
            val_paths.extend(paths[n_test: n_test + n_val])
            train_paths.extend(paths[n_test + n_val:])

        return train_paths, val_paths, test_paths

    else:
        raise ValueError(f"Unknown split mode: {mode}")


def get_dataloaders_v2(
    mode: str = "re_angle_stratified",
    holdout_geo: Optional[str] = None,
    seed: int = 42,
    norm_stats: Optional[Dict] = None,
    batch_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_p, val_p, test_p = get_split_paths_v2(mode, holdout_geo, seed)

    if norm_stats is None:
        norm_stats = compute_normalization_stats_v2(train_p)
        save_normalization_stats_v2(norm_stats)

    bs = batch_size or config_v2.batch_size

    train_ds = BifurcationWSSDatasetV2(train_p, norm_stats)
    val_ds   = BifurcationWSSDatasetV2(val_p,   norm_stats)
    test_ds  = BifurcationWSSDatasetV2(test_p,  norm_stats)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False)

    return train_loader, val_loader, test_loader


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bifurcation v2 dataset utilities")
    parser.add_argument("--process", action="store_true",
                        help="Build all v2 .pt graphs from raw OpenFOAM data")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if cached")
    parser.add_argument("--check", action="store_true",
                        help="Parse one sample and print v2 feature summary")
    parser.add_argument("--stats", action="store_true",
                        help="Compute & save v2 normalisation stats")
    args = parser.parse_args()

    if args.process:
        build_all_processed_v2(force=args.force)
        stats = compute_normalization_stats_v2()
        save_normalization_stats_v2(stats)

    elif args.check:
        geo     = "bifurcation_angle45_750_ascii"
        re_case = "Re500"
        print(f"Parsing (v2) {geo} / {re_case} ...")
        data = sample_to_pyg_v2(geo, re_case)
        print(f"  Nodes:           {data.x.shape[0]}")
        print(f"  Node features:   {data.x.shape}  (expected [N, 10])")
        print(f"  Edges:           {data.edge_index.shape[1]}")
        print(f"  WSS targets:     {data.y.shape}")
        print(f"  Re={data.re.item():.0f}  angle={data.angle.item():.0f}°")

        # Feature stats
        labels = ["x", "y", "z", "dist_junc", "arc_len", "branch_depth",
                  "nx", "ny", "nz", "curvature"]
        for i, lbl in enumerate(labels):
            col = data.x[:, i]
            print(f"  feat[{i}] {lbl:15s}: min={col.min():.4f}  max={col.max():.4f}  "
                  f"mean={col.mean():.4f}")
        print("OK")

    elif args.stats:
        stats = compute_normalization_stats_v2()
        save_normalization_stats_v2(stats)
        print(json.dumps(stats, indent=2))

    else:
        parser.print_help()

"""
Dataset module for bifurcation WSS prediction.

Parses raw OpenFOAM ASCII files (points, faces, boundary, wallShearStress),
builds PyTorch Geometric graph objects, and provides dataset / dataloader
utilities with normalization.

Usage:
    # Build all processed graphs from raw OpenFOAM data
    python -m Bifurcation.dataset --process

    # Quick sanity check (parse one sample)
    python -m Bifurcation.dataset --check
"""

import os
import re as _re
import math
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from Bifurcation.config import config


# ============================================================================
# OpenFOAM ASCII parsers
# ============================================================================

def parse_openfoam_vector_field(filepath: str) -> np.ndarray:
    """
    Parse an OpenFOAM vector field file into an (N, 3) numpy array.

    Works for ``points``, ``wallShearStress``, and any ``volVectorField``
    or ``vectorField`` stored in the standard OpenFOAM ASCII format::

        <N>
        (
        (vx vy vz)
        ...
        )
    """
    with open(filepath, "r") as f:
        content = f.read()

    match = _re.search(r"(\d+)\s*\n\s*\(\s*\n(.*?)\n\s*\)", content, _re.DOTALL)
    if match is None:
        raise ValueError(f"Could not parse vector field from {filepath}")

    n = int(match.group(1))
    block = match.group(2)
    vectors = _re.findall(r"\(\s*([^\)]+)\)", block)
    data = np.array([[float(x) for x in v.split()] for v in vectors])
    assert len(data) == n, f"Expected {n} vectors, parsed {len(data)} in {filepath}"
    return data


def parse_openfoam_faces(filepath: str) -> List[List[int]]:
    """
    Parse an OpenFOAM ``faces`` file into a list of point-index lists.

    Each line looks like ``3(111790 12494 24086)`` or ``4(0 1 5 4)``.
    """
    with open(filepath, "r") as f:
        content = f.read()

    match = _re.search(r"(\d+)\s*\n\s*\(\s*\n(.*?)\n\s*\)", content, _re.DOTALL)
    if match is None:
        raise ValueError(f"Could not parse faces from {filepath}")

    block = match.group(2)
    faces = []
    for line in block.strip().split("\n"):
        idx_match = _re.search(r"\d+\(([^)]+)\)", line.strip())
        if idx_match:
            indices = [int(x) for x in idx_match.group(1).split()]
            faces.append(indices)
    return faces


def parse_boundary(filepath: str) -> Dict[str, Dict]:
    """
    Parse an OpenFOAM ``boundary`` file.

    Returns a dict  ``patch_name → {type, nFaces, startFace}``,
    preserving insertion order (Python 3.7+).
    """
    with open(filepath, "r") as f:
        content = f.read()

    patches: Dict[str, Dict] = {}
    pattern = r"(\w[\w-]*)\s*\{[^}]*type\s+(\w+);\s*(?:inGroups[^;]*;\s*)?nFaces\s+(\d+);\s*startFace\s+(\d+);"
    for m in _re.finditer(pattern, content):
        name = m.group(1)
        patches[name] = {
            "type": m.group(2),
            "nFaces": int(m.group(3)),
            "startFace": int(m.group(4)),
        }
    return patches


# ============================================================================
# Graph construction
# ============================================================================

def load_sample(
    geometry_path: str,
    re_case: str,
    mesh_re: str = "Re100",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Dict]:
    """
    Load a single training sample as graph components.

    Parameters
    ----------
    geometry_path : str
        Path to a geometry folder, e.g. ``.../bifurcation_angle45_750_ascii/``
    re_case : str
        Reynolds-number subfolder, e.g. ``Re500``
    mesh_re : str
        Which Re subfolder stores the mesh (default ``Re100``)

    Returns
    -------
    node_coords : (N_wall, 3)   face-centre coordinates on wall faces
    wss_vectors : (N_wall, 3)   target WSS vectors
    edge_index  : (2, E)        undirected adjacency (wall-face connectivity)
    edge_attr   : (E, 4)        [distance, dx, dy, dz]
    re_number   : float
    boundary    : dict           parsed boundary info (needed for export)
    """
    geometry_path = Path(geometry_path)

    # ---- mesh (shared across all Re) ----
    poly = geometry_path / mesh_re / "constant" / "polyMesh"
    points = parse_openfoam_vector_field(str(poly / "points"))
    faces = parse_openfoam_faces(str(poly / "faces"))
    boundary = parse_boundary(str(poly / "boundary"))

    # ---- identify wall patches ----
    wall_patches = {k: v for k, v in boundary.items() if v["type"] == "wall"}
    if not wall_patches:
        raise ValueError(f"No wall patches found in {poly / 'boundary'}")

    # ---- compute wall face centres & track face indices ----
    wall_centres: List[np.ndarray] = []
    wall_face_indices: List[int] = []
    for _name, info in wall_patches.items():
        for i in range(info["startFace"], info["startFace"] + info["nFaces"]):
            face_pts = points[faces[i]]
            wall_centres.append(face_pts.mean(axis=0))
            wall_face_indices.append(i)
    wall_centres_arr = np.array(wall_centres, dtype=np.float32)

    # ---- build adjacency (two wall faces connected if they share ≥1 point) ----
    point_to_faces: Dict[int, set] = defaultdict(set)
    for local_idx, global_face_idx in enumerate(wall_face_indices):
        for pt in faces[global_face_idx]:
            point_to_faces[pt].add(local_idx)

    src_list, dst_list = [], []
    for _pt, face_set in point_to_faces.items():
        face_list = list(face_set)
        for i in range(len(face_list)):
            for j in range(i + 1, len(face_list)):
                src_list.extend([face_list[i], face_list[j]])
                dst_list.extend([face_list[j], face_list[i]])

    edge_index = np.array([src_list, dst_list], dtype=np.int64)

    # Remove duplicate edges
    if edge_index.shape[1] > 0:
        edge_pairs = edge_index.T
        _, unique_idx = np.unique(edge_pairs, axis=0, return_index=True)
        edge_index = edge_index[:, np.sort(unique_idx)]

    # ---- edge features: [distance, dx, dy, dz] ----
    if edge_index.shape[1] > 0:
        src_coords = wall_centres_arr[edge_index[0]]
        dst_coords = wall_centres_arr[edge_index[1]]
        diff = dst_coords - src_coords
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        edge_attr = np.concatenate([dist, diff], axis=1).astype(np.float32)
    else:
        edge_attr = np.zeros((0, 4), dtype=np.float32)

    # ---- load WSS from final timestep ----
    case_path = geometry_path / re_case
    timesteps = [
        d for d in os.listdir(case_path)
        if os.path.isdir(case_path / d) and d.isdigit() and int(d) > 0
    ]
    if not timesteps:
        raise FileNotFoundError(f"No timestep folders in {case_path}")
    final_time = max(timesteps, key=int)
    wss_path = case_path / final_time / "wallShearStress"
    wss_all = parse_openfoam_vector_field(str(wss_path))

    # ---- extract wall WSS (boundary file lists wall patch first with nFaces) ----
    # wallShearStress boundaryField stores per-patch data in the same order
    # as the boundary file.  We only need the wall patch values.
    # The first nFaces entries in the vector list correspond to the wall patch.
    wall_wss_parts: List[np.ndarray] = []
    for _name, info in wall_patches.items():
        wall_wss_parts.append(wss_all[: info["nFaces"]])
        wss_all = wss_all[info["nFaces"]:]          # consume
    wall_wss = np.vstack(wall_wss_parts).astype(np.float32)

    # ---- Reynolds number ----
    re_number = float(re_case.replace("Re", ""))

    return wall_centres_arr, wall_wss, edge_index, edge_attr, re_number, boundary


def sample_to_pyg(
    geometry_folder: str,
    re_case: str,
) -> Data:
    """
    Build a single PyG ``Data`` object from raw OpenFOAM files.
    """
    geo_path = config.get_geometry_path(geometry_folder)
    angle, mesh_level = config.parse_geometry_folder(geometry_folder)

    coords, wss, edge_index, edge_attr, re_num, _boundary = load_sample(
        str(geo_path), re_case, mesh_re=config.mesh_re
    )

    data = Data(
        x=torch.tensor(coords, dtype=torch.float32),               # (N, 3)
        pos=torch.tensor(coords, dtype=torch.float32),              # kept for viz
        y=torch.tensor(wss, dtype=torch.float32),                   # (N, 3)
        edge_index=torch.tensor(edge_index, dtype=torch.long),      # (2, E)
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),     # (E, 4)
        re=torch.tensor([re_num], dtype=torch.float32),
        angle=torch.tensor([angle], dtype=torch.float32),
        geo_name=geometry_folder,
        re_name=re_case,
    )
    return data


# ============================================================================
# Full-dataset builder (raw → .pt cache)
# ============================================================================

def build_all_processed(force: bool = False):
    """
    Parse every (geometry, Re) pair and save as ``.pt`` files under
    ``ProcessedData/<geometry>/Re<N>.pt``.
    """
    config.create_directories()
    total = config.total_samples
    built, skipped, failed = 0, 0, 0
    failed_cases: List[str] = []

    pbar = tqdm(total=total, desc="Building dataset")
    for geo in config.geometry_folders:
        out_dir = config.processed_data_dir / geo
        out_dir.mkdir(parents=True, exist_ok=True)

        for re_val in config.re_values:
            re_case = f"Re{re_val}"
            out_path = out_dir / f"{re_case}.pt"

            if out_path.exists() and not force:
                skipped += 1
                pbar.update(1)
                continue

            try:
                data = sample_to_pyg(geo, re_case)
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

def compute_normalization_stats(
    graph_paths: Optional[List[Path]] = None,
) -> Dict:
    """
    Compute mean / std for node features, edge features, and WSS targets
    (sign-preserving log1p when ``config.use_log_transform`` is set).

    Returns a dict that can be JSON-serialised and later re-loaded.
    """
    if graph_paths is None:
        graph_paths = sorted(config.processed_data_dir.rglob("*.pt"))

    all_x, all_edge, all_y = [], [], []
    for p in graph_paths:
        d = torch.load(p, weights_only=False)
        all_x.append(d.x)
        all_edge.append(d.edge_attr)
        all_y.append(d.y)

    all_x = torch.cat(all_x, 0)
    all_edge = torch.cat(all_edge, 0)
    all_y = torch.cat(all_y, 0)

    x_mean, x_std = all_x.mean(0), all_x.std(0)
    e_mean, e_std = all_edge.mean(0), all_edge.std(0)

    if config.use_log_transform:
        sign = torch.sign(all_y)
        log_y = sign * torch.log1p(torch.abs(all_y))
        y_mean, y_std = log_y.mean(0), log_y.std(0)
    else:
        y_mean, y_std = all_y.mean(0), all_y.std(0)

    # Clamp stds away from zero
    x_std = torch.clamp(x_std, min=1e-8)
    e_std = torch.clamp(e_std, min=1e-8)
    y_std = torch.clamp(y_std, min=1e-8)

    stats = {
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "edge_mean": e_mean.tolist(),
        "edge_std": e_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "use_log_transform": config.use_log_transform,
    }
    return stats


def save_normalization_stats(stats: Dict, path: Optional[Path] = None):
    path = path or (config.processed_data_dir / "normalization_stats.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved normalization stats → {path}")


def load_normalization_stats(path: Optional[Path] = None) -> Dict:
    path = path or (config.processed_data_dir / "normalization_stats.json")
    with open(path) as f:
        return json.load(f)


# ============================================================================
# PyG Dataset class (loads cached .pt files)
# ============================================================================

class BifurcationWSSDataset(Dataset):
    """
    PyTorch Dataset over cached ``.pt`` graph files.

    Applies z-score normalisation on-the-fly using pre-computed stats.
    """

    def __init__(
        self,
        graph_paths: List[Path],
        norm_stats: Optional[Dict] = None,
    ):
        super().__init__()
        self.graph_paths = list(graph_paths)
        self.norm_stats = norm_stats

        # Pre-compute tensors for fast normalisation
        if norm_stats is not None:
            self._x_mean = torch.tensor(norm_stats["x_mean"], dtype=torch.float32)
            self._x_std = torch.tensor(norm_stats["x_std"], dtype=torch.float32)
            self._e_mean = torch.tensor(norm_stats["edge_mean"], dtype=torch.float32)
            self._e_std = torch.tensor(norm_stats["edge_std"], dtype=torch.float32)
            self._y_mean = torch.tensor(norm_stats["y_mean"], dtype=torch.float32)
            self._y_std = torch.tensor(norm_stats["y_std"], dtype=torch.float32)

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx) -> Data:
        data = torch.load(self.graph_paths[idx], weights_only=False)

        if self.norm_stats is not None:
            # Normalise node features
            data.x = (data.x - self._x_mean) / self._x_std

            # Normalise edge features
            data.edge_attr = (data.edge_attr - self._e_mean) / self._e_std

            # Normalise targets (sign-preserving log1p + z-score)
            if self.norm_stats.get("use_log_transform", False):
                sign = torch.sign(data.y)
                data.y = sign * torch.log1p(torch.abs(data.y))
            data.y = (data.y - self._y_mean) / self._y_std

        return data


# ============================================================================
# Splitting & DataLoader helpers
# ============================================================================

def _geometry_from_path(p: Path) -> str:
    """``ProcessedData/<geo>/ReXXX.pt`` → ``<geo>``"""
    return p.parent.name


def get_split_paths(
    mode: str = "random",
    holdout_geo: Optional[str] = None,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Return (train, val, test) path lists.

    Parameters
    ----------
    mode : str
        ``"loocv-geo"`` – leave-one-geometry-out (holdout_geo required).
        ``"random"``    – random 80/10/10 stratified by geometry.
    holdout_geo : str
        Geometry folder name to hold out (for loocv-geo mode).
    """
    all_paths = sorted(config.processed_data_dir.rglob("*.pt"))
    if not all_paths:
        raise FileNotFoundError("No .pt files found – run dataset --process first")

    if mode == "loocv-geo":
        assert holdout_geo is not None
        test_paths = [p for p in all_paths if _geometry_from_path(p) == holdout_geo]
        remaining = [p for p in all_paths if _geometry_from_path(p) != holdout_geo]
        # Use one Re-sweep from a random remaining geometry as validation
        rng = np.random.RandomState(seed)
        remaining_geos = sorted(set(_geometry_from_path(p) for p in remaining))
        val_geo = rng.choice(remaining_geos)
        val_paths = [p for p in remaining if _geometry_from_path(p) == val_geo]
        train_paths = [p for p in remaining if _geometry_from_path(p) != val_geo]
        return train_paths, val_paths, test_paths

    elif mode == "random":
        rng = np.random.RandomState(seed)
        # Group by geometry so every Re for a geo stays together
        geo_groups: Dict[str, List[Path]] = defaultdict(list)
        for p in all_paths:
            geo_groups[_geometry_from_path(p)].append(p)

        train_paths, val_paths, test_paths = [], [], []
        for geo, paths in geo_groups.items():
            paths = sorted(paths)
            rng.shuffle(paths)
            n = len(paths)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train_paths.extend(paths[:n_train])
            val_paths.extend(paths[n_train: n_train + n_val])
            test_paths.extend(paths[n_train + n_val:])
        return train_paths, val_paths, test_paths

    else:
        raise ValueError(f"Unknown split mode: {mode}")


def get_dataloaders(
    mode: str = "random",
    holdout_geo: Optional[str] = None,
    seed: int = 42,
    norm_stats: Optional[Dict] = None,
    batch_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function returning (train_loader, val_loader, test_loader).
    """
    train_p, val_p, test_p = get_split_paths(mode, holdout_geo, seed)

    # Compute normalisation stats from training set only
    if norm_stats is None:
        norm_stats = compute_normalization_stats(train_p)
        save_normalization_stats(norm_stats)

    bs = batch_size or config.batch_size

    train_ds = BifurcationWSSDataset(train_p, norm_stats)
    val_ds = BifurcationWSSDataset(val_p, norm_stats)
    test_ds = BifurcationWSSDataset(test_p, norm_stats)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    return train_loader, val_loader, test_loader


# ============================================================================
# Denormalization (for inference)
# ============================================================================

def denormalize_wss(y_norm: torch.Tensor, stats: Dict) -> torch.Tensor:
    """
    Reverse the normalization applied to WSS targets.

    sign-preserving log1p + z-score  ⇒  physical WSS.
    """
    y_mean = torch.tensor(stats["y_mean"], dtype=y_norm.dtype, device=y_norm.device)
    y_std = torch.tensor(stats["y_std"], dtype=y_norm.dtype, device=y_norm.device)

    y = y_norm * y_std + y_mean

    if stats.get("use_log_transform", False):
        sign = torch.sign(y)
        y = sign * torch.expm1(torch.abs(y))

    return y


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bifurcation WSS dataset utilities")
    parser.add_argument("--process", action="store_true", help="Build all .pt graphs from raw data")
    parser.add_argument("--force", action="store_true", help="Re-process even if cached")
    parser.add_argument("--check", action="store_true", help="Parse one sample and print summary")
    parser.add_argument("--stats", action="store_true", help="Compute & save normalisation stats")
    args = parser.parse_args()

    if args.process:
        build_all_processed(force=args.force)
        # Also compute stats
        stats = compute_normalization_stats()
        save_normalization_stats(stats)

    elif args.check:
        geo = config.geometry_folders[4]  # angle45_750
        re_case = "Re500"
        print(f"Parsing {geo} / {re_case} ...")
        data = sample_to_pyg(geo, re_case)
        print(f"  Nodes (wall faces): {data.x.shape[0]}")
        print(f"  Node features:      {data.x.shape}")
        print(f"  Edges:              {data.edge_index.shape[1]}")
        print(f"  Edge features:      {data.edge_attr.shape}")
        print(f"  WSS targets:        {data.y.shape}")
        print(f"  WSS magnitude range: [{data.y.norm(dim=1).min():.6e}, {data.y.norm(dim=1).max():.6e}]")
        print(f"  Re={data.re.item():.0f}  angle={data.angle.item():.0f}°")
        print(f"  geo_name={data.geo_name}  re_name={data.re_name}")
        print("OK")

    elif args.stats:
        stats = compute_normalization_stats()
        save_normalization_stats(stats)
        print(json.dumps(stats, indent=2))

    else:
        parser.print_help()

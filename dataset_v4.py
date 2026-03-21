"""
Dataset module v4 for bifurcation WSS prediction (pulsatile flow).

New in v4 compared to dataset_v3.py:
  - parse_openfoam_vector_field_binary(): reads binary-format wallShearStress
      (OpenFOAM writes native-endian doubles; little-endian on x86-64 Linux/EC2)
  - parse_openfoam_vector_field_auto(): dispatches ASCII vs binary automatically
  - sample_to_pyg_v4(): loads geometry from steady-state path (shared polyMesh),
      WSS from a specific pulsatile timestep directory; adds data.phase
  - build_all_processed_v4(): caches 20 timesteps per (geo, Re) case
      ProcessedData_v4/<geo>/Re<N>/t<time>.pt
  - Normalization: x/edge stats computed at stride 20 (one snapshot per case);
      y stats computed over all 3780 samples (full temporal amplitude range)
  - get_split_paths_v4(): stratifies at (geo, Re) case level to prevent leakage,
      then expands to all 20 timestep files per case.

Usage:
    python -m Bifurcation.dataset_v4 --process   # build all cached .pt files
    python -m Bifurcation.dataset_v4 --check     # sanity check one sample
    python -m Bifurcation.dataset_v4 --stats     # compute & save norm stats
"""

import json
import math
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

from Bifurcation.config_v4 import config_v4
from Bifurcation.dataset import (
    parse_openfoam_vector_field,
    parse_openfoam_faces,
    parse_boundary,
    load_sample,
)
from Bifurcation.dataset_v2 import compute_physics_features_v2
from Bifurcation.dataset_v3 import rotate_z

def rotate_3d(data: Data) -> Data:
    """Uniform SO(3) rotation — makes model rotation-invariant."""
    M = torch.randn(3, 3, dtype=data.x.dtype)
    Q, R_mat = torch.linalg.qr(M)
    Q = Q * torch.sign(torch.diag(R_mat)).prod()
    data = data.clone()
    data.x[:, 0:3]         = data.x[:, 0:3]         @ Q.t()
    data.x[:, 6:9]         = data.x[:, 6:9]         @ Q.t()
    data.edge_attr[:, 1:4] = data.edge_attr[:, 1:4] @ Q.t()
    data.y                 = data.y                  @ Q.t()
    return data


# ============================================================================
# OpenFOAM binary reader
# ============================================================================

def parse_openfoam_vector_field_binary(filepath: str) -> np.ndarray:
    """
    Parse a binary-format OpenFOAM vector field (e.g. wallShearStress).

    OpenFOAM writes in the host architecture's native byte order.
    EC2 instances (x86-64 Linux) are little-endian, so doubles are '<f8'.

    Format after the FoamFile header:
        nonuniform List<vector>
        N
        (
        <N * 3 * 8 bytes of doubles>
        )

    Falls back to big-endian if little-endian values are physically unreasonable.
    """
    with open(filepath, 'rb') as f:
        raw = f.read()

    marker = b'nonuniform List<vector>'
    idx = raw.find(marker)
    if idx == -1:
        raise ValueError(f"No 'nonuniform List<vector>' in {filepath}")

    pos = idx + len(marker)

    # Skip whitespace
    while pos < len(raw) and raw[pos:pos+1] in (b' ', b'\t', b'\n', b'\r'):
        pos += 1

    # Read count N
    n_start = pos
    while pos < len(raw) and raw[pos:pos+1].isdigit():
        pos += 1
    if pos == n_start:
        raise ValueError(f"Could not parse count N in binary field: {filepath}")
    N = int(raw[n_start:pos])

    # Find the opening '(' immediately after N
    open_paren = raw.find(b'(', pos)
    if open_paren == -1:
        raise ValueError(f"Could not find '(' after count in {filepath}")

    # Binary data starts right after '(' (and optional newline)
    data_start = open_paren + 1
    if raw[data_start:data_start+1] == b'\n':
        data_start += 1

    n_bytes = N * 3 * 8
    data_bytes = raw[data_start:data_start + n_bytes]
    if len(data_bytes) != n_bytes:
        raise ValueError(
            f"Binary field truncated in {filepath}: "
            f"expected {n_bytes} bytes, got {len(data_bytes)}"
        )

    # Little-endian (native on x86-64 Linux / EC2)
    arr = np.frombuffer(data_bytes, dtype='<f8').reshape(N, 3).astype(np.float32)

    # Sanity check: WSS magnitudes should be well below 100 Pa for blood flow
    if np.abs(arr).max() > 1e4:
        # Try big-endian (unlikely but safe fallback)
        arr_be = np.frombuffer(data_bytes, dtype='>f8').reshape(N, 3).astype(np.float32)
        if np.abs(arr_be).max() < np.abs(arr).max():
            arr = arr_be

    return arr


def parse_openfoam_vector_field_auto(filepath: str) -> np.ndarray:
    """
    Auto-detect binary vs ASCII OpenFOAM vector field and dispatch accordingly.
    """
    with open(filepath, 'rb') as f:
        header = f.read(1024).decode('latin-1', errors='replace')
    if 'format      binary' in header or 'format binary' in header:
        return parse_openfoam_vector_field_binary(filepath)
    return parse_openfoam_vector_field(filepath)


# ============================================================================
# Pulsatile timestep directory lookup
# ============================================================================

def _is_float_str(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _find_timestep_dir(case_path: Path, t: float) -> Path:
    """
    Return the actual directory path for timestep t in a pulsatile case.

    OpenFOAM 'general' format strips trailing zeros: t=4.0 → dir '4', t=5.0 → '5'.
    Matches by closest float value to handle floating-point edge cases.
    Only considers directories with t > 3.0 (last 2 pulsatile cycles).
    """
    candidates = {
        float(d): d
        for d in os.listdir(case_path)
        if _is_float_str(d) and float(d) > 3.0 and (case_path / d).is_dir()
    }
    if not candidates:
        raise FileNotFoundError(
            f"No pulsatile time directories (t>3.0) found in {case_path}"
        )
    best = min(candidates, key=lambda k: abs(k - t))
    return case_path / candidates[best]


def _available_timesteps(case_path: Path) -> List[float]:
    """Return sorted list of available pulsatile timesteps (t > 3.0) in a case."""
    return sorted(
        float(d)
        for d in os.listdir(case_path)
        if _is_float_str(d) and float(d) > 3.0 and (case_path / d).is_dir()
    )


# ============================================================================
# V4 graph builder
# ============================================================================

def sample_to_pyg_v4(geometry_folder: str, re_case: str, timestep: float) -> Data:
    """
    Build a PyG Data object for one (geometry, Re, timestep) sample.

    Mesh (polyMesh) and graph structure come from the steady-state path —
    the pulsatile cases share the same mesh as steady-state cases.
    WSS target comes from the pulsatile timestep directory.

    Adds data.phase = tensor([φ]) where φ = timestep % 1.0 ∈ [0, 1).

    Parameters
    ----------
    geometry_folder : e.g. 'bifurcation_angle45_750_ascii'
    re_case         : e.g. 'Re500'
    timestep        : float, e.g. 3.1, 3.2, ..., 5.0
    """
    steady_geo_path  = config_v4.get_steady_geometry_path(geometry_folder)
    pulsatile_case   = config_v4.get_geometry_path(geometry_folder) / re_case
    angle, _mesh_lvl = config_v4.parse_geometry_folder(geometry_folder)

    # --- mesh + graph structure from steady-state path ---
    (wall_centres, _wss_steady, edge_index, edge_attr,
     re_num, boundary, surface_mesh_data) = load_sample(
        str(steady_geo_path), re_case,
        mesh_re=config_v4.mesh_re,
        return_surface_mesh=True,
    )
    all_points, wall_face_vertices, _wall_face_indices = surface_mesh_data

    poly_path = steady_geo_path / config_v4.mesh_re / "constant" / "polyMesh"
    all_faces = parse_openfoam_faces(str(poly_path / "faces"))

    # --- 10 physics node features (same as v3) ---
    x_physics = compute_physics_features_v2(
        wall_centres       = wall_centres,
        all_points         = all_points,
        wall_face_vertices = wall_face_vertices,
        boundary           = boundary,
        all_faces          = all_faces,
        edge_index         = edge_index,
    )  # (N, 10)

    # --- WSS target from pulsatile timestep ---
    t_dir = _find_timestep_dir(pulsatile_case, timestep)
    wss   = parse_openfoam_vector_field_auto(str(t_dir / "wallShearStress"))

    phi = config_v4.phase_from_time(timestep)  # t % 1.0

    return Data(
        x          = torch.tensor(x_physics,    dtype=torch.float32),
        pos        = torch.tensor(wall_centres,  dtype=torch.float32),
        y          = torch.tensor(wss,           dtype=torch.float32),
        edge_index = torch.tensor(edge_index,    dtype=torch.long),
        edge_attr  = torch.tensor(edge_attr,     dtype=torch.float32),
        re         = torch.tensor([re_num],      dtype=torch.float32),
        angle      = torch.tensor([float(angle)], dtype=torch.float32),
        phase      = torch.tensor([phi],         dtype=torch.float32),
        geo_name   = geometry_folder,
        re_name    = re_case,
    )


# ============================================================================
# Build v4 processed dataset cache
# ============================================================================

def build_all_processed_v4(force: bool = False):
    """
    Build and cache all (geometry, Re, timestep) graphs with v4 features.

    Output: ProcessedData_v4/<geometry>/Re<N>/t<time>.pt

    Only processes cases that have 20 pulsatile time directories available —
    silently skips cases still running or not yet started. This allows
    incremental builds as simulations complete.

    Re-run with --process when more simulations finish; existing .pt files
    are skipped unless --force is passed.
    """
    config_v4.create_directories()
    built, skipped, failed = 0, 0, 0
    failed_cases: List[str] = []

    # Estimate total work: count (geo, re, timestep) triples that exist
    total_expected = config_v4.num_geometries * config_v4.num_re * config_v4.num_timesteps
    pbar = tqdm(desc="Building v4 dataset (pulsatile)", unit="sample")

    for geo in config_v4.geometry_folders:
        out_geo_dir = config_v4.processed_data_dir / geo
        out_geo_dir.mkdir(parents=True, exist_ok=True)

        for re_val in config_v4.re_values:
            re_case    = f"Re{re_val}"
            pulsatile_case = config_v4.get_geometry_path(geo) / re_case

            if not pulsatile_case.exists():
                # Simulation not generated yet
                pbar.update(config_v4.num_timesteps)
                continue

            available_t = _available_timesteps(pulsatile_case)
            if len(available_t) < config_v4.num_timesteps:
                # Simulation still running — skip this case for now
                pbar.update(config_v4.num_timesteps)
                continue

            out_re_dir = out_geo_dir / re_case
            out_re_dir.mkdir(parents=True, exist_ok=True)

            # Build geometry graph once (reuse across all 20 timesteps)
            try:
                steady_geo_path = config_v4.get_steady_geometry_path(geo)
                angle, _        = config_v4.parse_geometry_folder(geo)

                (wall_centres, _wss_steady, edge_index, edge_attr,
                 re_num, boundary, surface_mesh_data) = load_sample(
                    str(steady_geo_path), re_case,
                    mesh_re=config_v4.mesh_re,
                    return_surface_mesh=True,
                )
                all_points, wall_face_vertices, _ = surface_mesh_data

                poly_path = steady_geo_path / config_v4.mesh_re / "constant" / "polyMesh"
                all_faces = parse_openfoam_faces(str(poly_path / "faces"))

                x_physics = compute_physics_features_v2(
                    wall_centres       = wall_centres,
                    all_points         = all_points,
                    wall_face_vertices = wall_face_vertices,
                    boundary           = boundary,
                    all_faces          = all_faces,
                    edge_index         = edge_index,
                )

                x_t     = torch.tensor(x_physics,    dtype=torch.float32)
                pos_t   = torch.tensor(wall_centres,  dtype=torch.float32)
                ei_t    = torch.tensor(edge_index,    dtype=torch.long)
                ea_t    = torch.tensor(edge_attr,     dtype=torch.float32)
                re_t    = torch.tensor([re_num],      dtype=torch.float32)
                ang_t   = torch.tensor([float(angle)], dtype=torch.float32)

            except Exception as e:
                failed += config_v4.num_timesteps
                failed_cases.append(f"{geo}/{re_case} (geometry): {e}")
                pbar.update(config_v4.num_timesteps)
                continue

            # Now process each timestep
            for t in config_v4.timestep_values:
                t_str    = f"t{t}"
                out_path = out_re_dir / f"{t_str}.pt"

                if out_path.exists() and not force:
                    skipped += 1
                    pbar.update(1)
                    continue

                try:
                    t_dir = _find_timestep_dir(pulsatile_case, t)
                    wss   = parse_openfoam_vector_field_auto(str(t_dir / "wallShearStress"))
                    phi   = config_v4.phase_from_time(t)

                    data = Data(
                        x          = x_t,
                        pos        = pos_t,
                        y          = torch.tensor(wss, dtype=torch.float32),
                        edge_index = ei_t,
                        edge_attr  = ea_t,
                        re         = re_t,
                        angle      = ang_t,
                        phase      = torch.tensor([phi], dtype=torch.float32),
                        geo_name   = geo,
                        re_name    = re_case,
                    )
                    torch.save(data, out_path)
                    built += 1
                except Exception as e:
                    failed += 1
                    failed_cases.append(f"{geo}/{re_case}/t{t}: {e}")

                pbar.update(1)

    pbar.close()
    print(f"\nBuilt {built} | Skipped (cached) {skipped} | Failed {failed}")
    if failed_cases:
        for msg in failed_cases[:10]:
            print(f"  FAIL: {msg}")
    if len(failed_cases) > 10:
        print(f"  ... and {len(failed_cases) - 10} more")


# ============================================================================
# Normalization helpers
# ============================================================================

def compute_normalization_stats_v4(
    graph_paths: Optional[List[Path]] = None,
) -> Dict:
    """
    Compute normalization statistics for v4 dataset.

    x and edge stats: sampled at stride 20 (one snapshot per (geo, Re) case)
    to avoid 20× over-counting of geometry features across timesteps.

    y stats: computed over all paths (full temporal amplitude range needed).
    """
    if graph_paths is None:
        graph_paths = sorted(config_v4.processed_data_dir.rglob("*.pt"))
    if not graph_paths:
        raise FileNotFoundError(
            "No v4 .pt files found — run: python -m Bifurcation.dataset_v4 --process"
        )

    # One geometry snapshot per (geo, Re) case for x/edge stats
    x_sample_paths = graph_paths[::20]

    all_x, all_edge = [], []
    for p in x_sample_paths:
        d = torch.load(p, weights_only=False)
        all_x.append(d.x)
        all_edge.append(d.edge_attr)

    all_y = []
    for p in graph_paths:
        d = torch.load(p, weights_only=False)
        all_y.append(d.y)

    all_x    = torch.cat(all_x,    0)
    all_edge = torch.cat(all_edge, 0)
    all_y    = torch.cat(all_y,    0)

    x_mean, x_std = all_x.mean(0), all_x.std(0)
    e_mean, e_std = all_edge.mean(0), all_edge.std(0)

    if config_v4.use_log_transform:
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
        "use_log_transform": config_v4.use_log_transform,
    }


def save_normalization_stats_v4(stats: Dict, path: Optional[Path] = None):
    path = path or (config_v4.processed_data_dir / "normalization_stats_v4.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved v4 normalization stats → {path}")


def load_normalization_stats_v4(path: Optional[Path] = None) -> Dict:
    path = path or (config_v4.processed_data_dir / "normalization_stats_v4.json")
    with open(path) as f:
        return json.load(f)


# ============================================================================
# De-normalisation (for inference)
# ============================================================================

def denormalize_wss_v4(y_norm: torch.Tensor, stats: Dict) -> torch.Tensor:
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

class BifurcationWSSDatasetV4(Dataset):
    """
    PyTorch Dataset loading v4 pulsatile .pt graph files (10-dim node features).

    Applies z-score normalisation on the fly. data.phase is a scalar per graph
    and is passed through unchanged (rotation augmentation does not affect phase).
    """

    def __init__(
        self,
        graph_paths: List[Path],
        norm_stats:  Optional[Dict] = None,
        augment:     bool = False,
        aug_angles:  tuple = (-15.0, -7.5, 7.5, 15.0),
        use_3d_aug:  bool = False,
    ):
        super().__init__()
        self.graph_paths = list(graph_paths)
        self.norm_stats  = norm_stats
        self.augment     = augment
        self.aug_angles  = list(aug_angles)
        self.use_3d_aug  = use_3d_aug

        if norm_stats is not None:
            self._x_mean = torch.tensor(norm_stats["x_mean"],   dtype=torch.float32)
            self._x_std  = torch.tensor(norm_stats["x_std"],    dtype=torch.float32)
            self._e_mean = torch.tensor(norm_stats["edge_mean"], dtype=torch.float32)
            self._e_std  = torch.tensor(norm_stats["edge_std"],  dtype=torch.float32)
            self._y_mean = torch.tensor(norm_stats["y_mean"],   dtype=torch.float32)
            self._y_std  = torch.tensor(norm_stats["y_std"],    dtype=torch.float32)

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx) -> Data:
        data = torch.load(self.graph_paths[idx], weights_only=False)

        if self.augment:
            if self.use_3d_aug:
                data = rotate_3d(data)
            else:
                theta = float(np.random.choice([0.0] + self.aug_angles))
                if theta != 0.0:
                    data = rotate_z(data, theta)
            # data.phase is unaffected by rotation (scalar cardiac phase)

        if self.norm_stats is not None:
            data.x         = (data.x        - self._x_mean) / self._x_std
            data.edge_attr = (data.edge_attr - self._e_mean) / self._e_std

            if self.norm_stats.get("use_log_transform", False):
                sign   = torch.sign(data.y)
                data.y = sign * torch.log1p(torch.abs(data.y))
            data.y = (data.y - self._y_mean) / self._y_std

        return data


# ============================================================================
# Stratified splitting  (case-level to prevent data leakage)
# ============================================================================

def _geo_from_path_v4(p: Path) -> str:
    """ProcessedData_v4/bifurcation_angle45_750_ascii/Re500/t3.1.pt → geo folder"""
    return p.parent.parent.name


def _re_from_path_v4(p: Path) -> int:
    """ProcessedData_v4/.../Re500/t3.1.pt → 500"""
    return int(p.parent.name.replace("Re", ""))


def _re_bracket_v4(re: int) -> int:
    for i, (lo, hi) in enumerate(config_v4.re_brackets):
        if lo <= re <= hi:
            return i
    return len(config_v4.re_brackets) - 1


def get_split_paths_v4(
    mode:        str = "re_angle_stratified",
    seed:        int = 42,
    train_ratio: float = 0.75,
    val_ratio:   float = 0.10,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split pulsatile dataset at the (geo, Re) case level, then expand to timesteps.

    All 20 timestep files for a given (geo, Re) case land in the same split —
    no data leakage between train/val/test from the same simulation.

    Stratification: cells = (angle_deg, re_bracket). Each cell is split
    proportionally across train/val/test.
    """
    all_paths = sorted(config_v4.processed_data_dir.rglob("*.pt"))
    if not all_paths:
        raise FileNotFoundError(
            "No v4 .pt files found — run: python -m Bifurcation.dataset_v4 --process"
        )

    # Group by (geo, Re) case → list of timestep paths
    case_to_paths: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for p in all_paths:
        geo = _geo_from_path_v4(p)
        re_name = p.parent.name   # e.g. "Re500"
        case_to_paths[(geo, re_name)].append(p)

    rng = np.random.RandomState(seed)

    if mode == "re_angle_stratified":
        # Group cases by (angle, re_bracket) cell
        cell_cases: Dict[Tuple, List[Tuple[str, str]]] = defaultdict(list)
        for (geo, re_name) in case_to_paths.keys():
            angle, _  = config_v4.parse_geometry_folder(geo)
            re_val    = int(re_name.replace("Re", ""))
            bracket   = _re_bracket_v4(re_val)
            cell_cases[(angle, bracket)].append((geo, re_name))

        train_paths, val_paths, test_paths = [], [], []
        for _cell, cases in cell_cases.items():
            cases = sorted(cases)
            rng.shuffle(cases)
            n      = len(cases)
            n_test = max(1, int(n * (1.0 - train_ratio - val_ratio)))
            n_val  = max(1, int(n * val_ratio))
            n_test = min(n_test, n - n_val - 1)

            test_cases  = cases[:n_test]
            val_cases   = cases[n_test : n_test + n_val]
            train_cases = cases[n_test + n_val:]

            for case in test_cases:
                test_paths.extend(case_to_paths[case])
            for case in val_cases:
                val_paths.extend(case_to_paths[case])
            for case in train_cases:
                train_paths.extend(case_to_paths[case])

        return train_paths, val_paths, test_paths

    else:
        raise ValueError(f"Unknown split mode: {mode!r}. Use 're_angle_stratified'.")


def get_dataloaders_v4(
    mode:       str = "re_angle_stratified",
    seed:       int = 42,
    norm_stats: Optional[Dict] = None,
    batch_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_p, val_p, test_p = get_split_paths_v4(mode, seed)

    if norm_stats is None:
        norm_stats = compute_normalization_stats_v4(train_p)
        save_normalization_stats_v4(norm_stats)

    bs = batch_size or config_v4.batch_size

    train_ds = BifurcationWSSDatasetV4(train_p, norm_stats, augment=True)
    val_ds   = BifurcationWSSDatasetV4(val_p,   norm_stats, augment=False)
    test_ds  = BifurcationWSSDatasetV4(test_p,  norm_stats, augment=False)

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

    parser = argparse.ArgumentParser(description="Bifurcation v4 pulsatile dataset utilities")
    parser.add_argument("--process", action="store_true",
                        help="Build all v4 .pt graphs (pulsatile, 20 timesteps per case)")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if cached .pt files exist")
    parser.add_argument("--check", action="store_true",
                        help="Parse one sample and print v4 feature summary")
    parser.add_argument("--stats", action="store_true",
                        help="Compute & save v4 normalisation stats")
    args = parser.parse_args()

    if args.process:
        build_all_processed_v4(force=args.force)
        print("\nComputing normalization stats...")
        stats = compute_normalization_stats_v4()
        save_normalization_stats_v4(stats)

    elif args.check:
        # Try to find a completed case
        geo     = "bifurcation_angle30_500_ascii"
        re_case = "Re500"
        t       = 3.1

        pulsatile_case = config_v4.get_geometry_path(geo) / re_case
        if not pulsatile_case.exists():
            print(f"Pulsatile case not found: {pulsatile_case}")
            print("Try another geometry/Re case, or run --process first.")
        else:
            print(f"Parsing (v4) {geo} / {re_case} / t={t} ...")
            data = sample_to_pyg_v4(geo, re_case, t)
            print(f"  Nodes:           {data.x.shape[0]}")
            print(f"  Node features:   {data.x.shape}  (expected [N, 10])")
            print(f"  Edges:           {data.edge_index.shape[1]}")
            print(f"  WSS targets:     {data.y.shape}")
            print(f"  Re:              {data.re.item():.0f}")
            print(f"  angle:           {data.angle.item():.0f}°")
            print(f"  phase (φ):       {data.phase.item():.3f}  (t={t} mod 1.0)")
            print(f"  WSS range:       [{data.y.min():.4e}, {data.y.max():.4e}]")

            labels = ["x", "y", "z", "dist_junc", "arc_len", "branch_depth",
                      "nx", "ny", "nz", "curvature"]
            for i, lbl in enumerate(labels):
                col = data.x[:, i]
                print(f"  feat[{i:2d}] {lbl:15s}: min={col.min():.4f}  max={col.max():.4f}")
            print("OK ✓")

    elif args.check:
        pass  # handled above

    elif args.stats:
        stats = compute_normalization_stats_v4()
        save_normalization_stats_v4(stats)
        print(json.dumps(stats, indent=2))

    else:
        parser.print_help()

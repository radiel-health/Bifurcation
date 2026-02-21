#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BIFURCATION TRAINING PREPROCESSOR
Fluent legacy ASCII .msh + wall_data_Re*.csv -> ProcessedData/Re*.pt

Key improvements:
- pandas whitespace parsing via sep=r"\\s+" (pandas 2/3 compatible)
- NO label dropping: we never discard CSV rows; tolerance is diagnostic only
- unit sanity check: auto-rescale CSV xyz if it looks like mm vs m mismatch
- better diagnostics: ok_frac, p95/p99, unique_hit_frac, max_dist
- Windows-safe: raw strings in USER SETTINGS
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# ======================================================================================
# USER SETTINGS (EDIT THESE ONCE)
# ======================================================================================

RESULTS_ROOT = Path(r"C:\\Users\\radie\\Desktop\\OLDrealbifurcationSIM\\results\\coronary_extracted_vessel")
MESH_PATH    = Path(r"C:\\Users\\radie\\Desktop\\OLDrealbifurcationSIM\\coronary_extracted_vessel.msh")
OUT_ROOT     = Path(r"C:\\Users\\radie\\Desktop\\trainingprocess\\ProcessedData\\coronary_extracted_vessel")

WALL_ZONE    = 19

# What y should contain:
#   "wss_vec3"      -> [x,y,z] wall-shear components
#   "wss_mag"       -> [mag]
#   "wss_mag_vec3"  -> [mag,x,y,z]
TARGET_MODE  = "wss_vec3"

# Diagnostic tolerance (NOT used to drop labels anymore):
# "ok_frac" = fraction of CSV points within tol_abs of nearest wall node.
TOL_FRAC     = 1e-4   # <- was 1e-5; 1e-4 is usually reasonable for exported coords

# Save per-Re metadata json next to each .pt?
WRITE_META   = False

# ======================================================================================


# ----------------------------
# Utilities
# ----------------------------

def _parse_int_auto(tok: str) -> int:
    """
    Fluent legacy .msh stores most indices (zone IDs, node IDs, face IDs, ranges)
    in HEX, even if the token contains only digits (e.g. '13' means 0x13 = 19).
    """
    t = tok.strip()
    if t == "":
        raise ValueError("Empty int token")
    sign = 1
    if t[0] == "-":
        sign = -1
        t = t[1:]
    return sign * int(t, 16)


def _strip_parens(line: str) -> str:
    return line.replace("(", " ").replace(")", " ").strip()


def _compute_vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals from triangle faces."""
    N = points.shape[0]
    normals = np.zeros((N, 3), dtype=np.float64)

    p0 = points[faces[:, 0]]
    p1 = points[faces[:, 1]]
    p2 = points[faces[:, 2]]

    fn = np.cross(p1 - p0, p2 - p0)  # [F,3]

    for k in range(3):
        np.add.at(normals, faces[:, k], fn)

    norm = np.linalg.norm(normals, axis=1)
    norm = np.where(norm > 1e-12, norm, 1.0)
    normals = (normals.T / norm).T
    return normals.astype(np.float32)


def _normalize_bbox(points: np.ndarray) -> np.ndarray:
    """Normalize xyz to [0,1] by bounding box."""
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    span = np.where((pmax - pmin) > 1e-12, (pmax - pmin), 1.0)
    return ((points - pmin) / span).astype(np.float32)


def _bbox_diag(points: np.ndarray) -> float:
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    return float(np.linalg.norm(pmax - pmin))


def _try_read_table(path: Path) -> pd.DataFrame:
    """
    Fluent export is typically whitespace-delimited with headers.
    Use sep=r"\\s+" and engine="python" for robust whitespace parsing.
    """
    # whitespace
    try:
        return pd.read_csv(path, sep=r"\s+", engine="python")
    except Exception:
        pass
    # comma fallback
    return pd.read_csv(path)


def _find_wall_csv(re_dir: Path, re_value: int) -> Path:
    """
    Finds wall_data file robustly inside Re### directory.
    Accepts:
      wall_data_Re75
      wall_data_Re75.csv
      wall_data_Re75.txt
      wall_data_Re75*
    """
    candidates: List[Path] = []
    patterns = [
        f"wall_data_Re{re_value}.csv",
        f"wall_data_Re{re_value}.txt",
        f"wall_data_Re{re_value}",
        f"wall_data_Re{re_value}.*",
        f"wall_data_Re{re_value}*",
    ]
    for pat in patterns:
        candidates.extend(sorted(re_dir.glob(pat)))
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    raise FileNotFoundError(f"Could not find wall_data for Re{re_value} in: {re_dir}")


def _extract_columns(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Flexible column picking.
    Expected (from your Fluent export):
      x-coordinate y-coordinate z-coordinate pressure wall-shear x-wall-shear y-wall-shear z-wall-shear
    """
    # map lower->original
    cols = {c.strip().lower(): c for c in df.columns}

    def pick(*names: str) -> str:
        # exact
        for n in names:
            n2 = n.lower()
            if n2 in cols:
                return cols[n2]
        # fuzzy contains
        for want in names:
            w = want.lower()
            for k, orig in cols.items():
                if w in k:
                    return orig
        raise KeyError(f"Missing column. Tried {names}. Available: {list(df.columns)}")

    xcol = pick("x-coordinate", "x", "x_coordinate")
    ycol = pick("y-coordinate", "y", "y_coordinate")
    zcol = pick("z-coordinate", "z", "z_coordinate")

    # WSS components
    wx = pick("x-wall-shear", "x_wall_shear", "wss_x", "x-wss")
    wy = pick("y-wall-shear", "y_wall_shear", "wss_y", "y-wss")
    wz = pick("z-wall-shear", "z_wall_shear", "wss_z", "z-wss")

    out = {
        "xyz": df[[xcol, ycol, zcol]].to_numpy(dtype=np.float32),
        "wss_vec3": df[[wx, wy, wz]].to_numpy(dtype=np.float32),
    }

    # optional magnitude / pressure
    for key, tries in [
        ("wss_mag", ("wall-shear", "wallshear", "wss", "wall_shear")),
        ("pressure", ("pressure",)),
    ]:
        try:
            c = pick(*tries)
            out[key] = df[[c]].to_numpy(dtype=np.float32).reshape(-1, 1)
        except Exception:
            pass

    return out


# ----------------------------
# Fluent .msh parser (ASCII): extract wall surface triangles
# ----------------------------

def read_fluent_ascii_msh_wall_surface(msh_path: Path, wall_zone_id: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    msh_path = Path(msh_path)
    if not msh_path.exists():
        raise FileNotFoundError(f"Input mesh not found: {msh_path}")

    node_total_start = None
    node_total_end = None
    node_blocks: List[Tuple[int, int, int, int, int, int]] = []
    face_blocks: List[Tuple[int, int, int, int, int, int]] = []
    scaling_factor = None

    with msh_path.open("r", errors="ignore") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()

            if s.startswith("(10 (0 "):
                parts = _strip_parens(s).split()
                if len(parts) >= 5:
                    node_total_start = _parse_int_auto(parts[2])
                    node_total_end = _parse_int_auto(parts[3])

            if s.startswith("(10 (") and not s.startswith("(10 (0 "):
                head = _strip_parens(s).split()
                if len(head) >= 6 and head[0] == "10":
                    zone = _parse_int_auto(head[1])
                    start = _parse_int_auto(head[2])
                    end = _parse_int_auto(head[3])
                    typ = _parse_int_auto(head[4])
                    dim = _parse_int_auto(head[5])
                    node_blocks.append((ln, zone, start, end, typ, dim))

            if s.startswith("(13 (") and not s.startswith("(13 (0 "):
                head = _strip_parens(s).split()
                if len(head) >= 6 and head[0] == "13":
                    zone = _parse_int_auto(head[1])
                    start = _parse_int_auto(head[2])
                    end = _parse_int_auto(head[3])
                    typ = _parse_int_auto(head[4])
                    etype = _parse_int_auto(head[5])
                    face_blocks.append((ln, zone, start, end, typ, etype))

            if "meshing-to-solver-scaling-factor" in s:
                toks = _strip_parens(s).split()
                try:
                    scaling_factor = float(toks[-1])
                except Exception:
                    pass

    if node_total_start is None or node_total_end is None:
        if not node_blocks:
            raise ValueError("Could not determine node range from mesh.")
        node_total_start = min(b[2] for b in node_blocks)
        node_total_end = max(b[3] for b in node_blocks)

    n_nodes = node_total_end - node_total_start + 1
    if n_nodes <= 0:
        raise ValueError(f"Invalid node count: start={node_total_start} end={node_total_end}")

    points_global = np.zeros((n_nodes, 3), dtype=np.float32)

    wall_block = None
    for blk in face_blocks:
        _, zone, *_ = blk
        if zone == wall_zone_id:
            wall_block = blk
            break
    if wall_block is None:
        zones = sorted(set(b[1] for b in face_blocks))
        raise ValueError(f"Wall zone {wall_zone_id} not found. Face zones present: {zones}")

    meta = {
        "input_mesh": str(msh_path),
        "node_total_start": int(node_total_start),
        "node_total_end": int(node_total_end),
        "n_nodes": int(n_nodes),
        "scaling_factor": scaling_factor,
        "wall_zone_id": int(wall_zone_id),
        "face_zones": sorted(set(b[1] for b in face_blocks)),
    }

    wall_faces: List[List[int]] = []

    node_block_headers_by_line = {b[0]: b for b in node_blocks}
    wall_face_header_line = wall_block[0]

    in_node_block = False
    current_node_expected = None
    current_node_end = None

    in_wall_face_block = False
    wall_faces_expected = None

    def set_node_xyz(node_id_1based: int, xyz: Tuple[float, float, float]):
        idx = node_id_1based - node_total_start
        if idx < 0 or idx >= n_nodes:
            raise IndexError(f"Node id {node_id_1based} out of bounds.")
        points_global[idx, :] = xyz

    with msh_path.open("r", errors="ignore") as f:
        ln = 0
        for line in f:
            ln += 1
            s = line.strip()

            if ln in node_block_headers_by_line:
                _, zone, start, end, typ, dim = node_block_headers_by_line[ln]
                if dim != 3:
                    raise ValueError(f"Expected 3D nodes, got dim={dim} in node zone={zone}")
                in_node_block = True
                current_node_expected = start
                current_node_end = end
                continue

            if ln == wall_face_header_line:
                _, zone, start, end, typ, etype = wall_block
                in_wall_face_block = True
                wall_faces_expected = end - start + 1
                continue

            if in_node_block:
                t = _strip_parens(s)
                if t:
                    vals = [float(x) for x in t.split()]
                    if len(vals) % 3 != 0:
                        raise ValueError(f"Node coord line not multiple of 3 floats at line {ln}")
                    for i in range(0, len(vals), 3):
                        if current_node_expected is None:
                            raise RuntimeError("current_node_expected None")
                        if current_node_expected > current_node_end:
                            break
                        set_node_xyz(current_node_expected, (vals[i], vals[i + 1], vals[i + 2]))
                        current_node_expected += 1

                if current_node_expected is not None and current_node_expected > current_node_end:
                    in_node_block = False
                    current_node_expected = None
                    current_node_end = None
                continue

            if in_wall_face_block:
                t = _strip_parens(s)
                if not t:
                    continue
                toks = t.split()

                try:
                    first_int = _parse_int_auto(toks[0])
                except Exception:
                    first_int = None

                if first_int in (3, 4):
                    nv = first_int
                    verts = toks[1:1 + nv]
                else:
                    nv = 3
                    verts = toks[0:nv]

                vid = [_parse_int_auto(v) for v in verts]

                if nv == 3:
                    wall_faces.append([vid[0], vid[1], vid[2]])
                else:
                    wall_faces.append([vid[0], vid[1], vid[2]])
                    wall_faces.append([vid[0], vid[2], vid[3]])

                if wall_faces_expected is not None and len(wall_faces) >= wall_faces_expected:
                    in_wall_face_block = False

    wall_faces_global = np.array(wall_faces, dtype=np.int64)
    wall_faces_global0 = wall_faces_global - node_total_start  # 0-based into points_global

    if scaling_factor is not None:
        points_global = (points_global * float(scaling_factor)).astype(np.float32)

    meta["n_wall_faces_raw"] = int(wall_faces_global0.shape[0])
    return points_global, wall_faces_global0, meta


# ----------------------------
# Build base wall surface graph
# ----------------------------

def build_wall_surface_graph(points_global: np.ndarray, wall_faces_global0: np.ndarray) -> Tuple[Data, Dict]:
    wall_node_ids = np.unique(wall_faces_global0.reshape(-1))
    wall_node_ids_sorted = np.sort(wall_node_ids)

    global_to_local = -np.ones(points_global.shape[0], dtype=np.int64)
    global_to_local[wall_node_ids_sorted] = np.arange(wall_node_ids_sorted.size, dtype=np.int64)

    faces_local = global_to_local[wall_faces_global0]
    wall_points = points_global[wall_node_ids_sorted].astype(np.float32)

    normals = _compute_vertex_normals(wall_points, faces_local.astype(np.int64))
    xyz_norm = _normalize_bbox(wall_points)
    is_wall = np.ones((wall_points.shape[0], 1), dtype=np.float32)

    x = np.concatenate([xyz_norm, normals, is_wall], axis=1).astype(np.float32)

    a, b, c = faces_local[:, 0], faces_local[:, 1], faces_local[:, 2]
    edges = np.vstack([
        np.stack([a, b], axis=1),
        np.stack([b, c], axis=1),
        np.stack([c, a], axis=1),
    ]).astype(np.int64)

    edges_undirected = np.vstack([edges, edges[:, ::-1]])
    edges_undirected = np.unique(edges_undirected, axis=0)

    edge_index = torch.from_numpy(edges_undirected.T).long()

    data = Data(
        pos=torch.from_numpy(wall_points).float(),
        x=torch.from_numpy(x).float(),
        edge_index=edge_index,
        face=torch.from_numpy(faces_local.T.astype(np.int64)).long(),
        num_nodes=int(wall_points.shape[0]),
    )
    data.global_node_ids = torch.from_numpy(wall_node_ids_sorted.astype(np.int64))  # local->global

    info = {
        "Nw": int(wall_points.shape[0]),
        "Fw": int(faces_local.shape[0]),
        "Ew": int(edges_undirected.shape[0]),
        "feature_dim": int(x.shape[1]),
        "wall_bbox_diag": _bbox_diag(wall_points),
    }
    return data, info


# ----------------------------
# Nearest neighbor mapping (CSV xyz -> wall xyz)
# ----------------------------

def _nearest_neighbor_map(query_xyz: np.ndarray, ref_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest neighbor indices + distances."""
    try:
        from scipy.spatial import cKDTree  # fast + standard
        tree = cKDTree(ref_xyz)
        dist, idx = tree.query(query_xyz, k=1)
        return idx.astype(np.int64), dist.astype(np.float32)
    except Exception:
        pass

    # fallback: brute force chunks
    M = query_xyz.shape[0]
    idx_out = np.empty(M, dtype=np.int64)
    dist_out = np.empty(M, dtype=np.float32)

    chunk = 1500
    for i0 in range(0, M, chunk):
        i1 = min(i0 + chunk, M)
        q = query_xyz[i0:i1]
        d2 = ((q[:, None, :] - ref_xyz[None, :, :]) ** 2).sum(axis=2)
        idx = np.argmin(d2, axis=1)
        dist = np.sqrt(d2[np.arange(idx.size), idx])
        idx_out[i0:i1] = idx
        dist_out[i0:i1] = dist.astype(np.float32)

    return idx_out, dist_out


def _auto_rescale_csv_xyz(csv_xyz: np.ndarray, wall_xyz: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    If CSV appears to be in different length units than wall_xyz, rescale.
    Uses bbox diagonal ratio.
    """
    diag_csv = _bbox_diag(csv_xyz)
    diag_wall = _bbox_diag(wall_xyz)
    if diag_csv < 1e-12 or diag_wall < 1e-12:
        return csv_xyz, {"auto_scale": 1.0, "diag_csv": diag_csv, "diag_wall": diag_wall}

    ratio = diag_wall / diag_csv  # multiply csv by ratio to match wall scale

    # common unit jumps: mm<->m is ~0.001 or ~1000
    if 0.0005 <= ratio <= 0.002:
        scale = ratio
    elif 500.0 <= ratio <= 2000.0:
        scale = ratio
    else:
        scale = 1.0

    return (csv_xyz * scale).astype(np.float32), {"auto_scale": float(scale), "diag_csv": float(diag_csv), "diag_wall": float(diag_wall), "diag_ratio": float(ratio)}


def align_wall_csv_to_graph(
    base_graph: Data,
    wall_csv_path: Path,
    tol_abs: float,
    target_mode: str,
) -> Tuple[torch.Tensor, Dict]:
    """
    Build y for each wall node by mapping every CSV row to nearest wall node.
    We DO NOT drop rows. tol_abs is used only for diagnostics (ok_frac).
    If multiple rows hit same node -> average.
    """

    df = _try_read_table(wall_csv_path)
    cols = _extract_columns(df)

    csv_xyz = cols["xyz"].astype(np.float32)      # [M,3]
    csv_wss = cols["wss_vec3"].astype(np.float32) # [M,3]

    wall_xyz = base_graph.pos.cpu().numpy().astype(np.float32)  # [Nw,3]

    # Unit sanity check
    csv_xyz2, scale_info = _auto_rescale_csv_xyz(csv_xyz, wall_xyz)

    nn_idx, nn_dist = _nearest_neighbor_map(csv_xyz2, wall_xyz)

    ok = (nn_dist <= tol_abs)
    ok_frac = float(ok.mean()) if ok.size else 0.0

    # Aggregate (ALL rows)
    Nw = wall_xyz.shape[0]
    sum_wss = np.zeros((Nw, 3), dtype=np.float64)
    cnt = np.zeros((Nw, 1), dtype=np.float64)

    np.add.at(sum_wss, nn_idx, csv_wss.astype(np.float64))
    np.add.at(cnt, nn_idx, 1.0)

    cnt_safe = np.where(cnt > 0, cnt, 1.0)
    mean_wss = (sum_wss / cnt_safe).astype(np.float32)

    # diagnostics
    unique_hit = int((cnt[:, 0] > 0).sum())
    unique_hit_frac = float(unique_hit / max(Nw, 1))

    # Build y
    if target_mode == "wss_vec3":
        y = mean_wss  # [Nw,3]
    elif target_mode == "wss_mag":
        y = np.linalg.norm(mean_wss, axis=1, keepdims=True).astype(np.float32)
    elif target_mode == "wss_mag_vec3":
        mag = np.linalg.norm(mean_wss, axis=1, keepdims=True).astype(np.float32)
        y = np.concatenate([mag, mean_wss], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unknown TARGET_MODE: {target_mode}")

    # quantiles
    if nn_dist.size:
        p95 = float(np.quantile(nn_dist, 0.95))
        p99 = float(np.quantile(nn_dist, 0.99))
        dmax = float(nn_dist.max())
        dmean = float(nn_dist.mean())
    else:
        p95 = p99 = dmax = dmean = float("nan")

    stats = {
        "wall_csv": str(wall_csv_path),
        "csv_rows": int(csv_xyz.shape[0]),
        "tol_abs": float(tol_abs),
        "match_ok_frac": ok_frac,
        "match_p95_dist": p95,
        "match_p99_dist": p99,
        "match_max_dist": dmax,
        "match_mean_dist": dmean,
        "unique_hit_frac": unique_hit_frac,
        "n_wall_nodes": int(Nw),
        **scale_info,
    }

    return torch.from_numpy(y).float(), stats


# ----------------------------
# Main
# ----------------------------

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("BIFURCATION TRAINING PREPROCESSOR: Fluent .msh + wall_data_Re*.csv -> ProcessedData/Re*.pt")
    print("=" * 90)
    print(f"RESULTS_ROOT : {RESULTS_ROOT}")
    print(f"MESH_PATH    : {MESH_PATH}")
    print(f"OUT_ROOT     : {OUT_ROOT}")
    print(f"WALL_ZONE    : {WALL_ZONE}")
    print(f"TARGET_MODE  : {TARGET_MODE}")
    print(f"TOL_FRAC     : {TOL_FRAC}")
    print(f"WRITE_META   : {WRITE_META}")
    print()

    # 1) Parse mesh & build base wall graph
    points_global, wall_faces_global0, meta_mesh = read_fluent_ascii_msh_wall_surface(
        msh_path=MESH_PATH,
        wall_zone_id=int(WALL_ZONE),
    )
    base_graph, info = build_wall_surface_graph(points_global, wall_faces_global0)

    bbox_diag = info["wall_bbox_diag"]
    tol_abs = float(TOL_FRAC) * float(bbox_diag)

    print("[OK] Built base wall graph from mesh")
    print(f"  Wall nodes: {info['Nw']}")
    print(f"  Wall faces: {info['Fw']}")
    print(f"  Wall edges: {info['Ew']}")
    print(f"  Feature dim: {info['feature_dim']}")
    print(f"  Wall bbox diag: {bbox_diag:.6e}  => diagnostic tol abs = {tol_abs:.6e}")
    if meta_mesh.get("scaling_factor", None) is not None:
        print(f"  Mesh scaling_factor applied: {meta_mesh['scaling_factor']}")
    print()

    # 2) Find Re folders
    if not RESULTS_ROOT.exists():
        raise FileNotFoundError(f"RESULTS_ROOT not found: {RESULTS_ROOT}")

    re_dirs = sorted([p for p in RESULTS_ROOT.iterdir() if p.is_dir() and re.match(r"^Re\d+$", p.name)])
    print(f"Found {len(re_dirs)} Reynolds folders to process.\n")

    failures = []
    successes = 0

    for re_dir in re_dirs:
        try:
            re_value = int(re_dir.name.replace("Re", ""))

            wall_csv = _find_wall_csv(re_dir, re_value)
            y, stats = align_wall_csv_to_graph(
                base_graph=base_graph,
                wall_csv_path=wall_csv,
                tol_abs=tol_abs,
                target_mode=TARGET_MODE,
            )

            data = Data(
                pos=base_graph.pos.clone(),
                x=base_graph.x.clone(),
                edge_index=base_graph.edge_index.clone(),
                face=base_graph.face.clone(),
                num_nodes=base_graph.num_nodes,
            )
            data.global_node_ids = base_graph.global_node_ids.clone()

            data.re = float(re_value)
            data.y = y

            out_path = OUT_ROOT / f"Re{re_value}.pt"
            torch.save(data, out_path)

            if WRITE_META:
                meta_out = OUT_ROOT / f"Re{re_value}_meta.json"
                with open(meta_out, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "re": re_value,
                            "mesh": str(MESH_PATH),
                            "results_dir": str(re_dir),
                            **meta_mesh,
                            **info,
                            **stats,
                            "y_shape": list(y.shape),
                        },
                        f,
                        indent=2,
                    )

            warn = []
            if stats["match_ok_frac"] < 0.98:
                warn.append(f"ok_frac={stats['match_ok_frac']:.3f}")
            if stats["unique_hit_frac"] < 0.98:
                warn.append(f"unique_hit_frac={stats['unique_hit_frac']:.3f}")
            if stats.get("auto_scale", 1.0) != 1.0:
                warn.append(f"auto_scale={stats['auto_scale']}")
            warn_txt = ("  [WARN] " + ", ".join(warn)) if warn else ""

            print(
                f"[OK] Re{re_value}: saved {out_path.name} | y={list(y.shape)} | "
                f"p99={stats['match_p99_dist']:.2e} | max={stats['match_max_dist']:.2e}{warn_txt}"
            )
            successes += 1

        except Exception as e:
            print(f"[FAIL] {re_dir.name}: {e}")
            failures.append(f"{re_dir.name}: {e}")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print(f"Output folder: {OUT_ROOT}")
    print(f"Successes: {successes}/{len(re_dirs)}")
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f" - {f}")


if __name__ == "__main__":
    main()

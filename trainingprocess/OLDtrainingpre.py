#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BIFURCATION TRAINING PREPROCESSOR
Fluent legacy ASCII .msh + wall_data_Re*.csv -> ProcessedData/Re*.pt

Key fix vs your previous version:
- DO NOT drop rows by an ultra-tight tolerance.
- Always map CSV rows to nearest wall node, then average if collisions occur.
- Use tolerance ONLY for diagnostics (match_ok_frac).
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

RESULTS_ROOT = Path(r"C:\Users\radie\Desktop\realbifurcationSIM\results\coronary_extracted_vessel")
MESH_PATH    = Path(r"C:\Users\radie\Desktop\realbifurcationSIM\coronary_extracted_vessel.msh")
OUT_ROOT     = Path(r"C:\Users\radie\Desktop\trainingprocess\ProcessedData\coronary_extracted_vessel")

WALL_ZONE    = 13

# What y should contain:
#   "wss_vec3" -> [x,y,z] wall-shear components
#   "wss_mag"  -> [mag]
#   "wss_mag_vec3" -> [mag,x,y,z]
TARGET_MODE  = "wss_vec3"

# Tolerance is now for DIAGNOSTICS only (match_ok_frac).
# Recommended: 1e-4 to 1e-3. Start with 1e-4.
TOL_FRAC     = 1e-4

# If True, writes Re###_meta.json next to each .pt
WRITE_META   = False

# ======================================================================================


# ----------------------------
# Utilities
# ----------------------------

def _parse_int_auto(tok: str) -> int:
    """
    Fluent ASCII .msh often uses hex-like integers without 0x prefix (e.g. '3392b', 'c2d').
    Parse base-16 if token has a-f, else base-10.
    """
    t = tok.strip()
    if not t:
        raise ValueError("Empty int token")

    sign = 1
    if t[0] == "-":
        sign = -1
        t = t[1:]

    base = 16 if any(c in "abcdefABCDEF" for c in t) else 10
    return sign * int(t, base)


def _strip_parens(line: str) -> str:
    return line.replace("(", " ").replace(")", " ").strip()


def _compute_vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals from triangle faces."""
    n = points.shape[0]
    normals = np.zeros((n, 3), dtype=np.float64)

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
    pandas>=2 works well with sep=r"\\s+".
    Also tries comma fallback if needed.
    """
    try:
        return pd.read_csv(path, sep=r"\s+", engine="python")
    except Exception:
        return pd.read_csv(path)


def _find_wall_csv(re_dir: Path, re_value: int) -> Path:
    """
    Finds:
      wall_data_Re75
      wall_data_Re75.csv
      wall_data_Re75.txt
      etc.
    """
    patterns = [
        f"wall_data_Re{re_value}.csv",
        f"wall_data_Re{re_value}.txt",
        f"wall_data_Re{re_value}",
        f"wall_data_Re{re_value}.*",
        f"wall_data_Re{re_value}*",
    ]
    for pat in patterns:
        for c in sorted(re_dir.glob(pat)):
            if c.is_file():
                return c
    raise FileNotFoundError(f"Could not find wall_data file for Re{re_value} in: {re_dir}")


def _extract_columns(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Matches columns flexibly. Expected from your .jou:
      x-coordinate y-coordinate z-coordinate pressure wall-shear x-wall-shear y-wall-shear z-wall-shear
    Some Fluent exports duplicate coordinate columns; we pick first match.
    """
    cols = {c.strip().lower(): c for c in df.columns}

    def pick(*names: str) -> str:
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
        raise KeyError(f"Missing expected column. Tried: {names}. Available: {list(df.columns)}")

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

    # Optional magnitude column
    try:
        wmag = pick("wall-shear", "wallshear", "wss", "wall_shear")
        out["wss_mag"] = df[[wmag]].to_numpy(dtype=np.float32).reshape(-1, 1)
    except Exception:
        pass

    return out


# ----------------------------
# Fluent .msh parser (ASCII)
# ----------------------------

def read_fluent_ascii_msh_wall_surface(
    msh_path: Path,
    wall_zone_id: int,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Extract global points and ONLY the wall zone triangles from Fluent legacy ASCII .msh.

    Returns:
      points_global: [N,3] float32 (all nodes, indexed from node_total_start..end)
      wall_faces_global0: [F,3] int64 (0-based into points_global)
      meta: dict
    """
    msh_path = Path(msh_path)
    if not msh_path.exists():
        raise FileNotFoundError(f"Input mesh not found: {msh_path}")

    node_total_start = None
    node_total_end = None
    node_blocks: List[Tuple[int,int,int,int,int,int]] = []
    face_blocks: List[Tuple[int,int,int,int,int,int]] = []
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
        "n_face_zones": int(len(face_blocks)),
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

    def set_node_xyz(node_id_1based: int, xyz: Tuple[float,float,float]):
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
                    raise ValueError(f"Expected 3D nodes (dim=3), got dim={dim} in node block zone={zone}")
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
                    toks = t.split()
                    vals = [float(x) for x in toks]
                    if len(vals) % 3 != 0:
                        raise ValueError(f"Node coord line not multiple of 3 floats at line {ln}")
                    for i in range(0, len(vals), 3):
                        if current_node_expected is None:
                            raise RuntimeError("current_node_expected None")
                        if current_node_expected > current_node_end:
                            break
                        set_node_xyz(current_node_expected, (vals[i], vals[i+1], vals[i+2]))
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
                    if len(toks) < 1 + nv + 2:
                        raise ValueError(f"Face line too short at {ln}: {s}")
                    verts = toks[1:1+nv]
                else:
                    nv = 3
                    if len(toks) < nv + 2:
                        raise ValueError(f"Face line too short at {ln}: {s}")
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
    wall_faces_global0 = wall_faces_global - node_total_start  # 0-based

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

    faces_local = global_to_local[wall_faces_global0]  # [F,3]
    wall_points = points_global[wall_node_ids_sorted].astype(np.float32)  # [Nw,3]

    normals = _compute_vertex_normals(wall_points, faces_local.astype(np.int64))
    xyz_norm = _normalize_bbox(wall_points)
    is_wall = np.ones((wall_points.shape[0], 1), dtype=np.float32)

    x = np.concatenate([xyz_norm, normals, is_wall], axis=1).astype(np.float32)  # dim=7

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
        face=torch.from_numpy(faces_local.T.astype(np.int64)).long(),  # [3,F]
        num_nodes=int(wall_points.shape[0]),
    )
    data.global_node_ids = torch.from_numpy(wall_node_ids_sorted.astype(np.int64))

    info = {
        "Nw": int(wall_points.shape[0]),
        "Fw": int(faces_local.shape[0]),
        "Ew": int(edges_undirected.shape[0]),
        "feature_dim": int(x.shape[1]),
        "wall_bbox_diag": _bbox_diag(wall_points),
    }
    return data, info


# ----------------------------
# Nearest-neighbor (fast if scipy/sklearn installed; fallback otherwise)
# ----------------------------

def _nearest_neighbor_map(query_xyz: np.ndarray, ref_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(ref_xyz)
        dist, idx = tree.query(query_xyz, k=1)
        return idx.astype(np.int64), dist.astype(np.float32)
    except Exception:
        pass

    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(ref_xyz)
        dist, idx = nn.kneighbors(query_xyz, return_distance=True)
        return idx[:, 0].astype(np.int64), dist[:, 0].astype(np.float32)
    except Exception:
        pass

    # brute force chunked fallback
    m = query_xyz.shape[0]
    idx_out = np.empty(m, dtype=np.int64)
    dist_out = np.empty(m, dtype=np.float32)
    chunk = 2000
    for i0 in range(0, m, chunk):
        i1 = min(i0 + chunk, m)
        q = query_xyz[i0:i1]
        d2 = ((q[:, None, :] - ref_xyz[None, :, :]) ** 2).sum(axis=2)
        idx = np.argmin(d2, axis=1)
        dist = np.sqrt(d2[np.arange(idx.size), idx])
        idx_out[i0:i1] = idx
        dist_out[i0:i1] = dist.astype(np.float32)
    return idx_out, dist_out


def align_wall_csv_to_graph(
    base_graph: Data,
    wall_csv_path: Path,
    tol_abs_diag: float,
    target_mode: str,
) -> Tuple[torch.Tensor, Dict]:
    """
    Assign each CSV row to nearest wall node (NO DROPPING).
    If multiple rows map to same node -> average.
    """
    df = _try_read_table(wall_csv_path)
    cols = _extract_columns(df)

    csv_xyz = cols["xyz"]        # [M,3]
    csv_wss = cols["wss_vec3"]   # [M,3]

    wall_xyz = base_graph.pos.cpu().numpy().astype(np.float32)  # [Nw,3]
    nn_idx, nn_dist = _nearest_neighbor_map(csv_xyz, wall_xyz)

    # Diagnostic only
    ok = nn_dist <= tol_abs_diag
    ok_frac = float(ok.mean()) if ok.size else 0.0

    # Aggregate all rows (no dropping)
    nw = wall_xyz.shape[0]
    sum_wss = np.zeros((nw, 3), dtype=np.float64)
    cnt = np.zeros((nw, 1), dtype=np.float64)

    np.add.at(sum_wss, nn_idx, csv_wss.astype(np.float64))
    np.add.at(cnt, nn_idx, 1.0)

    cnt_safe = np.where(cnt > 0, cnt, 1.0)
    mean_wss = (sum_wss / cnt_safe).astype(np.float32)

    n_unhit = int((cnt[:, 0] == 0).sum())

    if target_mode == "wss_vec3":
        y = mean_wss
    elif target_mode == "wss_mag":
        y = np.linalg.norm(mean_wss, axis=1, keepdims=True).astype(np.float32)
    elif target_mode == "wss_mag_vec3":
        mag = np.linalg.norm(mean_wss, axis=1, keepdims=True).astype(np.float32)
        y = np.concatenate([mag, mean_wss], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unknown TARGET_MODE: {target_mode}")

    stats = {
        "wall_csv": str(wall_csv_path),
        "csv_rows": int(csv_xyz.shape[0]),
        "tol_abs_diag": float(tol_abs_diag),
        "match_ok_frac": ok_frac,
        "match_max_dist": float(nn_dist.max()) if nn_dist.size else float("nan"),
        "match_p95_dist": float(np.quantile(nn_dist, 0.95)) if nn_dist.size else float("nan"),
        "match_mean_dist": float(nn_dist.mean()) if nn_dist.size else float("nan"),
        "n_wall_nodes": int(nw),
        "n_wall_nodes_unhit": n_unhit,
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
    print(f"TOL_FRAC     : {TOL_FRAC}  (diagnostic only)")
    print(f"WRITE_META   : {WRITE_META}")
    print()

    points_global, wall_faces_global0, meta_mesh = read_fluent_ascii_msh_wall_surface(
        msh_path=MESH_PATH,
        wall_zone_id=int(WALL_ZONE),
    )
    base_graph, info = build_wall_surface_graph(points_global, wall_faces_global0)

    bbox_diag = float(info["wall_bbox_diag"])
    tol_abs = float(TOL_FRAC) * bbox_diag

    print("[OK] Built base wall graph from mesh")
    print(f"  Wall nodes: {info['Nw']}")
    print(f"  Wall faces: {info['Fw']}")
    print(f"  Wall edges: {info['Ew']}")
    print(f"  Feature dim: {info['feature_dim']}")
    print(f"  Wall bbox diag: {bbox_diag:.6e}  => diagnostic tol abs = {tol_abs:.6e}")
    if meta_mesh.get("scaling_factor", None) is not None:
        print(f"  Mesh scaling_factor applied: {meta_mesh['scaling_factor']}")
    print()

    if not RESULTS_ROOT.exists():
        raise FileNotFoundError(f"RESULTS_ROOT not found: {RESULTS_ROOT}")

    re_dirs = sorted([p for p in RESULTS_ROOT.iterdir() if p.is_dir() and re.match(r"^Re\d+$", p.name)])
    print(f"Found {len(re_dirs)} Reynolds folders to process.\n")

    failures: List[str] = []
    successes = 0

    for re_dir in re_dirs:
        try:
            re_value = int(re_dir.name.replace("Re", ""))
            wall_csv = _find_wall_csv(re_dir, re_value)

            y, stats = align_wall_csv_to_graph(
                base_graph=base_graph,
                wall_csv_path=wall_csv,
                tol_abs_diag=tol_abs,
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
                warn.append(f"match_ok_frac={stats['match_ok_frac']:.3f}")
            if stats["match_p95_dist"] > 1e-4 * bbox_diag:
                warn.append(f"p95_dist={stats['match_p95_dist']:.2e}")
            if stats["n_wall_nodes_unhit"] > 0:
                warn.append(f"unhit_nodes={stats['n_wall_nodes_unhit']}")

            warn_txt = ("  [WARN] " + ", ".join(warn)) if warn else ""
            print(f"[OK] Re{re_value}: saved {out_path.name} | y={list(y.shape)}{warn_txt}")
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

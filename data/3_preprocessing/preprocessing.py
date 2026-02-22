#!/usr/bin/env python3
"""
Preprocess Fluent legacy ASCII .msh -> PyTorch Geometric Data (wall surface graph)

Goal:
- Nodes = wall surface nodes
- Faces = wall triangles
- Edges = triangle adjacency

Notes:
- Fluent legacy .msh can have face blocks in different encodings (some include a leading 'n' count per face).
  This parser supports both.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch_geometric.data import Data


# ----------------------------
# Utilities
# ----------------------------

def _parse_int_auto(tok: str) -> int:
    """
    Fluent ASCII .msh often uses hex-like integers without 0x prefix (e.g. '13' meaning 0x13).
    Some files also use plain decimal in some places. This function attempts to parse robustly.

    Heuristic:
    - If token has a-f letters -> parse base 16
    - Else parse base 10
    """
    t = tok.strip().lower()
    if any(c in t for c in "abcdef"):
        return int(t, 16)
    return int(t, 10)


def _strip_comments(line: str) -> str:
    # Fluent .msh typically uses ';' for comments
    if ";" in line:
        return line.split(";", 1)[0]
    return line


def _read_all_lines(msh_path: Path) -> List[str]:
    with open(msh_path, "r", errors="ignore") as f:
        lines = f.readlines()
    return [ln.rstrip("\n") for ln in lines]


def _find_scaling_factor(lines: List[str]) -> float:
    # (meshing-to-solver-scaling-factor 1)
    for ln in lines:
        s = _strip_comments(ln).strip()
        if s.startswith("(meshing-to-solver-scaling-factor"):
            parts = s.strip("()").split()
            if len(parts) >= 2:
                try:
                    return float(parts[-1])
                except Exception:
                    pass
    return 1.0


# ----------------------------
# Mesh parsing
# ----------------------------

def parse_fluent_ascii_msh_wall(
    msh_path: Path,
    wall_zone: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse Fluent legacy ASCII .msh:
      - nodes from (10 ...) blocks
      - wall faces from (13 ...) blocks with zone id == wall_zone
    Returns:
      nodes_xyz: [N,3] float
      faces: [F,3] int (global node indices, 0-based)
    """
    lines = _read_all_lines(msh_path)
    scale = _find_scaling_factor(lines)

    # --- Parse nodes ---
    nodes: Dict[int, np.ndarray] = {}

    i = 0
    while i < len(lines):
        raw = _strip_comments(lines[i]).strip()
        if raw.startswith("(10"):
            # header like: (10 (0 1 2 0 3))  OR (10 (2 1 12345 0 3))
            # then coordinates lines follow until a ')' on its own or closing
            hdr = raw
            # Sometimes header is multi-line; ensure we have matching parentheses
            while hdr.count("(") > hdr.count(")") and i + 1 < len(lines):
                i += 1
                hdr += " " + _strip_comments(lines[i]).strip()

            # Extract inside second parentheses: (zone start end ??? dim)
            # Example: (10 (2 1 100 0 3))
            try:
                inner = hdr.split("(10", 1)[1].strip()
                inner = inner.strip()
                # inner like: ((2 1 100 0 3)) or ( (2 1 100 0 3) )
                # pull the first '(' after (10
                p0 = inner.find("(")
                p1 = inner.find(")", p0 + 1)
                meta = inner[p0 + 1 : p1].split()
                if len(meta) >= 3:
                    zone = _parse_int_auto(meta[0])
                    start = _parse_int_auto(meta[1])
                    end = _parse_int_auto(meta[2])
                else:
                    zone = -1
                    start = -1
                    end = -1
            except Exception:
                zone, start, end = -1, -1, -1

            # read coords until we have (end-start+1) nodes or until ')'
            n_expected = max(0, end - start + 1) if (start > 0 and end >= start) else None
            coords = []
            j = i + 1
            while j < len(lines):
                s = _strip_comments(lines[j]).strip()
                if not s:
                    j += 1
                    continue
                if s.startswith(")"):
                    break
                # coordinates might be on same line multiple triples
                parts = s.replace("(", " ").replace(")", " ").split()
                # group into triples
                vals = []
                for p in parts:
                    try:
                        vals.append(float(p))
                    except Exception:
                        pass
                # chunk triples
                for k in range(0, len(vals), 3):
                    if k + 2 < len(vals):
                        coords.append(vals[k:k+3])
                j += 1
                if n_expected is not None and len(coords) >= n_expected:
                    # might still have more lines, but we got enough
                    # advance j until we hit a ')' to close the block
                    while j < len(lines):
                        s2 = _strip_comments(lines[j]).strip()
                        if s2.startswith(")"):
                            break
                        j += 1
                    break

            # assign coords to node ids start..end
            if start > 0 and end >= start and coords:
                nn = min(len(coords), end - start + 1)
                for idx in range(nn):
                    nid = start + idx
                    nodes[nid] = np.array(coords[idx], dtype=np.float64) * scale

            i = j  # continue from closing ')'
        i += 1

    if not nodes:
        raise RuntimeError(f"No nodes parsed from {msh_path}")

    # Build dense node array: Fluent node IDs are 1-based and (usually) contiguous
    max_id = max(nodes.keys())
    nodes_xyz = np.zeros((max_id, 3), dtype=np.float64)
    for nid, xyz in nodes.items():
        nodes_xyz[nid - 1, :] = xyz  # to 0-based

    # --- Parse faces for the requested wall zone ---
    faces: List[Tuple[int, int, int]] = []

    i = 0
    while i < len(lines):
        raw = _strip_comments(lines[i]).strip()
        if raw.startswith("(13"):
            hdr = raw
            while hdr.count("(") > hdr.count(")") and i + 1 < len(lines):
                i += 1
                hdr += " " + _strip_comments(lines[i]).strip()

            # header like: (13 (zone start end type ???))
            try:
                inner = hdr.split("(13", 1)[1].strip()
                p0 = inner.find("(")
                p1 = inner.find(")", p0 + 1)
                meta = inner[p0 + 1 : p1].split()
                zone = _parse_int_auto(meta[0]) if len(meta) > 0 else -1
                start = _parse_int_auto(meta[1]) if len(meta) > 1 else -1
                end = _parse_int_auto(meta[2]) if len(meta) > 2 else -1
            except Exception:
                zone, start, end = -1, -1, -1

            # Face connectivity lines follow until ')'
            j = i + 1
            if zone == wall_zone:
                while j < len(lines):
                    s = _strip_comments(lines[j]).strip()
                    if not s:
                        j += 1
                        continue
                    if s.startswith(")"):
                        break
                    # tokenization
                    parts = s.replace("(", " ").replace(")", " ").split()
                    if not parts:
                        j += 1
                        continue

                    # Two common patterns:
                    # A) n v1 v2 v3 [v4] c0 c1  (n=3 or 4)
                    # B) v1 v2 v3 c0 c1        (implicit triangles)
                    ints = []
                    for p in parts:
                        try:
                            ints.append(_parse_int_auto(p))
                        except Exception:
                            pass
                    if not ints:
                        j += 1
                        continue

                    if ints[0] in (3, 4):
                        n = ints[0]
                        verts = ints[1 : 1 + n]
                    else:
                        # assume triangle
                        verts = ints[0:3]
                        n = 3

                    if len(verts) >= 3:
                        # Fluent node ids are 1-based
                        v0, v1, v2 = verts[0] - 1, verts[1] - 1, verts[2] - 1
                        faces.append((v0, v1, v2))
                        if n == 4 and len(verts) == 4:
                            v3 = verts[3] - 1
                            # triangulate quad: (0,1,2) and (0,2,3)
                            faces.append((v0, v2, v3))
                    j += 1
            else:
                # skip lines until close
                while j < len(lines):
                    s = _strip_comments(lines[j]).strip()
                    if s.startswith(")"):
                        break
                    j += 1

            i = j
        i += 1

    if not faces:
        raise RuntimeError(
            f"No faces found for wall_zone={wall_zone} in {msh_path}. "
            f"Try a different --wall_zone."
        )

    faces_arr = np.array(faces, dtype=np.int64)
    return nodes_xyz, faces_arr


# ----------------------------
# Graph construction
# ----------------------------

def build_surface_graph(nodes_xyz: np.ndarray, faces: np.ndarray) -> Data:
    """
    Given global nodes and wall faces (global node indices), build a wall-only graph:
      - compress to wall nodes
      - edge_index from triangle adjacency
      - normals
      - x = [xyz_norm(3), normals(3), is_wall(1)]
    """
    # wall node set
    wall_nodes = np.unique(faces.reshape(-1))
    wall_nodes_sorted = np.sort(wall_nodes)
    global_to_local = {int(g): i for i, g in enumerate(wall_nodes_sorted)}

    pos = nodes_xyz[wall_nodes_sorted, :].astype(np.float32)  # [Nw,3]

    # remap faces to local indices
    local_faces = np.vectorize(lambda g: global_to_local[int(g)])(faces).astype(np.int64)  # [F,3]
    face = torch.from_numpy(local_faces.T.copy())  # PyG expects [3,F]

    # build unique undirected edges from faces
    e_set = set()
    for tri in local_faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in [(a, b), (b, c), (c, a)]:
            if u == v:
                continue
            if u < v:
                e_set.add((u, v))
            else:
                e_set.add((v, u))
    edges = np.array(list(e_set), dtype=np.int64)
    if edges.size == 0:
        raise RuntimeError("No edges constructed from faces.")
    # undirected edge_index
    edge_index = np.concatenate([edges, edges[:, ::-1]], axis=0).T  # [2, 2E]

    # normals (area-weighted vertex normals)
    vnorm = np.zeros_like(pos, dtype=np.float64)
    p = pos.astype(np.float64)
    for tri in local_faces:
        i0, i1, i2 = tri
        v0, v1, v2 = p[i0], p[i1], p[i2]
        n = np.cross(v1 - v0, v2 - v0)
        vnorm[i0] += n
        vnorm[i1] += n
        vnorm[i2] += n
    nrm = np.linalg.norm(vnorm, axis=1, keepdims=True)
    vnorm = vnorm / np.maximum(nrm, 1e-12)
    vnorm = vnorm.astype(np.float32)

    # bbox-normalized coordinates
    mn = pos.min(axis=0, keepdims=True)
    mx = pos.max(axis=0, keepdims=True)
    span = np.maximum(mx - mn, 1e-12)
    xyz_norm = ((pos - mn) / span).astype(np.float32)

    is_wall = np.ones((pos.shape[0], 1), dtype=np.float32)

    x = np.concatenate([xyz_norm, vnorm, is_wall], axis=1)  # [Nw,7]

    data = Data(
        x=torch.from_numpy(x),
        pos=torch.from_numpy(pos),
        edge_index=torch.from_numpy(edge_index.astype(np.int64)),
        face=face,
        global_node_ids=torch.from_numpy(wall_nodes_sorted.astype(np.int64)),
    )
    return data


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Preprocess Fluent legacy ASCII .msh -> wall surface graph (.pt)")
    ap.add_argument("--msh", required=True, type=str, help="Path to Fluent legacy ASCII .msh")
    ap.add_argument("--out_dir", required=True, type=str, help="Output directory")
    ap.add_argument("--out_name", default="graph.pt", type=str, help="Output .pt filename")
    ap.add_argument(
        "--wall_zone",
        default=19,
        type=int,
        help="Wall face zone id (Fluent face zone id). Default: 19",
    )
    ap.add_argument("--re", default=None, type=float, help="Optional Reynolds number to store in Data.re")
    ap.add_argument("--write_meta", action="store_true", help="Write meta.json with basic stats")
    args = ap.parse_args()

    msh_path = Path(args.msh)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes_xyz, faces = parse_fluent_ascii_msh_wall(msh_path, wall_zone=args.wall_zone)
    data = build_surface_graph(nodes_xyz, faces)

    if args.re is not None:
        data.re = torch.tensor(float(args.re), dtype=torch.float32)

    out_path = out_dir / args.out_name
    torch.save(data, out_path)

    if args.write_meta:
        meta = {
            "msh": str(msh_path),
            "wall_zone": int(args.wall_zone),
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.edge_index.shape[1]),
            "num_faces": int(data.face.shape[1]),
            "has_re": bool(args.re is not None),
            "out_pt": str(out_path),
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    print(f"[OK] Saved graph: {out_path}  (nodes={data.num_nodes}, edges={data.edge_index.shape[1]}, faces={data.face.shape[1]})")


if __name__ == "__main__":
    main()
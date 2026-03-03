"""
Rebuild carotid .pt graph edges with a smaller radius.

The original processing used neighbor_radius_multiplier=3.0, giving ~118 edges/node
and 50MB files. This script recomputes edges from the stored `pos` attribute using
radius_multiplier=1.5 (same logic as to_graph.py), giving ~8-12 edges/node and ~3MB files.

Run from repo root:
    python -m Bifurcation.rebuild_carotid_edges
"""

import sys
import torch
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

CAROTID_DIR = Path("Bifurcation/ProcessedData_carotid")
RADIUS_MULTIPLIER = 3.0  # gives ~10-12 edges/node (face adjacency + 1-ring)

def rebuild_edges(pos: torch.Tensor, multiplier: float):
    pts = pos.numpy()

    # Estimate mean face-centroid spacing from nearest neighbour distance
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)       # k=2: col 0 = self (0), col 1 = nearest
    mean_nn = float(dists[:, 1].mean())
    radius  = multiplier * mean_nn

    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if len(pairs) == 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 4), dtype=torch.float32)

    both = np.vstack([pairs, pairs[:, ::-1]])           # bidirectional
    edge_index = torch.from_numpy(both.T.astype(np.int64))

    # Edge attr: [distance, dx, dy, dz]
    src, dst    = both[:, 0], both[:, 1]
    diff        = pts[dst] - pts[src]
    dist        = np.linalg.norm(diff, axis=1, keepdims=True)
    edge_attr   = torch.from_numpy(np.hstack([dist, diff]).astype(np.float32))

    return edge_index, edge_attr


def main():
    all_pt = sorted(CAROTID_DIR.glob("*/systolic.pt"))
    print(f"Found {len(all_pt)} .pt files — rebuilding edges with radius_multiplier={RADIUS_MULTIPLIER}")

    total_before = total_after = 0

    for i, p in enumerate(all_pt):
        d = torch.load(p, weights_only=False)

        if not hasattr(d, "pos") or d.pos is None:
            print(f"  [{i+1}/{len(all_pt)}] SKIP {p.parent.name} — no pos attribute")
            continue

        old_edges = d.edge_index.shape[1]
        d.edge_index, d.edge_attr = rebuild_edges(d.pos, RADIUS_MULTIPLIER)
        new_edges = d.edge_index.shape[1]

        torch.save(d, p)

        size_mb = p.stat().st_size / 1024**2
        total_before += old_edges
        total_after  += new_edges

        sys.stdout.write(
            f"\r[{i+1}/{len(all_pt)}] {p.parent.name}: "
            f"{old_edges:,} -> {new_edges:,} edges  ({size_mb:.1f} MB)"
        )
        sys.stdout.flush()

    print(f"\n\nDone.")
    print(f"Edges/node before: {total_before / len(all_pt) / 15000:.0f}")
    print(f"Edges/node after:  {total_after  / len(all_pt) / 15000:.0f}")


if __name__ == "__main__":
    main()

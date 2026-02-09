"""
Preprocessing script for bifurcation WSS data.

Converts raw ANSYS mesh + Fluent CSV data into PyTorch Geometric graphs.

Process:
1. Load mesh file (.msh) → extract wall surface triangulation
2. For each Re: load CSV → match to mesh → extract WSS
3. Compute node features, edge features, targets
4. Save as ProcessedData/angle{A}_mesh{M}/Re{R}.pt
5. Compute normalization statistics from training set

Run: python -m Preprocessing.pre_process
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from Utils.mesh_utils import (
    load_fluent_csv,
    load_mesh_wall_surface,
    match_csv_to_mesh,
    create_pyg_graph,
    triangulate_point_cloud
)
import torch
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
import json


def process_one_case(angle: int, mesh_level: str, re: int, mesh_cache: dict) -> bool:
    """
    Process a single (angle, mesh_level, re) case.
    
    Args:
        angle: Bifurcation angle (30, 45, 60)
        mesh_level: "base", "750", or "1000"
        re: Reynolds number
        mesh_cache: Dictionary caching loaded meshes
        
    Returns:
        success: True if processing succeeded
    """
    # Get CSV path
    csv_path = config.get_csv_path(angle, mesh_level, re)
    if csv_path is None or not csv_path.exists():
        return False
    
    # Load CSV data first
    try:
        csv_coords, wss, pressure = load_fluent_csv(csv_path)
    except Exception as e:
        print(f"  WARNING: Error loading CSV: {e}")
        return False
    
    # Check if we have cached mesh for this (angle, mesh_level)
    mesh_key = (angle, mesh_level)
    
    # Try to load mesh file first, but fall back to CSV triangulation
    if mesh_key not in mesh_cache:
        vertices = None
        triangles = None
        
        # Attempt 1: Load from .msh file
        mesh_path = config.get_mesh_path(angle, mesh_level)
        if mesh_path is not None and mesh_path.exists():
            try:
                vertices, triangles = load_mesh_wall_surface(mesh_path)
            except Exception as e:
                print(f"  WARNING: Error loading mesh: {e}")
        
        # Attempt 2: Triangulate from CSV point cloud
        if vertices is None or triangles is None:
            # Use CSV coordinates directly
            vertices = csv_coords.copy()
            # Triangulate the 3D point cloud
            try:
                triangles = triangulate_point_cloud(csv_coords)
            except Exception as e:
                print(f"  WARNING: Error triangulating CSV: {e}")
                return False
        
        mesh_cache[mesh_key] = (vertices, triangles)
    
    vertices, triangles = mesh_cache[mesh_key]
    
    # Match CSV nodes to mesh vertices (for .msh case) or use directly (for CSV case)
    if len(vertices) == len(csv_coords):
        # CSV triangulation case - vertices are already matched
        wss_full = wss
    else:
        # .msh case - need to match
        try:
            indices = match_csv_to_mesh(csv_coords, vertices)
            wss_full = np.zeros((len(vertices), 3))
            wss_full[indices] = wss
        except Exception as e:
            print(f"  WARNING: Error matching CSV to mesh: {e}")
            return False
    
    # Create PyG graph
    graph_dict = create_pyg_graph(vertices, triangles, wss_full, re, angle)
    
    # Convert to PyG Data object
    data = Data(
        x=graph_dict['x'],
        edge_index=graph_dict['edge_index'],
        edge_attr=graph_dict['edge_attr'],
        y=graph_dict['y'],
        pos=graph_dict['pos'],
        flow_params=graph_dict['flow_params'],
        re=graph_dict['re'],
        angle=graph_dict['angle']
    )
    
    # Save to ProcessedData
    output_dir = config.processed_data_dir / f"angle{angle}_mesh{mesh_level}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"Re{re}.pt"
    
    torch.save(data, output_path)
    
    return True


def compute_normalization_stats():
    """
    Compute normalization statistics from all processed graphs.
    
    Computes mean and std for:
    - Node features (x)
    - Edge features (edge_attr)
    - Targets (y) - with sign-preserving log1p transform
    
    Saves to ProcessedData/normalization_stats.json
    """
    print("\nComputing normalization statistics...")
    
    # Collect all graphs
    all_x = []
    all_edge_attr = []
    all_y = []
    
    graph_files = list(config.processed_data_dir.rglob("*.pt"))
    
    for graph_path in tqdm(graph_files, desc="Loading graphs"):
        data = torch.load(graph_path, weights_only=False)
        all_x.append(data.x)
        all_edge_attr.append(data.edge_attr)
        all_y.append(data.y)
    
    # Concatenate
    all_x = torch.cat(all_x, dim=0)
    all_edge_attr = torch.cat(all_edge_attr, dim=0)
    all_y = torch.cat(all_y, dim=0)
    
    # Compute statistics
    x_mean = all_x.mean(dim=0)
    x_std = all_x.std(dim=0)
    
    edge_attr_mean = all_edge_attr.mean(dim=0)
    edge_attr_std = all_edge_attr.std(dim=0)
    
    # For targets: sign-preserving log1p transform first
    if config.use_log_transform:
        sign = torch.sign(all_y)
        log_y = sign * torch.log1p(torch.abs(all_y))
        y_mean = log_y.mean(dim=0)
        y_std = log_y.std(dim=0)
    else:
        y_mean = all_y.mean(dim=0)
        y_std = all_y.std(dim=0)
    
    # Package statistics
    stats = {
        'feature_mean': x_mean.tolist(),
        'feature_std': x_std.tolist(),
        'edge_attr_mean': edge_attr_mean.tolist(),
        'edge_attr_std': edge_attr_std.tolist(),
        'target_mean': y_mean.tolist(),
        'target_std': y_std.tolist(),
        'use_log_transform': config.use_log_transform
    }
    
    # Save
    stats_path = config.processed_data_dir / "normalization_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Saved normalization stats to {stats_path}")
    print(f"  Node features - mean: {x_mean.numpy()}")
    print(f"  Node features - std: {x_std.numpy()}")
    print(f"  Targets - mean: {y_mean.numpy()}")
    print(f"  Targets - std: {y_std.numpy()}")


def main():
    """Main preprocessing pipeline."""
    print("=" * 80)
    print("BIFURCATION WSS PREPROCESSING")
    print("=" * 80)
    print()
    
    # Create directories
    config.create_directories()
    
    # Get available cases
    cases = config.get_available_cases()
    print(f"Found {len(cases)} available cases")
    print()
    
    # Cache for loaded meshes (to avoid re-reading same mesh for each Re)
    mesh_cache = {}
    
    # Process all cases
    success_count = 0
    failed_cases = []
    
    for angle, mesh_level, re in tqdm(cases, desc="Processing cases"):
        success = process_one_case(angle, mesh_level, re, mesh_cache)
        if success:
            success_count += 1
        else:
            failed_cases.append((angle, mesh_level, re))
    
    print()
    print(f"[SUCCESS] Successfully processed: {success_count}/{len(cases)} cases")
    
    if failed_cases:
        print(f"[FAILED] Failed cases: {len(failed_cases)}")
        for angle, mesh_level, re in failed_cases[:10]:  # Show first 10
            print(f"    angle={angle}, mesh={mesh_level}, Re={re}")
        if len(failed_cases) > 10:
            print(f"    ... and {len(failed_cases) - 10} more")
    
    print()
    
    # Compute normalization statistics
    if success_count > 0:
        compute_normalization_stats()
    
    print()
    print("=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

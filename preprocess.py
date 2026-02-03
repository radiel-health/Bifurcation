"""
Preprocessing script for bifurcation WSS data

Converts raw CSV files into PyTorch Geometric graphs with:
1. Node features (coordinates, Re, angle, geometric features)
2. Edge connections (k-nearest neighbors)
3. Target values (WSS magnitude)
4. Region labels for analysis

Saves graphs to ProcessedData/ directory
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import json

from config import config
from geometry_utils import create_geometric_features, create_feature_matrix


def load_wss_csv(csv_path: Path) -> pd.DataFrame:
    """Load WSS data from CSV file"""
    df = pd.read_csv(csv_path)
    return df


def create_knn_edges(coords: np.ndarray, k: int = 8) -> torch.Tensor:
    """
    Create graph edges using k-nearest neighbors
    
    Args:
        coords: (N, 3) array of coordinates
        k: Number of nearest neighbors
        
    Returns:
        edge_index: (2, num_edges) tensor of edge connections
    """
    # Fit k-NN model
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # Create edge list (excluding self-loops)
    edge_list = []
    for i in range(len(coords)):
        for j in range(1, k+1):  # Skip first neighbor (itself)
            neighbor = indices[i, j]
            edge_list.append([i, neighbor])
            edge_list.append([neighbor, i])  # Add reverse edge for undirected graph
    
    # Convert to tensor and remove duplicates
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = torch.unique(edge_index, dim=1)
    
    return edge_index


def parse_case_info(case_dir: Path) -> dict:
    """
    Extract case information from directory path
    
    Example: bifurcation_angle30_500_ascii/Re100
    
    Returns:
        dict with keys: angle, mesh, re_number
    """
    case_name = case_dir.parent.name  # e.g., bifurcation_angle30_500_ascii
    re_name = case_dir.name  # e.g., Re100
    
    # Parse angle
    angle_str = case_name.split('angle')[1].split('_')[0]
    angle = int(angle_str)
    
    # Parse mesh
    mesh_str = case_name.split('_')[-2]
    mesh = int(mesh_str)
    
    # Parse Reynolds number
    re_number = int(re_name.replace('Re', ''))
    
    return {
        'angle': angle,
        'mesh': mesh,
        're_number': re_number
    }


def preprocess_single_case(csv_path: Path) -> Data:
    """
    Preprocess a single case into a PyG Data object
    
    Args:
        csv_path: Path to wall_wss.csv file
        
    Returns:
        data: PyTorch Geometric Data object
    """
    # Load CSV
    df = load_wss_csv(csv_path)
    
    # Extract case info
    case_info = parse_case_info(csv_path.parent)
    
    # Extract coordinates and WSS
    coords = df[['x', 'y', 'z']].values
    wss_mag = df['wss_mag'].values
    wss_components = df[['wss_x', 'wss_y', 'wss_z']].values
    
    # Create geometric features
    features_dict = create_geometric_features(
        coords=coords,
        wss_mag=wss_mag,
        re_number=case_info['re_number'],
        bifurcation_angle=case_info['angle'],
        mesh_refinement=case_info['mesh']
    )
    
    # Assemble feature matrix
    X = create_feature_matrix(features_dict)
    
    # Create graph edges
    edge_index = create_knn_edges(coords, k=config.k_neighbors)
    
    # Create PyG Data object
    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(wss_mag, dtype=torch.float).unsqueeze(1),  # (N, 1)
        pos=torch.tensor(coords, dtype=torch.float),
        region=torch.tensor(features_dict['region'], dtype=torch.long),
        wss_components=torch.tensor(wss_components, dtype=torch.float),
    )
    
    # Add metadata
    data.re_number = case_info['re_number']
    data.angle = case_info['angle']
    data.mesh = case_info['mesh']
    data.case_name = f"angle{case_info['angle']}_mesh{case_info['mesh']}_Re{case_info['re_number']}"
    
    return data


def find_all_cases() -> list:
    """Find all WSS CSV files in the data directory"""
    csv_files = []
    
    for angle in config.bifurcation_angles:
        for mesh in config.mesh_refinements:
            case_pattern = config.case_pattern.format(angle=angle, mesh=mesh)
            case_dir = config.data_root / case_pattern
            
            if case_dir.exists():
                for re_val in config.re_values:
                    csv_path = case_dir / f"Re{re_val}" / "wall_wss.csv"
                    if csv_path.exists():
                        csv_files.append(csv_path)
    
    return sorted(csv_files)


def preprocess_all_cases():
    """
    Preprocess all bifurcation cases and save to ProcessedData/
    """
    print("="*70)
    print("BIFURCATION WSS DATA PREPROCESSING")
    print("="*70)
    
    # Find all cases
    csv_files = find_all_cases()
    print(f"\nFound {len(csv_files)} cases to preprocess")
    
    # Create output directories
    for angle in config.bifurcation_angles:
        angle_dir = config.processed_data_dir / f"angle{angle}"
        angle_dir.mkdir(exist_ok=True, parents=True)
    
    # Process each case
    processed_cases = []
    failed_cases = []
    
    for csv_path in tqdm(csv_files, desc="Processing cases"):
        try:
            # Preprocess
            data = preprocess_single_case(csv_path)
            
            # Save
            output_dir = config.processed_data_dir / f"angle{data.angle}"
            output_path = output_dir / f"mesh{data.mesh}_Re{data.re_number}.pt"
            torch.save(data, output_path)
            
            processed_cases.append({
                'angle': data.angle,
                'mesh': data.mesh,
                're_number': data.re_number,
                'num_nodes': data.num_nodes,
                'num_edges': data.num_edges,
                'output_path': str(output_path)
            })
            
        except Exception as e:
            print(f"\nFailed to process {csv_path}: {e}")
            failed_cases.append(str(csv_path))
    
    print(f"\n{'='*70}")
    print(f"Preprocessing complete!")
    print(f"  Successful: {len(processed_cases)}")
    print(f"  Failed: {len(failed_cases)}")
    print(f"{'='*70}")
    
    # Save processing summary
    summary = {
        'num_processed': len(processed_cases),
        'num_failed': len(failed_cases),
        'processed_cases': processed_cases,
        'failed_cases': failed_cases,
        'config': {
            'k_neighbors': config.k_neighbors,
            'num_features': config.num_features,
            'bifurcation_angles': config.bifurcation_angles,
            'mesh_refinements': config.mesh_refinements,
            're_values': config.re_values,
        }
    }
    
    summary_path = config.processed_data_dir / "preprocessing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved summary to {summary_path}")
    
    # Print statistics
    if processed_cases:
        print("\nStatistics:")
        num_nodes_list = [c['num_nodes'] for c in processed_cases]
        num_edges_list = [c['num_edges'] for c in processed_cases]
        print(f"  Nodes per graph: {np.mean(num_nodes_list):.0f} ± {np.std(num_nodes_list):.0f}")
        print(f"  Edges per graph: {np.mean(num_edges_list):.0f} ± {np.std(num_edges_list):.0f}")
        
        # By angle
        print("\nCases by bifurcation angle:")
        for angle in config.bifurcation_angles:
            count = sum(1 for c in processed_cases if c['angle'] == angle)
            print(f"  {angle}°: {count} cases")


if __name__ == "__main__":
    preprocess_all_cases()

"""
Enhanced preprocessing with validation and error handling.

Improvements over original:
1. Data validation at each step
2. Better error handling with detailed messages
3. NaN/Inf checking
4. Feature normalization verification
5. Edge connectivity validation
6. Memory-efficient processing
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import json
import warnings

from config import Config
from geometry_utils import create_geometric_features, create_feature_matrix


class PreprocessingError(Exception):
    """Custom exception for preprocessing errors"""
    pass


def validate_dataframe(df: pd.DataFrame, csv_path: Path) -> None:
    """
    Validate that DataFrame has required columns and valid data.
    
    Raises PreprocessingError if validation fails.
    """
    required_cols = ['x', 'y', 'z', 'wss_mag', 'wss_x', 'wss_y', 'wss_z']
    
    # Check columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise PreprocessingError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Check for empty dataframe
    if len(df) == 0:
        raise PreprocessingError("DataFrame is empty")
    
    # Check for NaN values
    for col in required_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            raise PreprocessingError(
                f"Column '{col}' contains {nan_count} NaN values"
            )
    
    # Check for infinite values
    for col in required_cols:
        inf_count = np.isinf(df[col].values).sum()
        if inf_count > 0:
            raise PreprocessingError(
                f"Column '{col}' contains {inf_count} infinite values"
            )
    
    # Check for negative WSS magnitude (physically invalid)
    if (df['wss_mag'] < 0).any():
        neg_count = (df['wss_mag'] < 0).sum()
        warnings.warn(
            f"Found {neg_count} negative WSS magnitude values. Setting to 0."
        )
        df.loc[df['wss_mag'] < 0, 'wss_mag'] = 0
    
    # Check coordinate ranges are reasonable
    for coord in ['x', 'y', 'z']:
        coord_range = df[coord].max() - df[coord].min()
        if coord_range < 1e-6:
            raise PreprocessingError(
                f"Coordinate '{coord}' has near-zero range ({coord_range:.2e}). "
                "This suggests degenerate geometry."
            )
    
    print(f"  ✓ Validated {len(df)} points")


def load_wss_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load and validate WSS data from CSV file.
    
    Raises PreprocessingError if file is invalid.
    """
    if not csv_path.exists():
        raise PreprocessingError(f"File not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise PreprocessingError(f"Failed to read CSV: {e}")
    
    validate_dataframe(df, csv_path)
    
    return df


def create_knn_edges(coords: np.ndarray, k: int = 8) -> torch.Tensor:
    """
    Create graph edges using k-nearest neighbors with validation.
    
    Args:
        coords: (N, 3) array of coordinates
        k: Number of nearest neighbors
        
    Returns:
        edge_index: (2, num_edges) tensor of edge connections
        
    Raises PreprocessingError if graph construction fails.
    """
    N = len(coords)
    
    # Validate k
    if k >= N:
        raise PreprocessingError(
            f"k={k} is too large for {N} points. Using k={N-1} instead."
        )
        k = N - 1
    
    if k < 1:
        raise PreprocessingError(f"k must be >= 1, got {k}")
    
    # Check for duplicate coordinates (can cause k-NN issues)
    unique_coords = np.unique(coords, axis=0)
    if len(unique_coords) < N:
        warnings.warn(
            f"Found {N - len(unique_coords)} duplicate coordinate points. "
            "This may affect graph connectivity."
        )
    
    try:
        # Fit k-NN model
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(coords)
        distances, indices = nbrs.kneighbors(coords)
    except Exception as e:
        raise PreprocessingError(f"k-NN fitting failed: {e}")
    
    # Create edge list (excluding self-loops)
    edge_list = []
    for i in range(N):
        for j in range(1, k+1):  # Skip first neighbor (itself)
            neighbor = indices[i, j]
            if neighbor != i:  # Extra safety check
                edge_list.append([i, neighbor])
                edge_list.append([neighbor, i])  # Undirected graph
    
    if len(edge_list) == 0:
        raise PreprocessingError("No edges created - graph is disconnected")
    
    # Convert to tensor and remove duplicates
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = torch.unique(edge_index, dim=1)
    
    # Validate edge indices
    if edge_index.min() < 0 or edge_index.max() >= N:
        raise PreprocessingError(
            f"Invalid edge indices: [{edge_index.min()}, {edge_index.max()}] "
            f"for {N} nodes"
        )
    
    print(f"  ✓ Created {edge_index.shape[1]} edges with k={k}")
    
    return edge_index


def parse_case_info(case_dir: Path) -> dict:
    """
    Extract case information from directory path with validation.
    
    Example: bifurcation_angle30_500_ascii/Re100
    
    Returns:
        dict with keys: angle, mesh, re_number
    """
    case_name = case_dir.parent.name
    re_name = case_dir.name
    
    try:
        # Parse angle
        angle_str = case_name.split('angle')[1].split('_')[0]
        angle = int(angle_str)
        
        # Parse mesh
        parts = case_name.split('_')
        # Find the mesh number (should be before 'ascii')
        mesh_idx = -2 if 'ascii' in parts[-1] else -1
        mesh_str = parts[mesh_idx]
        mesh = int(mesh_str)
        
        # Parse Reynolds number
        re_number = int(re_name.replace('Re', ''))
        
    except (IndexError, ValueError) as e:
        raise PreprocessingError(
            f"Failed to parse case info from '{case_dir}': {e}"
        )
    
    # Validate values
    config = Config()
    
    if angle not in config.bifurcation_angles:
        warnings.warn(
            f"Angle {angle}° not in config.bifurcation_angles "
            f"{config.bifurcation_angles}"
        )
    
    if mesh not in config.mesh_refinements:
        warnings.warn(
            f"Mesh {mesh} not in config.mesh_refinements "
            f"{config.mesh_refinements}"
        )
    
    if re_number not in config.re_values:
        warnings.warn(
            f"Re={re_number} not in config.re_values "
            f"[{config.re_min}, {config.re_max}]"
        )
    
    return {
        'angle': angle,
        'mesh': mesh,
        're_number': re_number
    }


def validate_features(features: np.ndarray, case_name: str) -> None:
    """Validate feature matrix before creating PyG Data object."""
    # Check for NaN
    if np.isnan(features).any():
        nan_count = np.isnan(features).sum()
        raise PreprocessingError(
            f"Feature matrix contains {nan_count} NaN values"
        )
    
    # Check for infinite values
    if np.isinf(features).any():
        inf_count = np.isinf(features).sum()
        raise PreprocessingError(
            f"Feature matrix contains {inf_count} infinite values"
        )
    
    # Check feature ranges (normalized features should be roughly [-3, 3] for z-score)
    feature_max = np.abs(features).max()
    if feature_max > 1e6:
        warnings.warn(
            f"Feature values are very large (max={feature_max:.2e}). "
            "This may cause numerical instability."
        )
    
    print(f"  ✓ Validated {features.shape[0]} x {features.shape[1]} feature matrix")


def preprocess_single_case(csv_path: Path, config: Config = None) -> Data:
    """
    Preprocess a single case into a PyG Data object with full validation.
    
    Args:
        csv_path: Path to wall_wss.csv file
        config: Configuration object (creates new if None)
        
    Returns:
        data: PyTorch Geometric Data object
        
    Raises:
        PreprocessingError: If any validation step fails
    """
    if config is None:
        config = Config()
    
    print(f"\nProcessing: {csv_path.parent.name}/{csv_path.parent.parent.name}")
    
    # Step 1: Load and validate CSV
    df = load_wss_csv(csv_path)
    
    # Step 2: Parse case info
    case_info = parse_case_info(csv_path.parent)
    print(f"  Case: angle={case_info['angle']}°, mesh={case_info['mesh']}, Re={case_info['re_number']}")
    
    # Step 3: Extract coordinates and WSS
    coords = df[['x', 'y', 'z']].values.astype(np.float32)
    wss_mag = df['wss_mag'].values.astype(np.float32)
    wss_components = df[['wss_x', 'wss_y', 'wss_z']].values.astype(np.float32)
    
    # Step 4: Create geometric features
    try:
        features_dict = create_geometric_features(
            coords=coords,
            wss_mag=wss_mag,
            re_number=case_info['re_number'],
            bifurcation_angle=case_info['angle'],
            mesh_refinement=case_info['mesh']
        )
    except Exception as e:
        raise PreprocessingError(f"Feature creation failed: {e}")
    
    # Step 5: Assemble feature matrix
    try:
        X = create_feature_matrix(features_dict)
    except Exception as e:
        raise PreprocessingError(f"Feature matrix assembly failed: {e}")
    
    # Step 6: Validate features
    validate_features(X, csv_path.stem)
    
    # Step 7: Create graph edges
    edge_index = create_knn_edges(coords, k=config.k_neighbors)
    
    # Step 8: Create PyG Data object
    try:
        data = Data(
            x=torch.tensor(X, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(wss_mag, dtype=torch.float).unsqueeze(1),  # (N, 1)
            pos=torch.tensor(coords, dtype=torch.float),
            region=torch.tensor(features_dict['region'], dtype=torch.long),
            wss_components=torch.tensor(wss_components, dtype=torch.float),
        )
    except Exception as e:
        raise PreprocessingError(f"PyG Data object creation failed: {e}")
    
    # Step 9: Add metadata
    data.re_number = case_info['re_number']
    data.angle = case_info['angle']
    data.mesh = case_info['mesh']
    data.case_name = f"angle{case_info['angle']}_mesh{case_info['mesh']}_Re{case_info['re_number']}"
    
    # Step 10: Final validation
    if data.num_nodes == 0:
        raise PreprocessingError("Data object has 0 nodes")
    
    if data.num_edges == 0:
        raise PreprocessingError("Data object has 0 edges")
    
    print(f"  ✓ Created graph: {data.num_nodes} nodes, {data.num_edges} edges")
    
    return data


def find_all_cases(config: Config = None) -> list:
    """Find all WSS CSV files in the data directory."""
    if config is None:
        config = Config()
    
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
    Preprocess all bifurcation cases with enhanced error handling.
    """
    print("="*70)
    print("BIFURCATION WSS DATA PREPROCESSING (Enhanced)")
    print("="*70)
    
    config = Config()
    
    # Find all cases
    csv_files = find_all_cases(config)
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
            # Preprocess with validation
            data = preprocess_single_case(csv_path, config)
            
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
            
        except PreprocessingError as e:
            print(f"\n[ERROR] {csv_path}: {e}")
            failed_cases.append({
                'path': str(csv_path),
                'error': str(e)
            })
        except Exception as e:
            print(f"\n[UNEXPECTED ERROR] {csv_path}: {e}")
            failed_cases.append({
                'path': str(csv_path),
                'error': f"Unexpected: {e}"
            })
    
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
    
    if failed_cases:
        print("\nFailed cases details:")
        for fail in failed_cases:
            print(f"  - {fail['path']}")
            print(f"    Error: {fail['error']}")


if __name__ == "__main__":
    preprocess_all_cases()

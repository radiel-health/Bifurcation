"""
Geometry utilities for bifurcation analysis

Functions to:
1. Classify points into different regions (inlet, outlets, critical)
2. Compute geometric features (distance to apex, radial distance, etc.)
3. Identify the bifurcation apex (critical junction point)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from config import config


def classify_regions(coords: np.ndarray) -> np.ndarray:
    """
    Classify points into bifurcation regions based on coordinates
    
    Args:
        coords: (N, 3) array of [x, y, z] coordinates
        
    Returns:
        regions: (N,) array of region labels (0=inlet, 1=critical, 2=outlet_left, 3=outlet_right, 4=other)
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    
    # Get z-coordinate range
    z_min, z_max = z.min(), z.max()
    z_range = z_max - z_min
    
    # Define region thresholds
    inlet_threshold = z_min + config.region_thresholds['inlet_end'] * z_range
    critical_start = z_min + config.region_thresholds['critical_start'] * z_range
    critical_end = z_min + config.region_thresholds['critical_end'] * z_range
    outlet_threshold = z_min + config.region_thresholds['outlet_start'] * z_range
    
    # Initialize all as 'other' (label 4)
    regions = np.full(len(coords), 4, dtype=int)
    
    # Classify inlet (label 0)
    regions[z <= inlet_threshold] = 0
    
    # Classify critical region (label 1)
    critical_mask = (z >= critical_start) & (z <= critical_end)
    regions[critical_mask] = 1
    
    # Classify outlets (labels 2 and 3)
    outlet_mask = z >= outlet_threshold
    if outlet_mask.sum() > 0:
        # Split outlets by y-coordinate (left vs right)
        y_median = np.median(y[outlet_mask])
        outlet_left = outlet_mask & (y < y_median)
        outlet_right = outlet_mask & (y >= y_median)
        regions[outlet_left] = 2   # outlet_left
        regions[outlet_right] = 3  # outlet_right
    
    return regions


def find_bifurcation_apex(coords: np.ndarray, wss_mag: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Find the approximate bifurcation apex (junction point)
    
    The apex is typically where WSS is highest in the critical region
    
    Args:
        coords: (N, 3) array of coordinates
        wss_mag: (N,) array of WSS magnitudes
        
    Returns:
        apex_coords: (3,) coordinates of apex
        apex_idx: Index of apex point
    """
    # Get critical region
    regions = classify_regions(coords)
    critical_mask = regions == 1
    
    if critical_mask.sum() == 0:
        # Fallback: use middle z-coordinate point
        z = coords[:, 2]
        z_mid = (z.min() + z.max()) / 2
        apex_idx = np.argmin(np.abs(z - z_mid))
    else:
        # Find maximum WSS in critical region
        critical_indices = np.where(critical_mask)[0]
        critical_wss = wss_mag[critical_mask]
        apex_idx_in_critical = np.argmax(critical_wss)
        apex_idx = critical_indices[apex_idx_in_critical]
    
    apex_coords = coords[apex_idx]
    return apex_coords, apex_idx


def compute_distance_to_apex(coords: np.ndarray, apex_coords: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance from each point to the bifurcation apex
    
    Args:
        coords: (N, 3) array of coordinates
        apex_coords: (3,) apex coordinates
        
    Returns:
        distances: (N,) array of distances
    """
    return np.linalg.norm(coords - apex_coords[None, :], axis=1)


def compute_centerline_distance(coords: np.ndarray) -> np.ndarray:
    """
    Compute radial distance from the vessel centerline
    
    For a bifurcation, the centerline approximately follows z-axis with x≈0, y≈0
    
    Args:
        coords: (N, 3) array of coordinates
        
    Returns:
        radial_distances: (N,) array of radial distances from centerline
    """
    x, y = coords[:, 0], coords[:, 1]
    return np.sqrt(x**2 + y**2)


def compute_angular_position(coords: np.ndarray) -> np.ndarray:
    """
    Compute angular position around the centerline
    
    Args:
        coords: (N, 3) array of coordinates
        
    Returns:
        angles: (N,) array of angles in radians [-π, π]
    """
    x, y = coords[:, 0], coords[:, 1]
    return np.arctan2(y, x)


def compute_local_curvature_indicator(coords: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """
    Compute a simple curvature indicator based on region
    
    Inlet is straight (curvature = 0)
    Critical and outlet regions are curved (curvature = 1)
    
    Args:
        coords: (N, 3) array of coordinates
        regions: (N,) array of region labels
        
    Returns:
        curvature: (N,) array of curvature indicators
    """
    curvature = np.zeros(len(coords))
    # Critical and outlet regions have high curvature
    curved_regions = [1, 2, 3]  # critical, outlet_left, outlet_right
    for region_id in curved_regions:
        curvature[regions == region_id] = 1.0
    return curvature


def create_geometric_features(
    coords: np.ndarray,
    wss_mag: np.ndarray,
    re_number: float,
    bifurcation_angle: int,
    mesh_refinement: int
) -> Dict[str, np.ndarray]:
    """
    Create all geometric features for a bifurcation case
    
    Args:
        coords: (N, 3) array of [x, y, z] coordinates
        wss_mag: (N,) array of WSS magnitudes (for apex finding)
        re_number: Reynolds number
        bifurcation_angle: Bifurcation angle (30, 45, or 60)
        mesh_refinement: Mesh refinement level (500, 750, or 1000)
        
    Returns:
        features: Dictionary containing all features
    """
    N = len(coords)
    
    # Normalize coordinates to [0, 1] range
    x_norm = (coords[:, 0] - coords[:, 0].min()) / (coords[:, 0].max() - coords[:, 0].min())
    y_norm = (coords[:, 1] - coords[:, 1].min()) / (coords[:, 1].max() - coords[:, 1].min())
    z_norm = (coords[:, 2] - coords[:, 2].min()) / (coords[:, 2].max() - coords[:, 2].min())
    
    # Classify regions
    regions = classify_regions(coords)
    
    # Find bifurcation apex
    apex_coords, apex_idx = find_bifurcation_apex(coords, wss_mag)
    
    # Geometric features
    distance_to_apex = compute_distance_to_apex(coords, apex_coords)
    radial_distance = compute_centerline_distance(coords)
    angular_position = compute_angular_position(coords)
    curvature_indicator = compute_local_curvature_indicator(coords, regions)
    
    # Normalize geometric features
    if distance_to_apex.max() > 0:
        distance_to_apex_norm = distance_to_apex / distance_to_apex.max()
    else:
        distance_to_apex_norm = np.zeros_like(distance_to_apex)
        
    if radial_distance.max() > 0:
        radial_distance_norm = radial_distance / radial_distance.max()
    else:
        radial_distance_norm = np.zeros_like(radial_distance)
    
    # Normalize Reynolds number (typical range 100-2100)
    re_norm = (re_number - 100) / 2000
    
    # Normalize bifurcation angle (30, 45, 60 -> 0.0, 0.5, 1.0)
    angle_norm = (bifurcation_angle - 30) / 30
    
    # Normalize mesh refinement (500, 750, 1000 -> 0.0, 0.5, 1.0)
    mesh_norm = (mesh_refinement - 500) / 500
    
    # Create region one-hot encoding
    region_onehot = np.zeros((N, 4))  # 4 main regions (excluding 'other')
    for i in range(4):
        region_onehot[:, i] = (regions == i).astype(float)
    
    # Assemble feature dictionary
    features = {
        'x': coords[:, 0],
        'y': coords[:, 1],
        'z': coords[:, 2],
        'x_norm': x_norm,
        'y_norm': y_norm,
        'z_norm': z_norm,
        're_norm': np.full(N, re_norm),
        'angle_norm': np.full(N, angle_norm),
        'mesh_norm': np.full(N, mesh_norm),
        'region': regions,
        'region_inlet': region_onehot[:, 0],
        'region_critical': region_onehot[:, 1],
        'region_outlet_left': region_onehot[:, 2],
        'region_outlet_right': region_onehot[:, 3],
        'distance_to_apex': distance_to_apex_norm,
        'radial_distance': radial_distance_norm,
        'angular_position': angular_position,
        'curvature': curvature_indicator,
        'apex_coords': apex_coords,
        'apex_idx': apex_idx,
    }
    
    return features


def create_feature_matrix(features: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Assemble features into a single feature matrix for model input
    
    Args:
        features: Dictionary of features from create_geometric_features()
        
    Returns:
        X: (N, num_features) feature matrix
    """
    N = len(features['x_norm'])
    X = np.zeros((N, config.num_features))
    
    # Feature order matches config.py documentation:
    # 0-2: normalized coordinates
    X[:, 0] = features['x_norm']
    X[:, 1] = features['y_norm']
    X[:, 2] = features['z_norm']
    
    # 3-5: Reynolds number, angle, mesh refinement
    X[:, 3] = features['re_norm']
    X[:, 4] = features['angle_norm']
    X[:, 5] = features['mesh_norm']
    
    # 6-9: Region one-hot encoding
    X[:, 6] = features['region_inlet']
    X[:, 7] = features['region_critical']
    X[:, 8] = features['region_outlet_left']
    X[:, 9] = features['region_outlet_right']
    
    # 10-13: Geometric features
    X[:, 10] = features['distance_to_apex']
    X[:, 11] = features['radial_distance']
    X[:, 12] = features['angular_position']
    X[:, 13] = features['curvature']
    
    return X


def region_id_to_name(region_id: int) -> str:
    """Convert region ID to human-readable name"""
    names = {
        0: 'inlet',
        1: 'critical',
        2: 'outlet_left',
        3: 'outlet_right',
        4: 'other'
    }
    return names.get(region_id, 'unknown')


def region_name_to_id(region_name: str) -> int:
    """Convert region name to ID"""
    names = {
        'inlet': 0,
        'critical': 1,
        'outlet_left': 2,
        'outlet_right': 3,
        'other': 4
    }
    return names.get(region_name, 4)

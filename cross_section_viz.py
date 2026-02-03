"""
Cross-sectional visualization tools for bifurcation analysis

Takes 3D surface data and creates 2D cross-sectional views by slicing
at different z-positions. Useful for:
- Understanding WSS distribution around vessel perimeter
- Comparing inlet vs critical vs outlet regions
- Medical/clinical interpretation (like CT/MRI slices)

Note: This is for VISUALIZATION only - the model still works in full 3D
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from scipy.interpolate import griddata


def extract_cross_section(
    coords: np.ndarray,
    values: np.ndarray,
    z_position: float,
    z_tolerance: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract points near a specific z-plane
    
    Args:
        coords: (N, 3) array of [x, y, z] coordinates
        values: (N,) array of values (e.g., WSS magnitude)
        z_position: Z-coordinate to slice at
        z_tolerance: Distance tolerance for including points
        
    Returns:
        slice_coords: (M, 3) coordinates of points in slice
        slice_values: (M,) values at those points
    """
    z = coords[:, 2]
    
    # Find points close to this z-position
    mask = np.abs(z - z_position) <= z_tolerance
    
    slice_coords = coords[mask]
    slice_values = values[mask]
    
    return slice_coords, slice_values


def compute_angular_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Convert XY coordinates to angular positions around centerline
    
    Args:
        coords: (N, 3) array of coordinates
        
    Returns:
        angles: (N,) array of angles in degrees [0, 360]
    """
    x, y = coords[:, 0], coords[:, 1]
    angles = np.arctan2(y, x)  # [-π, π]
    angles_deg = np.degrees(angles)  # [-180, 180]
    angles_deg[angles_deg < 0] += 360  # [0, 360]
    return angles_deg


def visualize_cross_section(
    coords: np.ndarray,
    values: np.ndarray,
    z_position: float,
    title: str = "Cross Section",
    z_tolerance: float = 0.02,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Visualize a single cross-section as a 2D scatter plot
    
    Args:
        coords: (N, 3) array of coordinates
        values: (N,) array of values to plot
        z_position: Z-coordinate to slice at
        title: Plot title
        z_tolerance: Tolerance for slice extraction
        ax: Matplotlib axes (creates new if None)
        
    Returns:
        ax: Matplotlib axes with plot
    """
    # Extract slice
    slice_coords, slice_values = extract_cross_section(
        coords, values, z_position, z_tolerance
    )
    
    if len(slice_coords) == 0:
        print(f"Warning: No points found at z={z_position:.3f}")
        return ax
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot as 2D scatter (x, y plane)
    scatter = ax.scatter(
        slice_coords[:, 0], 
        slice_coords[:, 1],
        c=slice_values,
        s=20,
        cmap='hot',
        alpha=0.8
    )
    
    # Add centerline
    ax.plot(0, 0, 'k+', markersize=15, markeredgewidth=2, label='Centerline')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{title}\nz = {z_position:.3f} ({len(slice_coords)} points)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.colorbar(scatter, ax=ax, label='WSS Magnitude')
    
    return ax


def visualize_polar_cross_section(
    coords: np.ndarray,
    values: np.ndarray,
    z_position: float,
    title: str = "Cross Section (Polar)",
    z_tolerance: float = 0.02,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Visualize cross-section in polar coordinates (like looking down vessel)
    
    Shows WSS distribution around the vessel circumference
    
    Args:
        coords: (N, 3) array of coordinates
        values: (N,) array of values
        z_position: Z-coordinate to slice at
        title: Plot title
        z_tolerance: Tolerance for slice extraction
        ax: Matplotlib axes (creates new if None)
        
    Returns:
        ax: Matplotlib axes with polar plot
    """
    # Extract slice
    slice_coords, slice_values = extract_cross_section(
        coords, values, z_position, z_tolerance
    )
    
    if len(slice_coords) == 0:
        print(f"Warning: No points found at z={z_position:.3f}")
        return ax
    
    # Compute polar coordinates
    angles = compute_angular_coordinates(slice_coords)
    radii = np.sqrt(slice_coords[:, 0]**2 + slice_coords[:, 1]**2)
    
    # Create polar plot
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='polar')
    
    # Plot
    scatter = ax.scatter(
        np.radians(angles),
        radii,
        c=slice_values,
        s=20,
        cmap='hot',
        alpha=0.8
    )
    
    ax.set_title(f'{title}\nz = {z_position:.3f}', pad=20)
    
    plt.colorbar(scatter, ax=ax, label='WSS Magnitude', pad=0.1)
    
    return ax


def visualize_multiple_cross_sections(
    coords: np.ndarray,
    values: np.ndarray,
    z_positions: List[float],
    region_labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a figure with multiple cross-sections at different z-positions
    
    Args:
        coords: (N, 3) array of coordinates
        values: (N,) array of values
        z_positions: List of z-coordinates to slice at
        region_labels: Optional labels for each slice (e.g., "Inlet", "Apex")
        save_path: Optional path to save figure
        
    Returns:
        fig: Matplotlib figure
    """
    n_slices = len(z_positions)
    
    # Create subplot grid
    fig, axes = plt.subplots(2, n_slices, figsize=(5*n_slices, 10))
    if n_slices == 1:
        axes = axes.reshape(2, 1)
    
    for i, z_pos in enumerate(z_positions):
        # Label for this slice
        if region_labels and i < len(region_labels):
            label = region_labels[i]
        else:
            label = f"Slice {i+1}"
        
        # Cartesian view (top row)
        visualize_cross_section(
            coords, values, z_pos,
            title=f"{label} (Cartesian)",
            ax=axes[0, i]
        )
        
        # Polar view (bottom row)
        axes[1, i] = plt.subplot(2, n_slices, n_slices + i + 1, projection='polar')
        visualize_polar_cross_section(
            coords, values, z_pos,
            title=f"{label} (Polar)",
            ax=axes[1, i]
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved cross-sections to {save_path}")
    
    return fig


def create_angular_wss_profile(
    coords: np.ndarray,
    values: np.ndarray,
    z_position: float,
    z_tolerance: float = 0.02,
    n_bins: int = 36
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 1D profile of WSS around the vessel circumference
    
    Bins points by angle and averages WSS in each bin
    
    Args:
        coords: (N, 3) array of coordinates
        values: (N,) array of WSS values
        z_position: Z-coordinate to slice at
        z_tolerance: Tolerance for slice extraction
        n_bins: Number of angular bins (default 36 = 10° per bin)
        
    Returns:
        angles: (n_bins,) array of angles in degrees
        wss_profile: (n_bins,) array of average WSS at each angle
    """
    # Extract slice
    slice_coords, slice_values = extract_cross_section(
        coords, values, z_position, z_tolerance
    )
    
    if len(slice_coords) == 0:
        return np.array([]), np.array([])
    
    # Compute angles
    angles = compute_angular_coordinates(slice_coords)
    
    # Create bins
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Average WSS in each bin
    wss_profile = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (angles >= bin_edges[i]) & (angles < bin_edges[i+1])
        if mask.sum() > 0:
            wss_profile[i] = slice_values[mask].mean()
    
    return bin_centers, wss_profile


def plot_angular_profiles(
    coords: np.ndarray,
    values: np.ndarray,
    z_positions: List[float],
    region_labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot 1D angular WSS profiles for multiple cross-sections
    
    Shows how WSS varies around vessel circumference at each z-position
    
    Args:
        coords: (N, 3) array of coordinates
        values: (N,) array of WSS values
        z_positions: List of z-coordinates
        region_labels: Optional labels for each position
        save_path: Optional path to save figure
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, z_pos in enumerate(z_positions):
        # Compute profile
        angles, wss_profile = create_angular_wss_profile(coords, values, z_pos)
        
        if len(angles) == 0:
            continue
        
        # Label
        if region_labels and i < len(region_labels):
            label = f"{region_labels[i]} (z={z_pos:.2f})"
        else:
            label = f"z = {z_pos:.3f}"
        
        # Plot
        ax.plot(angles, wss_profile, marker='o', label=label, linewidth=2)
    
    ax.set_xlabel('Angle around centerline (degrees)')
    ax.set_ylabel('WSS Magnitude')
    ax.set_title('WSS Distribution Around Vessel Circumference')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 360)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved angular profiles to {save_path}")
    
    return fig


def analyze_case_with_cross_sections(
    csv_path: Path,
    output_dir: Optional[Path] = None
):
    """
    Complete cross-sectional analysis of a single case
    
    Creates visualizations at key z-positions:
    - Inlet (25% along z)
    - Critical/Apex (50% along z)  
    - Outlet (75% along z)
    
    Args:
        csv_path: Path to wall_wss.csv file
        output_dir: Directory to save visualizations
    """
    # Load data
    df = pd.read_csv(csv_path)
    coords = df[['x', 'y', 'z']].values
    wss_mag = df['wss_mag'].values
    
    # Get z-range
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    z_range = z_max - z_min
    
    # Key positions
    z_positions = [
        z_min + 0.25 * z_range,  # Inlet
        z_min + 0.50 * z_range,  # Critical/Apex
        z_min + 0.75 * z_range,  # Outlet
    ]
    region_labels = ['Inlet', 'Critical (Apex)', 'Outlet']
    
    # Output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "cross_section_results"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Case name
    case_name = csv_path.parent.parent.name + "_" + csv_path.parent.name
    
    print(f"\nAnalyzing {case_name}...")
    print(f"Z range: [{z_min:.3f}, {z_max:.3f}]")
    print(f"Cross-sections at: {[f'{z:.3f}' for z in z_positions]}")
    
    # 1. Multiple cross-sections (Cartesian + Polar)
    fig1 = visualize_multiple_cross_sections(
        coords, wss_mag, z_positions, region_labels,
        save_path=output_dir / f"{case_name}_cross_sections.png"
    )
    plt.close(fig1)
    
    # 2. Angular profiles
    fig2 = plot_angular_profiles(
        coords, wss_mag, z_positions, region_labels,
        save_path=output_dir / f"{case_name}_angular_profiles.png"
    )
    plt.close(fig2)
    
    print(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    # Example: analyze the same case we used for geometry analysis
    data_dir = Path("C:/Users/Rishabh/Downloads/Data/Bifurcation/results")
    csv_file = data_dir / "bifurcation_angle30_500_ascii" / "Re100" / "wall_wss.csv"
    
    if csv_file.exists():
        analyze_case_with_cross_sections(csv_file)
    else:
        print(f"File not found: {csv_file}")

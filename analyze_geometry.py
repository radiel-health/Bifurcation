"""
Analyze bifurcation geometry from WSS CSV files

This script helps understand:
1. Where different parts of the bifurcation are (inlet, outlets, critical region)
2. The coordinate ranges and bounds
3. Which points belong to which regions

The bifurcation has:
- One inlet (bottom/upstream)
- Two outlets (top branches)
- Critical region: The junction where the vessel splits (highest WSS typically)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D


def load_wss_data(csv_path):
    """Load WSS data from CSV file"""
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} wall points from {csv_path.name}")
    print(f"Columns: {list(df.columns)}")
    return df


def analyze_coordinates(df):
    """Analyze coordinate ranges to understand geometry"""
    print("\n" + "="*70)
    print("COORDINATE ANALYSIS")
    print("="*70)
    
    for coord in ['x', 'y', 'z']:
        print(f"\n{coord.upper()} coordinate:")
        print(f"  Range: [{df[coord].min():.6f}, {df[coord].max():.6f}]")
        print(f"  Mean: {df[coord].mean():.6f}")
        print(f"  Std: {df[coord].std():.6f}")
    
    print(f"\nTotal domain extent:")
    print(f"  X: {df['x'].max() - df['x'].min():.6f}")
    print(f"  Y: {df['y'].max() - df['y'].min():.6f}")
    print(f"  Z: {df['z'].max() - df['z'].min():.6f}")


def identify_bifurcation_regions(df):
    """
    Identify different regions of the bifurcation based on geometry
    
    Bifurcation structure (typical):
    - Inlet: Lower z values (upstream)
    - Outlets: Higher z values (two branches)
    - Critical region: Junction point (middle z, where it splits)
    """
    print("\n" + "="*70)
    print("BIFURCATION REGION IDENTIFICATION")
    print("="*70)
    
    # Analyze z-coordinate distribution (typically flow direction)
    z_min, z_max = df['z'].min(), df['z'].max()
    z_range = z_max - z_min
    
    # Define thresholds (these are heuristics that may need adjustment)
    inlet_threshold = z_min + 0.25 * z_range  # Bottom 25%
    critical_start = z_min + 0.35 * z_range   # Middle 35-60%
    critical_end = z_min + 0.60 * z_range
    outlet_threshold = z_min + 0.65 * z_range  # Top 65%+
    
    # Classify points
    df['region'] = 'other'
    df.loc[df['z'] <= inlet_threshold, 'region'] = 'inlet'
    df.loc[(df['z'] >= critical_start) & (df['z'] <= critical_end), 'region'] = 'critical'
    df.loc[df['z'] >= outlet_threshold, 'region'] = 'outlet'
    
    # For outlets, distinguish left and right based on y-coordinate
    y_median = df[df['region'] == 'outlet']['y'].median()
    df.loc[(df['region'] == 'outlet') & (df['y'] < y_median), 'region'] = 'outlet_left'
    df.loc[(df['region'] == 'outlet') & (df['y'] >= y_median), 'region'] = 'outlet_right'
    
    # Print statistics
    print("\nRegion distribution:")
    for region in ['inlet', 'critical', 'outlet_left', 'outlet_right', 'other']:
        count = (df['region'] == region).sum()
        pct = 100 * count / len(df)
        print(f"  {region:15s}: {count:5d} points ({pct:5.1f}%)")
    
    # Analyze WSS in each region
    print("\nWSS magnitude statistics by region:")
    for region in ['inlet', 'critical', 'outlet_left', 'outlet_right']:
        region_df = df[df['region'] == region]
        if len(region_df) > 0:
            wss_mean = region_df['wss_mag'].mean()
            wss_max = region_df['wss_mag'].max()
            wss_std = region_df['wss_mag'].std()
            print(f"  {region:15s}: mean={wss_mean:.6f}, max={wss_max:.6f}, std={wss_std:.6f}")
    
    return df


def visualize_geometry(df, output_path=None):
    """Create 3D visualization of the bifurcation with regions colored"""
    fig = plt.figure(figsize=(15, 12))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Color by region
    region_colors = {
        'inlet': 'blue',
        'critical': 'red',
        'outlet_left': 'green',
        'outlet_right': 'orange',
        'other': 'gray'
    }
    
    for region, color in region_colors.items():
        mask = df['region'] == region
        if mask.sum() > 0:
            ax1.scatter(df[mask]['x'], df[mask]['y'], df[mask]['z'], 
                       c=color, s=10, alpha=0.6, label=region)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z (flow direction)')
    ax1.set_title('Bifurcation Geometry - Regions')
    ax1.legend()
    
    # 3D scatter colored by WSS magnitude
    ax2 = fig.add_subplot(222, projection='3d')
    sc = ax2.scatter(df['x'], df['y'], df['z'], c=df['wss_mag'], 
                     s=10, alpha=0.6, cmap='hot')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z (flow direction)')
    ax2.set_title('WSS Magnitude Distribution')
    plt.colorbar(sc, ax=ax2, label='WSS Magnitude')
    
    # 2D projections
    # XY plane (top view)
    ax3 = fig.add_subplot(223)
    for region, color in region_colors.items():
        mask = df['region'] == region
        if mask.sum() > 0:
            ax3.scatter(df[mask]['x'], df[mask]['y'], 
                       c=color, s=5, alpha=0.5, label=region)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Top View (XY plane)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # YZ plane (side view)
    ax4 = fig.add_subplot(224)
    sc = ax4.scatter(df['y'], df['z'], c=df['wss_mag'], 
                     s=5, alpha=0.6, cmap='hot')
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z (flow direction)')
    ax4.set_title('Side View (YZ plane) - WSS Magnitude')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax4, label='WSS Magnitude')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_critical_region(df):
    """Deep dive into the critical bifurcation region"""
    print("\n" + "="*70)
    print("CRITICAL REGION ANALYSIS")
    print("="*70)
    
    critical_df = df[df['region'] == 'critical']
    
    if len(critical_df) == 0:
        print("No points in critical region!")
        return
    
    print(f"\nCritical region contains {len(critical_df)} points")
    
    # Find points with highest WSS (likely at the bifurcation apex)
    top_wss = critical_df.nlargest(10, 'wss_mag')
    print("\nTop 10 highest WSS points in critical region:")
    print(top_wss[['x', 'y', 'z', 'wss_mag']].to_string(index=False))
    
    # Analyze the apex/center of bifurcation
    apex_point = critical_df.loc[critical_df['wss_mag'].idxmax()]
    print(f"\nApproximate bifurcation apex (max WSS):")
    print(f"  Position: ({apex_point['x']:.6f}, {apex_point['y']:.6f}, {apex_point['z']:.6f})")
    print(f"  WSS magnitude: {apex_point['wss_mag']:.6f}")
    
    return critical_df


def main():
    """Main analysis workflow"""
    # Path to WSS data
    data_dir = Path("C:/Users/Rishabh/Downloads/Data/Bifurcation/results")
    
    # Analyze one case first (angle 30, 500 mesh, Re 100)
    wss_file = data_dir / "bifurcation_angle30_500_ascii" / "Re100" / "wall_wss.csv"
    
    if not wss_file.exists():
        print(f"ERROR: WSS file not found: {wss_file}")
        return
    
    # Load data
    df = load_wss_data(wss_file)
    
    # Analyze coordinates
    analyze_coordinates(df)
    
    # Identify regions
    df = identify_bifurcation_regions(df)
    
    # Analyze critical region
    critical_df = analyze_critical_region(df)
    
    # Visualize
    output_dir = Path(__file__).parent / "analysis_results"
    output_dir.mkdir(exist_ok=True)
    
    vis_path = output_dir / "bifurcation_geometry_analysis.png"
    visualize_geometry(df, vis_path)
    
    # Save annotated data
    output_csv = output_dir / "bifurcation_angle30_500_Re100_annotated.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved annotated data to {output_csv}")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()

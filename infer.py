"""
Inference Script: Run WSS predictions on preprocessed test cases

Usage:
    python infer_simple.py -re 1500 -ar 1x1 -output results/inference_test -plot

"""

import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from Models.model import WSSPredictor
from config import config


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = WSSPredictor(
        node_feature_dim=config.node_feature_dim,
        flow_param_dim=config.flow_param_dim,
        hidden_dim=config.hidden_dim,
        context_dim=config.context_dim,
        output_dim=config.target_dim,
        num_geom_layers=config.num_geom_layers,
        num_task_layers=config.num_task_layers,
        task_hidden_dim=config.task_hidden_dim,
        dropout=config.dropout_rate
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"[OK] Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    
    return model


def load_normalization_stats(filter_top_wall=False):
    """Load normalization statistics."""
    if filter_top_wall:
        stats_path = 'ProcessedData/normalization_stats_no_top.json'
    else:
        stats_path = 'ProcessedData/normalization_stats.json'
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return stats


def denormalize_wss(wss_normalized, norm_stats):
    """Denormalize WSS from log-space to physical units."""
    # Convert stats to tensors
    target_mean = torch.tensor(norm_stats['target_mean'], dtype=torch.float32)
    target_std = torch.tensor(norm_stats['target_std'], dtype=torch.float32)
    
    # Input is numpy array
    wss_normalized = torch.from_numpy(wss_normalized).float()
    
    # Reverse normalization: denormalized_log = normalized * std + mean
    sign = torch.sign(wss_normalized)
    log_mag = wss_normalized.abs() * target_std + target_mean
    
    # Reverse log1p transform: mag = exp(log) - 1
    mag = torch.expm1(log_mag)
    
    # Restore sign and convert to numpy
    wss_physical = (sign * mag).numpy()
    
    wss_x = wss_physical[:, 0]
    wss_y = wss_physical[:, 1]
    
    return wss_x, wss_y


def run_inference(model, graph_data, norm_stats, device, filter_top_wall=False):
    """Run model inference on a single graph."""
    # Apply top wall filtering if enabled
    if filter_top_wall:
        on_top = graph_data.x[:, 2].bool()  # Feature index 2 is 'on_top' flag
        keep_mask = ~on_top
        
        # Apply mask to node features, targets, and positions
        graph_data.x = graph_data.x[keep_mask]
        if hasattr(graph_data, 'y') and graph_data.y is not None:
            graph_data.y = graph_data.y[keep_mask]
        graph_data.pos = graph_data.pos[keep_mask]
        
        # Filter and remap edges
        from dataset import filter_edge_index
        graph_data.edge_index = filter_edge_index(graph_data.edge_index, keep_mask)
        
        # Update node count
        graph_data.num_nodes = keep_mask.sum().item()
    
    # Add batch attribute for single graph (all nodes belong to graph 0)
    graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)
    
    # Convert scalar attributes to tensors for batching compatibility
    if not isinstance(graph_data.re, torch.Tensor):
        graph_data.re = torch.tensor([graph_data.re], dtype=torch.float)
    if not isinstance(graph_data.lx, torch.Tensor):
        graph_data.lx = torch.tensor([graph_data.lx], dtype=torch.float)
    if not isinstance(graph_data.ly, torch.Tensor):
        graph_data.ly = torch.tensor([graph_data.ly], dtype=torch.float)
    
    graph_data = graph_data.to(device)
    
    with torch.no_grad():
        pred_normalized = model(graph_data).cpu().numpy()
    
    # Denormalize
    wss_x, wss_y = denormalize_wss(pred_normalized, norm_stats)
    wss_mag = np.sqrt(wss_x**2 + wss_y**2)
    
    # Extract coordinates
    coords = graph_data.pos.cpu().numpy()
    
    # Extract wall labels (indices 2-5: top, bottom, left, right)
    wall_labels = graph_data.x[:, 2:6].cpu().numpy()  # [top, bottom, left, right]
    
    # Identify which walls are present
    if filter_top_wall:
        walls_present = "3 walls (bottom/left/right)"
    else:
        walls_present = "4 walls (top/bottom/left/right)"
    
    results = {
        'coordinates': coords,
        'WSS_x': wss_x,
        'WSS_y': wss_y,
        'WSS_magnitude': wss_mag,
        'wall_labels': wall_labels,
        're': int(graph_data.re[0]),
        'lx': float(graph_data.lx[0]),
        'ly': float(graph_data.ly[0]),
        'filter_top_wall': filter_top_wall
    }
    
    print(f"\n[OK] Inference complete for Re={int(graph_data.re[0])}, AR={graph_data.lx[0]/graph_data.ly[0]:.2f} ({walls_present})")
    print(f"  WSS_x range: [{wss_x.min():.6e}, {wss_x.max():.6e}] Pa")
    print(f"  WSS_y range: [{wss_y.min():.6e}, {wss_y.max():.6e}] Pa")
    print(f"  Magnitude range: [{wss_mag.min():.6e}, {wss_mag.max():.6e}] Pa")
    
    return results


def save_results(results, output_dir):
    """Save predictions to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame({
        'x': results['coordinates'][:, 0],
        'y': results['coordinates'][:, 1],
        'WSS_x': results['WSS_x'],
        'WSS_y': results['WSS_y'],
        'WSS_magnitude': results['WSS_magnitude'],
        'wall_top': results['wall_labels'][:, 0],
        'wall_bottom': results['wall_labels'][:, 1],
        'wall_left': results['wall_labels'][:, 2],
        'wall_right': results['wall_labels'][:, 3]
    })
    
    re = results['re']
    ar = results['lx'] / results['ly']
    csv_path = output_dir / f'predictions_Re{int(re)}_AR{ar:.2f}.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"\n[OK] Results saved to: {csv_path}")
    return csv_path


def plot_results(results, output_dir):
    """Generate WSS visualization."""
    output_dir = Path(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    coords = results['coordinates']
    re = results['re']
    ar = results['lx'] / results['ly']
    
    # WSS_x
    ax = axes[0]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=results['WSS_x'],
                        cmap='RdBu_r', s=30, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title(f'WSS_x\\nRe={int(re)}, AR={ar:.2f}', fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='WSS_x (Pa)')
    
    # WSS_y
    ax = axes[1]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=results['WSS_y'],
                        cmap='RdBu_r', s=30, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title(f'WSS_y\\nRe={int(re)}, AR={ar:.2f}', fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='WSS_y (Pa)')
    
    # Magnitude
    ax = axes[2]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=results['WSS_magnitude'],
                        cmap='viridis', s=30, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title(f'WSS Magnitude\\nRe={int(re)}, AR={ar:.2f}', fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Magnitude (Pa)')
    
    plt.tight_layout()
    
    plot_path = output_dir / f'wss_prediction_Re{int(re)}_AR{ar:.2f}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Visualization saved to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='WSS Prediction Inference')
    parser.add_argument('-re', type=int, required=True, help='Reynolds number')
    parser.add_argument('-ar', type=str, required=True, choices=['1x1', '1x2', '2x1'],
                       help='Aspect ratio')
    parser.add_argument('-output', type=str, required=True, help='Output directory')
    parser.add_argument('-plot', action='store_true', help='Generate visualization')
    parser.add_argument('-model', type=str, default='Models/best_model.pt',
                       help='Model checkpoint path')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("WSS PREDICTION INFERENCE")
    print("="*70)
    print(f"Device: {device}")
    print(f"Reynolds: {args.re}")
    print(f"Aspect Ratio: {args.ar}")
    if config.filter_top_wall:
        print(f"⚠ Top wall filtering: ENABLED (3 walls only)")
    else:
        print(f"Top wall filtering: DISABLED (4 walls)")
    print("="*70)
    
    # Load model
    model = load_model(args.model, device)
    norm_stats = load_normalization_stats(filter_top_wall=config.filter_top_wall)
    
    # Load preprocessed graph
    graph_path = config.processed_data_dir / args.ar / f'Re{args.re}.pt'
    if not graph_path.exists():
        print(f"\\n Error: Preprocessed graph not found: {graph_path}")
        print(f"   Available graphs:")
        ar_dir = config.processed_data_dir / args.ar
        if ar_dir.exists():
            for pt_file in sorted(ar_dir.glob('Re*.pt')):
                print(f"     - {pt_file.name}")
        return
    
    print(f"\nLoading graph: {graph_path}")
    graph_data = torch.load(graph_path, weights_only=False)
    print(f"[OK] Loaded graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    # Run inference
    results = run_inference(model, graph_data, norm_stats, device, filter_top_wall=config.filter_top_wall)
    
    # Save results
    save_results(results, args.output)
    
    # Optional plot
    if args.plot:
        plot_results(results, args.output)
    
    print("\n" + "="*70)
    print("INFERENCE COMPLETE!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

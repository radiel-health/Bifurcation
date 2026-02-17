"""
Inference Script: Run WSS predictions on 3D point cloud test cases

Usage:
    python infer.py -re 1500 -graph mesh_Re1500.pt -output results/inference_test -plot
"""

import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from model import WSSPredictor
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


def load_normalization_stats():
    """Load normalization statistics."""
    stats_path = config.processed_data_dir / "normalization_stats.json"
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return stats


def denormalize_wss(wss_normalized, norm_stats):
    """Denormalize WSS from log-space to physical units."""
    target_mean = torch.tensor(norm_stats['target_mean'], dtype=torch.float32)
    target_std = torch.tensor(norm_stats['target_std'], dtype=torch.float32)
    
    wss_normalized = torch.from_numpy(wss_normalized).float()
    
    sign = torch.sign(wss_normalized)
    log_mag = wss_normalized.abs() * target_std + target_mean
    
    mag = torch.expm1(log_mag)
    
    wss_physical = (sign * mag).numpy()
    
    return wss_physical


def run_inference(model, graph_data, norm_stats, device):
    """Run model inference on a single graph."""
    num_nodes = graph_data.x.shape[0]
    
    graph_data.batch = torch.zeros(num_nodes, dtype=torch.long)
    
    if not isinstance(graph_data.flow_params, torch.Tensor):
        graph_data.flow_params = torch.tensor([graph_data.flow_params.item()], dtype=torch.float)
    
    graph_data = graph_data.to(device)
    
    with torch.no_grad():
        pred_normalized = model(graph_data).cpu().numpy()
    
    wss_physical = denormalize_wss(pred_normalized, norm_stats)
    
    wss_x = wss_physical[:, 0]
    wss_y = wss_physical[:, 1]
    wss_z = wss_physical[:, 2]
    wss_mag = np.sqrt(wss_x**2 + wss_y**2 + wss_z**2)
    
    coords = graph_data.pos.cpu().numpy()
    
    re = float(graph_data.flow_params[0])
    
    results = {
        'coordinates': coords,
        'WSS_x': wss_x,
        'WSS_y': wss_y,
        'WSS_z': wss_z,
        'WSS_magnitude': wss_mag,
        're': int(re),
    }
    
    print(f"\n[OK] Inference complete for Re={int(re)}")
    print(f"  WSS_x range: [{wss_x.min():.6e}, {wss_x.max():.6e}] Pa")
    print(f"  WSS_y range: [{wss_y.min():.6e}, {wss_y.max():.6e}] Pa")
    print(f"  WSS_z range: [{wss_z.min():.6e}, {wss_z.max():.6e}] Pa")
    print(f"  Magnitude range: [{wss_mag.min():.6e}, {wss_mag.max():.6e}] Pa")
    
    return results


def save_results(results, output_dir, graph_name):
    """Save predictions to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    coords = results['coordinates']
    
    df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'z': coords[:, 2],
        'WSS_x': results['WSS_x'],
        'WSS_y': results['WSS_y'],
        'WSS_z': results['WSS_z'],
        'WSS_magnitude': results['WSS_magnitude'],
    })
    
    csv_path = output_dir / f'predictions_{graph_name}.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"\n[OK] Results saved to: {csv_path}")
    return csv_path


def plot_results(results, output_dir, graph_name):
    """Generate WSS visualization (3D scatter plots)."""
    output_dir = Path(output_dir)
    coords = results['coordinates']
    re = results['re']
    
    fig = plt.figure(figsize=(18, 5))
    
    ax = fig.add_subplot(131, projection='3d')
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                        c=results['WSS_x'], cmap='RdBu_r', s=20, edgecolors='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'WSS_x\nRe={int(re)}')
    plt.colorbar(scatter, ax=ax, label='WSS_x (Pa)', shrink=0.7)
    
    ax = fig.add_subplot(132, projection='3d')
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                        c=results['WSS_y'], cmap='RdBu_r', s=20, edgecolors='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'WSS_y\nRe={int(re)}')
    plt.colorbar(scatter, ax=ax, label='WSS_y (Pa)', shrink=0.7)
    
    ax = fig.add_subplot(133, projection='3d')
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                        c=results['WSS_magnitude'], cmap='viridis', s=20, edgecolors='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'WSS Magnitude\nRe={int(re)}')
    plt.colorbar(scatter, ax=ax, label='Magnitude (Pa)', shrink=0.7)
    
    plt.tight_layout()
    
    plot_path = output_dir / f'wss_prediction_{graph_name}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Visualization saved to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='WSS Prediction Inference (3D Point Cloud)')
    parser.add_argument('-re', type=int, help='Reynolds number (optional if loading from filename)')
    parser.add_argument('-graph', type=str, required=True, help='Path to preprocessed graph .pt file')
    parser.add_argument('-output', type=str, default='results/inference', help='Output directory')
    parser.add_argument('-plot', action='store_true', help='Generate visualization')
    parser.add_argument('-model', type=str, default='Models/best_model.pt', help='Model checkpoint path')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("WSS PREDICTION INFERENCE (3D Point Cloud)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Graph: {args.graph}")
    print("="*70)
    
    model = load_model(args.model, device)
    norm_stats = load_normalization_stats()
    
    graph_path = Path(args.graph)
    if not graph_path.exists():
        print(f"\n[ERROR] Graph not found: {graph_path}")
        return
    
    print(f"\nLoading graph: {graph_path}")
    graph_data = torch.load(graph_path, weights_only=False)
    print(f"[OK] Loaded graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    results = run_inference(model, graph_data, norm_stats, device)
    
    graph_name = graph_path.stem
    save_results(results, args.output, graph_name)
    
    if args.plot:
        plot_results(results, args.output, graph_name)
    
    print("\n" + "="*70)
    print("INFERENCE COMPLETE!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

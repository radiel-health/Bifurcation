"""
Inference script for single-case WSS prediction.

Run: python infer.py --angle 30 --mesh 750 --re 100 --output predictions.csv
"""

import torch
import pandas as pd
import argparse
from pathlib import Path
import json

from Bifurcation.config import config
from dataset import BifurcationWSSDataset
from Models.model import BifurcationWSSPredictor
from evaluate import load_ensemble_models, ensemble_predict


def infer_single_case(angle, mesh_level,re, models, device, dataset):
    """
    Run inference on a single case.
    
    Args:
        angle: Bifurcation angle (30, 45, 60)
        mesh_level: "base", "750", or "1000"
        re: Reynolds number
        models: List of ensemble models
        device: torch device
        dataset: Dataset object for denormalization
        
    Returns:
        results_df: DataFrame with predictions
    """
    # Find the processed graph file
    graph_path = config.processed_data_dir / f"angle{angle}_mesh{mesh_level}" / f"Re{re}.pt"
    
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Preprocessed graph not found: {graph_path}\n"
            f"Run preprocessing first: python -m Preprocessing.pre_process"
        )
    
    # Load graph
    print(f"Loading graph: {graph_path}")
    data = torch.load(graph_path, weights_only=False)
    
    # Normalize (manually apply same normalization as dataset)
    stats_path = config.processed_data_dir / "normalization_stats.json"
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    feature_mean = torch.tensor(stats['feature_mean'])
    feature_std = torch.tensor(stats['feature_std'])
    edge_attr_mean = torch.tensor(stats['edge_attr_mean'])
    edge_attr_std = torch.tensor(stats['edge_attr_std'])
    
    data.x = (data.x - feature_mean) / (feature_std + 1e-8)
    data.edge_attr = (data.edge_attr - edge_attr_mean) / (edge_attr_std + 1e-8)
    
    # Move to device and add batch dimension
    data = data.to(device)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    
    # Ensemble prediction
    print("Running ensemble inference...")
    mean_pred, std_pred = ensemble_predict(models, data, device)
    
    # Denormalize predictions
    mean_pred_phys = dataset.denormalize_targets(mean_pred.cpu())
    std_pred_phys = std_pred.cpu()  # Std is already in meaningful units
    
    # Compute magnitude
    wss_mag = torch.norm(mean_pred_phys, dim=1)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'x': data.pos[:, 0].cpu().numpy(),
        'y': data.pos[:, 1].cpu().numpy(),
        'z': data.pos[:, 2].cpu().numpy(),
        'wss_x_pred': mean_pred_phys[:, 0].numpy(),
        'wss_y_pred': mean_pred_phys[:, 1].numpy(),
        'wss_z_pred': mean_pred_phys[:, 2].numpy(),
        'wss_mag_pred': wss_mag.numpy(),
        'wss_x_std': std_pred_phys[:, 0].numpy(),
        'wss_y_std': std_pred_phys[:, 1].numpy(),
        'wss_z_std': std_pred_phys[:, 2].numpy()
    })
    
    # Add ground truth if available
    if hasattr(data, 'y') and data.y is not None:
        y_true_phys = dataset.denormalize_targets(data.y.cpu())
        wss_true_mag = torch.norm(y_true_phys, dim=1)
        
        results_df['wss_x_true'] = y_true_phys[:, 0].numpy()
        results_df['wss_y_true'] = y_true_phys[:, 1].numpy()
        results_df['wss_z_true'] = y_true_phys[:, 2].numpy()
        results_df['wss_mag_true'] = wss_true_mag.numpy()
        
        # Compute errors
        results_df['error_mag'] = (results_df['wss_mag_pred'] - results_df['wss_mag_true']).abs()
    
    print(f"✓ Prediction complete: {len(results_df)} nodes")
    print(f"  WSS magnitude range: [{results_df['wss_mag_pred'].min():.4e}, {results_df['wss_mag_pred'].max():.4e}] Pa")
    
    return results_df


def main(args):
    """Main inference function."""
    print("=" * 80)
    print("BIFURCATION WSS PREDICTION - INFERENCE")
    print("=" * 80)
    print()
    
    print(f"Case: angle={args.angle}°, mesh={args.mesh}, Re={args.re}")
    print()
    
    # Device
    device = config.get_device()
    print(f"Using device: {device}")
    print()
    
    # Load models
    models = load_ensemble_models(device)
    print()
    
    # Create dataset object for denormalization
    dataset = BifurcationWSSDataset(split='train', strategy='random', normalize=False)
    # Load normalization stats manually
    stats_path = config.processed_data_dir / "normalization_stats.json"
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    dataset.target_mean = torch.tensor(stats['target_mean'])
    dataset.target_std = torch.tensor(stats['target_std'])
    dataset.use_log_transform = stats['use_log_transform']
    
    # Run inference
    results_df = infer_single_case(
        args.angle,
        args.mesh,
        args.re,
        models,
        device,
        dataset
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print()
    print(f"✓ Saved predictions to {output_path}")
    print()
    
    # Print summary statistics
    print("Summary Statistics:")
    print(f"  Mean WSS magnitude: {results_df['wss_mag_pred'].mean():.6f} Pa")
    print(f"  Std WSS magnitude:  {results_df['wss_mag_pred'].std():.6f} Pa")
    print(f"  Max WSS magnitude:  {results_df['wss_mag_pred'].max():.6f} Pa")
    
    if 'wss_mag_true' in results_df.columns:
        mae = results_df['error_mag'].mean()
        print(f"\n  MAE vs ground truth: {mae:.6f} Pa")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer WSS for a single bifurcation case")
    parser.add_argument('--angle', type=int, required=True, choices=[30, 45, 60],
                        help="Bifurcation angle (degrees)")
    parser.add_argument('--mesh', type=str, required=True, choices=['base', '750', '1000'],
                        help="Mesh refinement level")
    parser.add_argument('--re', type=int, required=True,
                        help="Reynolds number")
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help="Output CSV file path")
    
    args = parser.parse_args()
    main(args)

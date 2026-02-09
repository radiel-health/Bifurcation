"""
Evaluation script for bifurcation WSS prediction.

Loads ensemble models, evaluates on test set, and computes comprehensive metrics.

Run: python evaluate.py [--strategy re_interp] [--use_calibration]
"""

import torch
import numpy as np
import json
from pathlib import Path
import argparse
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

from Bifurcation.config import config
from dataset import BifurcationWSSDataset, get_dataloaders
from Models.model import BifurcationWSSPredictor


def load_ensemble_models(device):
    """
    Load all fold models for ensemble prediction.
    
    Returns:
        models: List of loaded models
    """
    models = []
    fold_paths = sorted(config.checkpoint_dir.glob("best_state_fold_*.pt"))
    
    if len(fold_paths) == 0:
        raise FileNotFoundError(
            f"No trained models found in {config.checkpoint_dir}. "
            f"Run training first: python train.py"
        )
    
    print(f"Loading {len(fold_paths)} fold models...")
    
    for fold_path in fold_paths:
        model = BifurcationWSSPredictor(
            original_node_feat_dim=config.node_feature_dim,
            edge_channels=config.edge_feature_dim,
            aggregated_edge_feat_dim=config.aggregated_edge_feat_dim,
            hidden_gcn_dim=config.hidden_gcn_dim,
            out_channels=config.out_channels,
            num_gcn_layers=config.num_gcn_layers,
            context_dim=config.context_dim
        ).to(device)
        
        checkpoint = torch.load(fold_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
    
    print(f"[OK] Loaded {len(models)} models")
    return models


def ensemble_predict(models, batch, device):
    """
    Ensemble prediction from multiple models.
    
    Args:
        models: List of trained models
        batch: PyG Batch object
        device: torch device
        
    Returns:
        mean_pred: [num_nodes, 3] mean prediction
        std_pred: [num_nodes, 3] prediction std (uncertainty)
    """
    predictions = []
    
    with torch.no_grad():
        for model in models:
            pred = model(batch)
            predictions.append(pred)
    
    predictions = torch.stack(predictions)  # [n_models, num_nodes, 3]
    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)
    
    return mean_pred, std_pred


def compute_metrics(y_true, y_pred, dataset):
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: [N, 3] ground truth WSS (normalized)
        y_pred: [N, 3] predicted WSS (normalized)
        dataset: Dataset object (for denormalization)
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Denormalize to physical units
    y_true_phys = dataset.denormalize_targets(y_true).cpu().numpy()
    y_pred_phys = dataset.denormalize_targets(y_pred).cpu().numpy()
    
    # Compute WSS magnitude
    wss_true_mag = np.linalg.norm(y_true_phys, axis=1)
    wss_pred_mag = np.linalg.norm(y_pred_phys, axis=1)
    
    metrics = {}
    
    # Component-wise metrics
    for i, component in enumerate(['wss_x', 'wss_y', 'wss_z']):
        metrics[component] = {
            'mae': mean_absolute_error(y_true_phys[:, i], y_pred_phys[:, i]),
            'rmse': np.sqrt(mean_squared_error(y_true_phys[:, i], y_pred_phys[:, i])),
            'r2': r2_score(y_true_phys[:, i], y_pred_phys[:, i])
        }
    
    # Magnitude metrics
    metrics['wss_magnitude'] = {
        'mae': mean_absolute_error(wss_true_mag, wss_pred_mag),
        'rmse': np.sqrt(mean_squared_error(wss_true_mag, wss_pred_mag)),
        'r2': r2_score(wss_true_mag, wss_pred_mag)
    }
    
    # Relative error (avoid division by very small values)
    mask = np.abs(wss_true_mag) > 1e-6
    if mask.sum() > 0:
        rel_error = np.abs(wss_pred_mag[mask] - wss_true_mag[mask]) / wss_true_mag[mask]
        metrics['relative_error'] = {
            'median': np.median(rel_error) * 100,  # Percentage
            'mean': np.mean(rel_error) * 100,
            'p95': np.percentile(rel_error, 95) * 100
        }
    
    return metrics, y_true_phys, y_pred_phys, wss_true_mag, wss_pred_mag


def plot_scatter(y_true_mag, y_pred_mag, save_path):
    """Create truth vs prediction scatter plot."""
    plt.figure(figsize=(8, 8))
    
    plt.scatter(y_true_mag, y_pred_mag, alpha=0.3, s=1)
    
    # 1:1 line
    max_val = max(y_true_mag.max(), y_pred_mag.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
    
    plt.xlabel('True WSS Magnitude (Pa)')
    plt.ylabel('Predicted WSS Magnitude (Pa)')
    plt.title('WSS Prediction: Truth vs Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"[OK] Saved scatter plot to {save_path}")


def evaluate(args):
    """Main evaluation function."""
    print("=" * 80)
    print("BIFURCATION WSS PREDICTION - EVALUATION")
    print("=" * 80)
    print()
    
    # Device
    device = config.get_device()
    print(f"Using device: {device}")
    print()
    
    # Load models
    models = load_ensemble_models(device)
    print()
    
    # Load test data
    print(f"Loading test data with strategy: {args.strategy}")
    test_dataset = BifurcationWSSDataset(split='test', strategy=args.strategy)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )
    print(f"Test set size: {len(test_dataset)} graphs")
    print()
    
    # Collect predictions
    print("Running inference on test set...")
    all_y_true = []
    all_y_pred = []
    all_y_std = []
    
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        
        # Ensemble prediction
        mean_pred, std_pred = ensemble_predict(models, batch, device)
        
        all_y_true.append(batch.y.cpu())
        all_y_pred.append(mean_pred.cpu())
        all_y_std.append(std_pred.cpu())
    
    # Concatenate
    y_true = torch.cat(all_y_true, dim=0)
    y_pred = torch.cat(all_y_pred, dim=0)
    y_std = torch.cat(all_y_std, dim=0)
    
    print(f"[OK] Collected {len(y_true)} node predictions")
    print()
    
    # Compute metrics
    print("Computing metrics...")
    metrics, y_true_phys, y_pred_phys, wss_true_mag, wss_pred_mag = compute_metrics(
        y_true, y_pred, test_dataset
    )
    
    # Print metrics
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    print()
    
    print("WSS Components:")
    for component in ['wss_x', 'wss_y', 'wss_z']:
        print(f"\n{component.upper()}:")
        print(f"  MAE:  {metrics[component]['mae']:.6f} Pa")
        print(f"  RMSE: {metrics[component]['rmse']:.6f} Pa")
        print(f"  R²:   {metrics[component]['r2']:.6f}")
    
    print(f"\nWSS MAGNITUDE:")
    print(f"  MAE:  {metrics['wss_magnitude']['mae']:.6f} Pa")
    print(f"  RMSE: {metrics['wss_magnitude']['rmse']:.6f} Pa")
    print(f"  R²:   {metrics['wss_magnitude']['r2']:.6f}")
    
    if 'relative_error' in metrics:
        print(f"\nRELATIVE ERROR:")
        print(f"  Median: {metrics['relative_error']['median']:.2f}%")
        print(f"  Mean:   {metrics['relative_error']['mean']:.2f}%")
        print(f"  95th:   {metrics['relative_error']['p95']:.2f}%")
    
    print()
    
    # Save metrics
    config.results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.results_dir / f"metrics_{args.strategy}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Saved metrics to {metrics_path}")
    
    # Create plots
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    scatter_path = config.figures_dir / f"scatter_{args.strategy}.png"
    plot_scatter(wss_true_mag, wss_pred_mag, scatter_path)
    
    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate bifurcation WSS prediction model")
    parser.add_argument('--strategy', type=str, default='re_interp',
                        choices=['re_interp', 'angle_transfer', 'random'],
                        help="Data split strategy (must match training)")
    parser.add_argument('--use_calibration', action='store_true',
                        help="Apply post-hoc calibration (if available)")
    
    args = parser.parse_args()
    evaluate(args)

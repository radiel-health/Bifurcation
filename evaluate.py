"""
Evaluation script for 3D point cloud WSS prediction model.
Computes comprehensive metrics and generates visualization plots.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from config import config
from dataset import get_dataloaders
from model import WSSPredictor


def load_best_model(checkpoint_path='Models/best_model.pt'):
    """Load the best trained model from checkpoint."""
    device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # UPDATED: Initialize leaner physics model
    model = WSSPredictor(
        node_feature_dim=config.node_feature_dim,
        # Removed the obsolete flow_param_dim and context_dim
        hidden_dim=config.hidden_dim,
        output_dim=config.target_dim,
        num_geom_layers=config.num_geom_layers,
        num_task_layers=config.num_task_layers,
        task_hidden_dim=config.task_hidden_dim,
        dropout=config.dropout_rate,
        output_range=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['val_loss']:.6f}")
    
    return model


def denormalize_wss(wss_normalized, stats):
    """Convert normalized WSS back to physical units."""
    # We now just multiply by the global scalar!
    target_scale = torch.tensor(stats['target_scale'], dtype=torch.float32)
    
    if isinstance(wss_normalized, np.ndarray):
        wss_normalized = torch.from_numpy(wss_normalized).float()
    
    # Linear scale restoration (no expm1 distortion!)
    wss_physical = (wss_normalized * target_scale).numpy()
    
    return wss_physical


def evaluate_model(model, test_loader, stats):
    """Comprehensive evaluation on test set."""
    model.eval()
    
    all_preds_norm = []
    all_targets_norm = []
    all_preds_phys = []
    all_targets_phys = []
    all_re = []
    all_coordinates = []
    test_loss = 0.0
    
    device = torch.device('cpu')
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            pred = model(batch)
            
            loss = torch.nn.functional.mse_loss(pred, batch.y)
            test_loss += loss.item()
            
            pred_phys = denormalize_wss(pred.cpu().numpy(), stats)
            target_phys = denormalize_wss(batch.y.cpu().numpy(), stats)
            
            all_preds_norm.append(pred.cpu().numpy())
            all_targets_norm.append(batch.y.cpu().numpy())
            all_preds_phys.append(pred_phys)
            all_targets_phys.append(target_phys)
            
            all_coordinates.append(batch.pos.cpu().numpy())
            
            # This safely handles the new [batch_size, 1] flow_params!
            flow_params = batch.flow_params.cpu().numpy()
            re_vals = flow_params[:, 0]
            
            batch_idx = batch.batch.cpu().numpy()
            for graph_idx in range(len(re_vals)):
                node_mask = (batch_idx == graph_idx)
                num_nodes_this_graph = node_mask.sum()
                all_re.extend([re_vals[graph_idx]] * num_nodes_this_graph)
    
    preds_norm = np.vstack(all_preds_norm)
    targets_norm = np.vstack(all_targets_norm)
    preds_phys = np.vstack(all_preds_phys)
    targets_phys = np.vstack(all_targets_phys)
    coordinates = np.vstack(all_coordinates)
    
    test_loss = test_loss / len(test_loader)
    
    mae_x = mean_absolute_error(targets_phys[:, 0], preds_phys[:, 0])
    mae_y = mean_absolute_error(targets_phys[:, 1], preds_phys[:, 1])
    mae_z = mean_absolute_error(targets_phys[:, 2], preds_phys[:, 2])
    
    rmse_x = np.sqrt(mean_squared_error(targets_phys[:, 0], preds_phys[:, 0]))
    rmse_y = np.sqrt(mean_squared_error(targets_phys[:, 1], preds_phys[:, 1]))
    rmse_z = np.sqrt(mean_squared_error(targets_phys[:, 2], preds_phys[:, 2]))
    
    r2_x = r2_score(targets_phys[:, 0], preds_phys[:, 0])
    r2_y = r2_score(targets_phys[:, 1], preds_phys[:, 1])
    r2_z = r2_score(targets_phys[:, 2], preds_phys[:, 2])
    
    target_mag = np.linalg.norm(targets_phys, axis=1)
    pred_mag = np.linalg.norm(preds_phys, axis=1)
    
    mae_mag = mean_absolute_error(target_mag, pred_mag)
    rmse_mag = np.sqrt(mean_squared_error(target_mag, pred_mag))
    r2_mag = r2_score(target_mag, pred_mag)
    
    epsilon = 1e-14
    rel_error_x = np.abs(targets_phys[:, 0] - preds_phys[:, 0]) / (np.abs(targets_phys[:, 0]) + epsilon) * 100
    rel_error_y = np.abs(targets_phys[:, 1] - preds_phys[:, 1]) / (np.abs(targets_phys[:, 1]) + epsilon) * 100
    rel_error_z = np.abs(targets_phys[:, 2] - preds_phys[:, 2]) / (np.abs(targets_phys[:, 2]) + epsilon) * 100
    rel_error_mag = np.abs(target_mag - pred_mag) / (target_mag + epsilon) * 100
    
    rel_error_x_filtered = rel_error_x[np.abs(targets_phys[:, 0]) > 1e-12]
    rel_error_y_filtered = rel_error_y[np.abs(targets_phys[:, 1]) > 1e-12]
    rel_error_z_filtered = rel_error_z[np.abs(targets_phys[:, 2]) > 1e-12]
    rel_error_mag_filtered = rel_error_mag[target_mag > 1e-12]
    
    metrics = {
        'test_loss_normalized': float(test_loss),
        'mae_x_Pa': float(mae_x),
        'mae_y_Pa': float(mae_y),
        'mae_z_Pa': float(mae_z),
        'mae_magnitude_Pa': float(mae_mag),
        'rmse_x_Pa': float(rmse_x),
        'rmse_y_Pa': float(rmse_y),
        'rmse_z_Pa': float(rmse_z),
        'rmse_magnitude_Pa': float(rmse_mag),
        'r2_x': float(r2_x),
        'r2_y': float(r2_y),
        'r2_z': float(r2_z),
        'r2_magnitude': float(r2_mag),
        'median_rel_error_x_percent': float(np.median(rel_error_x_filtered)),
        'median_rel_error_y_percent': float(np.median(rel_error_y_filtered)),
        'median_rel_error_z_percent': float(np.median(rel_error_z_filtered)),
        'median_rel_error_magnitude_percent': float(np.median(rel_error_mag_filtered)),
    }
    
    results = {
        'preds_norm': preds_norm,
        'targets_norm': targets_norm,
        'preds_phys': preds_phys,
        'targets_phys': targets_phys,
        'pred_mag': pred_mag,
        'target_mag': target_mag,
        'rel_error_x': rel_error_x,
        'rel_error_y': rel_error_y,
        'rel_error_z': rel_error_z,
        'rel_error_mag': rel_error_mag,
        're_values': np.array(all_re),
        'coordinates': coordinates,
    }
    
    return metrics, results


def print_metrics(metrics):
    """Print formatted metrics summary."""
    print("\n" + "="*60)
    print("TEST SET EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nNormalized Loss (MSE in log-space):")
    print(f"  Test Loss: {metrics['test_loss_normalized']:.6f}")
    
    print(f"\nPhysical Units - Mean Absolute Error (MAE):")
    print(f"  WSS_x:      {metrics['mae_x_Pa']:.6e} Pa")
    print(f"  WSS_y:      {metrics['mae_y_Pa']:.6e} Pa")
    print(f"  WSS_z:      {metrics['mae_z_Pa']:.6e} Pa")
    print(f"  Magnitude:  {metrics['mae_magnitude_Pa']:.6e} Pa")
    
    print(f"\nPhysical Units - Root Mean Squared Error (RMSE):")
    print(f"  WSS_x:      {metrics['rmse_x_Pa']:.6e} Pa")
    print(f"  WSS_y:      {metrics['rmse_y_Pa']:.6e} Pa")
    print(f"  WSS_z:      {metrics['rmse_z_Pa']:.6e} Pa")
    print(f"  Magnitude:  {metrics['rmse_magnitude_Pa']:.6e} Pa")
    
    print(f"\nR² Score:")
    print(f"  WSS_x:      {metrics['r2_x']:.6f}")
    print(f"  WSS_y:      {metrics['r2_y']:.6f}")
    print(f"  WSS_z:      {metrics['r2_z']:.6f}")
    print(f"  Magnitude:  {metrics['r2_magnitude']:.6f}")
    
    print("\n" + "="*60)


def plot_predictions_vs_truth(results, output_dir='results'):
    """Generate scatter plots comparing predictions to ground truth."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    components = ['X', 'Y', 'Z', 'Magnitude']
    targets_list = [
        results['targets_phys'][:, 0],
        results['targets_phys'][:, 1],
        results['targets_phys'][:, 2],
        results['target_mag']
    ]
    preds_list = [
        results['preds_phys'][:, 0],
        results['preds_phys'][:, 1],
        results['preds_phys'][:, 2],
        results['pred_mag']
    ]
    
    for idx, (ax, comp, target, pred) in enumerate(zip(axes.flat, components, targets_list, preds_list)):
        ax.scatter(target, pred, alpha=0.3, s=10, edgecolors='none')
        
        min_val = min(target.min(), pred.min())
        max_val = max(target.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel(f'True WSS_{comp} (Pa)', fontsize=12)
        ax.set_ylabel(f'Predicted WSS_{comp} (Pa)', fontsize=12)
        ax.set_title(f'WSS {comp}-Component', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_vs_truth.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'predictions_vs_truth.png'}")
    plt.close()


def analyze_by_reynolds(results, output_dir='results'):
    """Analyze error breakdown by Reynolds number."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    re_values = results['re_values']
    unique_re = np.unique(re_values)
    
    abs_error_mag = np.abs(results['target_mag'] - results['pred_mag'])
    
    re_stats = []
    for re in unique_re:
        mask = re_values == re
        if mask.sum() > 0:
            re_stats.append({
                'Re': int(re),
                'count': int(mask.sum()),
                'mae': float(abs_error_mag[mask].mean()),
                'rmse': float(np.sqrt((abs_error_mag[mask]**2).mean())),
                'r2': float(r2_score(results['target_mag'][mask], results['pred_mag'][mask])),
            })
    
    re_list = [s['Re'] for s in re_stats]
    mae_list = [s['mae'] for s in re_stats]
    
    sorted_stats = sorted(re_stats, key=lambda x: x['mae'], reverse=True)
    
    import csv
    with open(output_dir / 'error_by_reynolds.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Re', 'count', 'mae', 'rmse', 'r2'])
        writer.writeheader()
        writer.writerows(sorted_stats)
    print(f"Saved: {output_dir / 'error_by_reynolds.csv'}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.bar(re_list, mae_list, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Reynolds Number', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (Pa)', fontsize=12)
    ax.set_title('MAE by Reynolds Number', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1]
    r2_list = [s['r2'] for s in re_stats]
    ax.scatter(re_list, r2_list, s=50, c='darkgreen', alpha=0.7, edgecolors='black')
    ax.set_xlabel('Reynolds Number', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('R² Score by Reynolds Number', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='0.95 threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_by_reynolds.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'error_by_reynolds.png'}")
    plt.close()
    
    print("\n" + "="*60)
    print("REYNOLDS NUMBER PERFORMANCE BREAKDOWN")
    print("="*60)
    print(f"\nTotal unique Re values in test set: {len(unique_re)}")
    print(f"\nTop 5 WORST performers (highest MAE):")
    for i, stats in enumerate(sorted_stats[:5], 1):
        print(f"  {i}. Re={stats['Re']}: MAE={stats['mae']:.6e} Pa, R²={stats['r2']:.4f}")
    print(f"\nTop 5 BEST performers (lowest MAE):")
    for i, stats in enumerate(sorted_stats[-5:][::-1], 1):
        print(f"  {i}. Re={stats['Re']}: MAE={stats['mae']:.6e} Pa, R²={stats['r2']:.4f}")
    print("="*60)
    
    return re_stats


def main():
    """Main evaluation pipeline."""
    print("="*60)
    print("EVALUATING 3D POINT CLOUD WSS PREDICTION MODEL")
    print("="*60)
    
    stats_path = config.processed_data_dir / 'normalization_stats.json'
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    print(f"\nLoaded normalization stats from: {stats_path}")
    
    print("\nLoading test data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config.batch_size,
    )  
    print(f"Test set: {len(test_loader.dataset)} graphs, {len(test_loader)} batches")
    
    print("\nLoading trained model...")
    model = load_best_model('Models/best_model.pt')
    
    print("\nRunning evaluation on test set...")
    metrics, results = evaluate_model(model, test_loader, stats)
    
    print_metrics(metrics)
    
    output_dir = Path('results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = output_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")
    
    print("\nGenerating visualization plots...")
    plot_predictions_vs_truth(results, output_dir)
    analyze_by_reynolds(results, output_dir)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}/")
    print("  - test_metrics.json")
    print("  - predictions_vs_truth.png")
    print("  - error_by_reynolds.png & error_by_reynolds.csv")


if __name__ == '__main__':
    main()

"""
Evaluation script for trained bifurcation WSS prediction model.

Features:
- Comprehensive metrics (overall + per-region)
- Interactive 3D visualizations with plotly
- 2D cross-section analysis
- Error heatmaps and parity plots
- Worst-case analysis
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, List, Tuple

from config import Config
from model import create_model
from dataset import create_dataloaders, WSSDataset
from cross_section_viz import extract_cross_section


class ModelEvaluator:
    """Comprehensive model evaluation with visualizations."""
    
    def __init__(
        self,
        model_path: Path,
        config: Config,
        device: str = 'auto'
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            config: Configuration object
            device: Device to run on ('cuda', 'cpu', or 'auto')
        """
        self.config = config
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = create_model(config)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
        print(f"  Trained for {checkpoint['epoch']} epochs")
        print(f"  Best val loss: {checkpoint['best_val_loss']:.6f}")
        
        # Create results directory
        self.results_dir = config.results_dir / f"evaluation_epoch_{checkpoint['epoch']}"
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Store for later
        self.checkpoint = checkpoint
    
    @torch.no_grad()
    def predict_on_dataset(
        self,
        dataset: WSSDataset,
        dataloader
    ) -> Dict[str, np.ndarray]:
        """
        Run predictions on entire dataset.
        
        Returns:
            dict with keys: predictions, targets, coords, regions, cases, errors
        """
        all_predictions = []
        all_targets = []
        all_coords = []
        all_regions = []
        all_cases = []
        
        print("Running predictions...")
        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            
            # Predict
            pred_norm = self.model(batch.x, batch.edge_index, batch.batch)
            
            # Denormalize
            pred = dataset.denormalize_predictions(pred_norm.cpu())
            target = dataset.denormalize_predictions(batch.y.cpu())
            
            # Store
            all_predictions.append(pred.numpy())
            all_targets.append(target.numpy())
            all_coords.append(batch.pos.cpu().numpy())
            all_regions.append(batch.region.cpu().numpy())
            
            # Extract case names (need to handle batching)
            for i in range(batch.num_graphs):
                mask = (batch.batch == i).cpu().numpy()
                num_nodes = mask.sum()
                # Get case name from batch (stored in dataset)
                all_cases.extend([f"case_{len(all_cases)//35000}"] * num_nodes)
        
        # Concatenate all
        predictions = np.concatenate(all_predictions, axis=0).flatten()
        targets = np.concatenate(all_targets, axis=0).flatten()
        coords = np.concatenate(all_coords, axis=0)
        regions = np.concatenate(all_regions, axis=0)
        errors = predictions - targets
        
        return {
            'predictions': predictions,
            'targets': targets,
            'coords': coords,
            'regions': regions,
            'cases': np.array(all_cases),
            'errors': errors
        }
    
    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        regions: np.ndarray = None
    ) -> Dict[str, float]:
        """Compute comprehensive metrics."""
        metrics = {}
        
        # Overall metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        
        # R² score
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Relative error
        rel_error = np.mean(np.abs((predictions - targets) / (targets + 1e-8)))
        
        metrics['overall'] = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'relative_error': float(rel_error),
            'n_points': len(predictions)
        }
        
        # Per-region metrics
        if regions is not None:
            region_names = ['inlet', 'critical', 'outlet_left', 'outlet_right', 'other']
            
            for i, region_name in enumerate(region_names):
                mask = (regions == i)
                if mask.sum() > 0:
                    pred_region = predictions[mask]
                    target_region = targets[mask]
                    
                    mse_r = np.mean((pred_region - target_region) ** 2)
                    mae_r = np.mean(np.abs(pred_region - target_region))
                    rmse_r = np.sqrt(mse_r)
                    
                    ss_res_r = np.sum((target_region - pred_region) ** 2)
                    ss_tot_r = np.sum((target_region - np.mean(target_region)) ** 2)
                    r2_r = 1 - (ss_res_r / ss_tot_r)
                    
                    metrics[region_name] = {
                        'mse': float(mse_r),
                        'mae': float(mae_r),
                        'rmse': float(rmse_r),
                        'r2': float(r2_r),
                        'n_points': int(mask.sum())
                    }
        
        return metrics
    
    def plot_interactive_3d(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        title: str,
        colorbar_title: str = "WSS",
        filename: str = "3d_plot.html"
    ):
        """Create interactive 3D scatter plot with plotly."""
        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=colorbar_title),
                cmin=np.percentile(values, 1),  # Clip outliers
                cmax=np.percentile(values, 99),
            ),
            text=[f"WSS: {v:.6f}" for v in values],
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=1000,
            height=800
        )
        
        output_path = self.results_dir / filename
        fig.write_html(str(output_path))
        print(f"  Saved interactive plot: {output_path}")
        
        return fig
    
    def plot_parity(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        filename: str = "parity_plot.png"
    ):
        """Create parity plot (predicted vs actual)."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot
        ax.scatter(targets, predictions, alpha=0.3, s=1)
        
        # Perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
        
        # Labels
        ax.set_xlabel('Actual WSS', fontsize=12)
        ax.set_ylabel('Predicted WSS', fontsize=12)
        ax.set_title('Parity Plot: Predicted vs Actual WSS', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / filename, dpi=150)
        plt.close()
        
        print(f"  Saved parity plot: {filename}")
    
    def plot_error_distribution(
        self,
        errors: np.ndarray,
        regions: np.ndarray,
        filename: str = "error_distribution.png"
    ):
        """Plot error distribution by region."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Overall error histogram
        ax = axes[0, 0]
        ax.hist(errors, bins=100, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', label='Zero error')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Overall Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Error by region (boxplot)
        ax = axes[0, 1]
        region_names = ['inlet', 'critical', 'outlet_left', 'outlet_right', 'other']
        region_errors = [errors[regions == i] for i in range(5) if (regions == i).sum() > 0]
        region_labels = [region_names[i] for i in range(5) if (regions == i).sum() > 0]
        ax.boxplot(region_errors, labels=region_labels)
        ax.set_ylabel('Prediction Error')
        ax.set_title('Error by Region')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Absolute error histogram
        ax = axes[1, 0]
        ax.hist(np.abs(errors), bins=100, alpha=0.7, edgecolor='black', color='orange')
        ax.set_xlabel('Absolute Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Absolute Error Distribution')
        ax.grid(True, alpha=0.3)
        
        # Error vs target magnitude
        ax = axes[1, 1]
        # Subsample for clarity
        n_sample = min(10000, len(errors))
        idx = np.random.choice(len(errors), n_sample, replace=False)
        ax.scatter(errors[idx], np.abs(errors[idx]), alpha=0.3, s=1)
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Error Characteristics')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / filename, dpi=150)
        plt.close()
        
        print(f"  Saved error distribution: {filename}")
    
    def plot_region_performance(
        self,
        metrics: Dict,
        filename: str = "region_performance.png"
    ):
        """Bar plot of metrics by region."""
        regions = [k for k in metrics.keys() if k != 'overall']
        mae_values = [metrics[r]['mae'] for r in regions]
        r2_values = [metrics[r]['r2'] for r in regions]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # MAE by region
        ax1.bar(regions, mae_values, color='steelblue', edgecolor='black')
        ax1.set_ylabel('MAE', fontsize=12)
        ax1.set_title('Mean Absolute Error by Region', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # R² by region
        ax2.bar(regions, r2_values, color='coral', edgecolor='black')
        ax2.set_ylabel('R² Score', fontsize=12)
        ax2.set_title('R² Score by Region', fontsize=14)
        ax2.axhline(0, color='r', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / filename, dpi=150)
        plt.close()
        
        print(f"  Saved region performance: {filename}")
    
    def evaluate(self):
        """Run full evaluation pipeline."""
        print("="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        # Load test data
        print("\nLoading test dataset...")
        _, _, test_loader, norm_stats = create_dataloaders(
            self.config,
            batch_size=self.config.batch_size,
            num_workers=0
        )
        
        # Create test dataset for denormalization
        test_dataset = WSSDataset(
            root=self.config.data_root,
            split='test',
            normalize=self.config.normalize_features,
            normalization_stats=norm_stats
        )
        
        # Predict
        results = self.predict_on_dataset(test_dataset, test_loader)
        
        # Compute metrics
        print("\nComputing metrics...")
        metrics = self.compute_metrics(
            results['predictions'],
            results['targets'],
            results['regions']
        )
        
        # Print metrics
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print("\nOverall Performance:")
        for key, value in metrics['overall'].items():
            if key != 'n_points':
                print(f"  {key.upper():20s}: {value:.6f}")
            else:
                print(f"  {key.upper():20s}: {value}")
        
        print("\nPer-Region Performance:")
        for region in ['inlet', 'critical', 'outlet_left', 'outlet_right']:
            if region in metrics:
                print(f"\n  {region.upper()}:")
                for key, value in metrics[region].items():
                    if key != 'n_points':
                        print(f"    {key.upper():18s}: {value:.6f}")
                    else:
                        print(f"    {key.upper():18s}: {value}")
        
        # Save metrics
        metrics_path = self.results_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved metrics to {metrics_path}")
        
        # Save predictions
        print("\nSaving predictions...")
        predictions_df = pd.DataFrame({
            'x': results['coords'][:, 0],
            'y': results['coords'][:, 1],
            'z': results['coords'][:, 2],
            'wss_actual': results['targets'],
            'wss_predicted': results['predictions'],
            'error': results['errors'],
            'abs_error': np.abs(results['errors']),
            'region': results['regions']
        })
        predictions_df.to_csv(self.results_dir / "predictions.csv", index=False)
        print(f"  Saved predictions to predictions.csv")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        # 1. Interactive 3D plots
        print("  Creating 3D plots...")
        self.plot_interactive_3d(
            results['coords'],
            results['targets'],
            "Ground Truth WSS",
            "WSS (Pa)",
            "3d_ground_truth.html"
        )
        
        self.plot_interactive_3d(
            results['coords'],
            results['predictions'],
            "Predicted WSS",
            "WSS (Pa)",
            "3d_predictions.html"
        )
        
        self.plot_interactive_3d(
            results['coords'],
            np.abs(results['errors']),
            "Absolute Error",
            "Error (Pa)",
            "3d_error.html"
        )
        
        # 2. Parity plot
        print("  Creating parity plot...")
        self.plot_parity(results['predictions'], results['targets'])
        
        # 3. Error distribution
        print("  Creating error distribution...")
        self.plot_error_distribution(results['errors'], results['regions'])
        
        # 4. Region performance
        print("  Creating region performance...")
        self.plot_region_performance(metrics)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print(f"Results saved to: {self.results_dir}")
        print("="*70)
        
        return metrics, results


def main():
    """Main evaluation function."""
    config = Config()
    
    # Find best model
    model_path = config.checkpoint_dir / "best_model.pt"
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Please train a model first using: python train.py")
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(model_path, config)
    
    # Run evaluation
    metrics, results = evaluator.evaluate()


if __name__ == '__main__':
    main()

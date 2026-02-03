"""
Inference script for predicting WSS on new or existing bifurcation cases.

Usage:
    python predict.py --case path/to/case.pt
    python predict.py --csv path/to/wall_wss.csv --angle 30 --mesh 500 --re 100
"""

import argparse
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from config import Config
from model import create_model
from dataset import WSSDataset, create_dataloaders
from preprocess_enhanced import preprocess_single_case
from cross_section_viz import extract_cross_section


class WSSPredictor:
    """Predict WSS on bifurcation cases."""
    
    def __init__(self, model_path: Path, config: Config, device: str = 'auto'):
        """
        Args:
            model_path: Path to trained model checkpoint
            config: Configuration object
            device: Device to run on
        """
        self.config = config
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = create_model(config)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get normalization stats
        _, _, _, self.norm_stats = create_dataloaders(config, batch_size=1)
        
        print(f"Loaded model from {model_path}")
        print(f"Using device: {self.device}")
    
    def load_case_from_pt(self, pt_path: Path):
        """Load preprocessed case from .pt file."""
        data = torch.load(pt_path, weights_only=False)
        return data
    
    def load_case_from_csv(
        self,
        csv_path: Path,
        angle: int,
        mesh: int,
        re_number: int
    ):
        """Preprocess CSV file on-the-fly."""
        # Create temporary directory structure for preprocessing
        temp_dir = Path(csv_path.parent)
        case_name = f"bifurcation_angle{angle}_{mesh}_ascii"
        re_dir = temp_dir / case_name / f"Re{re_number}"
        re_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy CSV to expected location
        target_csv = re_dir / "wall_wss.csv"
        if csv_path != target_csv:
            import shutil
            shutil.copy(csv_path, target_csv)
        
        # Preprocess
        data = preprocess_single_case(target_csv, self.config)
        
        return data
    
    @torch.no_grad()
    def predict(self, data):
        """
        Run prediction on a single case.
        
        Args:
            data: PyG Data object
            
        Returns:
            dict with keys: predictions, coords, ground_truth (if available), regions
        """
        # Normalize features
        x_norm = (data.x - self.norm_stats['feature_mean']) / self.norm_stats['feature_std']
        
        # Move to device
        x_norm = x_norm.to(self.device)
        edge_index = data.edge_index.to(self.device)
        
        # Predict
        pred_norm = self.model(x_norm, edge_index)
        
        # Denormalize
        pred = (pred_norm.cpu() * self.norm_stats['target_std'] + 
                self.norm_stats['target_mean']).numpy().flatten()
        
        results = {
            'predictions': pred,
            'coords': data.pos.numpy(),
            'regions': data.region.numpy(),
            'case_name': data.case_name if hasattr(data, 'case_name') else 'unknown'
        }
        
        # Add ground truth if available
        if data.y is not None:
            gt = (data.y * self.norm_stats['target_std'] + 
                  self.norm_stats['target_mean']).numpy().flatten()
            results['ground_truth'] = gt
            results['error'] = pred - gt
        
        return results
    
    def visualize_results(
        self,
        results: dict,
        output_dir: Path,
        show_cross_sections: bool = True
    ):
        """Create comprehensive visualizations of prediction results."""
        output_dir.mkdir(exist_ok=True, parents=True)
        case_name = results['case_name']
        
        # 1. Interactive 3D prediction plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(
            x=results['coords'][:, 0],
            y=results['coords'][:, 1],
            z=results['coords'][:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=results['predictions'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Predicted WSS (Pa)"),
            ),
            name='Predictions',
            text=[f"WSS: {v:.6f}" for v in results['predictions']],
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title=f"Predicted WSS: {case_name}",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=1000,
            height=800
        )
        
        pred_path = output_dir / f"{case_name}_predictions.html"
        fig.write_html(str(pred_path))
        print(f"  Saved prediction plot: {pred_path}")
        
        # 2. If ground truth available, create comparison
        if 'ground_truth' in results:
            # Side-by-side comparison
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Ground Truth', 'Predictions', 'Absolute Error'),
                specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]]
            )
            
            coords = results['coords']
            
            # Ground truth
            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                    mode='markers',
                    marker=dict(size=1, color=results['ground_truth'], colorscale='Viridis'),
                    name='Ground Truth',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Predictions
            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                    mode='markers',
                    marker=dict(size=1, color=results['predictions'], colorscale='Viridis'),
                    name='Predictions',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Error
            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                    mode='markers',
                    marker=dict(size=1, color=np.abs(results['error']), colorscale='Reds'),
                    name='Error',
                    showlegend=False
                ),
                row=1, col=3
            )
            
            fig.update_layout(
                title=f"Comparison: {case_name}",
                height=600,
                width=1800
            )
            
            comp_path = output_dir / f"{case_name}_comparison.html"
            fig.write_html(str(comp_path))
            print(f"  Saved comparison plot: {comp_path}")
        
        # 3. Cross-sections if requested
        if show_cross_sections:
            self._plot_cross_sections(results, output_dir)
    
    def _plot_cross_sections(self, results: dict, output_dir: Path):
        """Plot 2D cross-sections at key locations."""
        coords = results['coords']
        z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
        z_range = z_max - z_min
        
        # Cross-section positions: inlet (25%), apex (50%), outlet (75%)
        z_positions = {
            'inlet': z_min + 0.25 * z_range,
            'apex': z_min + 0.50 * z_range,
            'outlet': z_min + 0.75 * z_range
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (name, z_pos) in enumerate(z_positions.items()):
            # Extract cross-section
            indices, _ = extract_cross_section(coords, results['predictions'], z_pos, z_tolerance=0.02)
            
            if len(indices) > 0:
                section_coords = coords[indices]
                section_wss = results['predictions'][indices]
                
                # Plot
                ax = axes[idx]
                scatter = ax.scatter(
                    section_coords[:, 0],
                    section_coords[:, 1],
                    c=section_wss,
                    s=20,
                    cmap='viridis'
                )
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(f'{name.capitalize()} (z={z_pos:.3f})')
                ax.set_aspect('equal')
                plt.colorbar(scatter, ax=ax, label='WSS (Pa)')
        
        plt.tight_layout()
        cs_path = output_dir / f"{results['case_name']}_cross_sections.png"
        plt.savefig(cs_path, dpi=150)
        plt.close()
        print(f"  Saved cross-sections: {cs_path}")
    
    def predict_and_save(
        self,
        case_path: Path,
        output_dir: Path = None,
        visualize: bool = True
    ):
        """
        Predict on a case and save results.
        
        Args:
            case_path: Path to .pt file
            output_dir: Output directory (creates if None)
            visualize: Whether to create visualizations
        """
        # Load case
        print(f"\nLoading case: {case_path}")
        data = self.load_case_from_pt(case_path)
        
        # Predict
        print("Running prediction...")
        results = self.predict(data)
        
        # Create output directory
        if output_dir is None:
            output_dir = self.config.results_dir / "predictions" / results['case_name']
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save predictions to CSV
        print("Saving results...")
        df = pd.DataFrame({
            'x': results['coords'][:, 0],
            'y': results['coords'][:, 1],
            'z': results['coords'][:, 2],
            'wss_predicted': results['predictions'],
            'region': results['regions']
        })
        
        if 'ground_truth' in results:
            df['wss_actual'] = results['ground_truth']
            df['error'] = results['error']
            df['abs_error'] = np.abs(results['error'])
        
        csv_path = output_dir / f"{results['case_name']}_predictions.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved CSV: {csv_path}")
        
        # Summary statistics
        stats = {
            'case_name': results['case_name'],
            'n_points': len(results['predictions']),
            'wss_min': float(results['predictions'].min()),
            'wss_max': float(results['predictions'].max()),
            'wss_mean': float(results['predictions'].mean()),
            'wss_std': float(results['predictions'].std()),
        }
        
        if 'ground_truth' in results:
            stats['mae'] = float(np.mean(np.abs(results['error'])))
            stats['rmse'] = float(np.sqrt(np.mean(results['error']**2)))
        
        stats_path = output_dir / f"{results['case_name']}_stats.json"
        import json
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved stats: {stats_path}")
        
        # Print summary
        print("\nPrediction Summary:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        
        # Visualize
        if visualize:
            print("\nCreating visualizations...")
            self.visualize_results(results, output_dir)
        
        print(f"\nResults saved to: {output_dir}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Predict WSS on bifurcation cases")
    parser.add_argument('--case', type=str, help="Path to preprocessed .pt file")
    parser.add_argument('--csv', type=str, help="Path to wall_wss.csv file")
    parser.add_argument('--angle', type=int, help="Bifurcation angle (for CSV input)")
    parser.add_argument('--mesh', type=int, help="Mesh refinement (for CSV input)")
    parser.add_argument('--re', type=int, help="Reynolds number (for CSV input)")
    parser.add_argument('--output', type=str, help="Output directory")
    parser.add_argument('--no-viz', action='store_true', help="Skip visualizations")
    
    args = parser.parse_args()
    
    config = Config()
    model_path = config.checkpoint_dir / "best_model.pt"
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Please train a model first using: python train.py")
        return
    
    # Create predictor
    predictor = WSSPredictor(model_path, config)
    
    # Determine input
    if args.case:
        case_path = Path(args.case)
        predictor.predict_and_save(
            case_path,
            output_dir=Path(args.output) if args.output else None,
            visualize=not args.no_viz
        )
    elif args.csv:
        if not all([args.angle, args.mesh, args.re]):
            print("Error: --csv requires --angle, --mesh, and --re")
            return
        
        # Load from CSV
        data = predictor.load_case_from_csv(
            Path(args.csv),
            args.angle,
            args.mesh,
            args.re
        )
        
        # Predict
        results = predictor.predict(data)
        
        # Save
        output_dir = Path(args.output) if args.output else config.results_dir / "predictions" / results['case_name']
        predictor.visualize_results(results, output_dir)
    else:
        print("Error: Provide either --case or --csv")
        print("\nExamples:")
        print("  python predict.py --case ProcessedData/angle30/mesh500_Re100.pt")
        print("  python predict.py --csv data/wall_wss.csv --angle 30 --mesh 500 --re 100")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()

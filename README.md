# Bifurcation WSS Prediction

Deep learning pipeline for predicting wall shear stress (WSS) on 3D bifurcation pipe geometries using graph neural networks.

## Overview

This project implements a hybrid GNN architecture combining:
- **AVFlow Gen 3 backbone**: EdgeUNetAggregator + 8×GCNConv for processing 3D triangulated wall surfaces
- **FiLM conditioning**: Flow parameter modulation (Reynolds number, bifurcation angle) for generalization

### Key Features

- **3D surface mesh processing**: Handles ANSYS .msh files and Fluent CSV outputs
- **Geometric edge features**: Dihedral angles, inner angles, edge-length ratios
- **Multi-parameter conditioning**: Generalizes across Reynolds numbers (100-2100) and bifurcation angles (30°, 45°, 60°)
- **Ensemble predictions**: LOO-CV training for robust uncertainty quantification
- **Clean modular structure**: Preprocessing → Training → Evaluation → Calibration → Inference

## Architecture

```
Flow Params [Re, angle] → FlowEncoder (MLP) → context [64D]
                                                   ↓ FiLM
Mesh:
  Node features [x,y,z,degree] → [4D]
  Edge features [dihedral, angles, ratios] → [5D]
        ↓
  EdgeUNetAggregator:
    LineGraph → GraphUNet(depth=4, pool=0.5) → Linear(5→16)
        ↓
  Cat[node_feat, edge_context] → [20D]
        ↓
  8× GCNConv(512D) + ReLU
        ↓
  h_geom [512D] ──────→ FiLMLayer: γ·h + β
                                ↓
                        Linear(512→3) → [wss_x, wss_y, wss_z]
```

**Model Size**: ~3.5M parameters

## Data

Bifurcation pipe CFD simulations from ANSYS Fluent:
- **Geometries**: 30°, 45°, 60° bifurcation angles
- **Mesh levels**: base (~500 cells), 750, 1000
- **Reynolds range**: 100 to 2100 (step 100)
- **Wall nodes**: ~17K per mesh
- **Total cases**: ~180 simulations

Data location: `../Data/Bifurcation/`

## Installation

### Requirements

```bash
# Core dependencies
pip install torch torch-geometric
pip install numpy scipy pandas matplotlib seaborn tqdm
pip install meshio  # For reading ANSYS .msh files

# Optional (for visualization)
pip install pyvista vtk
```

### Setup

```bash
cd LidDrivenHolder
git clone [this-repo] BifurcationWSS
cd BifurcationWSS
```

## Usage

### 1. Preprocessing

Convert raw ANSYS mesh + Fluent CSV to PyTorch Geometric graphs:

```bash
python -m Preprocessing.pre_process
```

This will:
- Load `.msh` files and extract wall surface triangulation
- Match CSV node data to mesh vertices
- Compute geometric edge features (5D)
- Generate node features (4D: x, y, z, degree)
- Save graphs to `ProcessedData/angle{A}_mesh{M}/Re{R}.pt`
- Compute and save normalization statistics

**Output**: ~180 `.pt` files + `normalization_stats.json`

### 2. Training

Train ensemble models with LOO-CV:

```bash
python train.py --folds 3 --strategy re_interp
```

**Arguments**:
- `--folds`: Number of ensemble folds (default: 3)
- `--strategy`: Split strategy
  - `re_interp`: Hold out odd-hundred Re values for interpolation testing
  - `angle_transfer`: Hold out one angle for generalization testing
  - `random`: Random stratified split

**Output**:
- `Models/best_state_fold_{i}.pt`: Checkpoint for each fold
- `Models/training_history.json`: Loss curves and LR schedule

**Training time**: ~2-4 hours per fold on GPU (NVIDIA RTX 3080)

### 3. Evaluation

Evaluate ensemble on test set:

```bash
python evaluate.py --strategy re_interp
```

Computes:
- **Physical-unit metrics**: MAE, RMSE, R² for WSS components and magnitude
- **Spatial analysis**: Error vs. distance from bifurcation point
- **Parametric analysis**: Error vs. Re and angle
- **Visualizations**: Scatter plots, error heatmaps

**Output**: `results/metrics.json`, `figures/*.png`

### 4. Post-hoc Calibration (Optional)

Apply residual calibration with uncertainty weighting (AVFlow Gen 3 method):

```bash
python calibrate.py
```

Implements:
- Linear + Isotonic regression on residuals
- Ensemble disagreement → uncertainty estimate
- Shrinkage: $\text{correction} \cdot \frac{1}{1 + \lambda\sigma}$
- Lambda tuning via nested CV

**Output**: `Models/calibrator.pkl`

### 5. Inference

Predict WSS for a specific case:

```bash
python infer.py --angle 30 --mesh 750 --re 150 --output results/predictions.csv
```

For cases not in the original dataset, preprocess first:
```bash
# Add new case to Data/Bifurcation/results/
python -m Preprocessing.pre_process  # Re-run to pick up new cases
python infer.py --angle 30 --mesh 750 --re 2200
```

**Output**: CSV with columns `[x, y, z, wss_x, wss_y, wss_z, wss_mag, uncertainty]`

### 6. Visualization

Generate 3D visualizations:

```bash
python visualize.py --angle 30 --mesh 750 --re 100
```

Creates side-by-side comparison: truth | prediction | error

**Requires**: `pyvista` for 3D rendering

## Configuration

All hyperparameters in [config.py](config.py):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `node_feature_dim` | 4 | x, y, z, degree |
| `edge_feature_dim` | 5 | Geometric features |
| `aggregated_edge_feat_dim` | 16 | Edge aggregator output |
| `hidden_gcn_dim` | 512 | GCN hidden dimension |
| `num_gcn_layers` | 8 | GCN depth |
| `context_dim` | 64 | FiLM context dimension |
| `batch_size` | 2 | Graphs per batch |
| `lr` | 1e-3 | Learning rate |
| `num_epochs` | 300 | Max epochs |
| `early_stop_patience` | 30 | Early stopping patience |

To modify, edit `config.py` and retrain.

## Project Structure

```
BifurcationWSS/
├── config.py                  # Central configuration
├── dataset.py                 # Data loading + normalization
├── train.py                   # Training script
├── evaluate.py                # Evaluation metrics
├── calibrate.py               # Post-hoc calibration
├── infer.py                   # Single-case inference
├── visualize.py               # 3D visualization
│
├── Models/
│   ├── model.py               # GNN architecture
│   ├── best_state_fold_*.pt   # Trained checkpoints
│   └── training_history.json  # Training logs
│
├── Preprocessing/
│   └── pre_process.py         # Mesh → graph conversion
│
├── Utils/
│   └── mesh_utils.py          # Mesh processing utilities
│
├── ProcessedData/              # Preprocessed graphs
│   ├── angle{A}_mesh{M}/Re{R}.pt
│   └── normalization_stats.json
│
├── results/                    # Evaluation outputs
└── figures/                    # Plots and visualizations
```

## Model Details

### Node Features (4D)
- `x_norm`, `y_norm`, `z_norm`: Normalized coordinates
- `degree`: Node degree in surface graph

### Edge Features (5D)
Computed from triangulated surface:
1. **Dihedral angle**: Angle between adjacent face normals
2. **Min inner angle**: Min angle at opposite vertex
3. **Max inner angle**: Max angle at opposite vertex
4. **Min edge ratio**: Min(edge_length / triangle_height)
5. **Max edge ratio**: Max(edge_length / triangle_height)

### Flow Parameters (2D)
- `Re_normalized`: (Re - 100) / 2000, normalized to [0, 1]
- `angle_radians`: Bifurcation angle in radians

### Targets (3D)
- `wss_x`, `wss_y`, `wss_z`: WSS components in Pa
- Normalized with sign-preserving log1p + z-score

## Expected Performance

Based on AVFlow Gen 3 results on vascular geometries:
- **R² > 0.90** for WSS magnitude on test set
- **Interpolation**: Held-out Re values should match training Re performance
- **Extrapolation**: Performance degrades for Re > 2100 or Re < 100
- **Transfer**: Generalization to held-out angle depends on geometric similarity

## Troubleshooting

### "No preprocessed graphs found"
Run preprocessing first: `python -m Preprocessing.pre_process`

### "meshio not installed"
Install: `pip install meshio`

### "CUDA out of memory"
Reduce `batch_size` in `config.py` to 1

### "CSV-to-mesh matching has large distance"
Check coordinate units consistency between .msh and .csv files

### Slow preprocessing
- Mesh loading is cached per (angle, mesh_level)
- Edge feature computation is O(E²) — expect ~5-10 min for all cases

## Citation

If you use this code, please cite:
- AVFlow team (Radiel Health) for the original GNN architecture
- LidDrivenCavity project for the FiLM conditioning approach

## License

[Specify license]

## Contact

[Your contact information]

## Acknowledgments

- **AVFlow project**: EdgeUNetAggregator and GCN backbone design
- **LidDrivenCavity project**: FiLM modulation and clean pipeline structure
- **PyTorch Geometric**: Graph neural network framework
- **meshio**: Mesh file I/O utilities

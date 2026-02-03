# Bifurcation WSS Prediction Model

Machine learning model for predicting wall shear stress (WSS) in bifurcation geometries using Graph Neural Networks.

## Overview

This project adapts the LidDrivenCavity GNN architecture for bifurcation vessel geometries. The model predicts WSS distribution based on:
- Reynolds number
- Bifurcation angle (30°, 45°, 60°)
- Mesh refinement level  
- Geometric features (position, region, curvature)

## Repository Structure

```
Bifurcation/
├── config.py                 # Central configuration file
├── geometry_utils.py         # Geometry analysis and feature extraction
├── preprocess.py            # Convert CSV data to PyG graphs
├── analyze_geometry.py      # Geometry visualization and analysis
├── ProcessedData/          # Preprocessed graph data
├── Models/                 # Saved model checkpoints
├── results/                # Evaluation results
└── analysis_results/       # Geometry analysis outputs
```

## Bifurcation Geometry

The bifurcation is divided into regions:

1. **Inlet** (bottom 25% by z-coordinate): Single upstream vessel
2. **Critical Region** (middle 35-60%): Junction where vessel splits - **highest WSS**
3. **Outlet Left/Right** (top 65%+): Two downstream branches

### Model Features (14 dimensions)

- Normalized coordinates (x, y, z)
- Reynolds number, bifurcation angle, mesh refinement
- Region indicators (inlet, critical, outlet_left, outlet_right)
- Distance to bifurcation apex
- Radial distance from centerline
- Angular position around centerline
- Curvature indicator

## Quick Start

### 1. Preprocess Data
```bash
python preprocess.py
```

### 2. Analyze Geometry (Optional)
```bash
python analyze_geometry.py
```

### 3. Train Model
```bash
python train.py
```

### 4. Evaluate
```bash
python evaluate.py --checkpoint Models/best_model.pt
```

## Key Findings from Geometry Analysis

- **34,907 wall points** per case (Re100, angle30, 500 mesh)
- **Critical region** contains ~16% of points but experiences highest WSS
- **Inlet** is largest region (~48% of points)
- **Outlets** split evenly left/right (~11% each)
- **Apex location**: (0.06, -0.05, 0.28) - bifurcation junction

## Differences from LidDrivenCavity Model

| Aspect | LidDrivenCavity | Bifurcation |
|--------|----------------|-------------|
| Geometry | 2D rectangular | 3D bifurcating vessel |
| Coordinates | (x, y) | (x, y, z) |
| Features | 10 | 14 |
| Critical Zone | Top moving wall | Bifurcation junction |
| Regions | 4 walls | Inlet + critical + 2 outlets |

## Status

- ✅ Geometry analysis complete
- ✅ Preprocessing pipeline ready
- ⏳ Training script (in progress)
- ⏳ Evaluation pipeline (in progress)

## Data

WSS data from OpenFOAM simulations:
- 3 angles × 3 mesh levels × 21 Reynolds numbers = 91 cases
- Each case: ~35,000 wall points
- Format: CSV with x,y,z coordinates and WSS components

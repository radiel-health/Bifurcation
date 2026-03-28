# Bifurcation WSS Predictor

GNN model that predicts wall shear stress (WSS) on arterial bifurcation geometries from CFD simulations. The latest version (v4) supports pulsatile flow — predicting WSS across 20 phases of a cardiac cycle and computing clinical hemodynamic indices (TAWSS, OSI).

## Model Versions

| Version | Architecture | Features | Notes |
|---------|-------------|----------|-------|
| v1 | GINEConv×6, hidden=256 | 3D coords | Baseline |
| v2 | GINEConv×8, hidden=384, BayesianLinear | 10 physics features | Re≥300 only |
| v3 | GATv2Conv×8, 4 heads, hidden=384 | 10 physics + 8 Laplacian PE | Attention aggregation |
| **v4** | GATv2Conv×8, 4 heads, hidden=384, FiLM | 10 physics features | **Pulsatile flow, current best** |

**v4 test set performance (21 cases × 20 timesteps = 418 snapshots):**
- WSS Magnitude R² = 0.967
- TAWSS R² = 0.975
- OSI R² = 0.776
- Median relative error = 11%

## Installation

```bash
pip install torch torch_geometric torchbnn tqdm scipy
```

## Quickstart (v4)

### 1. Build feature cache (required once, ~40 min)
```bash
python -m Bifurcation.dataset_v4 --process
```

### 2. Train
```bash
python -m Bifurcation.train_v4
```

### 3. Evaluate
```bash
python -m Bifurcation.evaluate_v4 --model Bifurcation/Models_v4/best_model_v4.pt --save --plot --clinical
```

### 4. Infer on a geometry (pulsatile, 20 cardiac phases)
```bash
python -m Bifurcation.infer_v4 \
    --model Bifurcation/Models_v4/best_model_v4.pt \
    --geometry bifurcation_angle45_750_ascii \
    --re 500 \
    --pulsatile \
    --paraview
```

Opens `predictions_v4/<geo>/Re<N>/prediction_v4.foam` in ParaView → color by `wallShearStress` → Play for animated cardiac cycle.

## Clinical Indices

- **TAWSS** (Time-Averaged WSS): mean wall stress magnitude across the cardiac cycle — low TAWSS indicates reduced mechanical stimulus, associated with plaque formation
- **OSI** (Oscillatory Shear Index): measures how much flow direction flips over the cycle — OSI=0 means unidirectional flow (healthy), OSI→0.5 means fully oscillating flow (high atherosclerosis risk)

## Directory Structure

```
Bifurcation/
├── Models/                  # Model architecture files (bif_v2.py, bif_v3.py, bif_v4.py)
├── config_v4.py             # Training config for v4
├── dataset_v4.py            # Data loading and graph construction
├── train_v4.py              # Training loop
├── evaluate_v4.py           # Evaluation metrics and plots
├── infer_v4.py              # Inference + ParaView export
├── demo_v4.py               # Interactive pulsatile WSS demo
└── upload_to_zenodo.sh      # Script to upload ProcessedData_v4 to Zenodo
```

## Data

ProcessedData_v4 (49 GB pulsatile graph dataset) is archived on Zenodo. Contact the team for access.

Raw CFD data (OpenFOAM): 7 bifurcation geometries × 19 Reynolds numbers (Re 300–2100) × 20 cardiac phases.

## GPU Notes

- **A100 (40 GB)**: no gradient checkpointing needed, ~28s/epoch
- **T4 (12 GB)**: enable gradient checkpointing in `bif_v4.py` forward(), ~346s/epoch

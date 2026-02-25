"""
Inference script v2 — with Monte Carlo uncertainty estimation.

Improvements over infer.py (v1):
  - MC sampling: N forward passes through the BayesianLinear head give per-node
    mean predictions and standard deviations (uncertainty maps).
  - Extended CSV output: x,y,z, wss_x/y/z, wss_mag, std_x/y/z, std_mag
  - Per-branch summary printed at inference time
  - ParaView export includes uncertainty field (std_mag)
  - Compatible with v1 checkpoints (auto-detects model version)

Usage:
    # Single prediction with uncertainty
    python -m Bifurcation.infer_v2 \\
        --model Models_v2/best_model_v2.pt \\
        --geometry bifurcation_angle45_750_ascii \\
        --re 500

    # Sweep all Re for one geometry
    python -m Bifurcation.infer_v2 \\
        --model Models_v2/best_model_v2.pt \\
        --geometry bifurcation_angle45_750_ascii \\
        --re-range 100 2100 100

    # Predict all 189 cases
    python -m Bifurcation.infer_v2 --model Models_v2/best_model_v2.pt --all

    # Disable MC (deterministic, faster)
    python -m Bifurcation.infer_v2 ... --mc-samples 1
"""

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from Bifurcation.config_v2 import config_v2
from Bifurcation.dataset import load_sample, parse_boundary, load_normalization_stats
from Bifurcation.dataset import parse_openfoam_faces
from Bifurcation.dataset_v2 import (
    denormalize_wss_v2,
    load_normalization_stats_v2,
    compute_physics_features_v2,
)
from Bifurcation.Models.bif_v2 import BifurcationWSSPredictorV2

# v1 model for comparison / fallback
from Bifurcation.model import BifurcationWSSPredictor
from Bifurcation.config import config as config_v1


# ============================================================================
# Model loading  (auto-detects v1 vs v2)
# ============================================================================

def load_model_v2(
    ckpt_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[BifurcationWSSPredictorV2, Dict]:
    """
    Load a v2 checkpoint.  Returns (model, norm_stats).

    The checkpoint's 'config' dict is used to reconstruct the model so that
    checkpoints remain portable across config changes.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg    = ckpt.get("config", {})

    model = BifurcationWSSPredictorV2(
        node_feat_dim  = cfg.get("node_feat_dim",  config_v2.node_feat_dim),
        edge_feat_dim  = cfg.get("edge_feat_dim",  config_v2.edge_feat_dim),
        hidden_dim     = cfg.get("hidden_dim",     config_v2.hidden_dim),
        out_channels   = cfg.get("output_dim",     config_v2.output_dim),
        num_layers     = cfg.get("num_layers",     config_v2.num_layers),
        context_dim    = cfg.get("context_dim",    config_v2.context_dim),
        flow_param_dim = cfg.get("flow_param_dim", config_v2.flow_param_dim),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    norm_stats = ckpt.get("norm_stats")
    if norm_stats is None:
        norm_stats = load_normalization_stats_v2()

    return model, norm_stats


# ============================================================================
# Graph builder for inference (applies v2 features)
# ============================================================================

def _build_inference_graph(
    geometry_folder: str,
    re_number:       float,
    norm_stats:      Dict,
    device:          torch.device,
) -> Tuple[np.ndarray, Data]:
    """
    Load mesh + compute v2 features → normalised PyG Data object.

    Returns
    -------
    coords : (N, 3)  raw face centroid coordinates (for CSV output)
    data   : normalised PyG Data ready for model forward pass
    """
    geo_path = str(config_v2.get_geometry_path(geometry_folder))
    angle, _ = config_v2.parse_geometry_folder(geometry_folder)
    re_case  = f"Re{int(re_number)}"

    # Load raw mesh + surface data
    (wall_centres, _wss_gt, edge_index, edge_attr,
     _re, boundary, surface_mesh_data) = load_sample(
        geo_path, re_case, mesh_re=config_v2.mesh_re, return_surface_mesh=True,
    )
    all_points, wall_face_vertices, _ = surface_mesh_data

    poly_path = config_v2.get_geometry_path(geometry_folder) / config_v2.mesh_re / "constant" / "polyMesh"
    all_faces = parse_openfoam_faces(str(poly_path / "faces"))

    # Compute v2 features
    x_raw = compute_physics_features_v2(
        wall_centres       = wall_centres,
        all_points         = all_points,
        wall_face_vertices = wall_face_vertices,
        boundary           = boundary,
        all_faces          = all_faces,
        edge_index         = edge_index,
    )

    # Normalise
    x_mean = torch.tensor(norm_stats["x_mean"],   dtype=torch.float32)
    x_std  = torch.tensor(norm_stats["x_std"],    dtype=torch.float32)
    e_mean = torch.tensor(norm_stats["edge_mean"], dtype=torch.float32)
    e_std  = torch.tensor(norm_stats["edge_std"],  dtype=torch.float32)

    x_t  = (torch.tensor(x_raw,      dtype=torch.float32) - x_mean) / x_std
    ei_t = torch.tensor(edge_index,   dtype=torch.long)
    ea_t = (torch.tensor(edge_attr,   dtype=torch.float32) - e_mean) / e_std

    data = Data(
        x          = x_t,
        edge_index = ei_t,
        edge_attr  = ea_t,
        re         = torch.tensor([re_number], dtype=torch.float32),
        angle      = torch.tensor([float(angle)], dtype=torch.float32),
    ).to(device)

    return wall_centres, data


# ============================================================================
# MC inference
# ============================================================================

def predict_single_v2(
    model:            BifurcationWSSPredictorV2,
    norm_stats:       Dict,
    geometry_folder:  str,
    re_number:        float,
    device:           Optional[torch.device] = None,
    mc_samples:       int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference for one (geometry, Re) pair with MC uncertainty.

    Parameters
    ----------
    mc_samples : int
        Number of stochastic forward passes (≥1).
        mc_samples=1 → deterministic (no uncertainty).

    Returns
    -------
    coords    : (N, 3)  wall face centroids (physical units)
    wss_mean  : (N, 3)  mean predicted WSS (physical units)
    wss_std   : (N, 3)  std across MC passes (physical units, same scale)
    """
    device = device or next(model.parameters()).device

    coords, data = _build_inference_graph(
        geometry_folder, re_number, norm_stats, device
    )

    if mc_samples > 1:
        model.enable_mc_mode()
    else:
        model.disable_mc_mode()

    all_preds = []
    with torch.no_grad():
        for _ in range(mc_samples):
            y_norm = model(data)                          # [N, 3] normalised
            y_phys = denormalize_wss_v2(y_norm, norm_stats)
            all_preds.append(y_phys.cpu())

    model.disable_mc_mode()

    preds    = torch.stack(all_preds, dim=0)             # [S, N, 3]
    wss_mean = preds.mean(0).numpy()                     # [N, 3]
    wss_std  = preds.std(0).numpy()                      # [N, 3]

    return coords, wss_mean, wss_std


# ============================================================================
# Branch summary helper
# ============================================================================

def _print_branch_summary(coords: np.ndarray, wss_mean: np.ndarray, wss_std: np.ndarray):
    """
    Print mean WSS magnitude per spatial region (above/below median Z).

    Simple proxy for parent vs daughter branch when branch_depth labels
    are not stored on the inference coords.
    """
    z = coords[:, 2]
    z_med   = np.median(z)
    parent  = z < z_med
    daughter = ~parent

    for label, mask in [("Parent (lower Z)", parent), ("Daughter (upper Z)", daughter)]:
        if mask.any():
            mag  = np.linalg.norm(wss_mean[mask], axis=1)
            smag = np.linalg.norm(wss_std[mask],  axis=1)
            print(f"  {label}: n={mask.sum():6d}  "
                  f"mean_mag={mag.mean():.4e}  "
                  f"mean_std={smag.mean():.4e}")


# ============================================================================
# OpenFOAM writer  (same as v1, reused here for self-containment)
# ============================================================================

def _write_openfoam_wss(filepath: str, wss_vectors: np.ndarray, boundary_info: dict):
    header = """\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    location    "1";
    object      wallShearStress;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{{
"""
    with open(filepath, "w") as f:
        f.write(header)
        offset = 0
        for name, info in boundary_info.items():
            if info["type"] == "wall":
                f.write(f"    {name}\n    {{\n")
                f.write(f"        type            calculated;\n")
                f.write(f"        value           nonuniform List<vector>\n")
                f.write(f"{info['nFaces']}\n(\n")
                for i in range(info["nFaces"]):
                    v = wss_vectors[offset + i]
                    f.write(f"({v[0]:.10e} {v[1]:.10e} {v[2]:.10e})\n")
                f.write(")\n;\n    }\n")
                offset += info["nFaces"]
            else:
                f.write(f"    {name}\n    {{\n")
                f.write("        type            calculated;\n")
                f.write("        value           uniform (0 0 0);\n")
                f.write("    }\n")
        f.write("}\n\n// ***************** //\n")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inference v2 — MC uncertainty")
    parser.add_argument("--model",      type=str,
                        default=str(config_v2.models_dir / "best_model_v2.pt"))
    parser.add_argument("--geometry",   type=str, default=None,
                        help="Geometry folder (e.g. bifurcation_angle45_750_ascii)")
    parser.add_argument("--re",         type=float, default=None,
                        help="Single Re to predict")
    parser.add_argument("--re-range",   nargs=3, type=int, default=None,
                        metavar=("START", "STOP", "STEP"))
    parser.add_argument("--all",        action="store_true",
                        help="Predict all 189 (geo, Re) pairs")
    parser.add_argument("--output",     type=str, default=None)
    parser.add_argument("--mc-samples", type=int, default=50,
                        help="MC forward passes (default 50; use 1 for deterministic)")
    parser.add_argument("--paraview",   action="store_true",
                        help="Export full ParaView case with uncertainty field")
    parser.add_argument("--device",     type=str, default=None)
    args = parser.parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    out_root = Path(args.output) if args.output else config_v2.predictions_dir
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model} …")
    model, norm_stats = load_model_v2(args.model, device)
    print(f"MC samples per prediction: {args.mc_samples}")

    # Determine (geo, re) pairs
    pairs = []
    if args.all:
        for geo in config_v2.geometry_folders:
            for re_val in config_v2.re_values:
                pairs.append((geo, float(re_val)))
    elif args.geometry:
        if args.re is not None:
            pairs.append((args.geometry, args.re))
        elif args.re_range is not None:
            start, stop, step = args.re_range
            for r in range(start, stop + 1, step):
                pairs.append((args.geometry, float(r)))
        else:
            for r in config_v2.re_values:
                pairs.append((args.geometry, float(r)))
    else:
        parser.error("Specify --geometry + (--re | --re-range), or --all")

    print(f"Predicting {len(pairs)} case(s) …\n")

    for geo, re_val in pairs:
        re_tag   = f"Re{int(re_val)}"
        case_dir = out_root / geo / re_tag
        case_dir.mkdir(parents=True, exist_ok=True)

        print(f"  {geo} / {re_tag} … ", end="", flush=True)

        coords, wss_mean, wss_std = predict_single_v2(
            model, norm_stats, geo, re_val, device, mc_samples=args.mc_samples
        )

        # Branch summary
        if args.mc_samples > 1:
            print()
            _print_branch_summary(coords, wss_mean, wss_std)

        # Save CSV
        mag_mean = np.linalg.norm(wss_mean, axis=1)
        mag_std  = np.linalg.norm(wss_std,  axis=1)
        csv_path = case_dir / "predicted_wss_v2.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "x", "y", "z",
                "wss_x", "wss_y", "wss_z", "wss_magnitude",
                "std_x", "std_y", "std_z", "std_magnitude",
            ])
            for c, wm, ws, mm, ms in zip(coords, wss_mean, wss_std, mag_mean, mag_std):
                w.writerow([*c, *wm, mm, *ws, ms])

        # Optional ParaView export with uncertainty
        if args.paraview:
            geo_src = config_v2.get_geometry_path(geo) / config_v2.mesh_re
            for sub in ["constant", "system", "0"]:
                src_sub = geo_src / sub
                dst_sub = case_dir / sub
                if src_sub.exists() and not dst_sub.exists():
                    shutil.copytree(str(src_sub), str(dst_sub))

            ts_dir = case_dir / "1"
            ts_dir.mkdir(exist_ok=True)
            boundary = parse_boundary(
                str(case_dir / "constant" / "polyMesh" / "boundary")
            )
            _write_openfoam_wss(
                str(ts_dir / "wallShearStress"), wss_mean, boundary
            )
            # Write std_magnitude as a scalar field
            _write_scalar_openfoam(
                str(ts_dir / "wss_uncertainty"), mag_std, boundary
            )
            (case_dir / "prediction_v2.foam").touch()

        print(f"done → {csv_path}")

    print(f"\nAll predictions → {out_root}/")


def _write_scalar_openfoam(filepath: str, scalar_field: np.ndarray, boundary_info: dict):
    """Write a scalar per-face field in OpenFOAM ASCII format (for uncertainty)."""
    header = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "1";
    object      wss_uncertainty;
}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
"""
    with open(filepath, "w") as f:
        f.write(header)
        offset = 0
        for name, info in boundary_info.items():
            if info["type"] == "wall":
                f.write(f"    {name}\n    {{\n")
                f.write("        type            calculated;\n")
                f.write(f"        value           nonuniform List<scalar>\n")
                f.write(f"{info['nFaces']}\n(\n")
                for i in range(info["nFaces"]):
                    f.write(f"{scalar_field[offset + i]:.10e}\n")
                f.write(")\n;\n    }\n")
                offset += info["nFaces"]
            else:
                f.write(f"    {name}\n    {{\n")
                f.write("        type            calculated;\n")
                f.write("        value           uniform 0;\n")
                f.write("    }\n")
        f.write("}\n\n// ***************** //\n")


if __name__ == "__main__":
    main()

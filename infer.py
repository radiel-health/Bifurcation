"""
Inference & ParaView export for bifurcation WSS predictions.

Usage:
    # Single prediction
    python -m Bifurcation.infer \\
        --model Models/best_model.pt \\
        --geometry bifurcation_angle45_750_ascii \\
        --re 500

    # Sweep all Re for one geometry
    python -m Bifurcation.infer \\
        --model Models/best_model.pt \\
        --geometry bifurcation_angle45_750_ascii \\
        --re-range 100 2100 100

    # Predict on every (geometry, Re) pair
    python -m Bifurcation.infer \\
        --model Models/best_model.pt \\
        --all
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
from torch_geometric.data import Data

from Bifurcation.config import config
from Bifurcation.dataset import (
    denormalize_wss,
    load_sample,
    parse_boundary,
    load_normalization_stats,
)
from Bifurcation.model import BifurcationWSSPredictor


# ============================================================================
# Model loading
# ============================================================================

def load_model(
    ckpt_path: str,
    device: torch.device | None = None,
) -> tuple[BifurcationWSSPredictor, Dict]:
    """
    Load a trained model + normalisation stats from a checkpoint.

    Returns (model, norm_stats).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = ckpt.get("config", {})
    model = BifurcationWSSPredictor(
        node_feat_dim=cfg.get("node_feat_dim", config.node_feat_dim),
        edge_feat_dim=cfg.get("edge_feat_dim", config.edge_feat_dim),
        hidden_dim=cfg.get("hidden_dim", config.hidden_dim),
        out_channels=cfg.get("output_dim", config.output_dim),
        num_layers=cfg.get("num_layers", config.num_layers),
        context_dim=cfg.get("context_dim", config.context_dim),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    norm_stats = ckpt.get("norm_stats")
    if norm_stats is None:
        norm_stats = load_normalization_stats()

    return model, norm_stats


# ============================================================================
# Single-sample prediction
# ============================================================================

@torch.no_grad()
def predict_single(
    model: BifurcationWSSPredictor,
    norm_stats: Dict,
    geometry_folder: str,
    re_number: float,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference for a single (geometry, Re) pair.

    Returns
    -------
    coords : (N, 3)   wall face centers
    wss    : (N, 3)   predicted WSS in physical units
    """
    device = device or next(model.parameters()).device
    geo_path = str(config.get_geometry_path(geometry_folder))
    angle, _mesh = config.parse_geometry_folder(geometry_folder)

    # Load raw mesh (no target needed – we predict it)
    re_case = f"Re{int(re_number)}"
    coords, _wss_gt, edge_index, edge_attr, _re, _boundary = load_sample(
        geo_path, re_case, mesh_re=config.mesh_re
    )

    # Build normalised graph
    x_mean = torch.tensor(norm_stats["x_mean"], dtype=torch.float32)
    x_std = torch.tensor(norm_stats["x_std"], dtype=torch.float32)
    e_mean = torch.tensor(norm_stats["edge_mean"], dtype=torch.float32)
    e_std = torch.tensor(norm_stats["edge_std"], dtype=torch.float32)

    x_t = (torch.tensor(coords, dtype=torch.float32) - x_mean) / x_std
    ei_t = torch.tensor(edge_index, dtype=torch.long)
    ea_t = (torch.tensor(edge_attr, dtype=torch.float32) - e_mean) / e_std

    data = Data(
        x=x_t,
        edge_index=ei_t,
        edge_attr=ea_t,
        re=torch.tensor([re_number], dtype=torch.float32),
        angle=torch.tensor([float(angle)], dtype=torch.float32),
    ).to(device)

    y_norm = model(data)                         # [N, 3] normalised
    y_phys = denormalize_wss(y_norm, norm_stats)  # [N, 3] physical units

    return coords, y_phys.cpu().numpy()


# ============================================================================
# OpenFOAM WSS writer
# ============================================================================

def write_openfoam_wss(
    filepath: str,
    wss_vectors: np.ndarray,
    boundary_info: dict,
):
    """
    Write predicted WSS vectors in OpenFOAM ASCII format.
    """
    header = """\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     |
    \\\\  /    A nd           |
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    location    "1";
    object      wallShearStress;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
"""

    with open(filepath, "w") as f:
        f.write(header)

        wss_offset = 0
        for name, info in boundary_info.items():
            if info["type"] == "wall":
                f.write(f"    {name}\n")
                f.write(f"    {{\n")
                f.write(f"        type            calculated;\n")
                f.write(f"        value           nonuniform List<vector>\n")
                f.write(f"{info['nFaces']}\n(\n")
                for i in range(info["nFaces"]):
                    v = wss_vectors[wss_offset + i]
                    f.write(f"({v[0]:.10e} {v[1]:.10e} {v[2]:.10e})\n")
                f.write(")\n;\n")
                f.write("    }\n")
                wss_offset += info["nFaces"]
            else:
                f.write(f"    {name}\n")
                f.write("    {\n")
                f.write("        type            calculated;\n")
                f.write("        value           uniform (0 0 0);\n")
                f.write("    }\n")

        f.write("}\n\n")
        f.write("// ************************************************************************* //\n")


# ============================================================================
# Full ParaView export
# ============================================================================

def export_prediction_for_paraview(
    pred_wss: np.ndarray,
    geometry_folder: str,
    output_dir: str,
    mesh_re: str = "Re100",
):
    """
    Create a complete ParaView-readable OpenFOAM case with predicted WSS.

    Copies constant/, system/, 0/ from the reference Re case, writes
    ``1/wallShearStress`` with the model's prediction, and creates an
    empty ``prediction.foam`` entry point.
    """
    geo_path = config.get_geometry_path(geometry_folder)
    src = geo_path / mesh_re

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy mesh, system, BCs
    for sub in ["constant", "system", "0"]:
        src_sub = src / sub
        dst_sub = output_dir / sub
        if src_sub.exists() and not dst_sub.exists():
            shutil.copytree(str(src_sub), str(dst_sub))

    # Write prediction at timestep "1"
    ts_dir = output_dir / "1"
    ts_dir.mkdir(exist_ok=True)

    boundary = parse_boundary(str(output_dir / "constant" / "polyMesh" / "boundary"))
    write_openfoam_wss(str(ts_dir / "wallShearStress"), pred_wss, boundary)

    # Empty .foam file (ParaView entry point)
    (output_dir / "prediction.foam").touch()

    print(f"Prediction exported → {output_dir}/")
    print(f"Open {output_dir / 'prediction.foam'} in ParaView to visualise.")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inference for bifurcation WSS model")
    parser.add_argument("--model", type=str, default=str(config.models_dir / "best_model.pt"))
    parser.add_argument("--geometry", type=str, default=None,
                        help="Geometry folder name (e.g. bifurcation_angle45_750_ascii)")
    parser.add_argument("--re", type=float, default=None, help="Single Re to predict")
    parser.add_argument("--re-range", nargs=3, type=int, default=None,
                        metavar=("START", "STOP", "STEP"),
                        help="Re sweep, e.g. --re-range 100 2100 100")
    parser.add_argument("--all", action="store_true",
                        help="Predict for all 189 (geo, Re) pairs")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: predictions/)")
    parser.add_argument("--paraview", action="store_true",
                        help="Export full ParaView case (not just CSV)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root = Path(args.output) if args.output else config.predictions_dir
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model} ...")
    model, norm_stats = load_model(args.model, device)

    # Determine which (geo, re) pairs to predict
    pairs = []
    if args.all:
        for geo in config.geometry_folders:
            for re_val in config.re_values:
                pairs.append((geo, float(re_val)))
    elif args.geometry:
        if args.re is not None:
            pairs.append((args.geometry, args.re))
        elif args.re_range is not None:
            start, stop, step = args.re_range
            for r in range(start, stop + 1, step):
                pairs.append((args.geometry, float(r)))
        else:
            # Default: all Re for this geometry
            for r in config.re_values:
                pairs.append((args.geometry, float(r)))
    else:
        parser.error("Specify --geometry + --re, --re-range, or --all")

    print(f"Predicting {len(pairs)} cases ...\n")

    for geo, re_val in pairs:
        re_tag = f"Re{int(re_val)}"
        print(f"  {geo} / {re_tag} ... ", end="", flush=True)

        coords, wss_pred = predict_single(model, norm_stats, geo, re_val, device)

        case_dir = out_root / geo / re_tag

        if args.paraview:
            export_prediction_for_paraview(wss_pred, geo, str(case_dir))
        else:
            # Save lightweight CSV
            case_dir.mkdir(parents=True, exist_ok=True)
            import csv
            csv_path = case_dir / "predicted_wss.csv"
            mag = np.linalg.norm(wss_pred, axis=1)
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["x", "y", "z", "wss_x", "wss_y", "wss_z", "wss_magnitude"])
                for c, wv, m in zip(coords, wss_pred, mag):
                    w.writerow([*c, *wv, m])

        print("done")

    print(f"\nAll predictions written to {out_root}/")


if __name__ == "__main__":
    main()

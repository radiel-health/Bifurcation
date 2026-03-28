"""
Inference script v4 — pulsatile WSS prediction with ParaView export.

Runs the phase-conditioned v4 model over any (geometry, Re) pair.
In --pulsatile mode, loops over 20 cardiac phases and writes one OpenFOAM
timestep directory per phase so ParaView can animate the full cardiac cycle.
Also computes TAWSS and OSI and writes them as scalar fields.

Usage:
    # Single phase (default φ=0.0, peak systole)
    python -m Bifurcation.infer_v4 \\
        --model Models_v4/best_model_v4.pt \\
        --geometry bifurcation_angle45_750_ascii \\
        --re 500

    # Full cardiac cycle → ParaView animation
    python -m Bifurcation.infer_v4 \\
        --model Models_v4/best_model_v4.pt \\
        --geometry bifurcation_angle45_750_ascii \\
        --re 500 --pulsatile --paraview

    # Sweep Re values
    python -m Bifurcation.infer_v4 \\
        --model Models_v4/best_model_v4.pt \\
        --geometry bifurcation_angle45_750_ascii \\
        --re-range 300 2100 100 --pulsatile --paraview

    Then open predictions_v4/bifurcation_angle45_750_ascii/Re500/prediction_v4.foam
    in ParaView → Apply → Play to animate the cardiac cycle.
"""

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from Bifurcation.config_v4 import config_v4
from Bifurcation.dataset import load_sample, parse_boundary
from Bifurcation.dataset import parse_openfoam_faces
from Bifurcation.dataset_v2 import compute_physics_features_v2
from Bifurcation.dataset_v4 import denormalize_wss_v4
from Bifurcation.Models.bif_v4 import BifurcationWSSPredictorV4


# ============================================================================
# Model loading
# ============================================================================

def load_model_v4(
    ckpt_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[BifurcationWSSPredictorV4, Dict]:
    """Load a v4 checkpoint. Returns (model, norm_stats)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg    = ckpt.get("config", {})

    model = BifurcationWSSPredictorV4(
        node_feat_dim  = cfg.get("node_feat_dim",  config_v4.node_feat_dim),
        edge_feat_dim  = cfg.get("edge_feat_dim",  config_v4.edge_feat_dim),
        hidden_dim     = cfg.get("hidden_dim",     config_v4.hidden_dim),
        num_heads      = cfg.get("num_heads",      config_v4.num_heads),
        out_channels   = cfg.get("output_dim",     config_v4.output_dim),
        num_layers     = cfg.get("num_layers",     config_v4.num_layers),
        context_dim    = cfg.get("context_dim",    config_v4.context_dim),
        flow_param_dim = cfg.get("flow_param_dim", config_v4.flow_param_dim),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    norm_stats = ckpt.get("norm_stats")
    if norm_stats is None:
        raise RuntimeError("Checkpoint has no norm_stats — retrain with train_v4.py")

    epoch = ckpt.get("epoch", "?")
    val   = ckpt.get("val_loss", float("nan"))
    print(f"  Loaded checkpoint: epoch={epoch}  val_loss={val:.4f}")
    return model, norm_stats


# ============================================================================
# Graph builder
# ============================================================================

def _build_inference_graph(
    geometry_folder: str,
    re_number:       float,
    norm_stats:      Dict,
    device:          torch.device,
) -> Tuple[np.ndarray, Data, Dict]:
    """
    Load steady-state mesh + compute 10-dim physics features → normalised Data.

    Returns
    -------
    coords   : (N, 3)  raw wall face centroid coordinates
    data     : normalised PyG Data (x, edge_index, edge_attr, re, angle)
               — phase is NOT set here; caller sets it per forward pass
    boundary : dict from parse_boundary (for OpenFOAM writer)
    """
    geo_path = str(config_v4.data_root_steady / geometry_folder)
    angle, _ = config_v4.parse_geometry_folder(geometry_folder)
    re_case  = f"Re{int(re_number)}"

    (wall_centres, _wss_gt, edge_index, edge_attr,
     _re, boundary, surface_mesh_data) = load_sample(
        geo_path, re_case, mesh_re=config_v4.mesh_re, return_surface_mesh=True,
    )
    all_points, wall_face_vertices, _ = surface_mesh_data

    poly_path = (config_v4.data_root_steady / geometry_folder
                 / config_v4.mesh_re / "constant" / "polyMesh")
    all_faces = parse_openfoam_faces(str(poly_path / "faces"))

    x_raw = compute_physics_features_v2(
        wall_centres       = wall_centres,
        all_points         = all_points,
        wall_face_vertices = wall_face_vertices,
        boundary           = boundary,
        all_faces          = all_faces,
        edge_index         = edge_index,
    )  # (N, 10)

    x_mean = torch.tensor(norm_stats["x_mean"][:10], dtype=torch.float32)
    x_std  = torch.tensor(norm_stats["x_std"][:10],  dtype=torch.float32)
    e_mean = torch.tensor(norm_stats["edge_mean"],    dtype=torch.float32)
    e_std  = torch.tensor(norm_stats["edge_std"],     dtype=torch.float32)

    x_t  = (torch.tensor(x_raw,    dtype=torch.float32) - x_mean) / x_std
    ei_t = torch.tensor(edge_index, dtype=torch.long)
    ea_t = (torch.tensor(edge_attr, dtype=torch.float32) - e_mean) / e_std

    data = Data(
        x          = x_t.to(device),
        edge_index = ei_t.to(device),
        edge_attr  = ea_t.to(device),
        re         = torch.tensor([re_number],    dtype=torch.float32).to(device),
        angle      = torch.tensor([float(angle)], dtype=torch.float32).to(device),
    )

    return wall_centres, data, boundary


# ============================================================================
# Inference
# ============================================================================

@torch.no_grad()
def predict_phase(
    model:      BifurcationWSSPredictorV4,
    norm_stats: Dict,
    data:       Data,
    phase:      float,
) -> np.ndarray:
    """Single forward pass at given cardiac phase. Returns WSS [N, 3] in Pa."""
    data.phase = torch.tensor([phase], dtype=torch.float32).to(data.x.device)
    y_norm = model(data)
    return denormalize_wss_v4(y_norm, norm_stats).cpu().numpy()


def predict_pulsatile(
    model:      BifurcationWSSPredictorV4,
    norm_stats: Dict,
    data:       Data,
    n_phases:   int = 20,
) -> Tuple[np.ndarray, List[float]]:
    """
    Run inference at n_phases evenly-spaced cardiac phases.

    Returns
    -------
    wss_all : (n_phases, N, 3)  WSS at each phase (Pa)
    phases  : list of phase values in [0, 1)
    """
    phases  = [i / n_phases for i in range(n_phases)]
    wss_all = np.stack([predict_phase(model, norm_stats, data, phi)
                        for phi in phases])
    return wss_all, phases


def compute_tawss_osi(wss_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    wss_all : (T, N, 3)
    Returns TAWSS (N,) and OSI (N,) in physical units.
    """
    tawss    = np.mean(np.linalg.norm(wss_all, axis=2), axis=0)        # (N,)
    mean_vec = np.mean(wss_all, axis=0)                                  # (N, 3)
    osi      = 0.5 * (1 - np.linalg.norm(mean_vec, axis=1) /
                      (tawss + 1e-10))                                   # (N,)
    return tawss, osi


# ============================================================================
# OpenFOAM writers  (reused verbatim from infer_v3.py)
# ============================================================================

def _write_openfoam_wss(filepath: str, wss_vectors: np.ndarray, boundary_info: dict):
    header = """\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    location    "1";
    object      wallShearStress;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];
internalField   uniform (0 0 0);
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
                f.write("        value           nonuniform List<vector>\n")
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


def _write_scalar_openfoam(filepath: str, scalar_field: np.ndarray,
                            boundary_info: dict, obj_name: str = "tawss"):
    header = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "1";
    object      {obj_name};
}}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{{
"""
    with open(filepath, "w") as f:
        f.write(header)
        offset = 0
        for name, info in boundary_info.items():
            if info["type"] == "wall":
                f.write(f"    {name}\n    {{\n")
                f.write("        type            calculated;\n")
                f.write("        value           nonuniform List<scalar>\n")
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


# ============================================================================
# ParaView case writer
# ============================================================================

def _setup_paraview_case(case_dir: Path, geo: str):
    """Copy constant/system/0 from steady-state mesh into case_dir."""
    geo_src = config_v4.data_root_steady / geo / config_v4.mesh_re
    for sub in ["constant", "system", "0"]:
        src_sub = geo_src / sub
        dst_sub = case_dir / sub
        if src_sub.exists() and not dst_sub.exists():
            shutil.copytree(str(src_sub), str(dst_sub))


def _write_paraview_pulsatile(
    case_dir:   Path,
    wss_all:    np.ndarray,
    phases:     List[float],
    tawss:      np.ndarray,
    osi:        np.ndarray,
    boundary:   dict,
):
    """
    Write one timestep directory per phase, plus a tawss_osi/ directory.
    Timestep dirs are named by phase value (e.g. "0.0", "0.05", ...).
    """
    for phi, wss in zip(phases, wss_all):
        ts_dir = case_dir / f"{phi:.4f}"
        ts_dir.mkdir(exist_ok=True)
        _write_openfoam_wss(str(ts_dir / "wallShearStress"), wss, boundary)

    # TAWSS and OSI in a separate timestep dir (99.0 so ParaView shows it last)
    clinical_dir = case_dir / "99.0"
    clinical_dir.mkdir(exist_ok=True)
    _write_scalar_openfoam(str(clinical_dir / "tawss"), tawss, boundary, "tawss")
    _write_scalar_openfoam(str(clinical_dir / "osi"),   osi,   boundary, "osi")

    (case_dir / "prediction_v4.foam").touch()
    print(f"    ParaView case → {case_dir}/prediction_v4.foam")
    print(f"    Open in ParaView → color by 'wallShearStress' magnitude → Play")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inference v4 — pulsatile WSS + ParaView")
    parser.add_argument("--model",      type=str,
                        default=str(config_v4.models_dir / "best_model_v4.pt"))
    parser.add_argument("--geometry",   type=str, default=None,
                        help="Geometry folder (e.g. bifurcation_angle45_750_ascii)")
    parser.add_argument("--re",         type=float, default=None,
                        help="Single Re to predict")
    parser.add_argument("--re-range",   nargs=3, type=int, default=None,
                        metavar=("START", "STOP", "STEP"))
    parser.add_argument("--all",        action="store_true",
                        help="Predict all (geo, Re) pairs")
    parser.add_argument("--phase",      type=float, default=0.25,
                        help="Cardiac phase φ∈[0,1) for single-phase mode (default 0.25 = peak)")
    parser.add_argument("--pulsatile",  action="store_true",
                        help="Loop over 20 phases instead of a single phase")
    parser.add_argument("--n-phases",   type=int, default=20,
                        help="Number of cardiac phases for --pulsatile (default 20)")
    parser.add_argument("--paraview",   action="store_true",
                        help="Export OpenFOAM case for ParaView (requires --pulsatile)")
    parser.add_argument("--output",     type=str, default=None)
    parser.add_argument("--device",     type=str, default=None)
    args = parser.parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    out_root = Path(args.output) if args.output else config_v4.predictions_dir
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model} …")
    model, norm_stats = load_model_v4(args.model, device)

    # Determine (geo, re) pairs
    pairs = []
    if args.all:
        for geo in config_v4.geometry_folders:
            for re_val in config_v4.re_values:
                pairs.append((geo, float(re_val)))
    elif args.geometry:
        if args.re is not None:
            pairs.append((args.geometry, args.re))
        elif args.re_range is not None:
            start, stop, step = args.re_range
            for r in range(start, stop + 1, step):
                pairs.append((args.geometry, float(r)))
        else:
            for r in config_v4.re_values:
                pairs.append((args.geometry, float(r)))
    else:
        parser.error("Specify --geometry + (--re | --re-range), or --all")

    print(f"Predicting {len(pairs)} case(s) "
          f"({'pulsatile' if args.pulsatile else f'phase={args.phase}'}) …\n")

    for geo, re_val in pairs:
        re_tag   = f"Re{int(re_val)}"
        case_dir = out_root / geo / re_tag
        case_dir.mkdir(parents=True, exist_ok=True)

        print(f"  {geo} / {re_tag} … ", end="", flush=True)

        coords, data, boundary = _build_inference_graph(geo, re_val, norm_stats, device)

        if args.pulsatile:
            wss_all, phases = predict_pulsatile(model, norm_stats, data, args.n_phases)
            tawss, osi = compute_tawss_osi(wss_all)

            # CSV: TAWSS and OSI per node
            csv_path = case_dir / "predicted_tawss_osi_v4.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["x", "y", "z", "tawss", "osi"])
                for c, t, o in zip(coords, tawss, osi):
                    w.writerow([*c, t, o])

            print(f"done  TAWSS_mean={tawss.mean():.4e}  OSI_mean={osi.mean():.4f}")
            print(f"    CSV -> {csv_path}")

            if args.paraview:
                _setup_paraview_case(case_dir, geo)
                _write_paraview_pulsatile(
                    case_dir, wss_all, phases, tawss, osi, boundary
                )

        else:
            wss = predict_phase(model, norm_stats, data, args.phase)
            mag = np.linalg.norm(wss, axis=1)

            csv_path = case_dir / f"predicted_wss_v4_phi{args.phase:.2f}.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["x", "y", "z", "wss_x", "wss_y", "wss_z", "wss_magnitude"])
                for c, ws, m in zip(coords, wss, mag):
                    w.writerow([*c, *ws, m])

            print(f"done  mag_mean={mag.mean():.4e}")
            print(f"    CSV -> {csv_path}")

            if args.paraview:
                _setup_paraview_case(case_dir, geo)
                ts_dir = case_dir / "1"
                ts_dir.mkdir(exist_ok=True)
                _write_openfoam_wss(str(ts_dir / "wallShearStress"), wss, boundary)
                (case_dir / "prediction_v4.foam").touch()
                print(f"    ParaView → {case_dir}/prediction_v4.foam")

    print(f"\nAll predictions → {out_root}/")


if __name__ == "__main__":
    main()

"""
demo_v4.py — Interactive pulsatile WSS demo for the v4 bifurcation model.

Two modes
---------
Interactive UI (default):
    python -m Bifurcation.demo_v4

ParaView export (writes 20 OpenFOAM phase directories):
    python -m Bifurcation.demo_v4 --export-paraview --angle 45 --re 500

Required files (download from RunPod via Jupyter Lab):
    Bifurcation/Models_v4/best_model_v4.pt
    Bifurcation/ProcessedData_v4/bifurcation_angle30_750_ascii/Re500/t4.5.pt
    Bifurcation/ProcessedData_v4/bifurcation_angle45_750_ascii/Re500/t4.5.pt
    Bifurcation/ProcessedData_v4/bifurcation_angle60_500_ascii/Re500/t4.5.pt
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from Bifurcation.config_v4 import config_v4
from Bifurcation.dataset_v4 import denormalize_wss_v4
from Bifurcation.Models.bif_v4 import BifurcationWSSPredictorV4


# ── graph file locations ──────────────────────────────────────────────────────
# angle60 uses _500 mesh only (750/1000 simulations failed)

_GEO_INFO = {
    30: ("bifurcation_angle30_750_ascii",
         config_v4.processed_data_dir / "bifurcation_angle30_750_ascii" / "Re500" / "t4.5.pt"),
    45: ("bifurcation_angle45_750_ascii",
         config_v4.processed_data_dir / "bifurcation_angle45_750_ascii" / "Re500" / "t4.5.pt"),
    60: ("bifurcation_angle60_500_ascii",
         config_v4.processed_data_dir / "bifurcation_angle60_500_ascii" / "Re500" / "t4.5.pt"),
}


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(
    ckpt_path: str = None,
    device: torch.device = None,
) -> Tuple[BifurcationWSSPredictorV4, Dict, int]:
    """Load v4 checkpoint. Returns (model, norm_stats, epoch)."""
    device    = device or torch.device("cpu")
    ckpt_path = Path(ckpt_path or config_v4.models_dir / "best_model_v4.pt")

    if not ckpt_path.exists():
        sys.exit(
            f"Model not found: {ckpt_path}\n"
            "Download best_model_v4.pt from RunPod via Jupyter Lab."
        )

    print(f"Loading model from {ckpt_path} …")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]

    model = BifurcationWSSPredictorV4(
        node_feat_dim  = cfg["node_feat_dim"],
        edge_feat_dim  = cfg["edge_feat_dim"],
        hidden_dim     = cfg["hidden_dim"],
        num_heads      = cfg["num_heads"],
        out_channels   = cfg["output_dim"],
        num_layers     = cfg["num_layers"],
        context_dim    = cfg["context_dim"],
        flow_param_dim = cfg["flow_param_dim"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    epoch = ckpt["epoch"]
    print(f"  Epoch {epoch}, val_loss={ckpt['val_loss']:.4f}")
    return model, ckpt["norm_stats"], epoch


# ── graph loading ─────────────────────────────────────────────────────────────

def load_graph(angle_deg: int) -> Data:
    geo_name, pt_path = _GEO_INFO[angle_deg]
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Missing: {pt_path}\n"
            f"  Download from RunPod: /Bifurcation/ProcessedData_v4/"
            f"{geo_name}/Re500/t4.5.pt"
        )
    return torch.load(str(pt_path), weights_only=False)


# ── inference ─────────────────────────────────────────────────────────────────

def _normalize(raw_data: Data, norm_stats: Dict):
    """Return normalized (x, edge_index, edge_attr) without modifying raw_data."""
    x_mean = torch.tensor(norm_stats["x_mean"][:10], dtype=torch.float32)
    x_std  = torch.tensor(norm_stats["x_std"][:10],  dtype=torch.float32)
    e_mean = torch.tensor(norm_stats["edge_mean"],    dtype=torch.float32)
    e_std  = torch.tensor(norm_stats["edge_std"],     dtype=torch.float32)
    return (
        (raw_data.x - x_mean) / x_std,
        raw_data.edge_index,
        (raw_data.edge_attr - e_mean) / e_std,
    )


@torch.no_grad()
def run_phases(
    model:      BifurcationWSSPredictorV4,
    raw_data:   Data,
    norm_stats: Dict,
    re_val:     float,
    angle_deg:  int,
    n_phases:   int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sequential inference across n_phases evenly-spaced cardiac phases.
    (One phase at a time — avoids batching N*n_phases nodes on CPU.)

    Returns
    -------
    wss_mag : (n_phases, N)  WSS magnitude in Pa
    phases  : (n_phases,)    phase values ∈ [0, 1)
    """
    x_norm, edge_index, e_norm = _normalize(raw_data, norm_stats)
    N        = x_norm.shape[0]
    phases_t = torch.linspace(0, 1, n_phases + 1)[:-1]

    frames = []
    for phi in phases_t:
        g = Data(
            x          = x_norm,
            edge_index = edge_index,
            edge_attr  = e_norm,
            re         = torch.tensor([re_val],           dtype=torch.float32),
            angle      = torch.tensor([float(angle_deg)], dtype=torch.float32),
            phase      = phi.unsqueeze(0).float(),
        )
        y_norm = model(g)                                    # [N, 3]
        y_phys = denormalize_wss_v4(y_norm, norm_stats)     # [N, 3]
        frames.append(torch.norm(y_phys, dim=1).numpy())    # [N]

    return np.stack(frames), phases_t.numpy()  # [n_phases, N], [n_phases]


# ── ParaView export ───────────────────────────────────────────────────────────

def _write_openfoam_wss(filepath: str, wss_vectors: np.ndarray, boundary_info: dict):
    """Write WSS vectors in OpenFOAM ASCII format (mirrors infer_v3.py)."""
    ts = Path(filepath).parent.name
    header = (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        "    class       volVectorField;\n"
        f'    location    "{ts}";\n'
        "    object      wallShearStress;\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
        "dimensions      [0 2 -2 0 0 0 0];\n"
        "internalField   uniform (0 0 0);\n"
        "boundaryField\n{\n"
    )
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


def export_paraview(
    model:      BifurcationWSSPredictorV4,
    raw_data:   Data,
    norm_stats: Dict,
    re_val:     float,
    angle_deg:  int,
    geo_folder: str,
    n_phases:   int = 20,
):
    """
    Write n_phases OpenFOAM timestep directories loadable in ParaView.

    Output: predictions_v4/<geo_folder>/Re<re_val>/
      → Open case.foam in ParaView, Apply, color by wallShearStress, Play.
    """
    from Bifurcation.dataset import parse_boundary

    geo_src = config_v4.get_steady_geometry_path(geo_folder) / config_v4.mesh_re
    if not geo_src.exists():
        sys.exit(
            f"Steady-state mesh not found: {geo_src}\n"
            "ParaView export requires the local OpenFOAM data directory."
        )

    out_dir = config_v4.predictions_dir / geo_folder / f"Re{int(re_val)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy case skeleton (constant/, system/, 0/) so ParaView can load the mesh
    for sub in ["constant", "system", "0"]:
        src, dst = geo_src / sub, out_dir / sub
        if src.exists() and not dst.exists():
            shutil.copytree(str(src), str(dst))

    boundary_info = parse_boundary(
        str(out_dir / "constant" / "polyMesh" / "boundary")
    )

    # Batched inference
    print(f"Running inference for {n_phases} phases (Re={re_val}, angle={angle_deg}°) …")
    x_norm, edge_index, e_norm = _normalize(raw_data, norm_stats)
    N        = x_norm.shape[0]
    phases_t = torch.linspace(0, 1, n_phases + 1)[:-1]

    graphs = [
        Data(
            x=x_norm, edge_index=edge_index, edge_attr=e_norm,
            re=torch.tensor([re_val],           dtype=torch.float32),
            angle=torch.tensor([float(angle_deg)], dtype=torch.float32),
            phase=phi.unsqueeze(0).float(),
        )
        for phi in phases_t
    ]
    with torch.no_grad():
        y_norm = model(Batch.from_data_list(graphs))
    y_phys = denormalize_wss_v4(y_norm, norm_stats).view(n_phases, N, 3)

    # Write one OpenFOAM timestep directory per phase
    for phi, wss in zip(phases_t.tolist(), y_phys):
        ts_dir = out_dir / f"{phi:.2f}"
        ts_dir.mkdir(exist_ok=True)
        _write_openfoam_wss(str(ts_dir / "wallShearStress"), wss.numpy(), boundary_info)
        print(f"  φ={phi:.2f}  max={wss.norm(dim=1).max():.4e} Pa")

    (out_dir / "case.foam").touch()
    print(f"\nDone → {out_dir}")
    print(f"ParaView: File → Open → case.foam → Apply → color by wallShearStress → Play")


# ── interactive UI ────────────────────────────────────────────────────────────

def run_interactive(model, norm_stats, epoch: int):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.widgets import Slider, RadioButtons

    N_PHASES = 10

    # Load all available graphs
    print("Loading graphs …")
    raw_graphs = {}
    for a in [30, 45, 60]:
        try:
            raw_graphs[a] = load_graph(a)
            print(f"  angle{a}°: {raw_graphs[a].x.shape[0]:,} nodes")
        except FileNotFoundError as e:
            print(f"  angle{a}°: not found ({e.args[0].split(chr(10))[0]})")

    if not raw_graphs:
        sys.exit("No graph files found. Download from RunPod first (see module docstring).")

    default_angle = 45 if 45 in raw_graphs else min(raw_graphs)
    state = {
        "angle":   default_angle,
        "re":      500.0,
        "frames":  None,   # [N_PHASES, N]
        "coords":  None,   # [N, 3]
        "phases":  None,   # [N_PHASES]
        "dirty":   True,
    }

    def compute_frames():
        raw = raw_graphs.get(state["angle"])
        if raw is None:
            return
        print(f"  Computing Re={state['re']:.0f}  angle={state['angle']}° … ",
              end="", flush=True)
        wss_mag, phases = run_phases(
            model, raw, norm_stats,
            re_val=state["re"], angle_deg=state["angle"], n_phases=N_PHASES,
        )
        state["coords"] = raw.pos.numpy()  # physical wall centroid coords
        state["frames"] = wss_mag
        state["phases"] = phases
        state["dirty"]  = False
        print("done")

    compute_frames()

    # ── figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7), facecolor="#0d1117")
    fig.suptitle(
        f"Pulsatile Bifurcation WSS  |  v4 model  |  epoch {epoch}",
        color="white", fontsize=12, y=0.97,
    )

    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.set_facecolor("#0d1117")
    for attr in ("xaxis", "yaxis", "zaxis"):
        getattr(ax3d, attr).pane.fill = False
        getattr(ax3d, attr).pane.set_edgecolor("#333")
    ax3d.tick_params(colors="#555", labelsize=6)
    ax3d.set_xlabel("x", color="#777", fontsize=7)
    ax3d.set_ylabel("y", color="#777", fontsize=7)
    ax3d.set_zlabel("z", color="#777", fontsize=7)

    coords = state["coords"]
    frames = state["frames"]
    sc = ax3d.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=frames[0], cmap="plasma", s=1.5,
        vmin=frames.min(), vmax=frames.max(),
    )
    cbar = fig.colorbar(sc, ax=ax3d, shrink=0.45, pad=0.05)
    cbar.set_label("WSS magnitude (Pa)", color="#aaa", fontsize=7)
    cbar.ax.yaxis.set_tick_params(color="#aaa", labelsize=6, labelcolor="#aaa")
    ax3d.set_title(
        f"Re={state['re']:.0f}  Angle={state['angle']}°",
        color="white", fontsize=10,
    )

    # ── control widgets ───────────────────────────────────────────────────
    ax_re    = fig.add_axes([0.58, 0.80, 0.36, 0.03], facecolor="#1a1a2e")
    ax_angle = fig.add_axes([0.58, 0.52, 0.13, 0.22], facecolor="#1a1a2e")
    ax_phase = fig.add_axes([0.74, 0.52, 0.22, 0.22], facecolor="#0d1117")
    ax_info  = fig.add_axes([0.58, 0.10, 0.36, 0.38], facecolor="#0d1117")
    ax_info.axis("off")

    for spine in ax_phase.spines.values():
        spine.set_color("#333")
    ax_phase.tick_params(colors="#555", labelsize=6)
    ax_phase.set_facecolor("#0d1117")
    ax_phase.set_title("Cardiac phase φ", color="#aaa", fontsize=8)
    ax_phase.set_xlim(0, 1)
    ax_phase.set_ylim(-1.5, 1.5)
    ax_phase.set_xlabel("φ", color="#777", fontsize=7)

    t_full = np.linspace(0, 1, 300)
    ax_phase.plot(t_full, np.sin(2 * np.pi * t_full), color="#3a3a6a", linewidth=1.5)
    phase_dot, = ax_phase.plot([], [], "o", color="#ff6b6b", markersize=9, zorder=5)
    phase_line = ax_phase.axvline(x=0, color="#ff6b6b", linewidth=0.8, alpha=0.5)

    sl_re = Slider(ax_re, "Re", 100, 2100, valinit=state["re"], valstep=100,
                   color="#2255aa", initcolor="none")
    sl_re.label.set_color("white")
    sl_re.valtext.set_color("white")

    avail_angles  = sorted(raw_graphs.keys())
    angle_labels  = [f"{a}°" for a in avail_angles]
    default_idx   = avail_angles.index(default_angle)
    rb_angle = RadioButtons(ax_angle, angle_labels, active=default_idx)
    for lbl in rb_angle.labels:
        lbl.set_color("white")
    rb_angle.ax.set_facecolor("#1a1a2e")

    info_text = ax_info.text(
        0.05, 0.95, "", transform=ax_info.transAxes,
        color="white", fontsize=9, va="top", family="monospace",
    )

    def _refresh_info():
        f = state["frames"]
        if f is None:
            return
        info_text.set_text(
            f"Re       = {state['re']:.0f}\n"
            f"Angle    = {state['angle']}°\n"
            f"Nodes    = {state['coords'].shape[0]:,}\n"
            f"─────────────────\n"
            f"Max WSS  = {f.max():.4e} Pa\n"
            f"Mean WSS = {f.mean():.4e} Pa\n"
            f"Min WSS  = {f.min():.4e} Pa\n"
            f"─────────────────\n"
            f"Epoch    = {epoch}\n"
            f"Phases   = {N_PHASES}"
        )

    _refresh_info()

    # ── callbacks ─────────────────────────────────────────────────────────
    def _recompute(new_re=None, new_angle=None):
        if new_re    is not None: state["re"]    = float(new_re)
        if new_angle is not None: state["angle"] = int(new_angle)
        ax3d.set_title("Computing…", color="yellow", fontsize=10)
        fig.canvas.draw_idle()
        compute_frames()
        f = state["frames"]
        sc.set_clim(f.min(), f.max())
        _refresh_info()
        ax3d.set_title(
            f"Re={state['re']:.0f}  Angle={state['angle']}°",
            color="white", fontsize=10,
        )

    sl_re.on_changed(lambda val: _recompute(new_re=val))
    rb_angle.on_clicked(lambda lbl: _recompute(new_angle=int(lbl.replace("°", ""))))

    # ── animation ─────────────────────────────────────────────────────────
    frame_counter = [0]

    def animate(_):
        if state["frames"] is None:
            return (sc, phase_dot, phase_line)

        i = frame_counter[0] % N_PHASES
        frame_counter[0] += 1

        f   = state["frames"]
        xyz = state["coords"]
        phi = state["phases"][i]

        sc._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
        sc.set_array(f[i])

        phase_dot.set_data([phi], [np.sin(2 * np.pi * phi)])
        phase_line.set_xdata([phi, phi])

        return (sc, phase_dot, phase_line)

    ani = animation.FuncAnimation(   # noqa: F841  (kept alive by reference)
        fig, animate, interval=150, blit=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V4 pulsatile WSS demo — interactive UI or ParaView export"
    )
    parser.add_argument("--model",           type=str,   default=None)
    parser.add_argument("--export-paraview", action="store_true",
                        help="Write 20 OpenFOAM phase dirs for ParaView")
    parser.add_argument("--angle",  type=int,   default=45, choices=[30, 45, 60])
    parser.add_argument("--re",     type=float, default=500.0)
    parser.add_argument("--n-phases", type=int, default=20,
                        help="Phases for ParaView export (default 20)")
    args = parser.parse_args()

    model, norm_stats, epoch = load_model(args.model)

    if args.export_paraview:
        geo_folder, _ = _GEO_INFO[args.angle]
        raw_data = load_graph(args.angle)
        export_paraview(
            model, raw_data, norm_stats,
            re_val=args.re, angle_deg=args.angle,
            geo_folder=geo_folder, n_phases=args.n_phases,
        )
    else:
        run_interactive(model, norm_stats, epoch)


if __name__ == "__main__":
    main()

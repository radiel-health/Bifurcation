import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
import re
from typing import TypedDict
import numpy.typing as npt

# --- Configuration ---
INPUT_DIR = "Data/openFoam(1e-3)"
OUTPUT_DIR = "ProcessedData/3D"
K_NEIGHBORS = 6
# ---------------------


class BoundaryData(TypedDict):
    """Type definition for boundary data."""

    coords: npt.NDArray[np.float32]  # Shape (N, 3)
    wss_mag: npt.NDArray[np.float32]  # Shape (N,)
    wss_x: npt.NDArray[np.float32]  # Shape (N,)
    wss_y: npt.NDArray[np.float32]  # Shape (N,)
    wss_z: npt.NDArray[np.float32]  # Shape (N,)
    pressure: npt.NDArray[np.float32]  # Shape (N,)


class FlowParams(TypedDict):
    """Type definitions for flow data"""

    # TODO: some aren't really 'flow parameters' so fix this semantic inconsistency
    re: np.float32
    angle: np.float32
    child_size: np.float32


def load_boundary_csv(csv_path: Path) -> BoundaryData:
    """Load boundary data from CSV file."""
    df = pd.read_csv(csv_path)
    return {
        "coords": df[["x", "y", "z"]].to_numpy(dtype=np.float32),
        "wss_mag": df["wss_mag"].to_numpy(dtype=np.float32),
        "wss_x": df["wss_x"].to_numpy(dtype=np.float32),
        "wss_y": df["wss_y"].to_numpy(dtype=np.float32),
        "wss_z": df["wss_z"].to_numpy(dtype=np.float32),
        "pressure": df["p"].to_numpy(dtype=np.float32),
    }


def create_edges(coords: npt.NDArray[np.float32], k: int = 6):
    """Create generic edge connectivity using K-Nearest Neighbors."""
    pos = torch.tensor(coords, dtype=torch.float32)
    return to_undirected(knn_graph(pos, k=k, loop=False))


def create_graph(data_dict: BoundaryData, flow_params: FlowParams):
    """Package generic CSV data into PyG Data object."""
    coords = data_dict["coords"]
    num_nodes = len(coords)

    edge_index = create_edges(coords, k=K_NEIGHBORS)

    # Features: [x, y, z, p]
    pressure = data_dict["pressure"].reshape(-1, 1)
    node_features = np.hstack([coords, pressure])
    x = torch.tensor(node_features, dtype=torch.float32)

    # Targets: [WSS_x, WSS_y, WSS_z]
    y_target = np.column_stack(
        [data_dict["wss_x"], data_dict["wss_y"], data_dict["wss_z"]]
    )

    fp = (
        torch.tensor(flow_params, dtype=torch.float32)
        if flow_params
        else torch.tensor([0.0])
    )

    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor(y_target, dtype=torch.float32),
        pos=torch.tensor(coords, dtype=torch.float32),
        flow_params=fp,
        num_nodes=num_nodes,
    )


if __name__ == "__main__":
    # 1. Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Get all wall_wss.csv files recursively in the input directory
    csv_files = list(Path(INPUT_DIR).rglob("wall_wss.csv"))
    print(f"Found {len(csv_files)} files to process in {INPUT_DIR}")

    for csv_path in csv_files:
        print(f"Processing: {csv_path}...", end=" ")

        try:
            # 3. Load and Convert
            raw_data = load_boundary_csv(csv_path)

            # Extract Reynolds number from parent directory (e.g., Re100 -> 100)
            # ---
            re_dir = csv_path.parent.name  # e.g., "Re100"
            assert re_dir.startswith("Re")
            re_val = np.float32(re_dir[2:])
            # ---

            # Extract bifurcation angle from grandparent directory
            # e.g., "bifurcation_angle30_1000_ascii" -> 30
            # ---
            angle_dir = (
                csv_path.parent.parent.name
            )  # e.g., "bifurcation_angle30_1000_ascii"
            pattern = r"^bifurcation_angle(?P<angle>\d+)_(?P<child_size>\d+)_ascii$"  # Define the expected pattern
            match = re.fullmatch(pattern, angle_dir)  # Match the pattern
            assert (
                match
            ), f"Directory name '{angle_dir}' does not match the expected format 'bifurcation_angle{{nat1}}_{{nat2}}_ascii'"
            angle_val = np.float32(
                match.group("angle")
            )  # Extract angle value and convert to float
            child_size_val = np.float32(match.group("child_size"))
            # TODO: think about how to better use angle_val and other params
            # ---

            # 4. Save the .pt file - preserve folder structure in filename
            # e.g., bifurcation_angle30_1000_ascii_Re100.pt
            graph_data = create_graph(
                raw_data,
                flow_params=FlowParams(
                    re=re_val, angle=angle_val, child_size=child_size_val
                ),
            )
            angle_name = csv_path.parent.parent.name
            re_name = csv_path.parent.name
            save_name = f"{angle_name}_{re_name}.pt"
            torch.save(graph_data, os.path.join(OUTPUT_DIR, save_name))
            print("Done.")

        except Exception as e:
            print(f"Failed! Error: {e}")

    print(f"\nPreprocessing complete. Files saved to {OUTPUT_DIR}")

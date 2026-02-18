import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected

# --- Configuration ---
INPUT_DIR = "RawData/3D_Meshes"
OUTPUT_DIR = "ProcessedData/3D"
K_NEIGHBORS = 6
# ---------------------


def load_boundary_csv(csv_path: Path):
    """Load boundary data from ANSYS CSV file."""
    df = pd.read_csv(csv_path)
    return {
        "coords": df[["x", "y", "z"]].values,
        "wss_mag": df["wss_mag"].values,
        "wss_x": df["wss_x"].values,
        "wss_y": df["wss_y"].values,
        "wss_z": df["wss_z"].values,
        "pressure": df["p"].values,
    }


def create_edges(coords, k: int = 6):
    """Create generic edge connectivity using K-Nearest Neighbors."""
    pos = torch.tensor(coords, dtype=torch.float32)
    return to_undirected(knn_graph(pos, k=k, loop=False))


def create_graph(data_dict, flow_params=None):
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

    # 2. Get all CSV files in the input directory
    csv_files = list(Path(INPUT_DIR).glob("*.csv"))
    print(f"Found {len(csv_files)} files to process in {INPUT_DIR}")

    for csv_path in csv_files:
        print(f"Processing: {csv_path.name}...", end=" ")

        try:
            # 3. Load and Convert
            raw_data = load_boundary_csv(csv_path)

            # Extract Reynolds number or other params from filename if needed
            # Example: "mesh_Re500.csv" -> 500
            # TODO: integrate real files here
            re_val = 0.0
            if "Re" in csv_path.stem:
                try:
                    re_val = float(csv_path.stem.split("Re")[-1])
                except ValueError:
                    pass

            graph_data = create_graph(raw_data, flow_params=[re_val])

            # 4. Save the .pt file
            save_name = csv_path.stem + ".pt"
            torch.save(graph_data, os.path.join(OUTPUT_DIR, save_name))
            print("Done.")

        except Exception as e:
            print(f"Failed! Error: {e}")

    print(f"\nPreprocessing complete. Files saved to {OUTPUT_DIR}")

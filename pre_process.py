import os
import re
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch_geometric.data import Data
from typing import TypedDict
from config import config
import gdown
import zipfile
# --- Configuration ---
OUTPUT_DIR = Path("ProcessedData/3D")
INPUT_DIR = Path(config.input_data_dir)
GOOGLE_DRIVE_ZIP_URL = config.google_drive_zip_url

class FlowParams(TypedDict):
    re: float
    angle: float
    child_size: float

# ============================================================================
# Robust OpenFOAM Parsers (Aligned with dataset.py logic)
# ============================================================================

def parse_openfoam_vector_field(filepath: str, is_result_file: bool = False) -> np.ndarray:
    """Parses OpenFOAM vector fields into an (N, 3) numpy array."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    if is_result_file:
        content = content.split("boundaryField")[-1]

    match = re.search(r"(\d+)\s*\n\s*\(\s*\n(.*?)\n\s*\)", content, re.DOTALL)
    if match is None:
        match = re.search(r"(\d+)\s*\(\s*\n(.*?)\n\s*\)", content, re.DOTALL)
        if match is None:
            raise ValueError(f"Could not parse vector field from {filepath}")

    block = match.group(2)
    vectors = re.findall(r"\(\s*([^\)]+)\)", block)
    return np.array([[float(x) for x in v.split()] for v in vectors], dtype=np.float32)

def parse_openfoam_faces(filepath: str) -> list[list[int]]:
    """Parses OpenFOAM faces into a list of point-index lists."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    match = re.search(r"(\d+)\s*\n\s*\(\s*\n(.*?)\n\s*\)", content, re.DOTALL)
    if match is None:
        match = re.search(r"(\d+)\s*\(\s*\n(.*?)\n\s*\)", content, re.DOTALL)
    if match is None:
        raise ValueError(f"Could not parse faces from {filepath}")

    block = match.group(2)
    faces = []
    for line in block.strip().split("\n"):
        idx_match = re.search(r"\d+\(([^)]+)\)", line.strip())
        if idx_match:
            indices = [int(x) for x in idx_match.group(1).split()]
            faces.append(indices)
    return faces

def parse_boundary(filepath: str) -> dict:
    """Parses boundary file into a dict of patch properties."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    patches = {}
    pattern = r"(\w[\w-]*)\s*\{[^}]*type\s+(\w+);\s*(?:inGroups[^;]*;\s*)?nFaces\s+(\d+);\s*startFace\s+(\d+);"
    for m in re.finditer(pattern, content):
        name = m.group(1)
        patches[name] = {
            "type": m.group(2),
            "nFaces": int(m.group(3)),
            "startFace": int(m.group(4)),
        }
    return patches

# ============================================================================
# Path & Mesh Helpers
# ============================================================================

def find_mesh_dir(case_path: Path) -> Path:
    re100_mesh = case_path.parent / "Re100" / "constant" / "polyMesh"
    local_mesh = case_path / "constant" / "polyMesh"

    if re100_mesh.exists(): return re100_mesh
    if local_mesh.exists(): return local_mesh
    raise FileNotFoundError(f"Mesh not found for {case_path}")

def get_latest_timestep(case_path: Path) -> Path:
    subdirs = [d for d in os.listdir(case_path) if os.path.isdir(case_path / d)]
    timesteps = [d for d in subdirs if d.replace('.','',1).isdigit() and float(d) > 0]
    if not timesteps: raise ValueError(f"No result timesteps in {case_path}")
    latest = max(timesteps, key=float)
    return case_path / latest

# ============================================================================
# Graph Construction
# ============================================================================

def create_graph(case_path: Path, flow_params: FlowParams):
    poly_mesh_dir = find_mesh_dir(case_path)

    # 1. Parse Mesh Elements
    points = parse_openfoam_vector_field(str(poly_mesh_dir / "points"), is_result_file=False)
    faces = parse_openfoam_faces(str(poly_mesh_dir / "faces"))
    boundary = parse_boundary(str(poly_mesh_dir / "boundary"))

    # 2. Identify all wall patches
    wall_patches = {k: v for k, v in boundary.items() if v["type"] == "wall"}
    if not wall_patches:
        raise ValueError("No wall patches found in boundary file.")

    wall_centres = []
    wall_face_indices = []
    wall_normals = []  # NEW: List to hold our normal vectors
    
    for _name, info in wall_patches.items():
        for i in range(info["startFace"], info["startFace"] + info["nFaces"]):
            face_pts = points[faces[i]]
            wall_centres.append(face_pts.mean(axis=0))
            wall_face_indices.append(i)
            
            # NEW: Compute Exact Face Normal using Cross Product
            # Using the first 3 vertices of the face
            v0 = face_pts[0]
            v1 = face_pts[1]
            v2 = face_pts[2]
            
            normal = np.cross(v1 - v0, v2 - v0)
            
            # Normalize to unit length
            magnitude = np.linalg.norm(normal)
            if magnitude > 1e-10:
                normal = normal / magnitude
                
            wall_normals.append(normal)
            
    wall_centres_arr = np.array(wall_centres, dtype=np.float32)
    wall_normals_arr = np.array(wall_normals, dtype=np.float32)  # NEW
    n_wall = len(wall_centres_arr)

    # 3. Adjacency via shared vertices
    point_to_faces = defaultdict(set)
    for local_idx, global_face_idx in enumerate(wall_face_indices):
        for pt in faces[global_face_idx]:
            point_to_faces[pt].add(local_idx)

    src_list, dst_list = [], []
    for _pt, face_set in point_to_faces.items():
        face_list = list(face_set)
        for i in range(len(face_list)):
            for j in range(i + 1, len(face_list)):
                src_list.extend([face_list[i], face_list[j]])
                dst_list.extend([face_list[j], face_list[i]])

    edge_index = np.array([src_list, dst_list], dtype=np.int64)

    # 4. Remove Duplicate Edges and Calculate Edge Attributes
    if edge_index.shape[1] > 0:
        edge_pairs = edge_index.T
        _, unique_idx = np.unique(edge_pairs, axis=0, return_index=True)
        edge_index = edge_index[:, np.sort(unique_idx)]

        # Edge features: [distance, dx, dy, dz]
        src_coords = wall_centres_arr[edge_index[0]]
        dst_coords = wall_centres_arr[edge_index[1]]
        diff = dst_coords - src_coords
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        edge_attr = np.concatenate([dist, diff], axis=1).astype(np.float32)
    else:
        edge_attr = np.zeros((0, 4), dtype=np.float32)

    # 5. Parse Result Fields (WSS)
    res_path = get_latest_timestep(case_path)
    wss_all = parse_openfoam_vector_field(str(res_path / "wallShearStress"), is_result_file=True)

    wall_wss_parts = []
    
    for _name, info in wall_patches.items():
        wall_wss_parts.append(wss_all[: info["nFaces"]])
        wss_all = wss_all[info["nFaces"]:]
        
    wall_wss = np.vstack(wall_wss_parts).astype(np.float32)

    # 6. Build Original Data Object Structure (Now including normals!)
    return Data(
        x=torch.tensor(wall_centres_arr, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor(wall_wss, dtype=torch.float32),
        pos=torch.tensor(wall_centres_arr, dtype=torch.float32),
        normals=torch.tensor(wall_normals_arr, dtype=torch.float32),  # NEW
        flow_params=torch.tensor([[flow_params['re'], flow_params['angle'], flow_params['child_size']]], dtype=torch.float32),
        num_nodes=n_wall
    )

def download_and_extract_data():
    """Download data from Google Drive and extract to the current directory."""
    # We extract to "." because the zip contains a "Data/" folder.
    # This ensures the final path is ./Data/...
    extract_path = "."

    print(f"Downloading data from Google Drive...")

    try:
        # gdown handles the 'large file' confirmation automatically.
        # fuzzy=True helps it find the ID even from a full URL.
        zip_path = gdown.download(GOOGLE_DRIVE_ZIP_URL, quiet=False, fuzzy=True)

        if not zip_path or not zipfile.is_zipfile(zip_path):
            raise RuntimeError("Downloaded file is not a valid zip. Check the File ID/URL.")

        print("Download complete. Extracting...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extracting to '.' merges the zip's 'Data/' folder
            # with your current working directory.
            zip_ref.extractall(extract_path)

        print(f"Data successfully extracted to {os.path.join(os.getcwd(), 'Data')}")

        # Clean up the temporary zip file
        os.remove(zip_path)
        print("Cleaned up temporary zip file.")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def check_and_download_data():
    """Check if INPUT_DIR has OpenFOAM cases, download if empty."""
    # Search for Reynolds folders (e.g., Re100, Re500)
    re_dirs = list(Path(INPUT_DIR).rglob("Re*"))

    if not re_dirs:
        print(f"No Reynolds folders found in {INPUT_DIR}. Downloading...")
        download_and_extract_data()

        re_dirs = list(Path(INPUT_DIR).rglob("Re*"))
        assert len(re_dirs) > 0, "INPUT_DIR is still empty after downloading and extracting data"

    print(f"Found {len(re_dirs)} cases in {INPUT_DIR}")

if __name__ == "__main__":
    check_and_download_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    re_dirs = sorted([p for p in Path(INPUT_DIR).rglob("Re*") if "constant" not in str(p) and "ProcessedData" not in str(p)])

    print(f"Found {len(re_dirs)} cases to process.")
    success_count = 0

    for re_path in re_dirs:
        print(f"Processing: {re_path.parent.name}/{re_path.name}...", end="")
        try:
            re_val = float(re_path.name[2:])
            angle_match = re.search(r"angle(\d+)_(\d+)", re_path.parent.name)
            angle_val, size_val = float(angle_match.group(1)), float(angle_match.group(2))

            data = create_graph(re_path, {"re": re_val, "angle": angle_val, "child_size": size_val})

            save_name = f"{re_path.parent.name}_{re_path.name}.pt"
            torch.save(data, OUTPUT_DIR / save_name)
            print(" Done.")
            success_count += 1
        except Exception as e:
            print(f" Failed! {e}")

    print(f"\nProcessing complete. Built {success_count} graphs in {OUTPUT_DIR}")

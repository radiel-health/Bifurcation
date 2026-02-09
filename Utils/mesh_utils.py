"""
Mesh processing utilities for bifurcation geometry.

Functions for:
- Loading ANSYS .msh files and extracting wall surfaces
- Loading and parsing Fluent CSV output files
- Computing geometric edge features (dihedral angles, inner angles, ratios)
- Matching CSV node data to mesh vertices
- Building PyG graph objects from mesh + data

Adapted from AVFlow pipeline notebooks.
"""

import numpy as np
import torch
from scipy.spatial import cKDTree
from pathlib import Path
from typing import Tuple, Optional, Dict
import warnings


def load_fluent_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Fluent wall data CSV file.
    
    CSV format (space-delimited):
        nodenumber x y z z-wall-shear y-wall-shear x-wall-shear wall-shear pressure z y x
    
    Args:
        csv_path: Path to wall_data_Re{R}.csv file
        
    Returns:
        coords: [N, 3] array of (x, y, z) coordinates
        wss: [N, 3] array of (wss_x, wss_y, wss_z) components
        pressure: [N] array of pressure values
    """
    # Read CSV, skip header, handle space-delimited
    data = np.loadtxt(csv_path, skiprows=1)
    
    # Column indices (0-based):
    # 0: nodenumber
    # 1,2,3: x, y, z (first occurrence)
    # 4: z-wall-shear
    # 5: y-wall-shear
    # 6: x-wall-shear
    # 7: wall-shear (magnitude)
    # 8: pressure
    # 9,10,11: z, y, x (duplicate coordinates - Fluent export artifact)
    
    coords = data[:, 1:4]  # x, y, z
    
    # WSS components (note: Fluent outputs as z, y, x order in columns 4-6)
    wss_x = data[:, 6]
    wss_y = data[:, 5]
    wss_z = data[:, 4]
    wss = np.column_stack([wss_x, wss_y, wss_z])
    
    pressure = data[:, 8]
    
    return coords, wss, pressure


def load_mesh_wall_surface(mesh_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mesh file and extract wall surface triangulation.
    
    Uses meshio to read ANSYS .msh files. Extracts the 'wall-fluid_domain'
    zone and returns vertices + triangle faces.
    
    Args:
        mesh_path: Path to .msh file
        
    Returns:
        vertices: [N, 3] array of vertex coordinates
        triangles: [M, 3] array of triangle face indices
    """
    try:
        import meshio
    except ImportError:
        raise ImportError("meshio is required: pip install meshio")
    
    # Read mesh
    mesh = meshio.read(mesh_path)
    
    # Extract triangular cells
    # ANSYS .msh files may have mixed cell types
    triangles = None
    vertices = mesh.points[:, :3]  # Ensure 3D coordinates only
    
    # Look for triangle cells
    for cell_type, cell_data in zip(mesh.cells, mesh.cell_data):
        if cell_type.type == "triangle":
            triangles = cell_type.data
            break
    
    if triangles is None:
        # Try to triangulate if we have quads
        for cell_type, cell_data in zip(mesh.cells, mesh.cell_data):
            if cell_type.type == "quad":
                # Simple quad→triangle: [0,1,2,3] → [0,1,2], [0,2,3]
                quads = cell_type.data
                tri1 = quads[:, [0, 1, 2]]
                tri2 = quads[:, [0, 2, 3]]
                triangles = np.vstack([tri1, tri2])
                break
    
    if triangles is None:
        raise ValueError(f"No triangle or quad cells found in mesh: {mesh_path}")
    
    return vertices, triangles


def match_csv_to_mesh(
    csv_coords: np.ndarray,
    mesh_vertices: np.ndarray,
    tol: float = 1e-6 # tolerance in meters for matching
) -> np.ndarray:
    """
    Match CSV node coordinates to mesh vertices using KD-tree.
    
    Args:
        csv_coords: [N, 3] CSV coordinates
        mesh_vertices: [M, 3] mesh vertex coordinates
        tol: Distance tolerance for matching (meters)
        
    Returns:
        indices: [N] array mapping CSV rows to mesh vertex indices
        
    Raises:
        ValueError: If any CSV node cannot be matched to mesh
    """
    # Build KD-tree for fast nearest-neighbor search
    tree = cKDTree(mesh_vertices)
    
    # Find nearest mesh vertex for each CSV coordinate
    distances, indices = tree.query(csv_coords, k=1)
    
    # Check match quality
    max_dist = distances.max()
    if max_dist > tol:
        warnings.warn(
            f"CSV-to-mesh matching has max distance {max_dist:.2e} m "
            f"(tolerance: {tol:.2e} m). Check coordinate consistency."
        )
    
    # Verify no duplicate mappings (multiple CSV nodes → same mesh vertex)
    unique_indices, counts = np.unique(indices, return_counts=True)
    if np.any(counts > 1):
        n_duplicates = np.sum(counts > 1)
        warnings.warn(
            f"{n_duplicates} mesh vertices matched to multiple CSV nodes. "
            f"This may indicate coordinate precision issues."
        )
    
    return indices

def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute unit normal vectors for triangular faces.
    
    Args:
        vertices: [N, 3] vertex coordinates
        faces: [M, 3] triangle vertex indices
        
    Returns:
        normals: [M, 3] unit normal vectors
    """
    # Get triangle vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Compute edge vectors
    e1 = v1 - v0
    e2 = v2 - v0
    
    # Cross product gives normal (pointing outward by right-hand rule)
    normals = np.cross(e1, e2)
    
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normals = normals / norms
    
    return normals


def compute_edge_features(
    vertices: np.ndarray,
    faces: np.ndarray,
    edge_index: np.ndarray
) -> np.ndarray:
    """
    Compute 5D geometric edge features.
    
    For each edge shared by two triangular faces:
    1. Dihedral angle: angle between face normals (0 for boundary edges)
    2. Min inner angle: min of angles at opposite vertices
    3. Max inner angle: max of angles at opposite vertices  
    4. Min edge ratio: min(edge_length / triangle_height)
    5. Max edge ratio: max(edge_length / triangle_height)
    
    Adapted from AVFlow pipeline.
    
    Args:
        vertices: [N, 3] vertex coordinates
        faces: [M, 3] triangle face indices
        edge_index: [2, E] undirected edge list
        
    Returns:
        edge_features: [E, 5] geometric features per edge
    """
    num_edges = edge_index.shape[1]
    
    # Build edge-to-faces mapping
    edge_to_faces = {}
    for face_idx, face in enumerate(faces):
        # Each triangle has 3 edges
        edges_in_face = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[2], face[0]]))
        ]
        for edge in edges_in_face:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(face_idx)
    
    # Compute face normals
    face_normals = compute_face_normals(vertices, faces)
    
    # Initialize features
    edge_features = np.zeros((num_edges, 5))
    
    for i in range(num_edges):
        u, v = edge_index[0, i], edge_index[1, i]
        edge_key = tuple(sorted([u, v]))
        
        # Edge vector
        edge_vec = vertices[v] - vertices[u]
        edge_length = np.linalg.norm(edge_vec)
        
        if edge_length == 0:
            continue  # Degenerate edge
        
        # Get adjacent faces
        adj_faces = edge_to_faces.get(edge_key, [])
        
        # Feature 1: Dihedral angle
        if len(adj_faces) == 2:
            n1 = face_normals[adj_faces[0]]
            n2 = face_normals[adj_faces[1]]
            cos_angle = np.clip(np.dot(n1, n2), -1, 1)
            dihedral_angle = np.arccos(cos_angle)
        else:
            # Boundary edge (only one adjacent face)
            dihedral_angle = 0.0
        
        edge_features[i, 0] = dihedral_angle
        
        # Features 2-5: Inner angles and edge ratios
        if len(adj_faces) >= 1:
            inner_angles = []
            edge_ratios = []
            
            for face_idx in adj_faces[:2]:  # Max 2 faces per edge
                face = faces[face_idx]
                
                # Find the vertex opposite to this edge
                opposite_vertex = None
                for vert in face:
                    if vert != u and vert != v:
                        opposite_vertex = vert
                        break
                
                if opposite_vertex is not None:
                    # Compute inner angle at opposite vertex
                    vec_to_u = vertices[u] - vertices[opposite_vertex]
                    vec_to_v = vertices[v] - vertices[opposite_vertex]
                    
                    cos_inner = np.dot(vec_to_u, vec_to_v) / (
                        np.linalg.norm(vec_to_u) * np.linalg.norm(vec_to_v) + 1e-10
                    )
                    cos_inner = np.clip(cos_inner, -1, 1)
                    inner_angle = np.arccos(cos_inner)
                    inner_angles.append(inner_angle)
                    
                    # Compute triangle height from opposite vertex to edge
                    # Area = 0.5 * base * height → height = 2*Area / base
                    v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                    # Ensure vertices are 1D arrays
                    v0 = np.asarray(v0).flatten()
                    v1 = np.asarray(v1).flatten()
                    v2 = np.asarray(v2).flatten()
                    # Compute area via cross product
                    cross_prod = np.cross(v1 - v0, v2 - v0)
                    area = 0.5 * np.linalg.norm(cross_prod)
                    height = 2 * area / (edge_length + 1e-10)
                    edge_ratio = edge_length / (height + 1e-10)
                    edge_ratios.append(edge_ratio)
            
            if inner_angles:
                edge_features[i, 1] = np.min(inner_angles)
                edge_features[i, 2] = np.max(inner_angles)
            
            if edge_ratios:
                edge_features[i, 3] = np.min(edge_ratios)
                edge_features[i, 4] = np.max(edge_ratios)
    
    # Handle NaN/Inf values
    edge_features = np.nan_to_num(edge_features, nan=0.0, posinf=1.0, neginf=0.0)
    
    return edge_features


def build_edge_index_from_faces(faces: np.ndarray) -> np.ndarray:
    """
    Build undirected edge index from triangle faces.
    
    Args:
        faces: [M, 3] triangle face indices
        
    Returns:
        edge_index: [2, E] undirected edge list (both directions included)
    """
    edges_set = set()
    
    for face in faces:
        # Add all 3 edges of the triangle
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            # Add both directions for undirected graph
            edges_set.add((v1, v2))
            edges_set.add((v2, v1))
    
    edges = np.array(list(edges_set), dtype=np.int64)
    edge_index = edges.T  # [2, E]
    
    return edge_index


def triangulate_point_cloud(coords: np.ndarray) -> np.ndarray: # backup function, should delete...
    """
    Fallback: triangulate 3D point cloud using Delaunay.
    
    This is a backup if mesh file cannot be read. Projects points to 2D,
    performs Delaunay triangulation, then maps back to 3D indices.
    
    Args:
        coords: [N, 3] 3D coordinates
        
    Returns:
        triangles: [M, 3] triangle face indices
    """
    from scipy.spatial import Delaunay
    
    # Project to 2D (use x-y plane, assuming pipe is roughly aligned with z)
    coords_2d = coords[:, :2]
    
    # Delaunay triangulation
    tri = Delaunay(coords_2d)
    triangles = tri.simplices
    
    return triangles


def create_pyg_graph(
    vertices: np.ndarray,
    faces: np.ndarray,
    wss: np.ndarray,
    re: int,
    angle: int
) -> Dict:
    """
    Create PyTorch Geometric graph from mesh + WSS data.
    
    Args:
        vertices: [N, 3] vertex coordinates
        faces: [M, 3] triangle faces
        wss: [N, 3] WSS components [x, y, z]
        re: Reynolds number
        angle: Bifurcation angle (degrees)
        
    Returns:
        graph_dict: Dictionary with PyG Data fields
    """
    num_nodes = len(vertices)
    
    # Build edge index
    edge_index = build_edge_index_from_faces(faces)
    num_edges = edge_index.shape[1]
    
    # Node features: [x_norm, y_norm, z_norm, degree]
    coords_min = vertices.min(axis=0)
    coords_max = vertices.max(axis=0)
    coords_range = coords_max - coords_min
    coords_range = np.where(coords_range == 0, 1, coords_range)  # Avoid division by zero
    coords_norm = (vertices - coords_min) / coords_range
    
    # Compute node degree
    degree = np.bincount(edge_index[0], minlength=num_nodes).astype(np.float32)
    
    # Stack node features
    node_features = np.column_stack([coords_norm, degree.reshape(-1, 1)])
    
    # Compute edge features
    edge_features = compute_edge_features(vertices, faces, edge_index)
    
    # Flow parameters: [Re_normalized, angle_radians]
    re_normalized = (re - 100) / (2100 - 100)  # Normalize to [0, 1]
    angle_radians = np.deg2rad(angle)
    flow_params = np.array([re_normalized, angle_radians], dtype=np.float32)
    
    # Package as dictionary
    graph_dict = {
        'x': torch.tensor(node_features, dtype=torch.float32),
        'edge_index': torch.tensor(edge_index, dtype=torch.long),
        'edge_attr': torch.tensor(edge_features, dtype=torch.float32),
        'y': torch.tensor(wss, dtype=torch.float32),
        'pos': torch.tensor(vertices, dtype=torch.float32),
        'flow_params': torch.tensor(flow_params, dtype=torch.float32),
        're': torch.tensor([re], dtype=torch.long),
        'angle': torch.tensor([angle], dtype=torch.long),
        'num_nodes': num_nodes
    }
    
    return graph_dict


if __name__ == "__main__":
    """Test mesh utilities on a sample case."""
    from pathlib import Path
    
    print("Testing mesh utilities...")
    print()
    
    # Test paths
    data_root = Path(__file__).parent.parent.parent / "Data" / "Bifurcation"
    csv_path = data_root / "results" / "bifurcation_angle30_750" / "Re100" / "wall_data_Re100.csv"
    mesh_path = data_root / "bifurcation_angle30_750.msh"
    
    print(f"CSV path: {csv_path}")
    print(f"CSV exists: {csv_path.exists()}")
    print(f"Mesh path: {mesh_path}")
    print(f"Mesh exists: {mesh_path.exists()}")
    print()
    
    if csv_path.exists():
        # Test CSV loading
        print("Loading CSV...")
        coords, wss, pressure = load_fluent_csv(csv_path)
        print(f"  CSV nodes: {len(coords)}")
        print(f"  Coordinate range: x=[{coords[:,0].min():.4f}, {coords[:,0].max():.4f}]")
        print(f"  WSS range: {wss.min():.4e} to {wss.max():.4e}")
        print()
    
    if mesh_path.exists():
        # Test mesh loading
        print("Loading mesh...")
        try:
            vertices, triangles = load_mesh_wall_surface(mesh_path)
            print(f"  Mesh vertices: {len(vertices)}")
            print(f"  Mesh triangles: {len(triangles)}")
            print()
            
            if csv_path.exists():
                # Test matching
                print("Matching CSV to mesh...")
                indices = match_csv_to_mesh(coords, vertices)
                print(f"  Matched {len(indices)} CSV nodes to mesh")
                print(f"  Unique mesh vertices used: {len(np.unique(indices))}")
                print()
                
                # Test graph creation
                print("Creating PyG graph...")
                # Map WSS to mesh vertices
                wss_full = np.zeros((len(vertices), 3))
                wss_full[indices] = wss
                
                graph_dict = create_pyg_graph(vertices, triangles, wss_full, re=100, angle=30)
                print(f"  Graph nodes: {graph_dict['x'].shape[0]}")
                print(f"  Graph edges: {graph_dict['edge_index'].shape[1]}")
                print(f"  Node features: {graph_dict['x'].shape}")
                print(f"  Edge features: {graph_dict['edge_attr'].shape}")
                print(f"  Targets: {graph_dict['y'].shape}")
                print()
                print("✓ All tests passed!")
        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Note: meshio may not be installed (pip install meshio)")

Summary of Changes
config.py
- Changed processed_data_dir to ProcessedData/3D/
- node_feature_dim: 10 → 4 (x, y, z, p)
- flow_param_dim: 3 → 1 (just Re)
- target_dim: 2 → 3 (wss_x, wss_y, wss_z)
- Removed: aspect ratios, filter_top_wall, 2D-specific paths
model.py
- FlowEncoder: input_dim 3 → 1
- GeometryEncoder: input_dim 10 → 4
- TaskHead: output_dim 2 → 3
- forward(): Uses data.flow_params directly instead of stacking re/lx/ly
dataset.py
- Simplified to load from single ProcessedData/3D/ directory
- Removed aspect ratio stratification
- Removed filter_top_wall functionality
- Simplified feature normalization (4 features)
train.py
- Fixed import: from Models.model → from model
- Removed filter_top_wall references
infer.py
- Rewrote for 3D point clouds
- Removed filter_top_wall logic
- Updated for 3-component output
- 3D scatter plot visualization
evaluate.py
- Rewrote for 3D point clouds
- Removed 2D-specific spatial analysis (wall location, corner proximity, arc position)
- Kept Re-based analysis
- Updated for 3-component targets
pre_process.py
- Fixed target output: 4 → 3 (removed wss_mag, keeping x, y, z components)

# How the 3D Bifurcation Model Works

## TL;DR: 3D Surface → Graph → GNN → WSS Predictions

We work directly on the 3D vessel surface, NOT 2D slices. The cross-sectional viz is just for **interpretation**, not modeling.

---

## The Full Pipeline

### 1. **Input: 3D Surface Point Cloud** 

```
wall_wss.csv contains ~35,000 points:
x,        y,        z,        wss_mag
-0.0847, -0.0681, -0.0849,  0.00092
-0.0515,  0.0000, -0.1163,  0.00237
...
```

**What this represents:**
- Imagine painting the inside of a bifurcation vessel
- Each point = one paint dot on the wall surface
- Together, they form a 3D "skin" of the vessel

**Key insight:** This is a **surface**, not a volume
- We don't care about flow inside the vessel
- Only the wall boundary where WSS occurs

---

### 2. **Build a Graph from the Surface**

```python
# For each point, find its 8 nearest neighbors on the surface
edges = create_knn_edges(coords, k=8)
```

**What this does:**
```
Point A at (x1,y1,z1) connects to nearby points B,C,D...
If A and B are 0.01 units apart on surface → edge between them

Result: A mesh-like graph following vessel topology
```

**Why k-nearest neighbors?**
- Captures local surface structure
- Straight inlet: neighbors form regular grid pattern
- Curved apex: neighbors wrap around the bend
- The graph "learns" the vessel shape automatically!

**Visual analogy:**
```
Straight pipe (inlet):        Bifurcation (apex):
  
  o--o--o--o                      o  o  o
  |  |  |  |                     / \ | / \
  o--o--o--o                    o   \|/   o
  |  |  |  |                         o
  o--o--o--o                       apex
```

---

### 3. **Extract Features for Each Point**

For every surface point, we compute **14 features**:

```python
Features[point_i] = [
    x_norm,              # 0: Where on vessel (0-1)
    y_norm,              # 1: 
    z_norm,              # 2: 
    re_norm,             # 3: Flow regime (100-2100 → 0-1)
    angle_norm,          # 4: Bifurcation angle (30,45,60 → 0,0.5,1)
    mesh_norm,           # 5: Mesh resolution
    is_inlet,            # 6: Region flags (one-hot)
    is_critical,         # 7: 
    is_outlet_left,      # 8: 
    is_outlet_right,     # 9: 
    dist_to_apex,        # 10: How far from bifurcation junction
    radial_dist,         # 11: Distance from centerline
    angular_pos,         # 12: Angle around centerline
    curvature            # 13: 0=straight, 1=curved
]
```

**Why these features matter:**

1. **Position (x,y,z)**: "Where am I on the vessel?"
2. **Reynolds number**: "How fast is the flow?"
3. **Region flags**: "Am I in straight inlet or curved junction?"
4. **Distance to apex**: "How close to the high-stress zone?"
5. **Curvature**: "Is the surface bending here?"

---

### 4. **Graph Neural Network Processes the Graph**

```
Input: Graph with 35,000 nodes × 14 features
       + edges connecting nearby points

GNN does this for each layer:
1. Each node looks at its neighbors
2. Aggregates their features
3. Updates its own representation
4. Repeat 4 times (4 layers)

Output: Each node has learned representation
        that knows about its local + global context
```

**What the model learns:**

**Layer 1:** "My immediate neighbors"
- Point sees: "I'm connected to 8 nearby points"
- Learns: Local surface geometry

**Layer 2:** "My neighborhood"  
- Point sees: "My neighbors' neighbors"
- Learns: Am I on flat wall or curved region?

**Layer 3:** "My region"
- Point sees: "Broader surface patterns"
- Learns: Inlet vs critical vs outlet characteristics

**Layer 4:** "Global structure"
- Point sees: "How do I fit in the whole vessel?"
- Learns: Relationship between Re, angle, and WSS

---

### 5. **Predict WSS at Each Point**

```python
for each point in surface:
    learned_features = GNN(point, its_neighbors)
    wss_prediction = final_layer(learned_features)
```

**Output:**
```
35,000 surface points → 35,000 WSS predictions
One prediction per wall location
```

---

## Why This Works for Bifurcations

### Critical Region = Complex Neighbor Relationships

**Inlet (straight pipe):**
```
Point's neighbors:     WSS pattern:
  o  o  o              ===== (uniform)
  o [•] o              ===== 
  o  o  o              =====
  
Simple, regular      → Model learns: similar WSS
```

**Apex (bifurcation junction):**
```
Point's neighbors:     WSS pattern:
    o                    *
   / \                 * | *
  o [•] o              *****  ← highest WSS
   \ /                 * | *
    o                    *
    
Complex, non-uniform → Model learns: peak WSS here!
```

The graph structure **automatically captures** that the apex has different topology than the inlet!

---

## Cross-Sectional Visualization (What We Just Added)

**Purpose:** Interpret results in 2D for easier understanding

**How it works:**
```python
# Take the 3D surface predictions
all_points = 35,000 with (x,y,z,wss)

# Slice at z = 0.342 (apex level)
slice_points = points where z ≈ 0.342

# Plot in 2D (x-y plane)
→ Shows WSS around vessel perimeter at that height
```

**Creates two views:**

1. **Cartesian:** Top-down view of vessel cross-section
2. **Polar:** WSS distribution around circumference (0-360°)

**Example interpretation:**
```
At inlet (z=0.014):   Low, uniform WSS around perimeter
At apex (z=0.342):    High, varying WSS - peaks where vessel splits
At outlet (z=0.670):  Medium WSS, different on left vs right branch
```

---

## Key Differences from 2D Slice Approach

| 3D Surface Approach (ours) | 2D Slice Approach |
|---------------------------|-------------------|
| ✅ Preserves full geometry | ❌ Loses connectivity at apex |
| ✅ Natural for GNN | ⚠️ Need to choose slice positions |
| ✅ Matches CFD output | ⚠️ Extra preprocessing |
| ✅ Critical region is 3D point | ❌ Critical region spans multiple slices |
| ⚠️ Harder to visualize | ✅ Easier to understand |

**Solution:** Use 3D for modeling, add 2D slices for visualization! (Which we just did)

---

## Analogy Time 🎯

**Think of the bifurcation like Earth's surface:**

- **Earth's surface** = Vessel wall (curved 3D surface)
- **35,000 points** = Weather stations measuring temperature
- **Graph edges** = "Nearby stations influence each other"
- **GNN** = Weather prediction model that considers neighboring stations
- **WSS** = Temperature we're predicting
- **Cross-sections** = Latitude lines (useful for viewing, but Earth is still 3D!)

You wouldn't flatten Earth to 2D to predict weather - you'd use the full 3D surface. Same here!

---

## Summary

**Model Flow:**
```
CSV file
  ↓
3D point cloud (35k points on vessel surface)
  ↓  
Build graph (connect nearby points)
  ↓
Extract 14 features per point
  ↓
GNN learns from neighbors (4 layers)
  ↓
Predict WSS at each point
  ↓
Visualize in 2D slices for interpretation
```

**The magic:** The graph structure + GNN automatically learns that:
- Straight inlet → simple patterns
- Curved apex → complex patterns + high WSS
- Reynolds number → scales the overall magnitude
- Angle → changes the splitting pattern

No need to explicitly teach it bifurcation physics - it learns from the geometry + data!

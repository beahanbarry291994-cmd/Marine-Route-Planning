# Marine Route Planning — Technical Report

## 1. Problem Statement

Given an electronic nautical chart covering a region of interest, plan a collision-free, smooth path from a specified origin to a destination, suitable for ship navigation.

**Constraints:**
- The path must be continuous and connect start to end
- The path must avoid all land obstacles with a configurable safety margin
- The path should be smooth, respecting ship motion characteristics

## 2. System Architecture

The system follows a five-stage pipeline:

```
Map Acquisition → Grid Rasterization → Coordinate Mapping → A* Pathfinding → B-spline Smoothing
```

Each stage is modular and can be independently tested or replaced.

---

## 3. Stage 1: Map Acquisition

### 3.1 Tile Source

We use **CartoDB Voyager (no labels)** as the base map provider. This tile set offers:
- Clean land/water color contrast (light land, dark water)
- No text labels that could interfere with image processing
- Free access via HTTPS tile server

### 3.2 Tile Coordinate System

Web Mercator tiles use a quadtree scheme. The conversion between geographic coordinates (latitude/longitude) and tile indices follows the standard OSM tile addressing:

```
x = floor((lon + 180) / 360 * 2^z)
y = floor((1 - asinh(tan(lat_rad)) / π) / 2 * 2^z)
```

where `z` is the zoom level. At zoom level 13, each tile covers approximately 0.044° × 0.044° (≈ 4.9 km × 3.5 km at this latitude).

### 3.3 Coverage and Caching

The system computes a bounding box that encompasses both start and goal coordinates with an 0.08° padding on each side. All tiles within this box are downloaded and stitched into a single image.

**Caching strategy:** The merged image and its geographic metadata (top-left and bottom-right coordinates) are saved to `data/map_cache.png` and `data/map_meta.json`. On subsequent runs, the cache is loaded directly, avoiding network requests.

---

## 4. Stage 2: Grid Rasterization

This is the most critical preprocessing step. The goal is to convert the color nautical chart into a binary grid where:
- `0` = navigable water
- `1` = obstacle (land + safety margin)

### 4.1 OTSU Auto-Thresholding

**Why OTSU?** The CartoDB Voyager map has a clear bimodal distribution: dark ocean pixels and bright land pixels. OTSU automatically finds the optimal threshold that maximizes inter-class variance, eliminating the need for manual threshold tuning.

```
gray = cvtColor(img, BGR2GRAY)
_, thresh = threshold(gray, 0, 255, BINARY_INV + OTSU)
land_mask = bitwise_not(thresh)
```

**Fallback:** If the OTSU result shows >90% land coverage (which can happen with unusual chart styles), the system falls back to a median-value threshold.

### 4.2 Morphological Denoising

**Opening (3×3 kernel):** Erosion followed by dilation. This removes thin bridge-like artifacts (e.g., causeways, pier structures) that incorrectly connect islands to the mainland, while preserving larger landmasses.

```
land_clean = morphologyEx(land_mask, MORPH_OPEN, 3×3_kernel)
```

**Why opening instead of closing?** Closing would fill gaps in water, potentially creating false water passages through land. Opening removes thin land bridges, which is the desired behavior.

### 4.3 Connected Component Analysis

After morphological cleaning, the water mask may contain multiple disconnected regions (ocean, lakes, rivers, noise). We retain only the **largest connected component** as the main navigable ocean.

```
water = (land_clean == 0)
n, labels, stats = connectedComponentsWithStats(water)
largest_label = argmax(stats[1:, AREA]) + 1
ocean = (labels == largest_label)
```

**Why this works:** In a coastal chart, the open ocean dominates the water area. Small inland water bodies (ponds, rivers) are either too small for navigation or irrelevant to the route. Removing them prevents the pathfinder from entering non-navigable dead ends.

### 4.4 Safety Margin Dilation

Ships cannot navigate arbitrarily close to shore. We apply a **3-pixel dilation** to the land mask, expanding obstacles into navigable water.

```
land_inflated = dilate(land_final, 3×3_kernel)
```

At zoom level 13, 3 pixels ≈ 15 meters, a reasonable clearance for small vessels. This parameter is configurable via `SAFETY_PX`.

---

## 5. Stage 3: Coordinate Mapping

### 5.1 Geographic to Pixel Conversion

The map image covers a known geographic bounding box (stored in `map_meta.json`). The conversion is a simple linear interpolation:

```
px_x = (lon - map_left) / (map_right - map_left) * image_width
px_y = (map_top - lat) / (map_top - map_bottom) * image_height
```

### 5.2 Ocean Snapping (BFS)

The user-specified start/goal coordinates may fall on land or outside the main ocean (due to coordinate precision or map resolution). The **BFS Snap** algorithm searches outward from the target point in 8 directions until it finds the nearest ocean pixel.

```
if point is on ocean: return point
else: BFS expand until ocean pixel found
```

**Why BFS instead of KD-tree?** BFS naturally handles the binary grid structure and guarantees finding the nearest valid pixel in Manhattan/Chebyshev distance. It's simple, correct, and fast for this grid size.

---

## 6. Stage 4: A* Pathfinding

### 6.1 Algorithm Design

We use the standard A* algorithm with the following configuration:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Movement | 8-directional | Allows diagonal movement for shorter paths |
| g-cost | 1.0 (cardinal), √2 (diagonal) | Accurate Euclidean step cost |
| h-cost | Euclidean distance to goal | Admissible heuristic, guarantees optimality |
| Data structure | Min-heap (binary heap) | O(log n) push/pop |

### 6.2 Why A* Over Other Algorithms?

- **Dijkstra:** Explores uniformly in all directions. A*'s heuristic guides search toward the goal, dramatically reducing explored nodes.
- **RRT/RRT\***: Better for high-dimensional continuous spaces, but overkill for a 2D grid. A* is guaranteed optimal on grids.
- **Theta\***: Produces smoother paths by allowing any-angle movement, but requires line-of-sight checks at each step. We achieve smoothness via post-processing (B-spline), which is simpler.

### 6.3 Path Reconstruction

When the goal is reached, the path is reconstructed by following `came_from` pointers back to the start, then reversing.

---

## 7. Stage 5: B-spline Trajectory Smoothing

### 7.1 Why Smoothing?

The A* output is a sequence of grid-aligned pixel coordinates. This produces:
- Zigzag patterns (especially on diagonal moves)
- Abrupt direction changes at grid boundaries
- Unsuitably jerky motion for real ships

### 7.2 B-spline Fitting

We use `scipy.interpolate.splprep` to fit a parametric B-spline through the discrete path points:

```python
tck, u = splprep([x, y], s=smoothing_factor)
u_new = linspace(0, 1, 300)
xs, ys = splev(u_new, tck)
```

**Smoothing factor `s`:** Set to `len(path) * 2.0`. This balances fidelity to the original path (avoiding obstacles) with smoothness (reducing curvature). A higher `s` produces smoother curves but may deviate from the optimal path.

### 7.3 Output

The smoothed trajectory is a continuous curve with 300 sample points, providing sufficient resolution for visualization and potential real-time navigation.

---

## 8. Visualization

### 8.1 Static Result (PNG)

A side-by-side figure showing:
- **Left:** Binary grid map with start/goal markers
- **Right:** Original chart overlay with A* path (dashed) and smoothed trajectory (solid)

### 8.2 Animated GIF

A ship marker (red triangle) moves along the smoothed trajectory, with a trailing path line. The animation is saved as GIF at 20 fps with ~80 frames (subsampled for manageable file size).

---

## 9. Iteration History

| Version | Key Improvement | File |
|---------|----------------|------|
| v0 | Early prototype, basic thresholding | `archive/v0_early_prototype.py` |
| v1 | OTSU auto-threshold + connected component | `archive/v1_basic_otsu.py` |
| v2 | Morphological bridge removal | `archive/v2_improved.py` |
| v3 | Dual-view visualization (grid + chart) | `archive/v3_dual_view.py` |
| v4 | HSV color prior + OTSU fusion pipeline | `archive/v4_fusion_pipeline.py` |
| **Final** | V1 rasterization + safety margin + GIF output | `main.py` |

**Design decision:** The final version adopts V1's simpler OTSU-based rasterization over V4's more complex HSV fusion. V4's green vegetation detection added complexity without significant improvement in the target region, where the CartoDB map already has clean land/water contrast.

---

## 10. Future Work

### 10.1 Dubins Path Integration

The current B-spline smoothing does not respect ship kinematic constraints. **Dubins curves** guarantee paths that satisfy:
- Minimum turning radius
- Forward-only motion (no reverse)
- Constant or bounded speed

Integration approach:
1. Extract waypoints from the smoothed trajectory at regular intervals
2. Connect consecutive waypoints with Dubins primitives (LSL, RSR, RSL, LSR, RLR, LRL)
3. Replace B-spline segments with Dubins arcs

### 10.2 Ship Dynamics Model

Real ships have constraints beyond turning radius:
- **Draft depth:** Shallow water regions become impassable
- **Beam width:** Wider ships need larger safety margins
- **Speed profiles:** Acceleration/deceleration limits affect maneuverability

These can be incorporated as additional cost terms in the A* heuristic or as post-processing constraints on the smoothed trajectory.

### 10.3 COLREGS Compliance

The International Regulations for Preventing Collisions at Sea (COLREGS) define:
- Head-on, crossing, and overtaking encounter rules
- Priority rules (stand-on vs. give-way vessels)
- Required actions (turn to starboard, reduce speed)

For multi-vessel scenarios, the path planner must generate COLREGS-compliant avoidance maneuvers.

### 10.4 Real-Time Replanning

For dynamic environments (moving obstacles, changing weather), the system could integrate:
- **D\* Lite** for incremental replanning when the grid changes
- **MPC (Model Predictive Control)** for continuous trajectory optimization
- **ROS integration** for sensor fusion and real-time updates

---

## 11. References

1. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100-107.
2. Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62-66.
3. Dubins, L. E. (1957). On Curves of Minimal Length with a Constraint on Average Curvature. *American Journal of Mathematics*, 79(3), 497-516.
4. de Berg, M., et al. (2008). *Computational Geometry: Algorithms and Applications*. Springer.

# Marine Route Planning

A* pathfinding + KT ship motion model with PID heading control, wind/current disturbance simulation, and B-spline trajectory smoothing on electronic nautical charts.

![result](output/result.png)
![ship_motion](output/result_ship.png)

## Features

- **Automatic map acquisition** from CartoDB raster tiles with local caching
- **Intelligent grid rasterization** via OTSU thresholding, morphological denoising, connected component analysis, and safety margin dilation
- **A\* shortest path search** with 8-directional movement and Euclidean heuristic
- **B-spline trajectory smoothing** for ship-navigable continuous paths
- **KT ship motion model** — nonlinear Nomoto equation (K=0.5, T=1.0) for realistic ship dynamics
- **PID heading control** — waypoint following with configurable KP/KI/KD gains
- **Wind/current disturbance** — constant wind (5 m/s NE) and current (0.5 m/s N) effects
- **Comprehensive visualization** — grid map, satellite overlay, KT trajectory, rudder angle, heading tracking, drift angle, speed profile

## Project Structure

```
Marine-Route-Planning/
├── main.py                  # Full pipeline: A* + KT model + visualization
├── requirements.txt
├── LICENSE
│
├── src/                     # Modular source code
│   ├── map_processor.py     # Map fetching, rasterization, coordinate tools
│   ├── astar.py             # A* algorithm, path smoothing
│   └── path_planner.py      # Pipeline orchestration, visualization
│
├── data/                    # Cached map tiles
├── output/                  # Example results
├── docs/                    # Technical report
└── archive/                 # Historical iterations (v0-v5)
```

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

Output files are generated in the current directory:
- `result*.png` — A* path on grid map and satellite overlay
- `result_ship*.png` — 4-panel: grid map, KT trajectory with heading arrows, rudder angle, heading tracking
- `result_motion_detail*.png` — heading error, speed profile, drift angle
- `animation*.gif` — animated ship movement along path

## Algorithm Overview

### 1. Map Acquisition
Download raster tiles from CartoDB Voyager (no labels), merge into a single image, and cache locally.

### 2. Grid Rasterization
OTSU auto-thresholding separates water (dark) from land (light). Morphological opening removes bridge artifacts. Connected component analysis retains only the largest water body. A 3-pixel safety dilation expands land obstacles.

### 3. A* Pathfinding
Standard A* with 8-directional movement on the binary grid. Diagonal moves cost sqrt(2), cardinal moves cost 1. Euclidean distance serves as the admissible heuristic.

### 4. B-spline Smoothing
The discrete pixel path is fitted with a B-spline curve, then uniformly sampled to extract equally-spaced waypoints (500m intervals).

### 5. KT Ship Motion Simulation
The ship follows waypoints using a PID heading controller. The KT (Nomoto) model governs ship dynamics:

```
dr/dt = (K * delta - (1 + a * |r|) * r) / T    # yaw rate
du/dt = -0.02 * u * |r| - 0.005 * u * delta^2   # speed loss
dv/dt = 0.15 * r * u - 0.25 * v                   # lateral velocity
```

Wind and current disturbances are applied as external forces on the lateral velocity and speed.

### Ship Parameters (Mariner class)

| Parameter | Value | Description |
|-----------|-------|-------------|
| L | 160 m | Ship length |
| B | 21 m | Beam |
| K | 0.5 | Maneuverability index |
| T | 1.0 | Responsiveness index |
| a | 0.4 | Nonlinear coefficient |
| V0 | 10 kn | Initial speed |
| delta_max | 35 deg | Maximum rudder angle |

## Roadmap

- [x] **KT ship motion model** — Nonlinear Nomoto dynamics with K/T indices
- [x] **PID heading control** — Waypoint following with configurable gains
- [x] **Wind/current disturbance** — Environmental disturbance simulation
- [ ] **MMG model integration** — Full hydrodynamic model for higher fidelity
- [ ] **COLREGS compliance** — Implement collision avoidance per maritime regulations
- [ ] **Multi-objective optimization** — Pareto-optimal trade-off between distance, safety, and fuel
- [ ] **Real-time replanning** — Dynamic obstacle avoidance with incremental A*

## License

[MIT](LICENSE)

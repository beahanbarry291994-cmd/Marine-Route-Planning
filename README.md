# Marine Route Planning

A* pathfinding + B-spline trajectory smoothing on electronic nautical charts.

![result](output/result.png)

## Features

- **Automatic map acquisition** from CartoDB raster tiles with local caching
- **Intelligent grid rasterization** via OTSU thresholding, morphological denoising, connected component analysis, and safety margin dilation
- **A\* shortest path search** with 8-directional movement and Euclidean heuristic
- **B-spline trajectory smoothing** for ship-navigable continuous paths
- **Dual visualization** — static PNG result + animated GIF

## Project Structure

```
Marine-Route-Planning/
├── main.py                  # Standalone entry point (all-in-one)
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
└── archive/                 # Historical iterations (v0-v4)
```

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run

```bash
# All-in-one version
python main.py

# Modular version
python -m src.path_planner
```

Output files (`result*.png`, `animation*.gif`) are generated in the current directory.

## Algorithm Overview

1. **Map Acquisition** — Download raster tiles from CartoDB Voyager (no labels), merge into a single image, and cache locally.
2. **Grid Rasterization** — OTSU auto-thresholding separates water (dark) from land (light). Morphological opening removes bridge artifacts. Connected component analysis retains only the largest water body. A 3-pixel safety dilation expands land obstacles.
3. **A\* Pathfinding** — Standard A\* with 8-directional movement on the binary grid. Diagonal moves cost √2, cardinal moves cost 1. Euclidean distance serves as the admissible heuristic.
4. **B-spline Smoothing** — The discrete pixel path is fitted with a B-spline curve, producing a smooth, continuous trajectory suitable for ship navigation.

## Roadmap

- [ ] **Dubins path integration** — Respect minimum turning radius constraints
- [ ] **Ship dynamics model** — Incorporate draft, beam, and speed constraints
- [ ] **COLREGS compliance** — Implement collision avoidance per maritime regulations
- [ ] **Multi-objective optimization** — Pareto-optimal trade-off between distance, safety, and fuel
- [ ] **Real-time replanning** — Dynamic obstacle avoidance with incremental A*

## License

[MIT](LICENSE)

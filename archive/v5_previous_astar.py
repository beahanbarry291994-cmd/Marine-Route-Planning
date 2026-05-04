"""
Marine Route Planning System
A* pathfinding + B-spline trajectory smoothing on electronic nautical charts.

Usage:
    python main.py
"""

import os
import re
import json
import math
import time
import ssl
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import splprep, splev
from collections import deque
import heapq
import urllib.request

if getattr(sys, 'frozen', False):
    os.chdir(os.path.dirname(sys.executable))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ──────────────────────── Parameters ────────────────────────
START_LONLAT = (122.295, 29.848)
GOAL_LONLAT  = (122.387, 29.9795)

ZOOM         = 13
SAFETY_PX    = 3
CACHE_IMG    = 'data/map_cache.png'
CACHE_META   = 'data/map_meta.json'


# ──────────────────────── Tile Utilities ────────────────────────
def deg2num(lat, lon, z):
    n = 2.0 ** z
    return int((lon + 180) / 360 * n), int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)

def num2deg(x, y, z):
    n = 2.0 ** z
    return (math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n)))),
            x / n * 360 - 180)


# ──────────────────────── Map Fetching ────────────────────────
def fetch_map(start, goal, zoom):
    global MAP_TL, MAP_BR
    if os.path.exists(CACHE_IMG) and os.path.exists(CACHE_META):
        meta = json.load(open(CACHE_META))
        MAP_TL, MAP_BR = tuple(meta['top_left']), tuple(meta['bottom_right'])
        img = cv2.imread(CACHE_IMG)
        if img is not None and np.std(img) > 10:
            print("[Map] Using local cache")
            return img

    print("[Map] Downloading CartoDB basemap...")
    d = 0.08
    x0, y1 = deg2num(min(start[1], goal[1]) - d, min(start[0], goal[0]) - d, zoom)
    x1, y0 = deg2num(max(start[1], goal[1]) + d, max(start[0], goal[0]) + d, zoom)

    MAP_TL = (num2deg(x0, y0, zoom)[1], num2deg(x0, y0, zoom)[0])
    MAP_BR = (num2deg(x1 + 1, y1 + 1, zoom)[1], num2deg(x1 + 1, y1 + 1, zoom)[0])

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    base = "https://basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png"
    gray_tile = np.full((256, 256, 3), 170, dtype=np.uint8)

    rows = []
    for y in range(y0, y1 + 1):
        cols = []
        for x in range(x0, x1 + 1):
            tile = None
            for _ in range(3):
                try:
                    req = urllib.request.Request(base.format(z=zoom, x=x, y=y),
                                                headers={'User-Agent': 'Mozilla/5.0'})
                    resp = urllib.request.urlopen(req, context=ctx, timeout=15)
                    tile = cv2.imdecode(np.frombuffer(resp.read(), np.uint8), cv2.IMREAD_COLOR)
                    if tile is not None:
                        break
                except Exception:
                    time.sleep(1.5)
            cols.append(tile if tile is not None else gray_tile)
        rows.append(np.concatenate(cols, axis=1))

    img = np.concatenate(rows, axis=0)
    os.makedirs(os.path.dirname(CACHE_IMG), exist_ok=True)
    cv2.imwrite(CACHE_IMG, img)
    json.dump({'top_left': MAP_TL, 'bottom_right': MAP_BR}, open(CACHE_META, 'w'))
    return img


# ──────────────────────── Grid Processing ────────────────────────
def process_map(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    land_mask = cv2.bitwise_not(thresh)

    land_ratio = np.sum(land_mask == 255) / land_mask.size
    if land_ratio > 0.90:
        median_val = np.median(gray)
        _, land_mask = cv2.threshold(gray, median_val, 255, cv2.THRESH_BINARY)

    land_clean = cv2.morphologyEx(land_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    water = (land_clean == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(water, connectivity=4)
    if n <= 1:
        return np.ones_like(land_clean, dtype=np.uint8), np.zeros_like(land_clean)

    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    main_ocean = (labels == largest_label).astype(np.uint8)
    land_final = ((land_clean == 255) | (main_ocean == 0)).astype(np.uint8)
    land_inflated = cv2.dilate(land_final, np.ones((SAFETY_PX, SAFETY_PX), np.uint8))
    return land_inflated, main_ocean


# ──────────────────────── Coordinate Tools ────────────────────────
def lonlat_to_px(lon, lat, shape):
    h, w = shape
    x = int((lon - MAP_TL[0]) / (MAP_BR[0] - MAP_TL[0]) * w)
    y = int((MAP_TL[1] - lat) / (MAP_TL[1] - MAP_BR[1]) * h)
    return max(0, min(x, w - 1)), max(0, min(y, h - 1))

def snap_to_ocean(grid, px, ocean):
    if grid[px[1], px[0]] == 0 and ocean[px[1], px[0]] == 1:
        return px
    h, w = grid.shape
    q, vis = deque([px]), {px}
    while q:
        cx, cy = q.popleft()
        if grid[cy, cx] == 0 and ocean[cy, cx] == 1:
            return (cx, cy)
        for dx, dy in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
            nxy = (cx + dx, cy + dy)
            if 0 <= nxy[0] < w and 0 <= nxy[1] < h and nxy not in vis:
                vis.add(nxy)
                q.append(nxy)
    return px


# ──────────────────────── A* Pathfinding ────────────────────────
def a_star(grid, start, goal):
    h, w = grid.shape
    open_set = [(0, start)]
    came_from, g = {}, {start: 0}
    dirs = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            return path[::-1]

        for dx, dy in dirs:
            nx, ny = cur[0] + dx, cur[1] + dy
            if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0:
                ng = g[cur] + (1.414 if dx and dy else 1.0)
                nb = (nx, ny)
                if ng < g.get(nb, float('inf')):
                    g[nb] = ng
                    came_from[nb] = cur
                    heapq.heappush(open_set, (ng + math.hypot(nx - goal[0], ny - goal[1]), nb))
    return None


# ──────────────────────── Path Smoothing ────────────────────────
def smooth_path(path, n_pts=300):
    x, y = zip(*path)
    tck, _ = splprep([x, y], s=len(path) * 2.0)
    u = np.linspace(0, 1, max(n_pts, len(path)))
    return splev(u, tck)


# ──────────────────────── Visualization ────────────────────────
def next_filename(prefix, ext):
    mx = 0
    for f in os.listdir('.'):
        m = re.match(rf'{re.escape(prefix)}(\d+)\.{re.escape(ext)}$', f)
        if m:
            mx = max(mx, int(m.group(1)))
    return f'{prefix}{mx + 1}.{ext}'


def visualize(grid, img, start, goal, path_x, path_y, xs, ys):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Static result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Marine Route Planning Result", fontsize=15, fontweight='bold', y=0.98)

    ax1.imshow(1 - grid, cmap='gray')
    ax1.plot(*start, 'go', ms=10, label='Start')
    ax1.plot(*goal, 'ro', ms=10, label='Goal')
    ax1.set_title("Grid Map (white=navigable, black=land+safety)", fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.axis('off')

    ax2.imshow(img_rgb)
    ax2.plot(*start, 'go', ms=10, label='Start')
    ax2.plot(*goal, 'ro', ms=10, label='Goal')
    ax2.plot(path_x, path_y, 'y--', alpha=0.5, lw=1, label='A* Path')
    ax2.plot(xs, ys, 'r-', lw=2, label='Smoothed Trajectory')
    ax2.plot(xs[-1], ys[-1], 'r^', ms=12, mec='black')
    ax2.set_title("A* + B-spline Smoothing", fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.axis('off')

    static_name = next_filename('result', 'png')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(static_name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Save] Static -> {static_name}")

    # Animated GIF
    fig2, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_rgb)
    ax.plot(*start, 'go', ms=10, label='Start')
    ax.plot(*goal, 'ro', ms=10, label='Goal')
    ax.plot(xs, ys, 'r-', lw=2, alpha=0.3, label='Smoothed Trajectory')
    ship, = ax.plot([], [], 'r^', ms=12, mec='black')
    trail, = ax.plot([], [], 'r-', lw=2)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title("Navigation Animation", fontsize=13, fontweight='bold')
    ax.axis('off')

    step = max(1, len(xs) // 80)
    frames = list(range(0, len(xs), step))
    if frames[-1] != len(xs) - 1:
        frames.append(len(xs) - 1)

    def animate(i):
        ship.set_data([xs[i]], [ys[i]])
        trail.set_data(xs[:i+1], ys[:i+1])
        return ship, trail

    ani = animation.FuncAnimation(fig2, animate, frames=frames,
                                  interval=50, blit=True, repeat=False)

    gif_name = next_filename('animation', 'gif')
    plt.tight_layout()
    ani.save(gif_name, writer='pillow', fps=20)
    plt.close(fig2)
    print(f"[Save] Animation -> {gif_name}")


# ──────────────────────── Main Pipeline ────────────────────────
def main():
    print("=" * 50)
    print("  Marine Route Planning System")
    print("  A* + B-spline on Electronic Nautical Chart")
    print("=" * 50)

    print("\n[1/5] Fetching map...")
    img = fetch_map(START_LONLAT, GOAL_LONLAT, ZOOM)

    print("[2/5] Processing grid map (OTSU + morphology)...")
    grid, ocean = process_map(img)

    print("[3/5] Coordinate mapping & snapping...")
    start = snap_to_ocean(grid, lonlat_to_px(*START_LONLAT, grid.shape), ocean)
    goal  = snap_to_ocean(grid, lonlat_to_px(*GOAL_LONLAT,  grid.shape), ocean)

    print("[4/5] A* pathfinding...")
    t0 = time.time()
    path = a_star(grid, start, goal)
    if path is None:
        print("[Error] No feasible path found")
        input("\nPress Enter to exit...")
        return
    print(f"        {time.time() - t0:.2f}s, {len(path)} waypoints")

    print("[5/5] B-spline smoothing & visualization...")
    xs, ys = smooth_path(path)
    path_x, path_y = zip(*path)
    visualize(grid, img, start, goal, path_x, path_y, xs, ys)

    print("\n[Done] Check result*.png and animation*.gif")


if __name__ == '__main__':
    main()

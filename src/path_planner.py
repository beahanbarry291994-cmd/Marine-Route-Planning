"""
航线规划主流程模块

功能:
  - 调用 map_processor 和 astar 模块完成端到端路径规划
  - 静态结果图 + 动态 GIF 可视化
"""

import os
import re
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .map_processor import fetch_map, process_map, lonlat_to_px, snap_to_ocean
from .astar import a_star, smooth_path

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def next_filename(prefix, ext):
    mx = 0
    for f in os.listdir('.'):
        m = re.match(rf'{re.escape(prefix)}(\d+)\.{re.escape(ext)}$', f)
        if m:
            mx = max(mx, int(m.group(1)))
    return f'{prefix}{mx + 1}.{ext}'


def visualize(grid, img, start, goal, path_x, path_y, xs, ys, output_dir='.'):
    """生成静态 PNG 和动态 GIF"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── 静态结果图 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Marine Route Planning Result", fontsize=15, fontweight='bold', y=0.98)

    ax1.imshow(1 - grid, cmap='gray')
    ax1.plot(*start, 'go', ms=10, label='Start')
    ax1.plot(*goal, 'ro', ms=10, label='Goal')
    ax1.set_title("Grid Map (white=navigable, black=land+safety margin)", fontsize=11)
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

    static_path = os.path.join(output_dir, next_filename('result', 'png'))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(static_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Save] Static image -> {static_path}")

    # ── 动态 GIF ──
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
        trail.set_data(xs[:i + 1], ys[:i + 1])
        return ship, trail

    ani = animation.FuncAnimation(fig2, animate, frames=frames,
                                  interval=50, blit=True, repeat=False)

    gif_path = os.path.join(output_dir, next_filename('animation', 'gif'))
    plt.tight_layout()
    ani.save(gif_path, writer='pillow', fps=20)
    plt.close(fig2)
    print(f"[Save] Animation -> {gif_path}")

    return static_path, gif_path


def run(start_lonlat, goal_lonlat, zoom=13, safety_px=3, output_dir='.'):
    """
    端到端航线规划流程

    Args:
        start_lonlat: (lon, lat) 起点
        goal_lonlat:  (lon, lat) 终点
        zoom:         地图缩放级别
        safety_px:    安全膨胀半径
        output_dir:   输出目录

    Returns:
        path: A* 原始路径
        xs, ys: 平滑轨迹
    """
    print("[1/5] Fetching map...")
    img, map_tl, map_br = fetch_map(start_lonlat, goal_lonlat, zoom)

    print("[2/5] Processing grid map...")
    grid, ocean = process_map(img, safety_px)

    print("[3/5] Coordinate mapping & snapping...")
    start = snap_to_ocean(grid, lonlat_to_px(*start_lonlat, grid.shape, map_tl, map_br), ocean)
    goal = snap_to_ocean(grid, lonlat_to_px(*goal_lonlat, grid.shape, map_tl, map_br), ocean)

    print("[4/5] A* pathfinding...")
    import time
    t0 = time.time()
    path = a_star(grid, start, goal)
    if path is None:
        print("[Error] No feasible path found")
        return None, None, None
    print(f"        {time.time() - t0:.2f}s, {len(path)} waypoints")

    print("[5/5] B-spline smoothing & visualization...")
    xs, ys = smooth_path(path)
    path_x, path_y = zip(*path)
    visualize(grid, img, start, goal, path_x, path_y, xs, ys, output_dir)

    return path, xs, ys

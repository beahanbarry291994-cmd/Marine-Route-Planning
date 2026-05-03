"""
A* 寻路与轨迹平滑模块

功能:
  - 八方向 A* 最短路径搜索
  - B-spline 轨迹平滑
"""

import math
import heapq

import numpy as np
from scipy.interpolate import splprep, splev


def a_star(grid, start, goal):
    """
    A* 最短路径算法 (八方向移动, 欧几里得启发式)

    Args:
        grid: 障碍栅格 (0=可通行, 1=障碍)
        start: (x, y) 起点像素坐标
        goal:  (x, y) 终点像素坐标

    Returns:
        path: [(x,y), ...] 路径点列表, 或 None (不可达)
    """
    h, w = grid.shape
    open_set = [(0, start)]
    came_from, g = {}, {start: 0}
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

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


def smooth_path(path, n_pts=300):
    """
    B-spline 轨迹平滑

    Args:
        path: [(x,y), ...] 离散路径点
        n_pts: 平滑后的采样点数

    Returns:
        xs, ys: 平滑后的坐标数组
    """
    x, y = zip(*path)
    tck, _ = splprep([x, y], s=len(path) * 2.0)
    u = np.linspace(0, 1, max(n_pts, len(path)))
    return splev(u, tck)

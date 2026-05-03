import math
import heapq
import requests
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union


# ================== 1. 自动下载地图数据 ==================
def download_osm_coastlines(south, west, north, east):
    """
    使用 Overpass API 下载指定矩形区域内的岛屿、礁石等陆地多边形。
    返回 Polygon/MultiPolygon 列表 (WGS84)。
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    # 查询 coastline, island, islet, reef 等
    query = f"""
    [out:json];
    (
      way["natural"="coastline"]({south},{west},{north},{east});
      relation["natural"="coastline"]({south},{west},{north},{east});
      way["place"="island"]({south},{west},{north},{east});
      relation["place"="island"]({south},{west},{north},{east});
      way["place"="islet"]({south},{west},{north},{east});
      way["natural"="reef"]({south},{west},{north},{east});
      way["natural"="rock"]({south},{west},{north},{east});
    );
    (._;>;);
    out body;
    """
    response = requests.get(overpass_url, params={'data': query})
    data = response.json()

    # 解析节点坐标
    nodes = {}
    for elem in data['elements']:
        if elem['type'] == 'node':
            nodes[elem['id']] = (elem['lon'], elem['lat'])

    # 构建多边形
    polygons = []
    for elem in data['elements']:
        if elem['type'] == 'way' and 'nodes' in elem:
            coords = [nodes[nid] for nid in elem['nodes'] if nid in nodes]
            if len(coords) > 3:
                poly = Polygon(coords)
                if poly.is_valid and not poly.is_empty:
                    polygons.append(poly)
    # 合并重叠多边形
    if polygons:
        return unary_union(polygons)
    return None


# ================== 2. 栅格化 ==================
def lonlat_to_rc(lon, lat, lon_min, lat_max, res):
    c = int((lon - lon_min) / res)
    r = int((lat_max - lat) / res)
    return r, c


def rc_to_lonlat(r, c, lon_min, lat_max, res):
    lon = lon_min + c * res
    lat = lat_max - r * res
    return lon, lat


def build_grid(land_poly, lon_min, lon_max, lat_min, lat_max, res):
    cols = int((lon_max - lon_min) / res) + 1
    rows = int((lat_max - lat_min) / res) + 1
    grid = [[0] * cols for _ in range(rows)]  # 0=水, 1=陆地

    # 对每个格点采样（使用中心点判断）
    for r in range(rows):
        for c in range(cols):
            lon = lon_min + c * res + res / 2
            lat = lat_max - r * res - res / 2
            if land_poly.contains(Point(lon, lat)):
                grid[r][c] = 1
    return grid, rows, cols


# ================== 3. A* 搜索（考虑缓冲区） ==================
def haversine(p1, p2):
    lon1, lat1 = p1
    lon2, lat2 = p2
    R = 6371.0
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def a_star(grid, start_rc, goal_rc, lon_min, lat_max, res):
    rows, cols = len(grid), len(grid[0])
    sr, sc = start_rc
    gr, gc = goal_rc

    g = [[float('inf')] * cols for _ in range(rows)]
    g[sr][sc] = 0.0
    parent = [[None] * cols for _ in range(rows)]

    open_heap = []
    h_start = haversine(rc_to_lonlat(sr, sc, lon_min, lat_max, res),
                        rc_to_lonlat(gr, gc, lon_min, lat_max, res))
    heapq.heappush(open_heap, (h_start, sr, sc))
    closed = set()

    # 8邻域方向
    dirs = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
            (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (1, 1, 1.414)]

    while open_heap:
        f, r, c = heapq.heappop(open_heap)
        if (r, c) == (gr, gc):
            # 回溯路径
            path = []
            while (r, c) is not None:
                path.append((r, c))
                prev = parent[r][c]
                if prev is None: break
                r, c = prev
            path.reverse()
            return path
        if (r, c) in closed:
            continue
        closed.add((r, c))

        for dr, dc, cost_ratio in dirs:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr][nc] == 1:  # 障碍物
                continue
            # 对角线移动需检查相邻两格不穿墙
            if cost_ratio > 1.1:
                if grid[r][nc] == 1 or grid[nr][c] == 1:
                    continue
            if (nr, nc) in closed:
                continue
            step_km = haversine(rc_to_lonlat(r, c, lon_min, lat_max, res),
                                rc_to_lonlat(nr, nc, lon_min, lat_max, res))
            new_g = g[r][c] + step_km
            if new_g < g[nr][nc]:
                g[nr][nc] = new_g
                parent[nr][nc] = (r, c)
                h = haversine(rc_to_lonlat(nr, nc, lon_min, lat_max, res),
                              rc_to_lonlat(gr, gc, lon_min, lat_max, res))
                heapq.heappush(open_heap, (new_g + h, nr, nc))
    return None


# ================== 4. 路径平滑（船舶运动特性） ==================
def smooth_path(rc_path, lon_min, lat_max, res, angle_limit_deg=30):
    """去除共线点并平滑转角，返回 (lon,lat) 列表"""
    if len(rc_path) < 3:
        return [rc_to_lonlat(r, c, lon_min, lat_max, res) for r, c in rc_path]

    # 转为经纬度
    pts = [rc_to_lonlat(r, c, lon_min, lat_max, res) for r, c in rc_path]

    # 共线点剔除
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    simplified = [pts[0]]
    for i in range(1, len(pts) - 1):
        if abs(cross(pts[i - 1], pts[i], pts[i + 1])) > 1e-8:
            simplified.append(pts[i])
    simplified.append(pts[-1])

    # 转角平滑
    def angle(p, q, r):
        v1 = (q[0] - p[0], q[1] - p[1])
        v2 = (r[0] - q[0], r[1] - q[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        n = math.hypot(*v1) * math.hypot(*v2)
        return math.degrees(math.acos(max(-1, min(1, dot / n))))

    final = [simplified[0]]
    for i in range(1, len(simplified) - 1):
        ang = angle(simplified[i - 1], simplified[i], simplified[i + 1])
        if ang > angle_limit_deg:
            p, q, r = simplified[i - 1], simplified[i], simplified[i + 1]
            # 插入两个中间点使转弯平滑
            t1 = (p[0] * 0.65 + q[0] * 0.35, p[1] * 0.65 + q[1] * 0.35)
            t2 = (q[0] * 0.65 + r[0] * 0.35, q[1] * 0.65 + r[1] * 0.35)
            final.extend([t1, q, t2])
        else:
            final.append(simplified[i])
    final.append(simplified[-1])
    return final


# ================== 5. 可视化 ==================
def plot_and_animate(grid, raw_path_rc, smooth_pts, lon_min, lon_max, lat_min, lat_max, res, start, goal):
    rows, cols = len(grid), len(grid[0])

    # 栅格地图颜色矩阵
    import numpy as np
    img = np.zeros((rows, cols, 3))
    img[grid == 1] = [0.2, 0.2, 0.2]  # 陆地深灰
    img[grid == 0] = [0.85, 0.95, 1.0]  # 水域浅蓝

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：静态栅格 + 平滑路径
    ax1.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max], origin='upper')
    ax1.plot([s[0] for s in smooth_pts], [s[1] for s in smooth_pts], 'r-', linewidth=2, label='Planned path')
    ax1.plot(start[0], start[1], 'go', markersize=10, label='Start (Taohua)')
    ax1.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal (Putuo Mtn)')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Grid Map with Smoothed Path')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 右图：动态轨迹动画
    ax2.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max], origin='upper')
    ax2.plot(start[0], start[1], 'go', markersize=8)
    ax2.plot(goal[0], goal[1], 'ro', markersize=8)
    ship, = ax2.plot([], [], 'ko', markersize=10, marker='s')  # 船舶方块
    trail, = ax2.plot([], [], 'y-', linewidth=2, alpha=0.8)  # 尾迹
    ax2.set_xlabel('Longitude')
    ax2.set_title('Ship Dynamic Trajectory')

    # 动画更新函数
    def update(frame):
        idx = min(frame, len(smooth_pts) - 1)
        ship.set_data([smooth_pts[idx][0]], [smooth_pts[idx][1]])
        if idx > 0:
            trail.set_data([p[0] for p in smooth_pts[:idx + 1]],
                           [p[1] for p in smooth_pts[:idx + 1]])
        return ship, trail

    ani = animation.FuncAnimation(fig, update, frames=len(smooth_pts),
                                  interval=200, blit=True, repeat=False)
    # 保存为GIF
    ani.save('ship_trajectory.gif', writer='pillow', fps=5)
    plt.tight_layout()
    plt.show()

    print("动画已保存为 ship_trajectory.gif")


# ================== 主程序 ==================
if __name__ == "__main__":
    # 作业指定目标区域（略扩大以包含全部岛屿）
    lon_min, lon_max = 122.27, 122.41
    lat_min, lat_max = 29.83, 30.00
    RES = 0.0008  # 约 89m/像素，精度足够

    # 起点/终点
    START = (122.295, 29.848)  # 桃花岛
    GOAL = (122.387, 29.9795)  # 普陀山港

    print("正在从OpenStreetMap下载岛屿数据...")
    land_poly = download_osm_coastlines(lat_min, lon_min, lat_max, lon_max)
    if land_poly is None:
        print("下载失败，将使用预定义多边形（仅演示）。")
        # 这里可以 fallback 到手动多边形，但为了演示，我们直接退出或给出提示
        # 真实作业建议提前下载好或使用离线地图瓦片
        print("请检查网络，或改用其他数据源。")
        exit(1)

    print("栅格化中...")
    grid, rows, cols = build_grid(land_poly, lon_min, lon_max, lat_min, lat_max, RES)
    print(f"栅格大小: {rows}×{cols}")

    # 起始栅格坐标
    sr, sc = lonlat_to_rc(START[0], START[1], lon_min, lat_max, RES)
    gr, gc = lonlat_to_rc(GOAL[0], GOAL[1], lon_min, lat_max, RES)
    # 确保起点/终点在水域
    grid[sr][sc] = 0
    grid[gr][gc] = 0

    print("A* 搜索中...")
    raw_path = a_star(grid, (sr, sc), (gr, gc), lon_min, lat_max, RES)
    if raw_path is None:
        print("未找到路径！")
        exit(1)

    print(f"原始路径点: {len(raw_path)}")
    smooth_pts = smooth_path(raw_path, lon_min, lat_max, RES)
    print(f"平滑后航路点: {len(smooth_pts)}")

    # 计算总航程
    dist = sum(haversine(smooth_pts[i], smooth_pts[i + 1]) for i in range(len(smooth_pts) - 1))
    print(f"总航程: {dist:.2f} km")

    # 输出航路点
    print("\n航路点 (lon, lat):")
    for i, pt in enumerate(smooth_pts):
        print(f"WP{i}: {pt[0]:.4f}, {pt[1]:.4f}")

    # 可视化
    plot_and_animate(grid, raw_path, smooth_pts, lon_min, lon_max, lat_min, lat_max, RES, START, GOAL)
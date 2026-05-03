import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import splprep, splev
import heapq
import math
import urllib.request
import ssl
import time
import os
import json
import random
import re
from collections import deque

# ---------- 中文字体支持 ----------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 参数配置区
# ==========================================
START_LONLAT = (122.295, 29.848)   # 桃花岛
GOAL_LONLAT  = (122.387, 29.9795)  # 普陀山

MAP_ZOOM_LEVEL = 13

CACHE_IMG_FILE = 'map_cache.png'
CACHE_META_FILE = 'map_meta.json'


# ==========================================
# 2. 地图下载（高德卫星图）
# ==========================================
def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def fetch_map_with_cache(start_lonlat, goal_lonlat, zoom):
    global MAP_TOP_LEFT, MAP_BOTTOM_RIGHT

    if os.path.exists(CACHE_IMG_FILE) and os.path.exists(CACHE_META_FILE):
        print("检测到本地地图缓存，验证有效性...")
        with open(CACHE_META_FILE, 'r') as f:
            meta = json.load(f)
        MAP_TOP_LEFT = tuple(meta['top_left'])
        MAP_BOTTOM_RIGHT = tuple(meta['bottom_right'])
        img = cv2.imread(CACHE_IMG_FILE)
        if img is not None and np.std(img) > 10:
            return img
        else:
            print("缓存无效，重新下载。")

    print("从高德卫星图下载瓦片...")
    min_lon = min(start_lonlat[0], goal_lonlat[0]) - 0.08
    max_lon = max(start_lonlat[0], goal_lonlat[0]) + 0.08
    min_lat = min(start_lonlat[1], goal_lonlat[1]) - 0.08
    max_lat = max(start_lonlat[1], goal_lonlat[1]) + 0.08

    x_min, y_max = deg2num(min_lat, min_lon, zoom)
    x_max, y_min = deg2num(max_lat, max_lon, zoom)

    MAP_TOP_LEFT = (num2deg(x_min, y_min, zoom)[1], num2deg(x_min, y_min, zoom)[0])
    MAP_BOTTOM_RIGHT = (num2deg(x_max + 1, y_max + 1, zoom)[1], num2deg(x_max + 1, y_max + 1, zoom)[0])

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    row_images = []
    max_tiles = 2 ** zoom
    failed_tiles = 0

    for y in range(y_min, y_max + 1):
        col_images = []
        for x in range(x_min, x_max + 1):
            y_amap = max_tiles - 1 - y
            server = random.randint(0, 3)
            url = f"https://webst0{server}.is.autonavi.com/appmaptile?style=6&x={x}&y={y_amap}&z={zoom}"

            tile = None
            for attempt in range(3):
                try:
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    resp = urllib.request.urlopen(req, context=ctx, timeout=15)
                    img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                    tile = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if tile is not None:
                        break
                except:
                    time.sleep(1.0)
            if tile is None:
                tile = np.ones((256, 256, 3), dtype=np.uint8) * 170
                failed_tiles += 1
            col_images.append(tile)
        row_images.append(np.concatenate(col_images, axis=1))

    if failed_tiles > 0:
        print(f"⚠ 有 {failed_tiles} 个瓦片下载失败，已用灰色填充。")

    full_img = np.concatenate(row_images, axis=0)
    cv2.imwrite(CACHE_IMG_FILE, full_img)
    with open(CACHE_META_FILE, 'w') as f:
        json.dump({'top_left': MAP_TOP_LEFT, 'bottom_right': MAP_BOTTOM_RIGHT}, f)
    print("地图缓存已更新。")
    return full_img


# ==========================================
# 3. 地图处理
# ==========================================
def process_map_to_grid(start_lonlat, goal_lonlat):
    img = fetch_map_with_cache(start_lonlat, goal_lonlat, MAP_ZOOM_LEVEL)
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    land_mask = cv2.bitwise_not(thresh)
    land_ratio = np.sum(land_mask == 255) / land_mask.size
    print(f"   [OTSU] 陆地占比: {land_ratio:.2%}")

    if land_ratio > 0.90:
        print("   ⚠ 陆地过多，改用中位数阈值...")
        median_val = np.median(gray)
        _, land_mask = cv2.threshold(gray, median_val, 255, cv2.THRESH_BINARY)
        land_ratio = np.sum(land_mask == 255) / land_mask.size
        print(f"   [中位数阈值 {median_val}] 陆地占比: {land_ratio:.2%}")

    kernel = np.ones((3, 3), np.uint8)
    land_no_bridges = cv2.morphologyEx(land_mask, cv2.MORPH_OPEN, kernel)

    water = (land_no_bridges == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(water, connectivity=4)
    print(f"   [连通域] 水域连通域数: {num_labels - 1}")

    if num_labels <= 1:
        grid_map = np.ones_like(land_no_bridges, dtype=np.uint8)
        return grid_map, img, (h, w), None, 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    print(f"   ✅ 最大海洋面积: {areas[largest_label-1]} 像素，占比 {areas[largest_label-1]/water.size:.2%}")

    real_sea = np.zeros_like(land_no_bridges, dtype=np.uint8)
    real_sea[labels == largest_label] = 255
    land_final = (land_no_bridges == 255) | (real_sea == 0)
    grid_map = land_final.astype(np.uint8)
    return grid_map, img, (h, w), labels, largest_label


# ==========================================
# 4. 坐标映射与吸附
# ==========================================
def lonlat_to_pixel(lon, lat, img_shape):
    h, w = img_shape
    min_lon, max_lat = MAP_TOP_LEFT
    max_lon, min_lat = MAP_BOTTOM_RIGHT
    x = int((lon - min_lon) / (max_lon - min_lon) * w)
    y = int((max_lat - lat) / (max_lat - min_lat) * h)
    return (max(0, min(x, w - 1)), max(0, min(y, h - 1)))

def find_nearest_water_in_ocean(grid, start_px, labels, ocean_label):
    h, w = grid.shape
    sx, sy = start_px
    if grid[sy, sx] == 0 and labels[sy, sx] == ocean_label:
        return start_px
    print(f"   * 坐标 {start_px} 不在主海洋，BFS 吸附中...")
    q = deque([(sx, sy)])
    visited = set([(sx, sy)])
    dirs = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    while q:
        cx, cy = q.popleft()
        if grid[cy, cx] == 0 and labels[cy, cx] == ocean_label:
            print(f"   * 吸附至 {(cx, cy)}")
            return (cx, cy)
        for dx, dy in dirs:
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append((nx, ny))
    return start_px


# ==========================================
# 5. 标准 A*
# ==========================================
def heuristic(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def a_star_standard(grid, start, goal):
    h, w = grid.shape
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    dirs = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        for dx, dy in dirs:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0:
                move_cost = 1.414 if dx*dy != 0 else 1.0
                tentative_g = g_score[current] + move_cost
                neighbor = (nx, ny)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None


# ==========================================
# 6. 自动文件名递增工具
# ==========================================
def get_next_image_name(prefix="image"):
    max_num = 0
    pattern = re.compile(rf"{prefix}(\d+)\.png")
    for fname in os.listdir('.'):
        m = pattern.match(fname)
        if m:
            num = int(m.group(1))
            if num > max_num:
                max_num = num
    return f"{prefix}{max_num + 1}.png"


# ==========================================
# 7. 主流程（双图合一 + 自动保存）
# ==========================================
def main():
    print("1. 地图获取与预处理...")
    grid_map, original_img, img_shape, labels, ocean_label = process_map_to_grid(START_LONLAT, GOAL_LONLAT)

    if ocean_label is None or labels is None:
        print("❌ 初始化失败：无法识别海洋。")
        return

    start_px = lonlat_to_pixel(START_LONLAT[0], START_LONLAT[1], img_shape)
    goal_px  = lonlat_to_pixel(GOAL_LONLAT[0], GOAL_LONLAT[1], img_shape)
    print(f"2. 起点像素: {start_px}, 终点像素: {goal_px}")

    start_px = find_nearest_water_in_ocean(grid_map, start_px, labels, ocean_label)
    goal_px  = find_nearest_water_in_ocean(grid_map, goal_px, labels, ocean_label)

    print("3. 标准 A* 寻路...")
    t0 = time.time()
    path_px = a_star_standard(grid_map, start_px, goal_px)
    print(f"   -> 耗时 {time.time()-t0:.2f} 秒")

    if not path_px:
        print("❌ 未找到路径。")
        return

    print(f"4. 原始路径点 {len(path_px)} 个，B 样条平滑...")
    x = [p[0] for p in path_px]
    y = [p[1] for p in path_px]
    tck, u = splprep([x, y], s=2.0)
    u_new = np.linspace(u.min(), u.max(), max(300, len(path_px)))
    x_smooth, y_smooth = splev(u_new, tck)

    # ---------- 创建合并画布 ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("桃花岛 → 普陀山 智能航行路径规划", fontsize=16)

    # 左图：栅格地图（白水黑陆）
    ax1.imshow(1 - grid_map, cmap='gray')
    ax1.plot(start_px[0], start_px[1], 'go', markersize=10, label='起点')
    ax1.plot(goal_px[0], goal_px[1], 'ro', markersize=10, label='终点')
    ax1.set_title("二值栅格地图（白=水，黑=陆）")
    ax1.legend()
    ax1.axis('off')

    # 右图：彩色原图 + 路径 + 船舶动画
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    ax2.imshow(img_rgb)
    ax2.plot(start_px[0], start_px[1], 'go', markersize=10, label='起点')
    ax2.plot(goal_px[0], goal_px[1], 'ro', markersize=10, label='终点')
    ax2.plot(x, y, 'y--', alpha=0.5, label='A* 离散路径')
    ax2.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='平滑轨迹')

    ship, = ax2.plot([], [], 'r^', markersize=12, markeredgecolor='black', label='船舶')
    ax2.set_title("动态航行视图")
    ax2.legend(loc='upper right')
    ax2.axis('off')

    # ---------- 先保存最终状态的结果图（船舶在终点） ----------
    ship.set_data([x_smooth[-1]], [y_smooth[-1]])   # 船舶放到终点
    save_name = get_next_image_name("image")
    fig.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"✅ 结果已保存为：{save_name}")

    # ---------- 然后开始动画（船舶从起点移动到终点） ----------
    ship.set_data([], [])   # 清空，准备动画

    def init():
        ship.set_data([], [])
        return ship,

    def animate(i):
        ship.set_data([x_smooth[i]], [y_smooth[i]])
        return ship,

    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(x_smooth), interval=30, blit=True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
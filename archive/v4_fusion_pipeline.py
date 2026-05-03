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
from collections import deque

# ---------- 中文字体支持 ----------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 参数配置
# ==========================================
START_LONLAT = (122.295, 29.848)   # 桃花岛
GOAL_LONLAT  = (122.387, 29.9795)  # 普陀山

MAP_ZOOM_LEVEL = 13
SAFETY_RADIUS_PX = 8               # 安全膨胀半径
CACHE_IMG_FILE = 'map_cache.png'
CACHE_META_FILE = 'map_meta.json'

# ==========================================
# 2. 地图下载（带缓存，同 V1）
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
        with open(CACHE_META_FILE, 'r') as f:
            meta = json.load(f)
        MAP_TOP_LEFT = tuple(meta['top_left'])
        MAP_BOTTOM_RIGHT = tuple(meta['bottom_right'])
        img = cv2.imread(CACHE_IMG_FILE)
        if img is not None and np.std(img) > 10:
            print("使用本地地图缓存")
            return img

    print("下载 CartoDB 无标签底图...")
    min_lon = min(start_lonlat[0], goal_lonlat[0]) - 0.08
    max_lon = max(start_lonlat[0], goal_lonlat[0]) + 0.08
    min_lat = min(start_lonlat[1], goal_lonlat[1]) - 0.08
    max_lat = max(start_lonlat[1], goal_lonlat[1]) + 0.08

    x_min, y_max = deg2num(min_lat, min_lon, zoom)
    x_max, y_min = deg2num(max_lat, max_lon, zoom)

    MAP_TOP_LEFT = num2deg(x_min, y_min, zoom)
    MAP_TOP_LEFT = (MAP_TOP_LEFT[1], MAP_TOP_LEFT[0])
    MAP_BOTTOM_RIGHT = num2deg(x_max + 1, y_max + 1, zoom)
    MAP_BOTTOM_RIGHT = (MAP_BOTTOM_RIGHT[1], MAP_BOTTOM_RIGHT[0])

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    base_url = "https://basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png"
    row_images = []
    for y in range(y_min, y_max + 1):
        col_images = []
        for x in range(x_min, x_max + 1):
            url = base_url.format(z=zoom, x=x, y=y)
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
                    time.sleep(1.5)
            if tile is None:
                tile = np.ones((256, 256, 3), dtype=np.uint8) * 170
            col_images.append(tile)
        row_images.append(np.concatenate(col_images, axis=1))
    full_img = np.concatenate(row_images, axis=0)
    cv2.imwrite(CACHE_IMG_FILE, full_img)
    with open(CACHE_META_FILE, 'w') as f:
        json.dump({'top_left': MAP_TOP_LEFT, 'bottom_right': MAP_BOTTOM_RIGHT}, f)
    return full_img

# ==========================================
# 3. 融合栅格化：颜色修正 + OTSU + 连通域 + 安全膨胀
# ==========================================
def process_map_fusion(img):
    """
    返回:
        grid_map : 0 可通行(海洋), 1 障碍(陆地+安全距离)
        ocean_mask : 0 陆地, 1 纯净海洋(无膨胀)
        其他用于可视化的中间结果
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3.1 颜色先验：植被（绿色）强制陆地，水体（蓝/青）强制海洋
    # 绿色范围 (H 35-85, S>40, V>40)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    # 蓝色/青色范围 (H 90-130, S>30, V>30) 用于识别明确水体
    blue_mask = cv2.inRange(hsv, (90, 30, 30), (130, 255, 255))

    # 3.2 OTSU 二值化（灰度）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 此时 thresh: 水域=255, 陆地=0 (与 V1 相同，后续反转)
    otsu_water = (thresh == 255)   # 布尔掩膜

    # 3.3 颜色修正 OTSU 结果
    # 策略：如果某像素被绿色掩膜命中，强制为陆地（即使 OTSU 误判为水）
    #        如果某像素被蓝色掩膜命中且 OTSU 判为陆地，则信任 OTSU（可能是浅水区）
    #        如果 OTSU 判为水但既不是绿色也不是蓝色，保持 OTSU 判断（可能是灰色浅滩）
    land_from_color = green_mask > 0
    water_from_color = blue_mask > 0

    # 最终海洋 = OTSU认为是水 且 不是绿色植被
    refined_water = otsu_water & (~land_from_color)

    # 如果 OTSU 认为水，且颜色先验也是水，则加强确信
    # 如果 OTSU 认为陆地，但颜色先验是水，则信任颜色（增加小河道连通性？这里暂时保留 OTSU 陆地，避免海洋侵入陆地）
    # 综合后海洋掩膜
    ocean_mask = refined_water.astype(np.uint8) * 255

    # 3.4 形态学去噪与断桥连接
    # 小开运算去除零星噪点
    kernel_noise = np.ones((3, 3), np.uint8)
    ocean_cleaned = cv2.morphologyEx(ocean_mask, cv2.MORPH_OPEN, kernel_noise)

    # 闭运算连接断裂的水域（如被桥梁隔断的航道，先闭后开，有助于打通）
    ocean_closed = cv2.morphologyEx(ocean_cleaned, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # 3.5 连通域分析，保留最大海洋（去除内陆湖、河流）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ocean_closed, connectivity=4)
    if num_labels <= 1:
        print("警告：未检测到任何水域。")
        grid_map = np.ones((h, w), dtype=np.uint8)
        return grid_map, img, (h, w), None, 0, ocean_mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    main_ocean = (labels == largest_label).astype(np.uint8)  # 主海洋为1，其余0

    # 陆地 = 非主海洋
    land_mask_raw = (main_ocean == 0).astype(np.uint8)  # 陆地=1, 海洋=0

    # 3.6 窄桥再处理：对原始陆地掩膜进行适度的开运算，去掉窄桥，防止主海洋被分割
    # 这里使用形态学开运算直接作用于 land_mask_raw，但为了不影响海岸线精度，核不能太大
    kernel_bridge = np.ones((5, 5), np.uint8)
    land_no_bridge = cv2.morphologyEx(land_mask_raw, cv2.MORPH_OPEN, kernel_bridge)
    # 重新生成海洋：非陆地为海洋
    ocean_final = (land_no_bridge == 0).astype(np.uint8)
    # 对新的海洋再做一次连通域保护，保证主海洋仍然最大
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(ocean_final, connectivity=4)
    if num_labels2 > 1:
        areas2 = stats2[1:, cv2.CC_STAT_AREA]
        largest_label2 = np.argmax(areas2) + 1
        main_ocean = (labels2 == largest_label2).astype(np.uint8)

    # 纯海洋掩膜（无安全膨胀）
    pure_ocean_mask = main_ocean

    # 3.7 安全距离膨胀：对陆地进行膨胀，形成防撞墙
    # 陆地 = 非主海洋
    land_for_inflation = (main_ocean == 0).astype(np.uint8)
    kernel_safety = np.ones((SAFETY_RADIUS_PX, SAFETY_RADIUS_PX), np.uint8)
    land_inflated = cv2.dilate(land_for_inflation, kernel_safety, iterations=1)

    # 可通行区域 = 非膨胀陆地
    grid_map = (land_inflated > 0).astype(np.uint8)

    return grid_map, img, (h, w), labels2, largest_label2, pure_ocean_mask

# ==========================================
# 4. 坐标映射与吸附（沿用 V1 逻辑）
# ==========================================
def lonlat_to_pixel(lon, lat, img_shape):
    h, w = img_shape
    min_lon, max_lat = MAP_TOP_LEFT
    max_lon, min_lat = MAP_BOTTOM_RIGHT
    x = int((lon - min_lon) / (max_lon - min_lon) * w)
    y = int((max_lat - lat) / (max_lat - min_lat) * h)
    return (max(0, min(x, w - 1)), max(0, min(y, h - 1)))

def find_nearest_water_in_ocean(grid, start_px, ocean_mask):
    """BFS 吸附到纯海洋中的可通行水域"""
    h, w = grid.shape
    sx, sy = start_px
    if grid[sy, sx] == 0 and ocean_mask[sy, sx] == 1:
        return start_px
    q = deque([(sx, sy)])
    visited = set([(sx, sy)])
    dirs = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    while q:
        cx, cy = q.popleft()
        if grid[cy, cx] == 0 and ocean_mask[cy, cx] == 1:
            return (cx, cy)
        for dx, dy in dirs:
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append((nx, ny))
    return start_px

# ==========================================
# 5. A* 寻路
# ==========================================
def heuristic(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def a_star(grid, start, goal):
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
# 6. 主程序与可视化
# ==========================================
def main():
    print("1. 地图获取与融合处理...")
    img = fetch_map_with_cache(START_LONLAT, GOAL_LONLAT, MAP_ZOOM_LEVEL)
    grid_map, original_img, img_shape, labels, ocean_label, pure_ocean = process_map_fusion(img)

    start_px = lonlat_to_pixel(START_LONLAT[0], START_LONLAT[1], img_shape)
    goal_px  = lonlat_to_pixel(GOAL_LONLAT[0], GOAL_LONLAT[1], img_shape)
    print(f"2. 起点像素: {start_px}, 终点像素: {goal_px}")

    # 吸附至主海洋
    start_px = find_nearest_water_in_ocean(grid_map, start_px, pure_ocean)
    goal_px  = find_nearest_water_in_ocean(grid_map, goal_px, pure_ocean)

    print("3. A* 寻路...")
    t0 = time.time()
    path_px = a_star(grid_map, start_px, goal_px)
    print(f"   耗时 {time.time()-t0:.2f}s")

    if not path_px:
        print("❌ 未找到路径，请检查起终点是否在可航行水域。")
        plt.imshow(1 - grid_map, cmap='gray')
        plt.plot(start_px[0], start_px[1], 'go', markersize=10)
        plt.plot(goal_px[0], goal_px[1], 'ro', markersize=10)
        plt.title("路径规划失败")
        plt.axis('off')
        plt.show()
        return

    print(f"4. 路径点数 {len(path_px)}，B 样条平滑...")
    x = [p[0] for p in path_px]
    y = [p[1] for p in path_px]
    tck, u = splprep([x, y], s=len(path_px)*2.0)
    u_new = np.linspace(u.min(), u.max(), max(300, len(path_px)))
    x_smooth, y_smooth = splev(u_new, tck)

    # --- 可视化 1：栅格地图（白水黑陆 + 路径）---
    plt.figure("Figure 1 - 融合栅格地图", figsize=(8,8))
    # 显示最终栅格（白=可通行，黑=障碍）
    plt.imshow(1 - grid_map, cmap='gray')
    plt.plot(start_px[0], start_px[1], 'go', markersize=10, label='起点')
    plt.plot(goal_px[0], goal_px[1], 'ro', markersize=10, label='终点')
    plt.title("融合栅格地图（白=水，黑=陆+安全距离）")
    plt.legend()
    plt.axis('off')

    # --- 可视化 2：原图 + 路径动画 ---
    fig2, ax2 = plt.subplots(num="Figure 2 - 动态航行", figsize=(10,8))
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    ax2.imshow(img_rgb)
    ax2.plot(start_px[0], start_px[1], 'go', markersize=10, label='起点')
    ax2.plot(goal_px[0], goal_px[1], 'ro', markersize=10, label='终点')
    ax2.plot(x, y, 'y--', alpha=0.5, label='A* 离散路径')
    ax2.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='平滑轨迹')

    ship, = ax2.plot([], [], 'r^', markersize=12, markeredgecolor='black', label='船舶')

    def init():
        ship.set_data([], [])
        return ship,

    def animate(i):
        ship.set_data([x_smooth[i]], [y_smooth[i]])
        return ship,

    ani = animation.FuncAnimation(fig2, animate, init_func=init,
                                  frames=len(x_smooth), interval=30, blit=True)
    ax2.set_title("桃花岛 → 普陀山 融合规划航行")
    ax2.legend(loc='upper right')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
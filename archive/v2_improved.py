import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import splprep, splev
import heapq
import math
import urllib.request
import ssl
import time  # 新增：用于网络重试延时

# ==========================================
# 1. 参数配置区 (全自动版，仅需输入经纬度)
# ==========================================
# 任务起终点
START_LONLAT = (122.295, 29.848)  # 桃花岛
GOAL_LONLAT = (122.387, 29.9795)  # 普陀山

# 膨胀安全半径 (像素)，适度降低以防港口被封死
SAFETY_RADIUS_PX = 8
MAP_ZOOM_LEVEL = 13  # 在线地图的自动缩放级别，13 对于这十几公里的距离非常合适


# ==========================================
# 2. 自动化在线地图获取与栅格化
# ==========================================
def deg2num(lat_deg, lon_deg, zoom):
    """经纬度转瓦片行列号"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile, ytile, zoom):
    """瓦片行列号转左上角经纬度"""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def fetch_online_map(start_lonlat, goal_lonlat, zoom):
    """全自动计算边界并从 CartoDB 下载无注记瓦片地图进行拼接"""
    # 增加 0.05 度的外扩视野，确保起终点不会贴在图片边缘
    min_lon = min(start_lonlat[0], goal_lonlat[0]) - 0.05
    max_lon = max(start_lonlat[0], goal_lonlat[0]) + 0.05
    min_lat = min(start_lonlat[1], goal_lonlat[1]) - 0.05
    max_lat = max(start_lonlat[1], goal_lonlat[1]) + 0.05

    x_min, y_max = deg2num(min_lat, min_lon, zoom)
    x_max, y_min = deg2num(max_lat, max_lon, zoom)

    # 动态记录并暴露精确的地图经纬度边界，供后续坐标映射使用！(完全替代人工测量)
    global MAP_TOP_LEFT, MAP_BOTTOM_RIGHT
    MAP_TOP_LEFT = num2deg(x_min, y_min, zoom)  # 返回(lat, lon)
    MAP_TOP_LEFT = (MAP_TOP_LEFT[1], MAP_TOP_LEFT[0])  # 统一转为 (lon, lat)
    MAP_BOTTOM_RIGHT = num2deg(x_max + 1, y_max + 1, zoom)
    MAP_BOTTOM_RIGHT = (MAP_BOTTOM_RIGHT[1], MAP_BOTTOM_RIGHT[0])

    # 忽略 SSL 证书验证以适配部分本地网络环境
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    row_images = []
    # 使用 CartoDB Voyager 的纯净底图服务 (免 API KEY，无烦人的地名文字干扰)
    base_url = "https://basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png"

    for y in range(y_min, y_max + 1):
        col_images = []
        for x in range(x_min, x_max + 1):
            url = base_url.format(z=zoom, x=x, y=y)

            # --- 修改部分：增加带超时的网络重试机制 ---
            max_retries = 3
            tile = None
            for attempt in range(max_retries):
                try:
                    req = urllib.request.Request(url,
                                                 headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
                    # 增加 timeout 防止死等，引发 IncompleteRead
                    resp = urllib.request.urlopen(req, context=ctx, timeout=10)
                    img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                    tile = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if tile is not None:
                        break  # 下载成功，跳出重试循环
                except Exception as e:
                    print(f"  > 瓦片 ({x},{y}) 下载异常: {e}，正在重试 {attempt + 1}/{max_retries}...")
                    time.sleep(1.5)  # 稍微等待后重试

            if tile is None:
                print(f"警告: 瓦片 ({x},{y}) 彻底加载失败，将以浅灰色占位以免产生错误边界")
                # 用接近水域的浅灰色占位，避免产生强烈的边缘被当做陆地
                tile = np.ones((256, 256, 3), dtype=np.uint8) * 170

            col_images.append(tile)
            # --- 修改结束 ---

        row_images.append(np.concatenate(col_images, axis=1))

    return np.concatenate(row_images, axis=0)


def process_map_to_grid(start_lonlat, goal_lonlat):
    img = fetch_online_map(start_lonlat, goal_lonlat, MAP_ZOOM_LEVEL)
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 采用 Canny 提取在线地图中所有岛屿和陆地的绝对轮廓
    edges = cv2.Canny(gray, 30, 100)

    # 闭运算连接断裂的边缘
    close_kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # 膨胀形成实体防撞墙 (起终点都在墙外的海里，A*自然会被圈在海里寻路，完美避障)
    dilate_kernel = np.ones((SAFETY_RADIUS_PX, SAFETY_RADIUS_PX), np.uint8)
    dilated_grid = cv2.dilate(closed_edges, dilate_kernel, iterations=1)

    # 0 为可通行(海洋)，1 为障碍(陆地及缓冲区)
    grid_map = (dilated_grid > 0).astype(np.uint8)
    return grid_map, img, (h, w)


# ==========================================
# 3. 坐标映射工具
# ==========================================
def lonlat_to_pixel(lon, lat, img_shape):
    h, w = img_shape
    min_lon, max_lat = MAP_TOP_LEFT
    max_lon, min_lat = MAP_BOTTOM_RIGHT

    # 线性插值
    x = int((lon - min_lon) / (max_lon - min_lon) * w)
    y = int((max_lat - lat) / (max_lat - min_lat) * h)

    # 边界限制
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    return (x, y)  # (列索引, 行索引)


# ==========================================
# 4. A* 路径规划
# ==========================================
def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def a_star(grid, start, goal):
    h, w = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}

    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    # 8 个移动方向
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # 反转，起点到终点

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            nx, ny = neighbor

            # 越界检查与障碍物检查
            if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0:
                # 对角线代价为 1.414，直线为 1
                cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g = g_score[current] + cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # 无法到达


# ==========================================
# 5. 主流程与动态可视化
# ==========================================
def main():
    print("1. 正在全自动下载在线地图并提取陆地边缘(请保持网络畅通)...")
    grid_map, original_img, img_shape = process_map_to_grid(START_LONLAT, GOAL_LONLAT)

    print("2. 正在进行坐标映射...")
    start_px = lonlat_to_pixel(START_LONLAT[0], START_LONLAT[1], img_shape)
    goal_px = lonlat_to_pixel(GOAL_LONLAT[0], GOAL_LONLAT[1], img_shape)

    print(f"像素起点: {start_px}, 像素终点: {goal_px}")

    # 检查起终点是否在障碍物内
    if grid_map[start_px[1], start_px[0]] == 1 or grid_map[goal_px[1], goal_px[0]] == 1:
        print("警告：起点或终点位于解析出的陆地/障碍物区域内！正在强制开辟出港航道...")
        # 强制清除起终点周围更大的障碍物以确保算法能顺利驶向深海
        cv2.circle(grid_map, start_px, SAFETY_RADIUS_PX + 5, 0, -1)
        cv2.circle(grid_map, goal_px, SAFETY_RADIUS_PX + 5, 0, -1)

    print("3. 开始 A* 路径搜索 (由于栅格精度高，可能需要几秒钟)...")
    path_px = a_star(grid_map, start_px, goal_px)

    if not path_px:
        print("寻路失败：无法找到有效路径。")
        return

    print(f"4. 寻路成功，找到离散点 {len(path_px)} 个。正在进行轨迹平滑...")
    x = [p[0] for p in path_px]
    y = [p[1] for p in path_px]

    # 使用 B 样条曲线进行平滑 (s 为平滑系数，需根据点数微调)
    tck, u = splprep([x, y], s=len(path_px) * 2.0)
    u_new = np.linspace(u.min(), u.max(), max(300, len(path_px)))
    x_smooth, y_smooth = splev(u_new, tck)

    print("5. 正在生成可视化结果...")
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制基础背景 (显示提取的障碍物栅格而非原图，更符合大作业严谨性)
    # cmap='Blues' 使得水域蓝色，障碍物深色
    ax.imshow(grid_map, cmap='Blues', alpha=0.6)

    # 绘制起点和终点
    ax.plot(start_px[0], start_px[1], 'go', markersize=10, label='Start (Taohua)')
    ax.plot(goal_px[0], goal_px[1], 'ro', markersize=10, label='Goal (Putuo)')

    # 绘制原始 A* 路径（虚线）和平滑后路径（实线）
    ax.plot(x, y, 'y--', alpha=0.5, label='A* Discrete Path')
    ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Smoothed Ship Trajectory')

    # 初始化动画船舶点
    ship_dot, = ax.plot([], [], 'b^', markersize=12, label='Ship Position')

    def init():
        ship_dot.set_data([], [])
        return ship_dot,

    def animate(i):
        # 更新船舶位置
        ship_dot.set_data([x_smooth[i]], [y_smooth[i]])
        return ship_dot,

    # 设置标题与图例
    ax.set_title("Ship Path Planning with Edge Obstacle Avoidance")
    ax.legend(loc='upper right')
    ax.axis('off')  # 隐藏坐标轴刻度

    # 创建动画
    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(x_smooth), interval=30, blit=True
    )

    # 可选：保存为 gif
    # print("正在保存动画至 ship_trajectory.gif...")
    # ani.save('ship_trajectory.gif', writer='pillow', fps=30)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
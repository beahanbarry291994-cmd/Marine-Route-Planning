"""
智能航海路径规划 — 最终版
融合 V1（OTSU+连通域栅格化）与 V3（双图合一可视化、动画播放）
地图源: CartoDB Voyager 无标签底图
算法:   A* + B-spline 轨迹平滑
"""

import os, re, json, math, time, ssl, sys
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

# ──────────────────────── 参数 ────────────────────────
START_LONLAT = (122.295, 29.848)    # 桃花岛
GOAL_LONLAT  = (122.387, 29.9795)   # 普陀山

ZOOM         = 13
SAFETY_PX    = 3                    # 安全膨胀半径 (像素)
CACHE_IMG    = 'map_cache.png'
CACHE_META   = 'map_meta.json'

# ──────────────────────── 船舶参数 (Mariner) ────────────────────────
SHIP_L       = 160.0                # 船长 (m)
SHIP_B       = 21.0                 # 船宽 (m)
SHIP_K       = 0.5                  # 旋回性指数
SHIP_T       = 1.0                  # 追随性指数
SHIP_A       = 0.4                  # 非线性系数
SHIP_U0      = 10.0 * 0.5144        # 初始航速 (m/s, 10节)
MAX_RUDDER   = np.radians(35)       # 最大舵角 (rad)
RUDDER_RATE  = np.radians(2.5)      # 舵机速率 (rad/s)

# PID 控制参数
PID_KP       = 1.0
PID_KI       = 0.01
PID_KD       = 5.0

# 风流干扰参数
V_WIND       = 5.0                  # 风速 (m/s)
PSI_WIND     = np.pi / 4            # 风向 (rad, 东北风)
V_CURRENT    = 0.5                  # 流速 (m/s, 约1节)
PSI_CURRENT  = 0.0                  # 流向 (rad, 正北)

# 仿真参数
SIM_DT       = 0.1                  # 仿真步长 (s)
WP_SWITCH_M  = 200.0                # 航路点切换半径 (m)
WP_INTERVAL  = 500.0                # 航路点间距 (m)


# ──────────────────────── 辅助函数 ────────────────────────
def normalize_angle(a):
    """归一化角度到 [-pi, pi]"""
    return (a + np.pi) % (2 * np.pi) - np.pi

def get_m_per_px(lat, zoom):
    """计算给定纬度和缩放级别下每像素代表的米数"""
    return 156543.03 * np.cos(np.radians(lat)) / (2 ** zoom)

M_PER_PX = get_m_per_px(29.9, ZOOM)  # 约19 m/px at zoom=13, lat=29.9°N

def px_to_m(px):
    return px * M_PER_PX

def m_to_px(m):
    return m / M_PER_PX


# ──────────────────────── 地图下载 ────────────────────────
def deg2num(lat, lon, z):
    n = 2.0 ** z
    return int((lon + 180) / 360 * n), int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)

def num2deg(x, y, z):
    n = 2.0 ** z
    return (math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n)))),
            x / n * 360 - 180)

def fetch_map(start, goal, zoom):
    global MAP_TL, MAP_BR
    if os.path.exists(CACHE_IMG) and os.path.exists(CACHE_META):
        meta = json.load(open(CACHE_META))
        MAP_TL, MAP_BR = tuple(meta['top_left']), tuple(meta['bottom_right'])
        img = cv2.imread(CACHE_IMG)
        if img is not None and np.std(img) > 10:
            print("[地图] 使用本地缓存")
            return img

    print("[地图] 下载 CartoDB 底图...")
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
    cv2.imwrite(CACHE_IMG, img)
    json.dump({'top_left': MAP_TL, 'bottom_right': MAP_BR}, open(CACHE_META, 'w'))
    return img


# ──────────────────────── 栅格化 ────────────────────────
def process_map(img):
    """
    OTSU 二值化 → 开运算去桥 → 连通域去假湖 → 小半径安全膨胀
    返回: grid_map (0=可通行, 1=障碍), main_ocean (主海洋掩膜)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OTSU 自动二值化（水域暗，陆地亮）
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    land_mask = cv2.bitwise_not(thresh)  # 陆地=255，水域=0

    # 陆地过多时回退到中位数阈值
    land_ratio = np.sum(land_mask == 255) / land_mask.size
    if land_ratio > 0.90:
        median_val = np.median(gray)
        _, land_mask = cv2.threshold(gray, median_val, 255, cv2.THRESH_BINARY)

    # 开运算去桥（3×3）
    land_clean = cv2.morphologyEx(land_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # 连通域: 保留最大水体作为主海洋
    water = (land_clean == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(water, connectivity=4)
    if n <= 1:
        return np.ones_like(land_clean, dtype=np.uint8), np.zeros_like(land_clean)

    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    main_ocean = (labels == largest_label).astype(np.uint8)

    # 最终陆地 = 原陆地 或 非主海洋区域
    land_final = ((land_clean == 255) | (main_ocean == 0)).astype(np.uint8)

    # 小半径安全膨胀
    land_inflated = cv2.dilate(land_final, np.ones((SAFETY_PX, SAFETY_PX), np.uint8))
    return land_inflated, main_ocean


# ──────────────────────── 坐标工具 ────────────────────────
def lonlat_to_px(lon, lat, shape):
    h, w = shape
    x = int((lon - MAP_TL[0]) / (MAP_BR[0] - MAP_TL[0]) * w)
    y = int((MAP_TL[1] - lat) / (MAP_TL[1] - MAP_BR[1]) * h)
    return max(0, min(x, w - 1)), max(0, min(y, h - 1))

def snap_to_ocean(grid, px, ocean):
    """BFS 吸附: 若 px 不在主海洋，搜索最近的海洋像素"""
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


# ──────────────────────── A* 寻路 ────────────────────────
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


# ──────────────────────── 平滑 ────────────────────────
def smooth_path(path, n_pts=300):
    x, y = zip(*path)
    tck, _ = splprep([x, y], s=len(path) * 2.0)
    u = np.linspace(0, 1, max(n_pts, len(path)))
    return splev(u, tck)


# ──────────────────────── 航路点提取 ────────────────────────
def extract_waypoints(path, interval_m=WP_INTERVAL):
    """从A*路径提取等间距航路点（像素坐标）"""
    xs, ys = smooth_path(path, n_pts=max(500, len(path) * 3))
    # 计算累积弧长
    dxs = np.diff(xs)
    dys = np.diff(ys)
    seg_len = np.sqrt(dxs**2 + dys**2)
    cum_len = np.concatenate([[0], np.cumsum(seg_len)])
    total_len = cum_len[-1]
    # 等间距采样
    interval_px = m_to_px(interval_m)
    n_wp = max(2, int(total_len / interval_px) + 1)
    sample_dist = np.linspace(0, total_len, n_wp)
    wp_x = np.interp(sample_dist, cum_len, xs)
    wp_y = np.interp(sample_dist, cum_len, ys)
    waypoints = list(zip(wp_x, wp_y))
    print(f"[航路点] 提取 {len(waypoints)} 个航路点, 间距≈{interval_m}m")
    return waypoints


# ──────────────────────── KT 模型仿真 ────────────────────────
def kt_simulate(waypoints, grid, start, goal):
    """
    KT模型 + PID航向控制 + 风流干扰
    模拟船舶逐点跟踪航路点
    """
    from scipy.integrate import odeint

    # 转换航路点为米制坐标（以起点为原点）
    wp_m = [(px_to_m(wp[0] - start[0]), px_to_m(wp[1] - start[1])) for wp in waypoints]
    # 起点设为 (0,0)
    wp_m[0] = (0.0, 0.0)

    # 初始状态: [v, r, psi, x, y, u]
    psi0 = np.arctan2(wp_m[1][1] - wp_m[0][1], wp_m[1][0] - wp_m[0][0])
    state = [0.0, 0.0, psi0, 0.0, 0.0, SHIP_U0]

    # 记录容器
    log_t, log_x, log_y, log_psi, log_u, log_delta, log_v = [], [], [], [], [], [], []
    log_psi_desired = []

    wp_idx = 1  # 当前目标航路点
    t = 0.0
    integral = 0.0
    prev_error = 0.0
    delta = 0.0  # 当前舵角

    max_steps = int(50000 / SIM_DT)  # 防止无限循环
    step = 0

    while wp_idx < len(wp_m) and step < max_steps:
        v, r, psi, x, y, u = state

        # 期望航向
        dx_wp = wp_m[wp_idx][0] - x
        dy_wp = wp_m[wp_idx][1] - y
        psi_desired = np.arctan2(dy_wp, dx_wp)

        # PID 航向控制
        error = normalize_angle(psi_desired - psi)
        integral += error * SIM_DT
        # 积分限幅防饱和
        integral = np.clip(integral, -np.pi, np.pi)
        derivative = (error - prev_error) / SIM_DT
        delta_cmd = PID_KP * error + PID_KI * integral + PID_KD * derivative
        delta_cmd = np.clip(delta_cmd, -MAX_RUDDER, MAX_RUDDER)
        prev_error = error

        # KT 运动方程（含风流干扰）
        def kt_eq(state, t_local, delta_cmd):
            v, r, psi, x, y, u = state

            # 舵角限幅
            delta_use = np.clip(delta_cmd, -MAX_RUDDER, MAX_RUDDER)

            # KT 方程
            drdt = (SHIP_K * delta_use - (1 + SHIP_A * abs(r)) * r) / SHIP_T

            # 航速损失
            dudt = -0.02 * u * abs(r) - 0.005 * u * delta_use**2

            # 横向速度
            dvdt = 0.15 * r * u - 0.25 * v

            # 风流干扰力
            # 风力（简化侧向力）
            Fy_wind = 0.5 * 1.225 * 0.8 * (SHIP_B * 5.0) * V_WIND**2 * np.sin(PSI_WIND - psi)
            # 流效应（速度叠加）
            u_eff = u + V_CURRENT * np.cos(PSI_CURRENT - psi)
            v_eff = v + V_CURRENT * np.sin(PSI_CURRENT - psi)

            # 风力修正横向速度
            dvdt += Fy_wind / (SHIP_L * SHIP_B * 8.0 * 1.025)  # 简化质量

            # 位置变化（含流）
            dpsidt = r
            dxdt = u_eff * np.cos(psi) - v_eff * np.sin(psi)
            dydt = u_eff * np.sin(psi) + v_eff * np.cos(psi)

            return [dvdt, drdt, dpsidt, dxdt, dydt, dudt]

        # 积分一步
        new_state = odeint(kt_eq, state, [0, SIM_DT], args=(delta_cmd,))[-1]
        state = new_state
        v, r, psi, x, y, u = state
        t += SIM_DT
        step += 1

        # 记录
        log_t.append(t)
        log_x.append(x)
        log_y.append(y)
        log_psi.append(np.degrees(psi))
        log_psi_desired.append(np.degrees(psi_desired))
        log_u.append(u)
        log_delta.append(np.degrees(delta_cmd))
        log_v.append(v)

        # 航路点切换
        dist_to_wp = np.sqrt((x - wp_m[wp_idx][0])**2 + (y - wp_m[wp_idx][1])**2)
        if dist_to_wp < WP_SWITCH_M:
            wp_idx += 1

        # 安全检查：仿真超时
        if t > 10000:
            print(f"[KT仿真] 警告：仿真超时 ({t:.0f}s)，提前终止")
            break

    # 转换回像素坐标（以起点为原点）
    log_x_px = [m_to_px(xm) + start[0] for xm in log_x]
    log_y_px = [m_to_px(ym) + start[1] for ym in log_y]

    result = {
        't': np.array(log_t),
        'x_px': np.array(log_x_px),
        'y_px': np.array(log_y_px),
        'x_m': np.array(log_x),
        'y_m': np.array(log_y),
        'psi': np.array(log_psi),
        'psi_desired': np.array(log_psi_desired),
        'u': np.array(log_u),
        'delta': np.array(log_delta),
        'v': np.array(log_v),
        'waypoints': waypoints,
        'wp_m': wp_m,
        'n_wp_reached': wp_idx,
        'total_wp': len(waypoints),
    }

    # 计算跟踪误差
    final_x, final_y = log_x[-1], log_y[-1]
    goal_m = (px_to_m(goal[0] - start[0]), px_to_m(goal[1] - start[1]))
    end_error = np.sqrt((final_x - goal_m[0])**2 + (final_y - goal_m[1])**2)

    print(f"[KT仿真] 完成 {step} 步, 耗时 {t:.1f}s")
    print(f"[KT仿真] 到达航路点 {wp_idx}/{len(waypoints)}")
    print(f"[KT仿真] 终点偏移 {end_error:.1f}m")
    print(f"[KT仿真] 舵角范围 [{min(log_delta):.1f}°, {max(log_delta):.1f}°]")

    return result


# ──────────────────────── 可视化 ────────────────────────
def next_filename(prefix, ext):
    mx = 0
    for f in os.listdir('.'):
        m = re.match(rf'{re.escape(prefix)}(\d+)\.{re.escape(ext)}$', f)
        if m:
            mx = max(mx, int(m.group(1)))
    return f'{prefix}{mx + 1}.{ext}'


def visualize(grid, img, start, goal, path_x, path_y, xs, ys):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── 1) 保存静态结果图 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("桃花岛→普陀山 智能航行路径规划",
                 fontsize=15, fontweight='bold', y=0.98)

    ax1.imshow(1 - grid, cmap='gray')
    ax1.plot(*start, 'go', ms=10, label='起点')
    ax1.plot(*goal,  'ro', ms=10, label='终点')
    ax1.set_title("栅格地图（白=水域，黑=陆地+安全距离）", fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.axis('off')

    ax2.imshow(img_rgb)
    ax2.plot(*start, 'go', ms=10, label='起点')
    ax2.plot(*goal,  'ro', ms=10, label='终点')
    ax2.plot(path_x, path_y, 'y--', alpha=0.5, lw=1, label='A* 路径')
    ax2.plot(xs, ys, 'r-', lw=2, label='平滑轨迹')
    ax2.plot(xs[-1], ys[-1], 'r^', ms=12, mec='black')
    ax2.set_title("A*寻路 + B-spline轨迹平滑", fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.axis('off')

    static_name = next_filename('result', 'png')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(static_name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[保存] 静态图 → {static_name}")

    # ── 2) 保存动态 GIF ──
    fig2, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_rgb)
    ax.plot(*start, 'go', ms=10, label='起点')
    ax.plot(*goal,  'ro', ms=10, label='终点')
    ax.plot(xs, ys, 'r-', lw=2, alpha=0.3, label='平滑轨迹')
    ship, = ax.plot([], [], 'r^', ms=12, mec='black')
    trail, = ax.plot([], [], 'r-', lw=2)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title("桃花岛→普陀山 航行动画", fontsize=13, fontweight='bold')
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
    print(f"[保存] 动态图 → {gif_name}")
    print(f"\n[完成] 请打开 {static_name} 和 {gif_name} 查看结果")


def visualize_enhanced(grid, img, start, goal, path, sim_result):
    """增强可视化：A*路径 + KT仿真轨迹 + 控制参数"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = sim_result

    # ── 图1: 4子图综合图 ──
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("A*路径规划 + KT船舶运动仿真", fontsize=15, fontweight='bold', y=0.98)

    # 左上: 栅格地图 + A*路径
    ax1 = axes[0, 0]
    ax1.imshow(1 - grid, cmap='gray')
    path_x, path_y = zip(*path)
    ax1.plot(path_x, path_y, 'y--', alpha=0.7, lw=1.5, label='A* 路径')
    ax1.plot(*start, 'go', ms=10, label='起点')
    ax1.plot(*goal, 'ro', ms=10, label='终点')
    ax1.set_title("栅格地图 + A*路径", fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.axis('off')

    # 右上: 卫星底图 + KT轨迹
    ax2 = axes[0, 1]
    ax2.imshow(img_rgb)
    ax2.plot(path_x, path_y, 'y--', alpha=0.5, lw=1, label='A* 路径')
    ax2.plot(res['x_px'], res['y_px'], 'b-', lw=2, label='KT仿真轨迹')
    # 航路点
    wp_x, wp_y = zip(*res['waypoints'])
    ax2.plot(wp_x, wp_y, 'g^', ms=8, label='航路点')
    # 航向箭头（每隔N步画一个）
    arrow_step = max(1, len(res['x_px']) // 30)
    for i in range(0, len(res['x_px']), arrow_step):
        psi_rad = np.radians(res['psi'][i])
        dx = 15 * np.cos(psi_rad)
        dy = 15 * np.sin(psi_rad)
        ax2.annotate('', xy=(res['x_px'][i] + dx, res['y_px'][i] + dy),
                     xytext=(res['x_px'][i], res['y_px'][i]),
                     arrowprops=dict(arrowstyle='->', color='cyan', lw=1.5))
    ax2.plot(*start, 'go', ms=10, label='起点')
    ax2.plot(*goal, 'ro', ms=10, label='终点')
    ax2.plot(res['x_px'][-1], res['y_px'][-1], 'b^', ms=12, mec='black', label='仿真终点')
    ax2.set_title("卫星底图 + KT仿真轨迹 + 航向", fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.axis('off')

    # 左下: 舵角时间序列
    ax3 = axes[1, 0]
    ax3.plot(res['t'], res['delta'], 'b-', lw=1.5)
    ax3.axhline(y=np.degrees(MAX_RUDDER), color='r', linestyle='--', alpha=0.5, label='舵角限幅')
    ax3.axhline(y=-np.degrees(MAX_RUDDER), color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('舵角 (°)')
    ax3.set_title('舵角指令 δ(t)', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)

    # 右下: 航向跟踪
    ax4 = axes[1, 1]
    ax4.plot(res['t'], res['psi'], 'r-', lw=1.5, label='实际航向 ψ')
    ax4.plot(res['t'], res['psi_desired'], 'b--', lw=1, alpha=0.7, label='期望航向 ψ_desired')
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('航向 (°)')
    ax4.set_title('航向跟踪', fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname1 = next_filename('result_ship', 'png')
    fig.savefig(fname1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[保存] 船舶运动综合图 → {fname1}")

    # ── 图2: 运动参数详情 ──
    fig2, (ax_a, ax_b, ax_c) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig2.suptitle("船舶运动参数详情", fontsize=14, fontweight='bold')

    # 航向误差
    heading_error = res['psi'] - res['psi_desired']
    heading_error = np.array([normalize_angle(np.radians(e)) for e in heading_error])
    heading_error = np.degrees(heading_error)
    ax_a.plot(res['t'], heading_error, 'r-', lw=1)
    ax_a.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax_a.set_ylabel('航向误差 (°)')
    ax_a.set_title('航向跟踪误差')
    ax_a.grid(True, alpha=0.3)

    # 航速
    ax_b.plot(res['t'], res['u'] / 0.5144, 'g-', lw=1.5)
    ax_b.set_ylabel('航速 (节)')
    ax_b.set_title('航速变化')
    ax_b.grid(True, alpha=0.3)

    # 漂角
    drift_angle = np.degrees(np.arctan2(res['v'], res['u']))
    ax_c.plot(res['t'], drift_angle, 'm-', lw=1)
    ax_c.set_xlabel('时间 (s)')
    ax_c.set_ylabel('漂角 (°)')
    ax_c.set_title('漂角变化')
    ax_c.grid(True, alpha=0.3)

    plt.tight_layout()
    fname2 = next_filename('result_motion_detail', 'png')
    fig2.savefig(fname2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"[保存] 运动参数详情 → {fname2}")


# ──────────────────────── 主流程 ────────────────────────
def main():
    print("=" * 40)
    print("  智能航海路径规划系统")
    print("  桃花岛 → 普陀山")
    print("=" * 40)

    print("\n[1/5] 地图获取...")
    img = fetch_map(START_LONLAT, GOAL_LONLAT, ZOOM)

    print("[2/5] 栅格化 (OTSU + 连通域 + 安全膨胀)...")
    grid, ocean = process_map(img)

    print("[3/5] 坐标映射与吸附...")
    start = snap_to_ocean(grid, lonlat_to_px(*START_LONLAT, grid.shape), ocean)
    goal  = snap_to_ocean(grid, lonlat_to_px(*GOAL_LONLAT,  grid.shape), ocean)

    print("[4/5] A* 寻路...")
    t0 = time.time()
    path = a_star(grid, start, goal)
    if path is None:
        print("[错误] 未找到可行路径")
        input("\n按回车键退出...")
        return
    print(f"       耗时 {time.time() - t0:.2f}s, 路径点 {len(path)}")

    print("[5/5] B-spline 平滑与可视化...")
    xs, ys = smooth_path(path)
    path_x, path_y = zip(*path)
    visualize(grid, img, start, goal, path_x, path_y, xs, ys)

    print("\n" + "=" * 40)
    print("  KT 船舶运动仿真")
    print("=" * 40)

    print("\n[6/6] 航路点提取...")
    waypoints = extract_waypoints(path, interval_m=WP_INTERVAL)

    print("[7/7] KT模型仿真 (PID航向控制 + 风流干扰)...")
    print(f"       风速={V_WIND}m/s(东北), 流速={V_CURRENT}m/s(正北)")
    sim_result = kt_simulate(waypoints, grid, start, goal)

    print("\n[8/8] 增强可视化...")
    visualize_enhanced(grid, img, start, goal, path, sim_result)

    # 打印仿真摘要
    print("\n" + "=" * 40)
    print("  仿真摘要")
    print("=" * 40)
    path_len_m = sum(np.sqrt((px_to_m(path[i+1][0]-path[i][0]))**2 +
                              (px_to_m(path[i+1][1]-path[i][1]))**2)
                     for i in range(len(path)-1))
    print(f"[船舶参数] 船长={SHIP_L}m, K={SHIP_K}, T={SHIP_T}, 航速={SHIP_U0/0.5144:.0f}节")
    print(f"[仿真结果] A*路径长度={path_len_m/1000:.1f}km, 仿真时间={sim_result['t'][-1]:.0f}s")
    print(f"[风流干扰] 风速={V_WIND}m/s({np.degrees(PSI_WIND):.0f}°), 流速={V_CURRENT}m/s({np.degrees(PSI_CURRENT):.0f}°)")
    print(f"[跟踪性能] 到达航路点 {sim_result['n_wp_reached']}/{sim_result['total_wp']}")
    end_err = np.sqrt((sim_result['x_m'][-1] - px_to_m(goal[0]-start[0]))**2 +
                      (sim_result['y_m'][-1] - px_to_m(goal[1]-start[1]))**2)
    print(f"[终点误差] {end_err:.1f}m")
    print(f"[舵角范围] [{min(sim_result['delta']):.1f}°, {max(sim_result['delta']):.1f}°]")


if __name__ == '__main__':
    main()

"""
船舶路径规划 + KT运动仿真 核心算法模块
从 main.py 抽取纯计算逻辑，无 matplotlib 依赖
"""

import os, json, math, time, ssl, base64, io
import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.integrate import odeint
from collections import deque
import heapq
import urllib.request


# ──────────────────────── 辅助函数 ────────────────────────

def normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def get_m_per_px(lat, zoom):
    return 156543.03 * np.cos(np.radians(lat)) / (2 ** zoom)


# ──────────────────────── 瓦片坐标 ────────────────────────

def deg2num(lat, lon, z):
    n = 2.0 ** z
    return int((lon + 180) / 360 * n), int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)

def num2deg(x, y, z):
    n = 2.0 ** z
    return (math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n)))),
            x / n * 360 - 180)


# ──────────────────────── 地图下载 ────────────────────────

def fetch_map(start_lonlat, goal_lonlat, zoom):
    cache_img = 'map_cache.png'
    cache_meta = 'map_meta.json'

    if os.path.exists(cache_img) and os.path.exists(cache_meta):
        meta = json.load(open(cache_meta))
        tl = tuple(meta['top_left'])
        br = tuple(meta['bottom_right'])
        img = cv2.imread(cache_img)
        if img is not None and np.std(img) > 10:
            return img, tl, br

    d = 0.08
    x0, y1 = deg2num(min(start_lonlat[1], goal_lonlat[1]) - d,
                      min(start_lonlat[0], goal_lonlat[0]) - d, zoom)
    x1, y0 = deg2num(max(start_lonlat[1], goal_lonlat[1]) + d,
                      max(start_lonlat[0], goal_lonlat[0]) + d, zoom)

    tl = (num2deg(x0, y0, zoom)[1], num2deg(x0, y0, zoom)[0])
    br = (num2deg(x1 + 1, y1 + 1, zoom)[1], num2deg(x1 + 1, y1 + 1, zoom)[0])

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
    cv2.imwrite(cache_img, img)
    json.dump({'top_left': tl, 'bottom_right': br}, open(cache_meta, 'w'))
    return img, tl, br


# ──────────────────────── 栅格化 ────────────────────────

def process_map(img, safety_px=3):
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
    land_inflated = cv2.dilate(land_final, np.ones((safety_px, safety_px), np.uint8))
    return land_inflated, main_ocean


# ──────────────────────── 坐标工具 ────────────────────────

def lonlat_to_px(lon, lat, shape, map_tl, map_br):
    h, w = shape
    x = int((lon - map_tl[0]) / (map_br[0] - map_tl[0]) * w)
    y = int((map_tl[1] - lat) / (map_tl[1] - map_br[1]) * h)
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
    k = min(3, len(path) - 1)  # 点数不足时降低样条阶数
    tck, _ = splprep([x, y], s=len(path) * 2.0, k=k)
    u = np.linspace(0, 1, max(n_pts, len(path)))
    return splev(u, tck)


# ──────────────────────── 航路点提取 ────────────────────────

def extract_waypoints(path, interval_m, m_per_px):
    xs, ys = smooth_path(path, n_pts=max(500, len(path) * 3))
    dxs = np.diff(xs)
    dys = np.diff(ys)
    seg_len = np.sqrt(dxs**2 + dys**2)
    cum_len = np.concatenate([[0], np.cumsum(seg_len)])
    total_len = cum_len[-1]
    interval_px = interval_m / m_per_px
    n_wp = max(2, int(total_len / interval_px) + 1)
    sample_dist = np.linspace(0, total_len, n_wp)
    wp_x = np.interp(sample_dist, cum_len, xs)
    wp_y = np.interp(sample_dist, cum_len, ys)
    waypoints = list(zip(wp_x.tolist(), wp_y.tolist()))
    return waypoints, [xs.tolist(), ys.tolist()]


# ──────────────────────── KT 模型仿真 ────────────────────────

def kt_simulate(waypoints, grid, start_px, goal_px, params, m_per_px):
    ship_k = params['ship_k']
    ship_t = params['ship_t']
    ship_a = params['ship_a']
    ship_u0 = params['ship_speed_kn'] * 0.5144
    max_rudder = np.radians(params['max_rudder_deg'])
    pid_kp = params['pid_kp']
    pid_ki = params['pid_ki']
    pid_kd = params['pid_kd']
    v_wind = params['v_wind']
    psi_wind = np.radians(params['psi_wind_deg'])
    v_current = params['v_current']
    psi_current = np.radians(params['psi_current_deg'])
    sim_dt = params['sim_dt']
    wp_switch_m = params['wp_switch_m']
    ship_l = params.get('ship_l', 160.0)
    ship_b = params.get('ship_b', 21.0)

    def px_to_m(px_val):
        return px_val * m_per_px

    wp_m = [(px_to_m(wp[0] - start_px[0]), px_to_m(wp[1] - start_px[1])) for wp in waypoints]
    wp_m[0] = (0.0, 0.0)

    psi0 = np.arctan2(wp_m[1][1] - wp_m[0][1], wp_m[1][0] - wp_m[0][0])
    state = [0.0, 0.0, psi0, 0.0, 0.0, ship_u0]

    log_t, log_x, log_y, log_psi, log_u, log_delta, log_v = [], [], [], [], [], [], []
    log_psi_desired = []

    wp_idx = 1
    t = 0.0
    integral = 0.0
    prev_error = 0.0
    max_steps = int(50000 / sim_dt)
    step = 0

    while wp_idx < len(wp_m) and step < max_steps:
        v, r, psi, x, y, u = state

        dx_wp = wp_m[wp_idx][0] - x
        dy_wp = wp_m[wp_idx][1] - y
        psi_desired = np.arctan2(dy_wp, dx_wp)

        error = normalize_angle(psi_desired - psi)
        integral += error * sim_dt
        integral = np.clip(integral, -np.pi, np.pi)
        derivative = (error - prev_error) / sim_dt
        delta_cmd = pid_kp * error + pid_ki * integral + pid_kd * derivative
        delta_cmd = np.clip(delta_cmd, -max_rudder, max_rudder)
        prev_error = error

        def kt_eq(state, t_local, delta_cmd):
            v, r, psi, x, y, u = state
            delta_use = np.clip(delta_cmd, -max_rudder, max_rudder)
            drdt = (ship_k * delta_use - (1 + ship_a * abs(r)) * r) / ship_t
            dudt = -0.02 * u * abs(r) - 0.005 * u * delta_use**2
            dvdt = 0.15 * r * u - 0.25 * v
            Fy_wind = 0.5 * 1.225 * 0.8 * (ship_b * 5.0) * v_wind**2 * np.sin(psi_wind - psi)
            u_eff = u + v_current * np.cos(psi_current - psi)
            v_eff = v + v_current * np.sin(psi_current - psi)
            dvdt += Fy_wind / (ship_l * ship_b * 8.0 * 1.025)
            dpsidt = r
            dxdt = u_eff * np.cos(psi) - v_eff * np.sin(psi)
            dydt = u_eff * np.sin(psi) + v_eff * np.cos(psi)
            return [dvdt, drdt, dpsidt, dxdt, dydt, dudt]

        new_state = odeint(kt_eq, state, [0, sim_dt], args=(delta_cmd,))[-1]
        state = new_state
        v, r, psi, x, y, u = state
        t += sim_dt
        step += 1

        log_t.append(t)
        log_x.append(x)
        log_y.append(y)
        log_psi.append(np.degrees(psi))
        log_psi_desired.append(np.degrees(psi_desired))
        log_u.append(u)
        log_delta.append(np.degrees(delta_cmd))
        log_v.append(v)

        dist_to_wp = np.sqrt((x - wp_m[wp_idx][0])**2 + (y - wp_m[wp_idx][1])**2)
        if dist_to_wp < wp_switch_m:
            wp_idx += 1

        if t > 10000:
            break

    def m_to_px_val(m_val):
        return m_val / m_per_px

    log_x_px = [m_to_px_val(xm) + start_px[0] for xm in log_x]
    log_y_px = [m_to_px_val(ym) + start_px[1] for ym in log_y]

    heading_error = [normalize_angle(np.radians(p - d)) for p, d in zip(log_psi, log_psi_desired)]
    heading_error_deg = [np.degrees(e) for e in heading_error]
    drift_angle = [np.degrees(np.arctan2(v, u)) if u > 0.1 else 0.0
                   for v, u in zip(log_v, log_u)]

    goal_m = (px_to_m(goal_px[0] - start_px[0]), px_to_m(goal_px[1] - start_px[1]))
    end_error = np.sqrt((log_x[-1] - goal_m[0])**2 + (log_y[-1] - goal_m[1])**2)

    path_len_m = 0.0
    for i in range(len(log_x) - 1):
        path_len_m += np.sqrt((log_x[i+1] - log_x[i])**2 + (log_y[i+1] - log_y[i])**2)

    return {
        'time': log_t,
        'x_px': log_x_px,
        'y_px': log_y_px,
        'x_m': log_x,
        'y_m': log_y,
        'rudder': log_delta,
        'heading': log_psi,
        'heading_desired': log_psi_desired,
        'speed_kn': [u / 0.5144 for u in log_u],
        'drift_angle': drift_angle,
        'heading_error': heading_error_deg,
        'metrics': {
            'path_length_km': path_len_m / 1000.0,
            'sim_time_s': t,
            'wp_reached': wp_idx,
            'wp_total': len(waypoints),
            'end_offset_m': float(end_error),
            'rudder_range': [float(min(log_delta)), float(max(log_delta))],
            'max_heading_error': float(max(abs(e) for e in heading_error_deg)),
        },
    }


# ──────────────────────── 降采样 ────────────────────────

def downsample(arr, max_pts=500):
    """将数组降采样到最多 max_pts 个点，保留首尾"""
    if len(arr) <= max_pts:
        return arr
    indices = np.linspace(0, len(arr) - 1, max_pts, dtype=int)
    return [arr[i] for i in indices]


# ──────────────────────── 图像编码 ────────────────────────

def img_to_base64(img_array):
    """将 numpy 图像编码为 base64 PNG"""
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
    else:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode('.png', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode('utf-8')


# ──────────────────────── 主流程 ────────────────────────

def run_pipeline(params):
    """完整流水线：地图获取 → 栅格化 → A* → 航路点 → KT仿真"""
    try:
        start_lonlat = (params['start_lon'], params['start_lat'])
        goal_lonlat = (params['goal_lon'], params['goal_lat'])
        zoom = params.get('zoom', 13)
        safety_px = params.get('safety_px', 3)

        # 1. 地图获取
        img, map_tl, map_br = fetch_map(start_lonlat, goal_lonlat, zoom)

        # 控制图像最大尺寸（降低前端数据传输量），缩放后栅格图与卫星图尺寸一致
        max_dim = 1000
        h_img, w_img = img.shape[:2]
        if max(h_img, w_img) > max_dim:
            scale = max_dim / max(h_img, w_img)
            img = cv2.resize(img, (int(w_img * scale), int(h_img * scale)))

        # 2. 栅格化
        grid, ocean = process_map(img, safety_px)

        # 3. 坐标映射
        m_per_px = get_m_per_px(29.9, zoom)

        start_px = snap_to_ocean(grid,
            lonlat_to_px(*start_lonlat, grid.shape, map_tl, map_br), ocean)
        goal_px = snap_to_ocean(grid,
            lonlat_to_px(*goal_lonlat, grid.shape, map_tl, map_br), ocean)

        # 4. A* 寻路
        path = a_star(grid, start_px, goal_px)
        if path is None:
            return {'status': 'error', 'message': '未找到可行路径'}

        # 5. 航路点提取
        wp_interval = params.get('wp_interval_m', 500.0)
        waypoints, smooth_xy = extract_waypoints(path, wp_interval, m_per_px)

        # 6. KT 仿真
        sim_params = {
            'ship_k': params.get('ship_k', 0.5),
            'ship_t': params.get('ship_t', 1.0),
            'ship_a': params.get('ship_a', 0.4),
            'ship_l': params.get('ship_l', 160.0),
            'ship_b': params.get('ship_b', 21.0),
            'ship_speed_kn': params.get('ship_speed_kn', 10.0),
            'max_rudder_deg': params.get('max_rudder_deg', 35.0),
            'pid_kp': params.get('pid_kp', 1.0),
            'pid_ki': params.get('pid_ki', 0.01),
            'pid_kd': params.get('pid_kd', 5.0),
            'v_wind': params.get('v_wind', 5.0),
            'psi_wind_deg': params.get('psi_wind_deg', 45.0),
            'v_current': params.get('v_current', 0.5),
            'psi_current_deg': params.get('psi_current_deg', 0.0),
            'sim_dt': params.get('sim_dt', 0.1),
            'wp_switch_m': params.get('wp_switch_m', 200.0),
        }
        sim_result = kt_simulate(waypoints, grid, start_px, goal_px, sim_params, m_per_px)

        # 7. 编码图像（卫星图与栅格图尺寸一致）
        grid_vis = ((1 - grid) * 255).astype(np.uint8)
        grid_b64 = img_to_base64(grid_vis)
        map_b64 = img_to_base64(img)

        # 8. 构建返回（降采样减少数据量）
        # A* 路径降采样
        path_ds = downsample([[p[0], p[1]] for p in path], 300)

        # KT 轨迹降采样
        kt_traj = list(zip(sim_result['x_px'], sim_result['y_px']))
        kt_traj_ds = downsample(kt_traj, 500)

        # 时间序列降采样
        n_pts = 500
        time_ds = downsample(sim_result['time'], n_pts)
        rudder_ds = downsample(sim_result['rudder'], n_pts)
        heading_ds = downsample(sim_result['heading'], n_pts)
        heading_desired_ds = downsample(sim_result['heading_desired'], n_pts)
        speed_ds = downsample(sim_result['speed_kn'], n_pts)
        drift_ds = downsample(sim_result['drift_angle'], n_pts)
        heading_err_ds = downsample(sim_result['heading_error'], n_pts)

        wp_pixels = [[w[0], w[1]] for w in waypoints]

        return {
            'status': 'ok',
            'grid_w': int(grid.shape[1]),
            'grid_h': int(grid.shape[0]),
            'path_pixels': path_ds,
            'smooth_pixels': smooth_xy,
            'waypoints': wp_pixels,
            'kt_trajectory': kt_traj_ds,
            'time': time_ds,
            'rudder': rudder_ds,
            'heading': heading_ds,
            'heading_desired': heading_desired_ds,
            'speed_kn': speed_ds,
            'drift_angle': drift_ds,
            'heading_error': heading_err_ds,
            'metrics': sim_result['metrics'],
            'map_image_b64': map_b64,
            'grid_image_b64': grid_b64,
            'start_px': list(start_px),
            'goal_px': list(goal_px),
            'map_tl': list(map_tl),
            'map_br': list(map_br),
        }

    except Exception as e:
        import traceback
        return {'status': 'error', 'message': str(e), 'traceback': traceback.format_exc()}

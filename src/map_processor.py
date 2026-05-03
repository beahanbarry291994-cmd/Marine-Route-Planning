"""
地图获取与栅格化模块

功能:
  - 从 CartoDB 下载瓦片地图并缓存
  - OTSU 二值化 + 形态学处理生成可通行栅格
  - 经纬度 ↔ 像素坐标转换
  - BFS 吸附到主海洋
"""

import os
import json
import math
import ssl
import time

import cv2
import numpy as np
from collections import deque

# ──────────────────────── 瓦片坐标工具 ────────────────────────

def deg2num(lat, lon, z):
    n = 2.0 ** z
    return int((lon + 180) / 360 * n), int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)


def num2deg(x, y, z):
    n = 2.0 ** z
    return (math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n)))),
            x / n * 360 - 180)


# ──────────────────────── 地图下载 ────────────────────────

def fetch_map(start, goal, zoom, cache_img='map_cache.png', cache_meta='map_meta.json'):
    """
    获取地图图像。优先读取本地缓存，否则从 CartoDB 下载。

    Args:
        start: (lon, lat) 起点坐标
        goal:  (lon, lat) 终点坐标
        zoom:  瓦片缩放级别
        cache_img:  缓存图像路径
        cache_meta: 缓存元数据路径

    Returns:
        img: BGR 地图图像
        map_tl: (lon, lat) 地图左上角
        map_br: (lon, lat) 地图右下角
    """
    if os.path.exists(cache_img) and os.path.exists(cache_meta):
        meta = json.load(open(cache_meta))
        map_tl = tuple(meta['top_left'])
        map_br = tuple(meta['bottom_right'])
        img = cv2.imread(cache_img)
        if img is not None and np.std(img) > 10:
            print("[地图] 使用本地缓存")
            return img, map_tl, map_br

    print("[地图] 下载 CartoDB 底图...")
    d = 0.08
    x0, y1 = deg2num(min(start[1], goal[1]) - d, min(start[0], goal[0]) - d, zoom)
    x1, y0 = deg2num(max(start[1], goal[1]) + d, max(start[0], goal[0]) + d, zoom)

    map_tl = (num2deg(x0, y0, zoom)[1], num2deg(x0, y0, zoom)[0])
    map_br = (num2deg(x1 + 1, y1 + 1, zoom)[1], num2deg(x1 + 1, y1 + 1, zoom)[0])

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
    json.dump({'top_left': map_tl, 'bottom_right': map_br}, open(cache_meta, 'w'))
    return img, map_tl, map_br


# ──────────────────────── 栅格化 ────────────────────────

def process_map(img, safety_px=3):
    """
    将彩色地图转换为可通行栅格。

    流程: OTSU 二值化 → 开运算去桥 → 连通域去假湖 → 安全膨胀

    Args:
        img: BGR 地图图像
        safety_px: 安全膨胀半径 (像素)

    Returns:
        grid: 障碍栅格 (0=可通行, 1=障碍)
        ocean: 主海洋掩膜 (0/1)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OTSU 自动二值化
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    land_mask = cv2.bitwise_not(thresh)

    # 陆地过多时回退到中位数阈值
    land_ratio = np.sum(land_mask == 255) / land_mask.size
    if land_ratio > 0.90:
        median_val = np.median(gray)
        _, land_mask = cv2.threshold(gray, median_val, 255, cv2.THRESH_BINARY)

    # 开运算去桥
    land_clean = cv2.morphologyEx(land_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # 连通域: 保留最大水体
    water = (land_clean == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(water, connectivity=4)
    if n <= 1:
        return np.ones_like(land_clean, dtype=np.uint8), np.zeros_like(land_clean)

    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    ocean = (labels == largest_label).astype(np.uint8)

    # 最终陆地 + 安全膨胀
    land_final = ((land_clean == 255) | (ocean == 0)).astype(np.uint8)
    grid = cv2.dilate(land_final, np.ones((safety_px, safety_px), np.uint8))

    return grid, ocean


# ──────────────────────── 坐标映射 ────────────────────────

def lonlat_to_px(lon, lat, shape, map_tl, map_br):
    h, w = shape
    x = int((lon - map_tl[0]) / (map_br[0] - map_tl[0]) * w)
    y = int((map_tl[1] - lat) / (map_tl[1] - map_br[1]) * h)
    return max(0, min(x, w - 1)), max(0, min(y, h - 1))


def snap_to_ocean(grid, px, ocean):
    """BFS 吸附: 若目标点不在主海洋，搜索最近的海洋像素"""
    if grid[px[1], px[0]] == 0 and ocean[px[1], px[0]] == 1:
        return px
    h, w = grid.shape
    q, vis = deque([px]), {px}
    while q:
        cx, cy = q.popleft()
        if grid[cy, cx] == 0 and ocean[cy, cx] == 1:
            return (cx, cy)
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nxy = (cx + dx, cy + dy)
            if 0 <= nxy[0] < w and 0 <= nxy[1] < h and nxy not in vis:
                vis.add(nxy)
                q.append(nxy)
    return px

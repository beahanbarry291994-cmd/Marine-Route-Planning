"""
Marine Route Planning — Flask Web GUI
"""

import sys
import os

# 确保 engine.py 可以被导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 加载 .env 文件（不依赖 python-dotenv）
def _load_dotenv(path=None):
    """读取 .env 文件并注入到 os.environ，不覆盖已有环境变量"""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, val = line.partition('=')
            key, val = key.strip(), val.strip()
            if key and key not in os.environ:
                os.environ[key] = val

_load_dotenv()

import time
import json
import urllib.request
import ssl
from flask import Flask, render_template, request, jsonify
from engine import run_pipeline

# PyInstaller bundle path detection
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static'))

# DeepSeek API 配置 — Key 从环境变量读取，不要硬编码在代码里
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_API_URL = 'https://api.deepseek.com/chat/completions'
DEEPSEEK_MODEL = 'deepseek-v4-flash'

# 禁用所有缓存，确保前端每次都能拿到最新文件
@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def index():
    # 通过模板变量传入时间戳用于缓存破坏
    return render_template('index.html', cache_buster=int(time.time()))


@app.route('/api/simulate', methods=['POST'])
def simulate():
    params = request.get_json()
    if not params:
        return jsonify({'status': 'error', 'message': '未收到参数'}), 400
    result = run_pipeline(params)
    if result.get('status') == 'ok':
        print(f"[DEBUG] grid_image_b64: {len(result.get('grid_image_b64',''))} chars")
        print(f"[DEBUG] map_image_b64:  {len(result.get('map_image_b64',''))} chars")
        print(f"[DEBUG] grid size: {result['grid_w']}x{result['grid_h']}")
    return jsonify(result)


@app.route('/api/analyze', methods=['POST'])
def ai_analyze():
    """接收仿真数据 + 分析类型，调用 DeepSeek API 生成报告"""
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': '未收到数据'}), 400

    analysis_type = data.get('analysis_type', 'comprehensive')
    m = data.get('metrics', {})
    params = data.get('parameters', {})

    # 数据摘要
    data_block = f"""## 仿真数据
- 航线距离: {m.get('path_length_km')} km | 仿真时长: {m.get('sim_time_s')} s
- 航路点达成: {m.get('wp_reached')}/{m.get('wp_total')}
- 终点偏移: {m.get('end_offset_m')} m | 舵角范围: {m.get('rudder_range')} deg
- 最大航向误差: {m.get('max_heading_error')} deg

## 船舶参数
- K={params.get('ship_k')}, T={params.get('ship_t')}, a={params.get('ship_a')}
- 船长 {params.get('ship_l')}m | 船宽 {params.get('ship_b')}m | 航速 {params.get('ship_speed_kn')}kn
- 最大舵角 {params.get('max_rudder_deg')}deg | PID: KP={params.get('pid_kp')} KI={params.get('pid_ki')} KD={params.get('pid_kd')}

## 环境
- 风: {params.get('v_wind')}m/s @ {params.get('psi_wind_deg')}° | 流: {params.get('v_current')}m/s @ {params.get('psi_current_deg')}°"""

    prompts = {
        'comprehensive': f"""你是船舶导航专家。请进行综合航行分析，直接输出以下五段报告（每段3-5句）：

{data_block}

### 1. 航线规划合理性
### 2. 船舶操纵性与控制表现
### 3. 环境因素影响
### 4. 潜在风险点
### 5. 优化建议

总字数400-600字，直接输出正文。""",

        'route': f"""你是船舶导航专家。请仅评估航线规划质量，直接输出3-5句话：

{data_block}

分析航线长度合理性、航路点达成情况、终点偏移原因。""",

        'maneuver': f"""你是船舶操纵性专家。请分析船舶操纵表现，直接输出3-5句话：

{data_block}

根据K/T比值（当前K={params.get('ship_k')},T={params.get('ship_t')},K/T≈{round(float(params.get('ship_k',0.5))/max(float(params.get('ship_t',1)),0.1),1)}）判断回转/跟随特性；评估舵角使用是否合理；评价最大航向误差{m.get('max_heading_error')}°的控制质量。""",

        'environment': f"""你是航海环境专家。请分析环境因素影响，直接输出3-5句话：

{data_block}

计算横风/横流分量，分析对航迹偏移的贡献，评估舵角补偿幅度是否合理。""",

        'risk': f"""你是航海安全专家。请识别航行风险并给出优化建议，直接输出3-5句话：

{data_block}

识别大角度转向点、近岸浅水风险；给出PID参数调整或航线优化建议。""",
    }

    prompt = prompts.get(analysis_type, prompts['comprehensive'])

    try:
        req_body = json.dumps({
            'model': DEEPSEEK_MODEL,
            'messages': [
                {'role': 'system', 'content': '你是资深船舶导航与航海安全专家。直接输出分析正文，不说引导语。'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.7,
            'max_tokens': 1500,
            'stream': False
        }).encode('utf-8')

        ctx = ssl.create_default_context()
        api_req = urllib.request.Request(DEEPSEEK_API_URL, data=req_body, headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
        })

        with urllib.request.urlopen(api_req, context=ctx, timeout=60) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            report = result['choices'][0]['message']['content']
            return jsonify({'status': 'ok', 'report': report, 'analysis_type': analysis_type})

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'AI 分析失败: {str(e)}'})


if __name__ == '__main__':
    print("=" * 50)
    print("  Marine Route Planning — Web GUI")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)

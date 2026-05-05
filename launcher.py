"""Launcher for packaged .exe — starts Flask and opens browser."""
import sys, os, threading, webbrowser, time, shutil

if getattr(sys, 'frozen', False):
    # _MEIPASS: temp dir where PyInstaller extracts bundled files (templates, static)
    # exe_dir: the directory containing MarineRoutePlanning.exe (for cache, logs, etc.)
    meipass = sys._MEIPASS
    exe_dir = os.path.dirname(sys.executable)
    os.chdir(exe_dir)
    # Copy bundled cache files to exe directory on first run
    for f in ['map_cache.png', 'map_meta.json']:
        src = os.path.join(meipass, f)
        dst = os.path.join(exe_dir, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
else:
    meipass = os.path.dirname(os.path.abspath(__file__))
    exe_dir = meipass
    os.chdir(exe_dir)

sys.path.insert(0, meipass)

from app import app

def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    host = '127.0.0.1'
    port = 5000
    print("=" * 50)
    print("  Marine Route Planning — Web GUI")
    print(f"  Opening http://{host}:{port}")
    print("=" * 50)
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host=host, port=port, debug=False, use_reloader=False)

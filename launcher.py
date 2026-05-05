"""Launcher for packaged .exe — starts Flask and opens browser."""
import sys, os, threading, webbrowser, time

# Handle PyInstaller bundle paths
if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, base_dir)
os.chdir(base_dir)

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

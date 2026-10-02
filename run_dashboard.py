"""
run_dashboard.py
================
One-command launcher for the Quantum RL Decoder Web Dashboard.

Usage:
    python run_dashboard.py
"""

import os
import sys
import subprocess
import time
import webbrowser

def check_dependencies():
    missing = []
    try:
        import fastapi
    except ImportError:
        missing.append("fastapi")
    try:
        import uvicorn
    except ImportError:
        missing.append("uvicorn")
    
    if missing:
        print("=" * 65)
        print("  [!] Missing dashboard dependencies: " + ", ".join(missing))
        print("  Installing automatically via pip...")
        print("=" * 65)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("  [✓] Dependencies installed successfully!")
        except Exception as e:
            print(f"  [X] Failed to install automatically: {e}")
            print(f"  Please run: pip install {' '.join(missing)}")
            sys.exit(1)

def main():
    check_dependencies()
    import uvicorn

    port = 8000
    url = f"http://127.0.0.1:{port}"

    print("\n" + "=" * 65)
    print("   QUANTUM RL DECODER — ADAPTIVE QEC BENCHMARK COCKPIT")
    print("=" * 65)
    print(f"   ► Web Dashboard URL : {url}")
    print(f"   ► Interactive Arena : 3-Qubit Circuit & 4-Way Decoder Duel")
    print(f"   ► Hardware Cockpit  : Noise Drift & 7.2x LER Recovery")
    print(f"   ► Policy Inspector  : 4x4 Q-Table Explainability Matrix")
    print("=" * 65)
    print("   Opening browser in 1.5 seconds... Press Ctrl+C to terminate.\n")

    # Give uvicorn a brief moment and open browser
    def open_browser():
        time.sleep(1.2)
        webbrowser.open(url)

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Launch uvicorn server
    uvicorn.run("api:app", host="127.0.0.1", port=port, reload=True)

if __name__ == "__main__":
    main()

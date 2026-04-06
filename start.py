"""
start.py - Single launcher for HireFlow-AI
==========================================
Runs both the Flask backend (port 5001) and the Vite frontend dev server
from a single command. Works on Windows.

Usage:
    python start.py          # runs both backend + frontend
    python start.py --backend   # runs only the backend
    python start.py --frontend  # runs only the frontend
"""

import subprocess
import sys
import os
import signal
import time
import argparse

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

# Colors for terminal output
class Colors:
    HEADER  = '\033[95m'
    BLUE    = '\033[94m'
    CYAN    = '\033[96m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    RED     = '\033[91m'
    BOLD    = '\033[1m'
    RESET   = '\033[0m'

processes = []

def print_banner():
    print(f"""
{Colors.BOLD}{Colors.GREEN}╔══════════════════════════════════════════╗
║          🚀  HireFlow AI  🚀            ║
║      AI-Powered Resume Screening        ║
╚══════════════════════════════════════════╝{Colors.RESET}
""")

def start_backend():
    """Start Flask backend server."""
    print(f"{Colors.CYAN}[Backend]{Colors.RESET} Starting Flask on http://localhost:5001 ...")
    
    # Check if app.py exists
    app_path = os.path.join(BACKEND_DIR, "app.py")
    if not os.path.exists(app_path):
        print(f"{Colors.RED}[Backend] ERROR: {app_path} not found!{Colors.RESET}")
        return None
    
    proc = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=BACKEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    processes.append(("Backend", proc))
    return proc


def start_frontend():
    """Start Vite dev server for the React frontend."""
    print(f"{Colors.YELLOW}[Frontend]{Colors.RESET} Starting Vite dev server ...")
    
    # Check if package.json exists
    pkg_path = os.path.join(FRONTEND_DIR, "package.json")
    if not os.path.exists(pkg_path):
        print(f"{Colors.RED}[Frontend] ERROR: {pkg_path} not found!{Colors.RESET}")
        return None
    
    # Check if node_modules exists, if not run npm install
    nm_path = os.path.join(FRONTEND_DIR, "node_modules")
    if not os.path.exists(nm_path):
        print(f"{Colors.YELLOW}[Frontend]{Colors.RESET} node_modules not found, running npm install...")
        install_proc = subprocess.run(
            ["npm", "install"],
            cwd=FRONTEND_DIR,
            shell=True,
        )
        if install_proc.returncode != 0:
            print(f"{Colors.RED}[Frontend] npm install failed!{Colors.RESET}")
            return None
    
    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=True,
    )
    processes.append(("Frontend", proc))
    return proc


def stream_output(name, proc, color):
    """Read and print output from a subprocess line-by-line."""
    try:
        for line in iter(proc.stdout.readline, ''):
            if line:
                print(f"{color}[{name}]{Colors.RESET} {line}", end='')
    except Exception:
        pass


def cleanup(*args):
    """Kill all child processes on exit."""
    print(f"\n{Colors.RED}[Shutdown]{Colors.RESET} Stopping all services...")
    for name, proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
            print(f"  ✓ {name} stopped")
        except Exception:
            try:
                proc.kill()
                print(f"  ✗ {name} force-killed")
            except Exception:
                pass
    print(f"{Colors.GREEN}[Done]{Colors.RESET} All services stopped. Goodbye!")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="HireFlow-AI Launcher")
    parser.add_argument("--backend", action="store_true", help="Run only the backend")
    parser.add_argument("--frontend", action="store_true", help="Run only the frontend")
    args = parser.parse_args()
    
    # If neither flag is set, run both
    run_backend = args.backend or (not args.backend and not args.frontend)
    run_frontend = args.frontend or (not args.backend and not args.frontend)
    
    print_banner()
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    backend_proc = None
    frontend_proc = None
    
    if run_backend:
        backend_proc = start_backend()
    
    if run_frontend:
        frontend_proc = start_frontend()
    
    # Give processes a moment to start
    time.sleep(2)
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.RESET}")
    if run_backend:
        print(f"  {Colors.CYAN}Backend  →{Colors.RESET}  http://localhost:5001")
    if run_frontend:
        print(f"  {Colors.YELLOW}Frontend →{Colors.RESET}  http://localhost:5173")
    print(f"{Colors.BOLD}{Colors.GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.RESET}")
    print(f"\n  Press {Colors.BOLD}Ctrl+C{Colors.RESET} to stop all services\n")
    
    # Stream output from both processes interleaved
    import threading
    
    threads = []
    if backend_proc:
        t = threading.Thread(target=stream_output, args=("Backend", backend_proc, Colors.CYAN), daemon=True)
        t.start()
        threads.append(t)
    
    if frontend_proc:
        t = threading.Thread(target=stream_output, args=("Frontend", frontend_proc, Colors.YELLOW), daemon=True)
        t.start()
        threads.append(t)
    
    # Wait for either process to exit (or Ctrl+C)
    try:
        while True:
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"\n{Colors.RED}[{name}]{Colors.RESET} Process exited with code {proc.returncode}")
                    cleanup()
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()

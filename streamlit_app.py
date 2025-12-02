"""
Entry point for Streamlit Cloud.

This wrapper ensures Streamlit can run the frontend app even though the project
is nested inside the `Arm-Detection/frontend` directory.
"""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "Arm-Detection" / "frontend"
APP_PATH = FRONTEND_DIR / "app.py"

if not FRONTEND_DIR.exists():
    raise FileNotFoundError(f"Frontend directory not found at {FRONTEND_DIR}")

# Ensure the frontend directory is importable (for module-relative imports).
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

runpy.run_path(str(APP_PATH), run_name="__main__")


"""Compatibility wrapper for scripts.compare_solvers."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from scripts.compare_solvers import *  # noqa: F401,F403

if __name__ == "__main__":
    from scripts.compare_solvers import main

    main()

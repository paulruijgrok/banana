#!/usr/bin/env python3
"""
Run BLI analysis from YAML config (same as CLI banana-bli).

  python examples/run_BLI.py path/to/Nb6_folder -c path/to/config.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from banana.cli.run_bli import main

if __name__ == "__main__":
    sys.exit(main())

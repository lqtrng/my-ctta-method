from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CT_MLFLOW"] = "0"

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

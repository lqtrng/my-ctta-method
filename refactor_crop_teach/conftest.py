"""Lets pytest import the modules of this folder (and tests/helpers.py) by their plain names."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "tests")]

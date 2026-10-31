"""Lets the scripts import the signbangla package when run as `python scripts/xx.py` from anywhere."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

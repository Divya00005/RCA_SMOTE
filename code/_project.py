"""Portable project locations; no model or metric changes."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'code'
sys.path.insert(0,str(ROOT/'.cache'/'aw_baseline'))

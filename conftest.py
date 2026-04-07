"""
conftest.py — Project root pytest configuration

This file must live at D:\RagANI\anime_rag\conftest.py (project root).
It ensures all phase* packages are importable regardless of how pytest is invoked.
"""
import sys
from pathlib import Path

# Add project root to sys.path BEFORE any test collection
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
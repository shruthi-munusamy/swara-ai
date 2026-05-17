"""
Pytest configuration.

Adds the project root to sys.path so tests can import from src/ without
requiring an editable install.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
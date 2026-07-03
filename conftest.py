"""Гарантирует, что корень репозитория в sys.path при запуске pytest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

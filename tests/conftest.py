"""Put the repository root on sys.path and import the studies package, which adds every study folder,
so tests can import the study scripts by their plain names (import _study3_common)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import studies  # noqa: E402,F401

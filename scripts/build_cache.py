"""Convert the 2022 GeoTIFFs into normalised per-split arrays under data/cache/.

Usage: .venv\\Scripts\\python scripts\\build_cache.py [--force]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallmines.data import build_cache  # noqa: E402

if __name__ == "__main__":
    build_cache(force="--force" in sys.argv)
    print("OK: cache ready")

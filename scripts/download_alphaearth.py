"""Download the 2022 AlphaEarth Foundations embeddings under every SmallMinesDS 2022 patch.

Source: the AlphaEarth Foundations annual Satellite Embedding dataset ("The AlphaEarth
Foundations Satellite Embedding dataset is produced by Google and Google DeepMind", CC-BY 4.0),
read from its public Cloud-Optimized GeoTIFF mirror on Source Cooperative
(https://source.coop/tge-labs/aef). Each pixel is a 64-dim embedding at 10 m, stored as int8.

All SmallMinesDS patches lie in UTM zone 30N on a 10 m grid aligned with the embedding grid, so
each patch is cut out without resampling. The four 2022 files covering the study area were found
with the mirror's index (v1/annual/aef_index.csv, year 2022, zone 30N). The files are stored
bottom-up (origin at the south-west corner, positive y resolution) in 1024 x 1024 blocks, one
band at a time; the script reads the 1024 x 1024 cells that contain patches (56 cells, about
2-3 GB), flips them to north-up and cuts every patch out of them.

Output: data/alphaearth/aef_2022.npz with
    ids : (2135,) SmallMinesDS patch index
    emb : (2135, 64, 128, 128) int8, raw quantised values; -128 = no data.
De-quantise with (v / 127.5) ** 2 * sign(v); scripts/run_alphaearth.py uses the raw values (trees are
unaffected by this monotonic map).

Usage: python scripts\\download_alphaearth.py   (2-3 GB download, 2.2 GB output)
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from smallmines.data import IMG_DIR, IMG_SIZE, YEAR  # noqa: E402

BASE = "https://data.source.coop/tge-labs/aef/v1/annual/2022/30N/"
TILES = {  # file name: (west, south) of its pixel array in EPSG:32630; each is 8192 x 8192 px
    "xxjkjz72qg2kq51ku-0000008192-0000008192": (581920, 573440),
    "xsrzp43zqueiaoxuj-0000000000-0000008192": (581920, 655360),
    "xnh3v2bbgvomh1qaq-0000008192-0000000000": (663840, 573440),
    "x9lr7mfwshvirawd8-0000000000-0000000000": (663840, 655360),
}
TILE_PX, CELL_PX, RES = 8192, 1024, 10
OUT = ROOT / "data" / "alphaearth" / f"aef_{YEAR}.npz"
N_DIM = 64
NODATA = -128

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")


def patch_bounds(idx: int):
    with rasterio.open(IMG_DIR / f"IMG_GH_{idx:04d}_{YEAR}.tif") as r:
        assert r.crs.to_epsg() == 32630 and r.res == (10.0, 10.0)
        return r.bounds


def cells_of(b) -> set[tuple[str, int, int]]:
    """(tile, cell column, cell row counted from the south) touched by a patch."""
    out = set()
    for x in (b.left, b.right - RES):
        for y in (b.bottom, b.top - RES):
            for name, (w, s) in TILES.items():
                if w <= x < w + TILE_PX * RES and s <= y < s + TILE_PX * RES:
                    out.add((name, int((x - w) // (CELL_PX * RES)), int((y - s) // (CELL_PX * RES))))
    return out


def read_cell(cell: tuple[str, int, int]) -> np.ndarray:
    """One 1024 x 1024 cell, all 64 bands, flipped to north-up."""
    name, cx, cy = cell
    with rasterio.open(f"/vsicurl/{BASE}{name}.tiff") as t:
        data = t.read(window=Window(cx * CELL_PX, cy * CELL_PX, CELL_PX, CELL_PX))
    return data[:, ::-1, :]


def paste(out: np.ndarray, b, cell: tuple[str, int, int], data: np.ndarray) -> None:
    name, cx, cy = cell
    w, s = TILES[name]
    c_left, c_bottom = w + cx * CELL_PX * RES, s + cy * CELL_PX * RES
    c_top = c_bottom + CELL_PX * RES
    left, right = max(b.left, c_left), min(b.right, c_left + CELL_PX * RES)
    bottom, top = max(b.bottom, c_bottom), min(b.top, c_top)
    if left >= right or bottom >= top:
        return
    ncol, nrow = int((right - left) // RES), int((top - bottom) // RES)
    pc, pr = int((left - b.left) // RES), int((b.top - top) // RES)
    cc, cr = int((left - c_left) // RES), int((c_top - top) // RES)
    out[:, pr:pr + nrow, pc:pc + ncol] = data[:, cr:cr + nrow, cc:cc + ncol]


def main() -> int:
    ids = sorted(int(p.stem.split("_")[2]) for p in IMG_DIR.glob(f"IMG_GH_*_{YEAR}.tif"))
    bounds = [patch_bounds(i) for i in ids]
    need = [cells_of(b) for b in bounds]
    cells = sorted(set().union(*need))
    print(f"{len(ids)} patches in {len(cells)} cells of {CELL_PX} x {CELL_PX} px")
    emb = np.full((len(ids), N_DIM, IMG_SIZE, IMG_SIZE), NODATA, np.int8)
    with ThreadPoolExecutor(4) as ex:
        for k, (cell, data) in enumerate(zip(cells, ex.map(read_cell, cells)), 1):
            for i, (b, nd) in enumerate(zip(bounds, need)):
                if cell in nd:
                    paste(emb[i], b, cell, data)
            print(f"  cell {k}/{len(cells)} {cell}", flush=True)
    missing = (emb[:, 0] == NODATA).mean()
    print(f"no-data pixels: {missing:.4%}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, ids=np.array(ids), emb=emb)
    print("OK:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())

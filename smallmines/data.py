"""SmallMinesDS 2022: loading, normalisation, caching, subset selection and D4 augmentation.

Band handling
    GeoTIFF bands 0-9  : Sentinel-2 L2A  B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12  (kept)
    GeoTIFF bands 10-11: Sentinel-1 RTC  VV,VH, stored linear -> converted to dB (kept)
    GeoTIFF band 12    : Copernicus DEM (dropped, per the experiment plan)
    Both modalities are standardised with TerraMind v1 pretraining statistics, for every condition.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
YEAR = 2022
IMG_DIR = DATA_DIR / "SmallMinesDS" / str(YEAR) / "IMAGE"
MASK_DIR = DATA_DIR / "SmallMinesDS" / str(YEAR) / "MASK"
SPLIT_CSV = DATA_DIR / "data_splits" / f"train_test_splits_{YEAR}.csv"
CACHE_DIR = DATA_DIR / "cache"
IMG_SIZE = 128

S2_TIF_BANDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
S1_TIF_BANDS = [10, 11]
# TerraMind names for the dataset's S2 bands, in GeoTIFF order
S2_BANDS = ["BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3",
            "NIR_BROAD", "NIR_NARROW", "SWIR_1", "SWIR_2"]
S1_BANDS = ["VV", "VH"]
N_CHANNELS = len(S2_BANDS) + len(S1_BANDS)

# TerraMind v1 pretraining statistics (terratorch: terramind_register.py). S1RTC values are dB.
_TM_S2_BANDS = ["COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3",
                "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "SWIR_1", "SWIR_2"]
_TM_S2_MEAN = [1390.458, 1503.317, 1718.197, 1853.91, 2199.1, 2779.975,
               2987.011, 3083.234, 3132.22, 3162.988, 2424.884, 1857.648]
_TM_S2_STD = [2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926,
              1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311]
S2_MEAN = np.array([_TM_S2_MEAN[_TM_S2_BANDS.index(b)] for b in S2_BANDS], np.float32)
S2_STD = np.array([_TM_S2_STD[_TM_S2_BANDS.index(b)] for b in S2_BANDS], np.float32)
S1_MEAN = np.array([-10.93, -17.329], np.float32)
S1_STD = np.array([4.391, 4.459], np.float32)

# Low-N subset rule: the first N_FROM_POOL images of every seed's ordering are drawn from
# training images whose mine-pixel fraction lies in MINE_POOL_RANGE (inclusive).
MINE_POOL_RANGE = (0.25, 0.75)
N_FROM_POOL = 10


def _cache_path(split: str) -> Path:
    return CACHE_DIR / f"{YEAR}_{split}.npz"


def split_table() -> pd.DataFrame:
    df = pd.read_csv(SPLIT_CSV)
    # CSV names masks MASK_<idx>_<year>.tif; files on disk are MASK_GH_<idx>_<year>.tif
    df["idx"] = df["patch_name"].str.extract(r"MASK_(\d+)_")[0].astype(int)
    return df


def build_cache(force: bool = False) -> None:
    """Read every 2022 GeoTIFF once and store normalised arrays per split."""
    import rasterio

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = split_table()
    for split in ("train", "test"):
        out = _cache_path(split)
        if out.exists() and not force:
            print(f"  {out.name}: already present")
            continue
        sub = df[df["split"] == split].sort_values("idx").reset_index(drop=True)
        x = np.empty((len(sub), N_CHANNELS, IMG_SIZE, IMG_SIZE), np.float32)
        m = np.empty((len(sub), IMG_SIZE, IMG_SIZE), np.uint8)
        for i, idx in enumerate(sub["idx"]):
            with rasterio.open(IMG_DIR / f"IMG_GH_{idx:04d}_{YEAR}.tif") as r:
                a = r.read()
            s2 = a[S2_TIF_BANDS]
            s1 = 10.0 * np.log10(np.clip(a[S1_TIF_BANDS], 1e-6, None))  # linear -> dB
            x[i, :10] = (s2 - S2_MEAN[:, None, None]) / S2_STD[:, None, None]
            x[i, 10:] = (s1 - S1_MEAN[:, None, None]) / S1_STD[:, None, None]
            with rasterio.open(MASK_DIR / f"MASK_GH_{idx:04d}_{YEAR}.tif") as r:
                m[i] = r.read(1)
            if i % 200 == 0:
                print(f"\r  {split}: {i}/{len(sub)}", end="", flush=True)
        print(f"\r  {split}: {len(sub)}/{len(sub)}")
        assert np.isfinite(x).all(), "non-finite values after normalisation"
        assert set(np.unique(m)) <= {0, 1}, "mask is not binary"
        mine_frac = m.reshape(len(sub), -1).mean(1)
        # The CSV's class_percentage should agree with the masks on disk
        assert np.abs(mine_frac * 100 - sub["class_percentage"].values).max() < 0.01
        np.savez(out, x=x, mask=m, ids=sub["idx"].values, mine_frac=mine_frac)
    (CACHE_DIR / "normalisation.json").write_text(json.dumps({
        "s2_bands": S2_BANDS, "s2_mean": S2_MEAN.tolist(), "s2_std": S2_STD.tolist(),
        "s1_bands": S1_BANDS, "s1_unit": "dB", "s1_mean": S1_MEAN.tolist(), "s1_std": S1_STD.tolist(),
    }, indent=2))


@dataclass
class Split:
    x: torch.Tensor      # (n, 12, 128, 128) float32, normalised
    mask: torch.Tensor   # (n, 128, 128) int64, 1 = mine
    ids: np.ndarray      # SmallMinesDS patch index per row
    mine_frac: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)

    def subset(self, rows: np.ndarray) -> "Split":
        rows = np.asarray(rows)
        return Split(self.x[rows], self.mask[rows], self.ids[rows], self.mine_frac[rows])

    def to(self, device) -> "Split":
        return Split(self.x.to(device), self.mask.to(device), self.ids, self.mine_frac)


def load_split(split: str) -> Split:
    p = _cache_path(split)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run scripts/build_cache.py first")
    z = np.load(p)
    return Split(torch.from_numpy(z["x"]), torch.from_numpy(z["mask"]).long(), z["ids"], z["mine_frac"])


def as_model_input(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split the 12-channel tensor into the modality dict TerraMind expects."""
    return {"S2L2A": x[:, :10], "S1RTC": x[:, 10:]}


def training_order(mine_frac: np.ndarray, seed: int) -> np.ndarray:
    """Per-seed ordering of the training pool; subset for budget N is the first N rows.

    The first N_FROM_POOL rows are a random draw from images with mine fraction in
    MINE_POOL_RANGE, the rest is a random permutation of all remaining images. Budgets are
    therefore nested, and identical across the four conditions for a given seed.
    """
    rng = np.random.default_rng(seed)
    lo, hi = MINE_POOL_RANGE
    pool = np.flatnonzero((mine_frac >= lo) & (mine_frac <= hi))
    first = rng.choice(pool, N_FROM_POOL, replace=False)
    rest = rng.permutation(np.setdiff1d(np.arange(len(mine_frac)), first))
    return np.concatenate([first, rest])


def d4_transform(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """One of the 8 square symmetries: rotate by k*90 degrees, then optionally mirror left-right."""
    x = torch.rot90(x, k, dims=(-2, -1))
    return x.flip(-1) if flip else x


def augment_batch(x: torch.Tensor, mask: torch.Tensor, gen: torch.Generator):
    """Apply an independent uniformly drawn D4 symmetry to each image and its mask."""
    t = torch.randint(8, (x.shape[0],), generator=gen)
    x = x.clone()
    mask = mask.clone()
    for i, ti in enumerate(t.tolist()):
        k, flip = ti % 4, bool(ti // 4)
        x[i] = d4_transform(x[i], k, flip)
        mask[i] = d4_transform(mask[i], k, flip)
    return x, mask

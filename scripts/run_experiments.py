"""Run the full reported grid and append results to CSV.

Grid: N in {1, 3, 5, 10, 20, 50, 100, 250} with seeds 0-4, plus N in {500, 1000, 1491 (all)} with
seeds 0-2; 3 conditions each (TerraMind pretrained frozen + aug, TerraMind random init trainable
+ aug, plain UNet of the same size + aug) -> 147 runs.
Each run trains until test IoU stops improving (see smallmines/train.py).

Usage
    full experiment:  python scripts\\run_experiments.py
    options:          --seeds 0 1 ...  --budgets 1 10 ...  --encoders unet  --norm groupnorm  --out <runs.csv>

Resumable: runs whose name already appears in the CSV are skipped. The two conditions of each
(seed, N) run back to back so paired comparisons are adjacent in the output.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Must be set before CUDA is initialised for deterministic cuBLAS
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from smallmines.data import load_split  # noqa: E402
from smallmines.train import (  # noqa: E402
    CONDITIONS, PROTOCOL, RunConfig, configure_determinism, report_grid, run_one, subset_rows,
)


def environment_info() -> dict:
    import importlib.metadata as md

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        commit = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "terratorch": md.version("terratorch"),
        "git_commit": commit,
        "protocol": PROTOCOL,
    }


def done_runs(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {row["run"] for row in csv.DictReader(f)}


def append_row(path: Path, rec: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec.keys()))
        if new:
            w.writeheader()
        w.writerow(rec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="restrict to these seeds (default: the full reported grid)")
    ap.add_argument("--budgets", type=int, nargs="+", default=None,
                    help="restrict to these N (default: the full reported grid)")
    ap.add_argument("--norm", default="batchnorm", choices=["batchnorm", "groupnorm"])
    ap.add_argument("--encoders", nargs="+", default=None, choices=[e for e, _ in CONDITIONS],
                    help="restrict to these conditions (default: all)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "main" / "runs.csv")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA GPU required (see scripts/check_gpu.py)")
        return 1
    configure_determinism()
    device = torch.device("cuda")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.parent / "environment.json").write_text(json.dumps(environment_info(), indent=2))

    train = load_split("train")
    test = load_split("test").to(device)

    # Default: the full reported grid (N <= 250 with 5 seeds, N >= 500 with 3 seeds).
    # --seeds / --budgets restrict it; a pair outside the grid runs only when both are given.
    grid = report_grid()
    if args.seeds is not None and args.budgets is not None:
        grid = [(s, n) for n in args.budgets for s in args.seeds]
    elif args.seeds is not None:
        grid = [(s, n) for s, n in grid if s in args.seeds]
    elif args.budgets is not None:
        grid = [(s, n) for s, n in grid if n in args.budgets]

    # Record exactly which patches each (seed, N) subset contains
    subsets_path = args.out.parent / "subsets.json"
    subsets = json.loads(subsets_path.read_text()) if subsets_path.exists() else {}
    for seed, n in grid:
        rows = subset_rows(train, RunConfig(seed, n, "pretrained"))
        subsets.setdefault(str(seed), {})[str(n)] = [int(i) for i in train.ids[rows]]
    subsets_path.write_text(json.dumps(subsets, indent=1))

    done = done_runs(args.out)
    todo = [RunConfig(s, n, enc, aug, norm=args.norm) for s, n in grid for enc, aug in CONDITIONS
            if args.encoders is None or enc in args.encoders]
    todo = [c for c in todo if c.name not in done]
    print(f"{len(done)} runs already done, {len(todo)} to go (norm: {args.norm})")
    print("protocol:", PROTOCOL)

    t0 = time.perf_counter()
    for i, cfg in enumerate(todo, 1):
        rec = run_one(cfg, train, test, device)
        append_row(args.out, rec)
        print(f"[{i}/{len(todo)}] {cfg.name:28s} iou={rec['iou']:.3f} best_ep={rec['best_epoch']:<4d} "
              f"stop_ep={rec['epochs_done']:<4d} steps={rec['steps_done']:<6d} "
              f"{rec['flops_total'] / 1e12:7.1f} TFLOP {rec['seconds']:6.0f}s  "
              f"(elapsed {(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    print("OK: all runs complete ->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

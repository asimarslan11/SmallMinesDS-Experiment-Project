"""Run the full grid (8 budgets x 4 conditions x 5 seeds = 160 runs) and append results to CSV.

Usage: .venv\\Scripts\\python scripts\\run_experiments.py [--seeds 0 1] [--budgets 1 10] [--out results/runs.csv]

Resumable: runs whose name already appears in the CSV are skipped. The four conditions of each
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
    BUDGETS, CONDITIONS, SEEDS, RunConfig, configure_determinism, run_one, subset_rows,
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
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--budgets", type=int, nargs="+", default=BUDGETS)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "runs.csv")
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

    # Record exactly which patches each (seed, N) subset contains
    subsets_path = args.out.parent / "subsets.json"
    subsets = json.loads(subsets_path.read_text()) if subsets_path.exists() else {}
    for seed in args.seeds:
        for n in args.budgets:
            rows = subset_rows(train, RunConfig(seed, n, "pretrained", False))
            subsets.setdefault(str(seed), {})[str(n)] = [int(i) for i in train.ids[rows]]
    subsets_path.write_text(json.dumps(subsets, indent=1))

    done = done_runs(args.out)
    todo = [RunConfig(s, n, enc, aug) for s in args.seeds for n in args.budgets for enc, aug in CONDITIONS]
    todo = [c for c in todo if c.name not in done]
    print(f"{len(done)} runs already done, {len(todo)} to go")

    t0 = time.perf_counter()
    for i, cfg in enumerate(todo, 1):
        rec = run_one(cfg, train, test, device)
        append_row(args.out, rec)
        print(f"[{i}/{len(todo)}] {cfg.name:32s} iou={rec['iou']:.3f} f1={rec['f1']:.3f} "
              f"prec={rec['precision']:.3f} rec={rec['recall']:.3f} loss={rec['train_loss_last']:.3f} "
              f"{rec['seconds']:.0f}s  (elapsed {(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    print("OK: all runs complete ->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

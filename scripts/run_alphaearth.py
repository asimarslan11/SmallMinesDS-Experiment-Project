"""AlphaEarth embeddings + XGBoost: per-pixel mine classification on the same splits and subsets.

Every pixel of a patch is one sample: its 64-dim 2022 AlphaEarth embedding (from
scripts/download_alphaearth.py) and its mine mask label. Training uses the pixels of exactly the
same (seed, N) patch subsets as the deep-learning runs (smallmines.train.subset_rows); the test
set is all pixels of the same 644 test patches, scored with the same pooled mine-class IoU.

Model: XGBoost gradient-boosted trees, binary logistic loss, hist method on CPU, learning rate
0.1, max depth 6, all other parameters at their defaults; seeded by the run seed. The raw int8
embedding values are used as features: the de-quantisation map is monotonic per channel, so
trees are unaffected by it. Pixels without an embedding (no data) are left out of training and
predicted as background.

Stopping rule, mirroring the deep-learning protocol: test IoU is checked every CHECK_ROUNDS (25)
boosting rounds; a check improves when IoU exceeds the best by more than 1e-3; training stops at
the first check PATIENCE_ROUNDS (100) or more rounds after the best check (4 checks), or at
MAX_ROUNDS; the model at the best check is reported. Training pixels are capped at MAX_PIXELS
(2^22 = 256 patches' worth), drawn uniformly at random with the run seed; this only affects
N >= 500.

Usage
    python scripts\\run_alphaearth.py                  # 49 runs -> results/alphaearth/runs.csv
    options: --seeds 0 1 ...  --budgets 1 10 ...  --out <runs.csv>
Resumable: runs already in the CSV are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from smallmines.data import load_split  # noqa: E402
from smallmines.metrics import summarise  # noqa: E402
from smallmines.train import MIN_DELTA, RunConfig, report_grid, subset_rows  # noqa: E402

EMB_PATH = ROOT / "data" / "alphaearth" / "aef_2022.npz"
NODATA = -128
CHECK_ROUNDS = 25
PATIENCE_ROUNDS = 100
MAX_ROUNDS = 3000
MAX_PIXELS = 2 ** 22
N_THREADS = max(1, (os.cpu_count() or 2) - 4)  # leave cores for the GPU job running alongside
PARAMS = {"objective": "binary:logistic", "eta": 0.1, "max_depth": 6, "tree_method": "hist",
          "disable_default_eval_metric": 1, "nthread": N_THREADS}
ENCODER = "alphaearth_xgb"
PROTOCOL = (f"AlphaEarth 2022 embeddings + XGBoost (eta 0.1, depth 6); test IoU every {CHECK_ROUNDS} rounds, "
            f"patience {PATIENCE_ROUNDS} rounds, min delta {MIN_DELTA:g}; at most {MAX_PIXELS} training pixels")


def pixels(emb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(n, 64, H, W) int8 + (n, H, W) labels -> (pixels, 64) features, labels, valid flags."""
    x = emb.transpose(0, 2, 3, 1).reshape(-1, emb.shape[1])
    y = mask.reshape(-1).astype(np.float32)
    return x, y, x[:, 0] != NODATA


class Stopper(xgb.callback.TrainingCallback):
    """Applies the check / patience rule to the test IoU logged every round."""

    def __init__(self):
        self.best_iou, self.best_round, self.history = -1.0, 0, []

    def after_iteration(self, model, epoch, evals_log) -> bool:
        rnd = epoch + 1
        if rnd % CHECK_ROUNDS:
            return False
        iou = evals_log["test"]["iou"][-1]
        self.history.append((rnd, iou))
        if iou > self.best_iou + MIN_DELTA:
            self.best_iou, self.best_round = iou, rnd
            return False
        return rnd - self.best_round >= PATIENCE_ROUNDS


def run_one(seed: int, n: int, train, emb_train, emb_test, test_mask) -> dict:
    t0 = time.perf_counter()
    rows = subset_rows(train, RunConfig(seed, n, "pretrained"))
    x, y, ok = pixels(emb_train[rows], train.mask[rows].numpy())
    x, y = x[ok], y[ok]
    if len(y) > MAX_PIXELS:
        keep = np.random.default_rng(seed).choice(len(y), MAX_PIXELS, replace=False)
        x, y = x[keep], y[keep]
    xt, yt, okt = pixels(emb_test, test_mask)
    dtrain = xgb.DMatrix(x, label=y, nthread=N_THREADS)
    dtest = xgb.DMatrix(xt[okt], label=yt[okt], nthread=N_THREADS)
    n_invalid_mine = int(yt[~okt].sum())  # no-data test pixels are predicted background

    rounds_seen = [0]
    true_test = yt[okt] > 0.5

    def iou_metric(prob, dm):
        # called after every round; only check rounds are read by Stopper, so skip the rest
        rounds_seen[0] += 1
        if rounds_seen[0] % CHECK_ROUNDS:
            return "iou", 0.0
        true = true_test
        pred = prob > 0.5
        tp = int((pred & true).sum())
        fp = int((pred & ~true).sum())
        fn = int((~pred & true).sum()) + n_invalid_mine
        return "iou", tp / (tp + fp + fn) if tp + fp + fn else 0.0

    stop = Stopper()
    bst = xgb.train({**PARAMS, "seed": seed}, dtrain, MAX_ROUNDS, evals=[(dtest, "test")],
                    custom_metric=iou_metric, callbacks=[stop], verbose_eval=False)
    rounds = bst.num_boosted_rounds()
    prob = bst.predict(dtest, iteration_range=(0, stop.best_round))
    pred = np.zeros(len(yt), bool)
    pred[okt] = prob > 0.5
    true = yt > 0.5
    counts = [int((pred & true).sum()), int((pred & ~true).sum()),
              int((~pred & true).sum()), int((~pred & ~true).sum())]
    seconds = time.perf_counter() - t0
    rec = {"seed": seed, "n_train": n, "encoder": ENCODER, "augment": False, "norm": "",
           "run": f"s{seed}_n{n}_{ENCODER}", "protocol": PROTOCOL, **summarise(counts)}
    rec.update({
        "best_round": stop.best_round, "rounds_done": rounds, "checks_done": len(stop.history),
        "stop_reason": "patience" if rounds < MAX_ROUNDS else "max_rounds",
        "iou_history": ";".join(f"{r}:{v:.4f}" for r, v in stop.history),
        "train_pixels": len(y), "train_mine_pixel_frac": float(y.mean()),
        "test_nodata_pixels": int((~okt).sum()), "seconds": seconds, "cpu_threads": N_THREADS,
    })
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--budgets", type=int, nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "alphaearth" / "runs.csv")
    args = ap.parse_args()
    if not EMB_PATH.exists():
        print(f"FAIL: {EMB_PATH} missing; run scripts/download_alphaearth.py first")
        return 1

    z = np.load(EMB_PATH)
    pos = {int(i): k for k, i in enumerate(z["ids"])}
    train, test = load_split("train"), load_split("test")
    emb = z["emb"]
    emb_train = emb[[pos[int(i)] for i in train.ids]]
    emb_test = emb[[pos[int(i)] for i in test.ids]]
    test_mask = test.mask.numpy()
    print(f"embeddings: train {emb_train.shape}, test {emb_test.shape}; "
          f"no-data pixels train {np.mean(emb_train[:, 0] == NODATA):.4%}, test {np.mean(emb_test[:, 0] == NODATA):.4%}")

    grid = report_grid()
    if args.seeds is not None:
        grid = [(s, n) for s, n in grid if s in args.seeds]
    if args.budgets is not None:
        grid = [(s, n) for s, n in grid if n in args.budgets]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.parent / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(),
        "xgboost": xgb.__version__, "numpy": np.__version__, "protocol": PROTOCOL, "params": PARAMS,
    }, indent=2))
    done = set()
    if args.out.exists():
        with open(args.out, newline="") as f:
            done = {r["run"] for r in csv.DictReader(f)}
    todo = [(s, n) for s, n in grid if f"s{s}_n{n}_{ENCODER}" not in done]
    print(f"{len(done)} runs already done, {len(todo)} to go\nprotocol: {PROTOCOL}")
    t0 = time.perf_counter()
    for k, (s, n) in enumerate(todo, 1):
        rec = run_one(s, n, train, emb_train, emb_test, test_mask)
        new = not args.out.exists()
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rec.keys()))
            if new:
                w.writeheader()
            w.writerow(rec)
        print(f"[{k}/{len(todo)}] {rec['run']:28s} iou={rec['iou']:.3f} best_round={rec['best_round']:<5d} "
              f"rounds={rec['rounds_done']:<5d} pixels={rec['train_pixels']:<8d} {rec['seconds']:6.0f}s "
              f"(elapsed {(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    print("OK: all runs complete ->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

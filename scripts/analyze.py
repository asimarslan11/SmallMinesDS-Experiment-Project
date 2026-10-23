"""Summarise a runs.csv: IoU and training cost vs N, seed variance, and tables.

Usage: .venv\\Scripts\\python scripts\\analyze.py [--runs results/main/runs.csv]

Outputs (next to the runs file)
    summary.csv            mean/std over seeds of every metric and cost column per (N, condition)
    summary.md             markdown tables: IoU, and per-condition cost (steps, epochs, GPU hours, FLOPs)
    figures/iou_vs_n.png   mine-class IoU vs N, mean +- std over seeds
    figures/variance.png   every run as a dot, mean line and +-std band, one panel per condition
    figures/cost_vs_n.png  optimiser steps, GPU hours and FLOPs vs N
    aug_gain.csv           only when both augmented and unaugmented runs are present
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRAIN_SIZE = 1491  # all SmallMinesDS 2022 training patches
METRICS = ["iou", "f1", "precision", "recall"]
COST = ["steps_done", "epochs_done", "best_epoch", "checks_done", "gpu_hours", "flops_train", "flops_total"]
ENCODER_LABEL = {"pretrained": "TerraMind pretrained, frozen", "random": "Random init, trainable"}
AUG_LABEL = {False: "no aug", True: "D4 aug"}

# Light-mode palette: encoder -> hue, augmentation -> line style
COLOR = {"pretrained": "#2a78d6", "random": "#eb6834"}
STYLE = {False: "--", True: "-"}
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"

BUDGETS: list[int] = []


def n_label(n: int) -> str:
    return "all" if n == TRAIN_SIZE else str(n)


def cond_label(enc: str, aug: bool) -> str:
    return f"{ENCODER_LABEL[enc]}, {AUG_LABEL[aug]}"


def load_runs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["augment"] = df["augment"].astype(str).str.lower().eq("true")
    return df


def conditions(df: pd.DataFrame) -> list[tuple[str, bool]]:
    order = [("pretrained", False), ("pretrained", True), ("random", False), ("random", True)]
    present = set(zip(df["encoder"], df["augment"]))
    return [c for c in order if c in present]


def variant_label(df: pd.DataFrame) -> str:
    norm = df["norm"].iloc[0] if "norm" in df else "batchnorm"
    short = "early stopping on test IoU (patience 4, min delta 1e-3), lr 5e-3 halved on plateau"
    if "protocol" in df and "early stopping" not in str(df["protocol"].iloc[0]):
        short = str(df["protocol"].iloc[0])
    return f"{short} · {'GroupNorm' if norm == 'groupnorm' else 'BatchNorm'} decoder"


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = [c for c in METRICS + COST + ["train_loss_last", "pred_mine_frac", "seconds"] if c in df]
    for (n, enc, aug), sub in df.groupby(["n_train", "encoder", "augment"]):
        for m in cols:
            v = sub[m].astype(float)
            rows.append({"n_train": n, "encoder": enc, "augment": aug, "metric": m,
                         "mean": v.mean(), "std": v.std(ddof=0), "n_seeds": int(v.notna().sum()),
                         "n_na": int(v.isna().sum())})
    return pd.DataFrame(rows)


def paired_gain(df: pd.DataFrame) -> pd.DataFrame | None:
    """aug - noaug within each (seed, N, encoder); None when one side is absent."""
    if not (df["augment"].any() and (~df["augment"]).any()):
        return None
    key = ["seed", "n_train", "encoder"]
    d = (df[df["augment"]].set_index(key)[METRICS] - df[~df["augment"]].set_index(key)[METRICS]).reset_index()
    rows = []
    for (n, enc), sub in d.groupby(["n_train", "encoder"]):
        for m in METRICS:
            v = sub[m].astype(float)
            rows.append({"n_train": n, "encoder": enc, "metric": m, "mean_gain": v.mean(),
                         "std_gain": v.std(ddof=0), "n_positive": int((v > 0).sum()),
                         "n_seeds": int(v.notna().sum())})
    return pd.DataFrame(rows)


def _style_axis(ax, title: str, log_y: bool = False) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([n_label(b) for b in BUDGETS], color=TEXT2)
    ax.tick_params(axis="y", colors=TEXT2)
    ax.minorticks_off()
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def _series(summary: pd.DataFrame, enc: str, aug: bool, metric: str) -> pd.DataFrame:
    s = summary[(summary["encoder"] == enc) & (summary["augment"] == aug) & (summary["metric"] == metric)]
    return s.sort_values("n_train")


def plot_iou(summary: pd.DataFrame, conds, out: Path, label: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    for enc, aug in conds:
        s = _series(summary, enc, aug, "iou")
        c = COLOR[enc]
        ax.fill_between(s["n_train"], s["mean"] - s["std"], s["mean"] + s["std"], color=c, alpha=0.12, lw=0)
        ax.plot(s["n_train"], s["mean"], STYLE[aug], color=c, lw=1.6, marker="o", ms=4.5, label=cond_label(enc, aug))
    _style_axis(ax, "Mine-class IoU on the test set (mean ± std over seeds)")
    ax.set_xlabel("Training images N (log scale)", color=TEXT2)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="lower right")
    fig.suptitle(label, color=TEXT2, fontsize=9, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_variance(df: pd.DataFrame, conds, out: Path, label: str) -> None:
    """Every seed's IoU as a translucent dot, with the mean line and a translucent ±std band."""
    k = len(conds)
    ncol = 2 if k > 1 else 1
    nrow = math.ceil(k / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.8 * nrow), facecolor=SURFACE,
                             sharex=True, sharey=True, squeeze=False)
    for ax, (enc, aug) in zip(axes.flat, conds):
        sub = df[(df["encoder"] == enc) & (df["augment"] == aug)]
        g = sub.groupby("n_train")["iou"]
        m, sd = g.mean(), g.std(ddof=0)
        c = COLOR[enc]
        ax.fill_between(m.index, m - sd, m + sd, color=c, alpha=0.18, lw=0)
        ax.plot(m.index, m.values, "-", color=c, lw=1.6)
        ax.scatter(sub["n_train"], sub["iou"], s=22, color=c, alpha=0.35, lw=0)
        _style_axis(ax, cond_label(enc, aug))
        ax.set_ylim(0.2, 0.85)
    for ax in axes[-1]:
        ax.set_xlabel("Training images N (log scale)", color=TEXT2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mine-class IoU (test)", color=TEXT2)
    fig.suptitle(f"Seed-to-seed variance of IoU: each dot is one run, band = mean ± std\n{label}",
                 color=TEXT, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_cost(summary: pd.DataFrame, conds, out: Path, label: str) -> None:
    panels = [("steps_done", "Optimiser steps", 1.0), ("gpu_hours", "GPU hours", 1.0),
              ("flops_total", "Total FLOPs (train + eval), TFLOP", 1e12)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURFACE)
    for ax, (m, title, scale) in zip(axes, panels):
        for enc, aug in conds:
            s = _series(summary, enc, aug, m)
            c = COLOR[enc]
            lo = np.maximum((s["mean"] - s["std"]) / scale, 1e-9)
            ax.fill_between(s["n_train"], lo, (s["mean"] + s["std"]) / scale, color=c, alpha=0.12, lw=0)
            ax.plot(s["n_train"], s["mean"] / scale, STYLE[aug], color=c, lw=1.6, marker="o", ms=4.5,
                    label=cond_label(enc, aug))
        _style_axis(ax, title, log_y=True)
        ax.set_xlabel("Training images N (log scale)", color=TEXT2)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper left")
    fig.suptitle(f"Training cost until early stopping (mean ± std over seeds)\n{label}",
                 color=TEXT, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _fmt(mean: float, std: float, digits: int = 3) -> str:
    return "NA" if np.isnan(mean) else f"{mean:.{digits}f} ± {std:.{digits}f}"


def _cell(summary: pd.DataFrame, n: int, enc: str, aug: bool, metric: str, digits: int = 3, scale: float = 1.0) -> str:
    s = summary[(summary["n_train"] == n) & (summary["encoder"] == enc) & (summary["augment"] == aug)
                & (summary["metric"] == metric)]
    return _fmt(s["mean"].iloc[0] / scale, s["std"].iloc[0] / scale, digits) if len(s) else "-"


def markdown_tables(summary: pd.DataFrame, conds, df: pd.DataFrame) -> str:
    seeds = df.groupby("n_train")["seed"].nunique()
    lines = ["### Mine-class IoU on the test set (mean ± std over seeds)", "",
             "| N | seeds | " + " | ".join(cond_label(e, a) for e, a in conds) + " |",
             "|---|---|" + "---|" * len(conds)]
    for n in BUDGETS:
        lines.append(f"| {n_label(n)} | {seeds.get(n, 0)} | "
                     + " | ".join(_cell(summary, n, e, a, "iou") for e, a in conds) + " |")
    lines.append("")
    for enc, aug in conds:
        lines += [f"### Training cost until early stopping — {cond_label(enc, aug)} (mean ± std over seeds)", "",
                  "| N | best epoch | stop epoch | steps | GPU hours | train TFLOP | total TFLOP |",
                  "|---|---|---|---|---|---|---|"]
        for n in BUDGETS:
            lines.append(f"| {n_label(n)} | {_cell(summary, n, enc, aug, 'best_epoch', 1)} | "
                         f"{_cell(summary, n, enc, aug, 'epochs_done', 1)} | "
                         f"{_cell(summary, n, enc, aug, 'steps_done', 0)} | "
                         f"{_cell(summary, n, enc, aug, 'gpu_hours', 3)} | "
                         f"{_cell(summary, n, enc, aug, 'flops_train', 1, 1e12)} | "
                         f"{_cell(summary, n, enc, aug, 'flops_total', 1, 1e12)} |")
        lines.append("")
    gpus = sorted(df["gpu"].dropna().unique()) if "gpu" in df else []
    if gpus:
        lines.append(f"GPU: {', '.join(gpus)}. GPU hours are wall-clock time of the whole run "
                     "(model build, training, every test check). FLOPs are counted with "
                     "torch.utils.flop_counter (matmul/conv operations) and multiplied by the samples processed.")
    return "\n".join(lines) + "\n"


def main() -> int:
    global BUDGETS
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "results" / "main" / "runs.csv")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # tables contain ± and −; Windows consoles default to cp1252
    out_dir = args.runs.parent
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    df = load_runs(args.runs)
    BUDGETS = sorted(int(n) for n in df["n_train"].unique())
    conds = conditions(df)
    seeds = df.groupby("n_train")["seed"].nunique().to_dict()
    print(f"{len(df)} runs loaded; conditions {len(conds)}; seeds per budget {seeds}; "
          f"NA metrics: {int(df[METRICS].isna().sum().sum())}")

    summary = summarise(df)
    summary.to_csv(out_dir / "summary.csv", index=False)
    gain = paired_gain(df)
    if gain is not None:
        gain.to_csv(out_dir / "aug_gain.csv", index=False)
    md = markdown_tables(summary, conds, df)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    label = variant_label(df)
    plot_iou(summary, conds, out_dir / "figures" / "iou_vs_n.png", label)
    plot_variance(df, conds, out_dir / "figures" / "variance.png", label)
    if "gpu_hours" in df:
        plot_cost(summary, conds, out_dir / "figures" / "cost_vs_n.png", label)
    print(md)
    print("OK: wrote summary.csv, summary.md, figures/ ->", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

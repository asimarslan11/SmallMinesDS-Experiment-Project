"""Summarise a runs.csv: curves vs N, paired augmentation gains, and tables.

Usage: .venv\\Scripts\\python scripts\\analyze.py [--runs results/main/runs.csv]

Outputs (next to the runs file)
    summary.csv        mean/std over seeds of every metric per (N, encoder, augmentation)
    aug_gain.csv       per-seed paired gain (aug - no aug) summarised per (N, encoder)
    summary.md         compact markdown tables for the README
    figures/curves.png IoU, F1, precision, recall vs N, mean +- std across seeds
    figures/aug_gain.png paired augmentation gain in IoU and F1 vs N
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from smallmines.train import BUDGETS, CONDITIONS  # noqa: E402

TRAIN_SIZE = 1491  # all SmallMinesDS 2022 training patches


def n_label(n: int) -> str:
    return "all" if n == TRAIN_SIZE else str(n)

METRICS = ["iou", "f1", "precision", "recall"]
ENCODER_LABEL = {"pretrained": "TerraMind pretrained, frozen", "random": "Random init, trainable"}
AUG_LABEL = {False: "no aug", True: "D4 aug"}

# Light-mode palette: encoder -> hue (categorical slots 1 and 2), augmentation -> line style
COLOR = {"pretrained": "#2a78d6", "random": "#eb6834"}
STYLE = {False: "--", True: "-"}
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"


def load_runs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["augment"] = df["augment"].astype(str).str.lower().eq("true")
    return df


def variant_label(df: pd.DataFrame) -> str:
    """Short description of the protocol/decoder used in a runs file, for figure titles."""
    protocol = df["protocol"].iloc[0] if "protocol" in df else "10 epochs"
    norm = df["norm"].iloc[0] if "norm" in df else "batchnorm"
    return f"{protocol}, {'GroupNorm' if norm == 'groupnorm' else 'BatchNorm'} decoder"


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["n_train", "encoder", "augment"])
    rows = []
    for (n, enc, aug), sub in g:
        for m in METRICS + ["train_loss_last", "pred_mine_frac", "seconds"]:
            v = sub[m].astype(float)
            rows.append({"n_train": n, "encoder": enc, "augment": aug, "metric": m,
                         "mean": v.mean(), "std": v.std(ddof=0), "n_seeds": int(v.notna().sum()),
                         "n_na": int(v.isna().sum())})
    return pd.DataFrame(rows)


def paired_gain(df: pd.DataFrame) -> pd.DataFrame:
    """aug - noaug within each (seed, N, encoder); NA if either side is NA."""
    key = ["seed", "n_train", "encoder"]
    a = df[df["augment"]].set_index(key)[METRICS]
    b = df[~df["augment"]].set_index(key)[METRICS]
    d = (a - b).reset_index()
    rows = []
    for (n, enc), sub in d.groupby(["n_train", "encoder"]):
        for m in METRICS:
            v = sub[m].astype(float)
            rows.append({"n_train": n, "encoder": enc, "metric": m, "mean_gain": v.mean(),
                         "std_gain": v.std(ddof=0), "n_positive": int((v > 0).sum()),
                         "n_seeds": int(v.notna().sum())})
    return pd.DataFrame(rows)


def _style_axis(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.set_xscale("log")
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


def plot_curves(summary: pd.DataFrame, out: Path, label: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), facecolor=SURFACE, sharex=True)
    for ax, m in zip(axes.flat, METRICS):
        for enc, aug in CONDITIONS:
            s = summary[(summary["encoder"] == enc) & (summary["augment"] == aug) & (summary["metric"] == m)]
            s = s.sort_values("n_train")
            c = COLOR[enc]
            ax.fill_between(s["n_train"], s["mean"] - s["std"], s["mean"] + s["std"], color=c, alpha=0.12, lw=0)
            ax.plot(s["n_train"], s["mean"], STYLE[aug], color=c, lw=1.6, marker="o", ms=4.5,
                    label=f"{ENCODER_LABEL[enc]}, {AUG_LABEL[aug]}")
        _style_axis(ax, f"Mine-class {m.upper() if m == 'iou' else m.capitalize()}")
        ax.set_ylim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("Training images N (log scale)", color=TEXT2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9,
               labelcolor=TEXT, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Test performance vs training budget (mean ± std over 5 seeds)\n{label}",
                 color=TEXT, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_gain(gain: pd.DataFrame, out: Path, label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), facecolor=SURFACE, sharex=True)
    for ax, m in zip(axes, ["iou", "f1"]):
        ax.axhline(0, color=TEXT2, lw=0.8)
        for enc in ("pretrained", "random"):
            s = gain[(gain["encoder"] == enc) & (gain["metric"] == m)].sort_values("n_train")
            c = COLOR[enc]
            ax.fill_between(s["n_train"], s["mean_gain"] - s["std_gain"], s["mean_gain"] + s["std_gain"],
                            color=c, alpha=0.12, lw=0)
            ax.plot(s["n_train"], s["mean_gain"], "-", color=c, lw=1.6, marker="o", ms=4.5, label=ENCODER_LABEL[enc])
        _style_axis(ax, f"Paired gain in {m.upper() if m == 'iou' else 'F1'} from D4 augmentation")
        ax.set_xlabel("Training images N (log scale)", color=TEXT2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=TEXT,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Augmented minus unaugmented run, same seed and subset (mean ± std over 5 seeds)\n{label}",
                 color=TEXT, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.92))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_variance(df: pd.DataFrame, out: Path, label: str) -> None:
    """Every seed's IoU as a translucent dot, with the mean line and a translucent ±std band."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), facecolor=SURFACE, sharex=True, sharey=True)
    for ax, (enc, aug) in zip(axes.flat, CONDITIONS):
        sub = df[(df["encoder"] == enc) & (df["augment"] == aug)]
        g = sub.groupby("n_train")["iou"]
        m, sd = g.mean(), g.std(ddof=0)
        c = COLOR[enc]
        ax.fill_between(m.index, m - sd, m + sd, color=c, alpha=0.18, lw=0)
        ax.plot(m.index, m.values, "-", color=c, lw=1.6)
        ax.scatter(sub["n_train"], sub["iou"], s=22, color=c, alpha=0.35, lw=0)
        _style_axis(ax, f"{ENCODER_LABEL[enc]}, {AUG_LABEL[aug]}")
        ax.set_ylim(0.2, 0.85)
    for ax in axes[1]:
        ax.set_xlabel("Training images N (log scale)", color=TEXT2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mine-class IoU (test)", color=TEXT2)
    fig.suptitle(f"Seed-to-seed variance of IoU: each dot is one run, band = mean ± std\n{label}",
                 color=TEXT, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _fmt(mean: float, std: float) -> str:
    return "NA" if np.isnan(mean) else f"{mean:.3f} ± {std:.3f}"


def markdown_tables(summary: pd.DataFrame, gain: pd.DataFrame) -> str:
    lines = []
    for m in METRICS:
        lines += [f"### Mine-class {m} (mean ± std over seeds)", "",
                  "| N | " + " | ".join(f"{ENCODER_LABEL[e]}, {AUG_LABEL[a]}" for e, a in CONDITIONS) + " |",
                  "|---|" + "---|" * len(CONDITIONS)]
        for n in BUDGETS:
            cells = []
            for enc, aug in CONDITIONS:
                s = summary[(summary["n_train"] == n) & (summary["encoder"] == enc)
                            & (summary["augment"] == aug) & (summary["metric"] == m)]
                cells.append(_fmt(s["mean"].iloc[0], s["std"].iloc[0]) if len(s) else "-")
            lines.append(f"| {n_label(n)} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["### Paired augmentation gain in IoU (aug − no aug, same seed and subset)", "",
              "| N | " + " | ".join(f"{ENCODER_LABEL[e]} (seeds improved)" for e in ("pretrained", "random")) + " |",
              "|---|---|---|"]
    for n in BUDGETS:
        cells = []
        for enc in ("pretrained", "random"):
            s = gain[(gain["n_train"] == n) & (gain["encoder"] == enc) & (gain["metric"] == "iou")]
            if len(s):
                r = s.iloc[0]
                cells.append(f"{_fmt(r['mean_gain'], r['std_gain'])} ({r['n_positive']}/{r['n_seeds']})")
            else:
                cells.append("-")
        lines.append(f"| {n_label(n)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "results" / "main" / "runs.csv")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # tables contain ± and −; Windows consoles default to cp1252
    out_dir = args.runs.parent
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    df = load_runs(args.runs)
    global BUDGETS
    BUDGETS = sorted(int(n) for n in df["n_train"].unique())  # every budget present in the runs file
    seeds = df.groupby("n_train")["seed"].nunique().to_dict()
    print(f"{len(df)} runs loaded; seeds per budget {seeds}; NA metrics: {int(df[METRICS].isna().sum().sum())}")

    summary = summarise(df)
    gain = paired_gain(df)
    summary.to_csv(out_dir / "summary.csv", index=False)
    gain.to_csv(out_dir / "aug_gain.csv", index=False)
    (out_dir / "summary.md").write_text(markdown_tables(summary, gain), encoding="utf-8")
    label = variant_label(df)
    plot_curves(summary, out_dir / "figures" / "curves.png", label)
    plot_gain(gain, out_dir / "figures" / "aug_gain.png", label)
    plot_variance(df, out_dir / "figures" / "variance.png", label)
    print(markdown_tables(summary, gain))
    print("OK: wrote summary.csv, aug_gain.csv, summary.md, figures/ ->", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

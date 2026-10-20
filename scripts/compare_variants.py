"""Side-by-side IoU curves and a table for the main experiment and the supplementary variants.

Usage: .venv\\Scripts\\python scripts\\compare_variants.py [--root results/v1_10ep]

Reads <root>/<variant>/summary.csv for every variant that exists and writes
<root>/figures/variants.png and <root>/variants.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analyze import AUG_LABEL, COLOR, ENCODER_LABEL, STYLE, SURFACE, TEXT, TEXT2, _style_axis  # noqa: E402
from smallmines.train import BUDGETS, CONDITIONS  # noqa: E402

VARIANTS = [  # (directory, title)
    ("main", "Main protocol, BatchNorm decoder"),
    ("groupnorm", "A: same budget, GroupNorm decoder"),
    ("steps320", "B: 320 steps for every N, GroupNorm decoder"),
]


def load(root: Path) -> list[tuple[str, str, pd.DataFrame]]:
    out = []
    for d, title in VARIANTS:
        p = root / d / "summary.csv"
        if p.exists():
            s = pd.read_csv(p)
            s["augment"] = s["augment"].astype(str).str.lower().eq("true")
            out.append((d, title, s))
    return out


def plot(variants, out: Path) -> None:
    fig, axes = plt.subplots(1, len(variants), figsize=(4.4 * len(variants), 4.2), facecolor=SURFACE, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (_, title, s) in zip(axes, variants):
        for enc, aug in CONDITIONS:
            r = s[(s["encoder"] == enc) & (s["augment"] == aug) & (s["metric"] == "iou")].sort_values("n_train")
            c = COLOR[enc]
            ax.fill_between(r["n_train"], r["mean"] - r["std"], r["mean"] + r["std"], color=c, alpha=0.12, lw=0)
            ax.plot(r["n_train"], r["mean"], STYLE[aug], color=c, lw=1.6, marker="o", ms=4.5,
                    label=f"{ENCODER_LABEL[enc]}, {AUG_LABEL[aug]}")
        _style_axis(ax, title)
        ax.title.set_fontsize(9.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Training images N (log scale)", color=TEXT2)
    axes[0].set_ylabel("Mine-class IoU (test)", color=TEXT2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=TEXT,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Mine-class IoU vs N across protocol variants (mean ± std over 5 seeds)",
                 color=TEXT, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def table(variants) -> str:
    lines = ["### Mine-class IoU (mean ± std over 5 seeds) per variant", ""]
    for _, title, s in variants:
        lines += [f"**{title}**", "",
                  "| N | " + " | ".join(f"{ENCODER_LABEL[e]}, {AUG_LABEL[a]}" for e, a in CONDITIONS) + " |",
                  "|---|" + "---|" * len(CONDITIONS)]
        for n in BUDGETS:
            cells = []
            for enc, aug in CONDITIONS:
                r = s[(s["n_train"] == n) & (s["encoder"] == enc) & (s["augment"] == aug) & (s["metric"] == "iou")]
                cells.append(f"{r['mean'].iloc[0]:.3f} ± {r['std'].iloc[0]:.3f}" if len(r) else "-")
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    import argparse

    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    variants = load(args.root)
    if not variants:
        print(f"no {args.root}/<variant>/summary.csv found; run scripts/analyze.py first")
        return 1
    (args.root / "figures").mkdir(exist_ok=True)
    plot(variants, args.root / "figures" / "variants.png")
    md = table(variants)
    (args.root / "variants.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"OK: {args.root}/figures/variants.png, {args.root}/variants.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

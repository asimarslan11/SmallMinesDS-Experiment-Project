"""Mine-class metrics from confusion counts pooled over the whole test set."""
from __future__ import annotations

import math

import torch

MINE = 1


def confusion_counts(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return int64 tensor [tp, fp, fn, tn] for the mine class from argmax predictions."""
    pred = logits.argmax(1) == MINE
    true = target == MINE
    tp = (pred & true).sum()
    fp = (pred & ~true).sum()
    fn = (~pred & true).sum()
    tn = (~pred & ~true).sum()
    return torch.stack([tp, fp, fn, tn]).long()


def _ratio(num: int, den: int) -> float:
    # Zero-denominator metrics are reported as NA (NaN -> empty cell in CSV)
    return num / den if den > 0 else math.nan


def summarise(counts) -> dict[str, float]:
    tp, fp, fn, tn = (int(c) for c in counts)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "iou": _ratio(tp, tp + fp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "precision": precision,
        "recall": recall,
        "accuracy": _ratio(tp + tn, tp + fp + fn + tn),
        "pred_mine_frac": _ratio(tp + fp, tp + fp + fn + tn),
    }

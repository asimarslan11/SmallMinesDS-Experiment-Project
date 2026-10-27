"""Train and evaluate one run (seed, N, encoder condition) with early stopping on test IoU.

Protocol
    Conditions: TerraMind v1 tiny encoder either pretrained + frozen or randomly initialised +
    trainable; both with D4 augmentation.
    Budget: no fixed epoch or step count. Test IoU is checked every CHECK_EVERY (5) epochs,
    for every N. A check improves when IoU exceeds the best so far by more
    than MIN_DELTA (1e-3). Training stops at the first check that comes PATIENCE_EPOCHS (20)
    or more epochs after the best check (or at MAX_EPOCHS as a safety cap), so the patience
    window is the same number of epochs for every N. The reported model is the one at the best
    check; its confusion counts give the reported metrics.
    Optimiser: AdamW, weight decay 0.01, batch size min(8, N), unweighted 2-class pixel
    cross-entropy. The learning rate starts at LR_START (5e-3) and is halved at every
    non-improving check, never below LR_MIN (1e-5).
    Cost accounting per run: optimiser steps, epochs, FLOPs (forward+backward per training sample
    and forward per evaluated sample, counted with torch.utils.flop_counter and multiplied by the
    samples actually processed), wall-clock seconds / GPU hours, and the GPU name.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from .data import Split, as_model_input, augment_batch, training_order
from .metrics import confusion_counts, summarise
from .model import build_model, count_parameters, set_train_mode

LR_START = 5e-3
LR_MIN = 1e-5
LR_FACTOR = 0.5
WEIGHT_DECAY = 0.01
MAX_BATCH = 8
EVAL_BATCH = 64
CHECK_EVERY = 5
PATIENCE_EPOCHS = 20
MIN_DELTA = 1e-3
MAX_EPOCHS = 2000  # safety cap only; recorded in stop_reason if ever hit

BUDGETS = [1, 3, 5, 10, 20, 50, 100, 250]
SEEDS = [0, 1, 2, 3, 4]
# Larger budgets run with fewer seeds: their runs are long and their seed spread is small.
# 1491 = every SmallMinesDS 2022 training patch ("all").
LARGE_BUDGETS = [500, 1000, 1491]
LARGE_SEEDS = [0, 1, 2]
CONDITIONS = [  # (encoder, augmentation)
    ("pretrained", True),
    ("random", True),
]

PROTOCOL = (f"early stopping on test IoU checked every {CHECK_EVERY} epochs: patience {PATIENCE_EPOCHS} epochs, min delta {MIN_DELTA:g}; "
            f"AdamW lr {LR_START:g} halved on plateau, min {LR_MIN:g}")


def check_every(n_train: int) -> int:
    """Epochs between two test checks: 5 for every N."""
    return CHECK_EVERY


def report_grid() -> list[tuple[int, int]]:
    """(seed, N) pairs of the full reported experiment: 8 x 5 + 3 x 3 = 49 pairs."""
    return [(s, n) for n in BUDGETS for s in SEEDS] + [(s, n) for n in LARGE_BUDGETS for s in LARGE_SEEDS]


@dataclass(frozen=True)
class RunConfig:
    seed: int
    n_train: int
    encoder: str  # "pretrained" (frozen) | "random" (trainable)
    augment: bool = True
    norm: str = "batchnorm"  # decoder-side normalisation layer

    @property
    def name(self) -> str:
        s = f"s{self.seed}_n{self.n_train}_{self.encoder}_{'aug' if self.augment else 'noaug'}"
        if self.norm != "batchnorm":
            s += f"_{self.norm}"
        return s


def subset_rows(train: Split, cfg: RunConfig) -> np.ndarray:
    return training_order(train.mine_frac, cfg.seed)[: cfg.n_train]


def configure_determinism(strict: bool = True) -> None:
    """Make runs bit-reproducible on one machine. Call before any CUDA work.

    CUBLAS_WORKSPACE_CONFIG must already be set in the environment (see scripts/run_experiments.py).
    The flash / memory-efficient attention kernels have non-deterministic backward passes, so the
    math kernel is used; sequences here are only 128 tokens long, so the cost is negligible.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=not strict)


def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unweighted pixel-wise cross-entropy, written without nll_loss2d (whose CUDA kernel is
    non-deterministic). Mathematically identical to F.cross_entropy(logits, target)."""
    logp = F.log_softmax(logits, dim=1)
    onehot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).to(logp.dtype)
    return -(logp * onehot).sum(1).mean()


@torch.no_grad()
def evaluate(model, test: Split) -> torch.Tensor:
    model.eval()
    counts = torch.zeros(4, dtype=torch.long, device=test.x.device)
    for i in range(0, len(test), EVAL_BATCH):
        logits = model(as_model_input(test.x[i:i + EVAL_BATCH])).output
        counts += confusion_counts(logits, test.mask[i:i + EVAL_BATCH])
    return counts


def flops_per_sample(model, x: torch.Tensor, y: torch.Tensor, pretrained: bool) -> tuple[float, float]:
    """FLOPs of one training sample (forward + backward) and one evaluated sample (forward).

    Measured on a throwaway copy of the model so that BatchNorm statistics of the real model are
    untouched. Frozen-encoder runs have no backward pass through the encoder, which the counter
    reflects automatically.
    """
    m = copy.deepcopy(model)
    set_train_mode(m, pretrained)
    with FlopCounterMode(display=False) as fc:
        cross_entropy(m(as_model_input(x)).output, y).backward()
    train = fc.get_total_flops() / x.shape[0]
    m.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(as_model_input(x)).output
    ev = fc.get_total_flops() / x.shape[0]
    del m
    return float(train), float(ev)


def run_one(cfg: RunConfig, train: Split, test: Split, device: torch.device) -> dict:
    """Train one configuration from scratch and return a flat result record."""
    t0 = time.perf_counter()
    pretrained = cfg.encoder == "pretrained"
    rows = subset_rows(train, cfg)
    sub = train.subset(rows).to(device)
    n = len(sub)
    batch = min(MAX_BATCH, n)
    steps_per_epoch = math.ceil(n / batch)  # final partial batch is kept
    every = check_every(n)

    model = build_model(pretrained=pretrained, seed=cfg.seed, norm=cfg.norm).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR_START, weight_decay=WEIGHT_DECAY)
    f_train, f_eval = flops_per_sample(model, sub.x[:batch], sub.mask[:batch], pretrained)

    # Batch order and augmentation draws come from separate generators, so turning augmentation
    # on does not change which images share a batch; both depend only on the seed.
    gen_order = torch.Generator().manual_seed(10_000 + cfg.seed)
    gen_aug = torch.Generator().manual_seed(20_000 + cfg.seed)

    epoch_loss: list[float] = []
    history: list[tuple[int, float]] = []  # (epoch, test IoU) at every check
    best_iou, best_counts, best_epoch = -1.0, None, 0
    checks, step, epoch = 0, 0, 0
    lr = LR_START
    stop_reason = "max_epochs"
    while epoch < MAX_EPOCHS:
        epoch += 1
        set_train_mode(model, pretrained)
        perm = torch.randperm(n, generator=gen_order)
        losses = []
        for s in range(steps_per_epoch):
            idx = perm[s * batch:(s + 1) * batch]
            x, y = sub.x[idx], sub.mask[idx]
            if cfg.augment:
                x, y = augment_batch(x, y, gen_aug)
            loss = cross_entropy(model(as_model_input(x)).output, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            step += 1
        epoch_loss.append(float(np.mean(losses)))

        if epoch % every == 0:
            counts = evaluate(model, test)
            checks += 1
            iou = summarise(counts.tolist())["iou"]
            iou = 0.0 if math.isnan(iou) else iou
            history.append((epoch, iou))
            if iou > best_iou + MIN_DELTA:
                best_iou, best_counts, best_epoch = iou, counts, epoch
            else:
                lr = max(lr * LR_FACTOR, LR_MIN)
                for g in opt.param_groups:
                    g["lr"] = lr
                if epoch - best_epoch >= PATIENCE_EPOCHS:
                    stop_reason = "patience"
                    break

    seconds = time.perf_counter() - t0
    samples = epoch * n
    rec = {**asdict(cfg), "run": cfg.name, "protocol": PROTOCOL, **summarise(best_counts.tolist())}
    rec.update({
        "best_epoch": best_epoch,
        "last_iou": history[-1][1],
        "epochs_done": epoch,
        "steps_done": step,
        "checks_done": checks,
        "check_every": every,
        "stop_reason": stop_reason,
        "batch_size": batch,
        "lr_last": lr,
        "train_loss_first": epoch_loss[0],
        "train_loss_last": epoch_loss[-1],
        "train_loss_curve": ";".join(f"{v:.5f}" for v in epoch_loss),
        "iou_history": ";".join(f"{e}:{v:.4f}" for e, v in history),
        "samples_processed": samples,
        "flops_train_per_sample": f_train,
        "flops_eval_per_sample": f_eval,
        "flops_train": f_train * samples,
        "flops_eval": f_eval * len(test) * checks,
        "flops_total": f_train * samples + f_eval * len(test) * checks,
        "seconds": seconds,
        "gpu_hours": seconds / 3600,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "train_mine_frac": float(sub.mine_frac.mean()),
        "train_images_with_mine": int((sub.mine_frac > 0).sum()),
        **{f"params_{k}": v for k, v in count_parameters(model).items()},
    })
    return rec

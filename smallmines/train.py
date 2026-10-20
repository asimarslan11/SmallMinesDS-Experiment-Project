"""Train and evaluate one run (seed, N, encoder condition, augmentation) per the experiment plan.

Fixed protocol: 10 epochs, evaluate the epoch-10 weights; AdamW lr 1e-4 constant, weight decay
0.01; batch size min(8, N); unweighted 2-class pixel cross-entropy; reshuffle every epoch, keep
the final partial batch; no early stopping, validation or checkpoint selection.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .data import Split, as_model_input, augment_batch, training_order
from .metrics import confusion_counts, summarise
from .model import build_model, count_parameters, set_train_mode

EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 0.01
MAX_BATCH = 8
EVAL_BATCH = 64
BUDGETS = [1, 3, 5, 10, 20, 50, 100, 250]
SEEDS = [0, 1, 2, 3, 4]
CONDITIONS = [  # (encoder, augmentation)
    ("pretrained", False),
    ("pretrained", True),
    ("random", False),
    ("random", True),
]


@dataclass(frozen=True)
class RunConfig:
    seed: int
    n_train: int
    encoder: str  # "pretrained" (frozen) | "random" (trainable)
    augment: bool

    @property
    def name(self) -> str:
        return f"s{self.seed}_n{self.n_train}_{self.encoder}_{'aug' if self.augment else 'noaug'}"


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


def run_one(cfg: RunConfig, train: Split, test: Split, device: torch.device) -> dict:
    """Train one configuration from scratch and return a flat result record."""
    t0 = time.perf_counter()
    pretrained = cfg.encoder == "pretrained"
    rows = subset_rows(train, cfg)
    sub = train.subset(rows).to(device)
    n = len(sub)
    batch = min(MAX_BATCH, n)
    steps_per_epoch = math.ceil(n / batch)  # final partial batch is kept

    model = build_model(pretrained=pretrained, seed=cfg.seed).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    # Batch order and augmentation draws come from separate generators, so turning augmentation
    # on does not change which images share a batch; both depend only on the seed.
    gen_order = torch.Generator().manual_seed(10_000 + cfg.seed)
    gen_aug = torch.Generator().manual_seed(20_000 + cfg.seed)

    epoch_loss = []
    for _ in range(EPOCHS):
        set_train_mode(model, pretrained)
        perm = torch.randperm(n, generator=gen_order)
        losses = []
        for s in range(steps_per_epoch):
            idx = perm[s * batch:(s + 1) * batch]
            x, y = sub.x[idx], sub.mask[idx]
            if cfg.augment:
                x, y = augment_batch(x, y, gen_aug)
            logits = model(as_model_input(x)).output
            loss = cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        epoch_loss.append(float(np.mean(losses)))

    counts = evaluate(model, test)
    rec = {**asdict(cfg), "run": cfg.name, **summarise(counts.tolist())}
    rec.update({
        "steps": EPOCHS * steps_per_epoch,
        "batch_size": batch,
        "train_loss_first": epoch_loss[0],
        "train_loss_last": epoch_loss[-1],
        "train_loss_curve": ";".join(f"{v:.5f}" for v in epoch_loss),
        "train_mine_frac": float(sub.mine_frac.mean()),
        "train_images_with_mine": int((sub.mine_frac > 0).sum()),
        "seconds": time.perf_counter() - t0,
        **{f"params_{k}": v for k, v in count_parameters(model).items()},
    })
    return rec

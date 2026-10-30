# TerraMind v1 Tiny on SmallMinesDS

Training-data efficiency for binary mine segmentation: a frozen, pretrained
[TerraMind v1 tiny](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny) encoder versus
the same encoder architecture trained from scratch (Experiment 2), and a plain UNet with the same
number of parameters trained from scratch (Experiment 3), all with D4 augmentation, on
[SmallMinesDS](https://huggingface.co/datasets/ellaampy/SmallMinesDS) (2022, Sentinel-1 + Sentinel-2).
Each run trains until its test IoU stops improving, and reports IoU together with its cost:
optimiser steps, FLOPs, GPU hours and the GPU used.

Everything needed to reproduce the 147 runs is in this repository: pinned dependencies, a pinned
dataset revision, deterministic training, and the analysis scripts.

## Requirements

- Windows/Linux with an NVIDIA GPU (tested: RTX 3080 Laptop 8 GB) and a driver that supports CUDA 12.8
- Python 3.12
- ~6 GB disk: 1.6 GB dataset zip, 1.8 GB extracted, 1.7 GB cache

## Setup

On Windows, clone into a short path (e.g. `C:\projects\`) or enable long-path support:
some PyTorch/TerraTorch package files exceed the default 260-character path limit and
`pip install` fails in deeply nested folders.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\check_gpu.py        # must print "OK: GPU works with PyTorch."
python scripts\download_data.py    # SmallMinesDS from Hugging Face, pinned revision, checksum verified
python scripts\build_cache.py      # normalised per-split arrays in data/cache/
```

## Run

```powershell
python scripts\run_experiments.py  # full experiment: 147 runs, ~6 h on an RTX 3080 Laptop; resumable
python scripts\analyze.py          # tables + figures in results/main/

# partial runs, e.g. only N <= 250 or one seed
python scripts\run_experiments.py --budgets 1 3 5 10 20 50 100 250
python scripts\run_experiments.py --seeds 0
python scripts\run_experiments.py --encoders unet   # only the UNet (Experiment 3)
```

`run_experiments.py` appends one row per run to the runs file and skips runs already present,
so it can be interrupted and restarted. The repository ships the reported results in
`results/main/runs.csv`, so on a fresh clone the default command finds every run done; to
rerun from scratch, write to a new file, e.g. `python scripts\run_experiments.py --out results/rerun/runs.csv`
(then `python scripts\analyze.py --runs results/rerun/runs.csv`). TerraMind weights are
downloaded from Hugging Face on first use.

## Experiment

Research question: how does training-set size affect mine segmentation for a frozen pretrained
TerraMind v1 tiny encoder versus the same architecture trained from scratch, how do both compare
with a plain UNet of the same size trained from scratch, and what does each cost to train?

| | Model | Init | Training | Augmentation |
|---|---|---|---|---|
| A | TerraMind v1 tiny + UNet decoder (6.90 M) | encoder pretrained | encoder frozen | random D4 symmetry |
| B | TerraMind v1 tiny + UNet decoder (6.90 M) | random | all trainable | random D4 symmetry |
| C | plain UNet (6.83 M) | random | all trainable | random D4 symmetry |

A and B form Experiment 2; C is Experiment 3, run under exactly the same protocol, data,
subsets and seeds.

- Budgets N ∈ {1, 3, 5, 10, 20, 50, 100, 250} training images with 5 seeds, plus
  N ∈ {500, 1000, all = 1491} with 3 seeds; 3 conditions → 147 runs.
- Subsets are nested within a seed and identical across the three conditions.
- **Stopping rule (no fixed epoch or step count).** Test IoU is checked every 5 epochs for
  every N. A check counts as an improvement when IoU exceeds the best so far by more than
  10⁻³. Training stops at the first check that comes 20 or more epochs after the best check
  (4 non-improving checks). The reported model is the one at the best check.
- **Optimiser.** AdamW, weight decay 0.01, batch size min(8, N), unweighted 2-class pixel
  cross-entropy, reshuffle every epoch, final partial batch kept. The learning rate starts at
  5e-3 and is halved at every non-improving check, never below 1e-5.
- **D4 augmentation:** one of the 8 square symmetries (rotations by 0/90/180/270°, each with or
  without a horizontal flip), uniform, re-drawn per image presentation, applied identically to
  S1, S2 and the mask. Test images are never augmented.
- **Metric:** mine-class IoU from confusion counts pooled over the fixed 644-image test set,
  at the best check. F1, precision and recall from the same counts are in `results/main/runs.csv`.
- **Cost per run:** optimiser steps, epochs (best and stop), FLOPs, wall-clock GPU hours and the
  GPU name. FLOPs are counted once per run with `torch.utils.flop_counter` for one training
  sample (forward + backward; the frozen encoder has no backward pass) and one evaluated sample
  (forward), then multiplied by the samples actually processed; `flops_total` includes every
  test check. GPU hours are the wall-clock time of the whole run (model build, training, checks).
- Note: the stopping rule reads the test set, so the reported IoU is the best test value seen
  during training; this was the agreed protocol and applies equally to all conditions.

### Data

- SmallMinesDS 2022: 2135 patches of 128×128 px, train 1491 / test 644 (the dataset's own split).
- Inputs: Sentinel-2 L2A bands B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 and Sentinel-1 RTC
  VV, VH. The DEM band is not used.
- S1 is stored linear in the GeoTIFFs and converted to dB (10·log10) because TerraMind was
  pretrained on dB. Both modalities are standardised with TerraMind's pretraining mean/std
  (`data/cache/normalisation.json`), in every condition.
- Images are used at their native 128×128 (no resizing). TerraMind's fixed sin-cos positional
  table built for 128 px is exactly the top-left 8×8 crop of the 224 px table, so pretrained
  weights load unchanged.

### Model

Conditions A and B, built with [TerraTorch](https://github.com/IBM/terratorch) 1.2.13, identical in both:

- Encoder `terramind_v1_tiny` (5.92 M params): S2L2A + S1RTC tokens are processed as one joint
  sequence; per-layer outputs of the two modalities are averaged (`merge_method="mean"`).
- Decoder side (0.97 M params, randomly initialised in every run): encoder layers 2/5/8/11 →
  8×8 maps → learned pyramid (32/16/8/4 px) → 4-block UNet decoder with 128/64/32/16 channels →
  1×1 conv to 2 classes → bilinear ×2 to 128×128. Normalisation layers are BatchNorm2d
  (`--norm groupnorm` swaps them for GroupNorm).

Condition C is a plain UNet (Ronneberger et al. 2015, `smallmines/model.py`) with 6.83 M params,
within 1 % of A/B's 6.90 M. The 12 input channels are concatenated. Encoder (4.15 M): 5
double-conv stages (3×3 conv + BatchNorm + ReLU, twice) with 30/60/120/240/480 channels and
2×2 max-pooling between them, so 128×128 goes down to an 8×8 bottleneck, the same grid as
TerraMind's tokens. Decoder (2.68 M): 4 transposed-conv upsamplings, each concatenated with the
skip connection and followed by a double conv, then a 1×1 conv to 2 classes. All weights use
PyTorch's default initialisation from the seed.

### Choices the plan left open

- **Low-N subsets.** 34 % of the training images contain no mine pixels. Per seed, the first 10
  images of the ordering are drawn at random from the 140 training images whose mine fraction
  lies in [0.25, 0.75]; the remaining images follow in random order. So N ≤ 10 budgets consist
  entirely of such images and larger budgets contain at least those 10.
  `results/main/subsets.json` lists the patch ids of every (seed, N) subset.
- **Seed.** One seed fixes the subset, the decoder initialisation and the per-epoch batch
  order, so the conditions of a seed are paired; augmentation draws come from a separate
  seeded stream. The random-init encoder and the UNet are also initialised from the seed.
- **Determinism.** cuDNN deterministic mode, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  `torch.use_deterministic_algorithms(True)`, math attention kernel, and a cross-entropy
  written without the non-deterministic `nll_loss2d` kernel. Repeated runs are bit-identical
  on the same machine/software stack (verified on a clean clone of this repository).

## Results

All 147 runs: `results/main/runs.csv` (one row per run with confusion counts, the IoU of every
check, per-epoch training loss and all cost columns), `results/main/summary.md`,
`results/main/environment.json`. Total compute: 6.0 h on an RTX 3080 Laptop GPU
(1.43 GPU hours for the frozen encoder, 2.69 for the trainable one, 1.88 for the UNet). Every run stopped by the
patience rule; no metric is NA.

![IoU vs N](results/main/figures/iou_vs_n.png)
![Seed variance](results/main/figures/variance.png)

### Mine-class IoU on the test set (mean ± std over seeds)

| N | seeds | A: pretrained, frozen + aug | B: random init, trainable + aug | C: plain UNet + aug |
|---|---|---|---|---|
| 1 | 5 | 0.402 ± 0.096 | 0.395 ± 0.041 | 0.579 ± 0.057 |
| 3 | 5 | 0.539 ± 0.020 | 0.482 ± 0.057 | 0.639 ± 0.040 |
| 5 | 5 | 0.563 ± 0.019 | 0.500 ± 0.040 | 0.661 ± 0.031 |
| 10 | 5 | 0.616 ± 0.023 | 0.539 ± 0.027 | 0.687 ± 0.008 |
| 20 | 5 | 0.651 ± 0.017 | 0.541 ± 0.012 | 0.696 ± 0.008 |
| 50 | 5 | 0.666 ± 0.009 | 0.550 ± 0.009 | 0.702 ± 0.009 |
| 100 | 5 | 0.694 ± 0.008 | 0.585 ± 0.034 | 0.723 ± 0.012 |
| 250 | 5 | 0.712 ± 0.003 | 0.616 ± 0.028 | 0.753 ± 0.013 |
| 500 | 3 | 0.729 ± 0.004 | 0.663 ± 0.007 | 0.767 ± 0.005 |
| 1000 | 3 | 0.740 ± 0.002 | 0.680 ± 0.006 | 0.781 ± 0.001 |
| all (1491) | 3 | 0.747 ± 0.001 | 0.688 ± 0.008 | 0.789 ± 0.003 |

### Training cost until early stopping (mean ± std over seeds)

![Cost vs N](results/main/figures/cost_vs_n.png)

A: pretrained, frozen + aug:

| N | best epoch | stop epoch | steps | GPU hours | train TFLOP | total TFLOP |
|---|---|---|---|---|---|---|
| 1 | 43.0 ± 28.0 | 63.0 ± 28.0 | 63 ± 28 | 0.003 ± 0.001 | 0.1 ± 0.1 | 15.4 ± 6.8 |
| 3 | 39.0 ± 9.2 | 59.0 ± 9.2 | 59 ± 9 | 0.002 ± 0.000 | 0.4 ± 0.1 | 14.7 ± 2.3 |
| 5 | 55.0 ± 8.9 | 75.0 ± 8.9 | 75 ± 9 | 0.003 ± 0.000 | 0.9 ± 0.1 | 19.0 ± 2.3 |
| 10 | 63.0 ± 10.8 | 83.0 ± 10.8 | 166 ± 22 | 0.005 ± 0.001 | 2.0 ± 0.3 | 22.0 ± 2.9 |
| 20 | 76.0 ± 13.9 | 96.0 ± 13.9 | 288 ± 42 | 0.006 ± 0.001 | 4.6 ± 0.7 | 27.7 ± 4.0 |
| 50 | 65.0 ± 17.0 | 85.0 ± 17.0 | 595 ± 119 | 0.010 ± 0.002 | 10.1 ± 2.0 | 30.6 ± 6.1 |
| 100 | 57.0 ± 6.8 | 77.0 ± 6.8 | 1001 ± 88 | 0.015 ± 0.001 | 18.3 ± 1.6 | 36.9 ± 3.2 |
| 250 | 62.0 ± 14.4 | 82.0 ± 14.4 | 2624 ± 459 | 0.034 ± 0.006 | 48.7 ± 8.5 | 68.4 ± 12.0 |
| 500 | 53.3 ± 15.5 | 73.3 ± 15.5 | 4620 ± 974 | 0.059 ± 0.012 | 87.0 ± 18.3 | 104.7 ± 22.1 |
| 1000 | 53.3 ± 4.7 | 73.3 ± 4.7 | 9167 ± 589 | 0.116 ± 0.007 | 174.1 ± 11.2 | 191.8 ± 12.3 |
| all (1491) | 55.0 ± 7.1 | 75.0 ± 7.1 | 14025 ± 1322 | 0.172 ± 0.017 | 265.4 ± 25.0 | 283.5 ± 26.7 |

B: random init, trainable + aug:

| N | best epoch | stop epoch | steps | GPU hours | train TFLOP | total TFLOP |
|---|---|---|---|---|---|---|
| 1 | 34.0 ± 23.5 | 54.0 ± 23.5 | 54 ± 24 | 0.002 ± 0.001 | 0.3 ± 0.1 | 13.3 ± 5.8 |
| 3 | 49.0 ± 28.5 | 69.0 ± 28.5 | 69 ± 29 | 0.003 ± 0.001 | 1.1 ± 0.5 | 17.8 ± 7.4 |
| 5 | 40.0 ± 11.4 | 60.0 ± 11.4 | 60 ± 11 | 0.003 ± 0.001 | 1.7 ± 0.3 | 16.1 ± 3.1 |
| 10 | 25.0 ± 17.6 | 45.0 ± 17.6 | 90 ± 35 | 0.004 ± 0.002 | 2.5 ± 1.0 | 13.4 ± 5.2 |
| 20 | 35.0 ± 27.9 | 55.0 ± 27.9 | 165 ± 84 | 0.005 ± 0.002 | 6.1 ± 3.1 | 19.4 ± 9.8 |
| 50 | 39.0 ± 5.8 | 59.0 ± 5.8 | 413 ± 41 | 0.010 ± 0.001 | 16.4 ± 1.6 | 30.6 ± 3.0 |
| 100 | 49.0 ± 26.3 | 69.0 ± 26.3 | 897 ± 342 | 0.021 ± 0.008 | 38.3 ± 14.6 | 54.9 ± 21.0 |
| 250 | 52.0 ± 19.1 | 72.0 ± 19.1 | 2304 ± 612 | 0.047 ± 0.012 | 99.8 ± 26.5 | 117.2 ± 31.1 |
| 500 | 73.3 ± 12.5 | 93.3 ± 12.5 | 5880 ± 786 | 0.122 ± 0.015 | 258.7 ± 34.6 | 281.3 ± 37.6 |
| 1000 | 75.0 ± 25.5 | 95.0 ± 25.5 | 11875 ± 3187 | 0.247 ± 0.069 | 526.7 ± 141.4 | 549.7 ± 147.5 |
| all (1491) | 78.3 ± 8.5 | 98.3 ± 8.5 | 18388 ± 1589 | 0.370 ± 0.031 | 812.9 ± 70.3 | 836.6 ± 72.3 |

C: plain UNet + aug:

| N | best epoch | stop epoch | steps | GPU hours | train TFLOP | total TFLOP |
|---|---|---|---|---|---|---|
| 1 | 54.0 ± 7.3 | 74.0 ± 7.3 | 74 ± 7 | 0.003 ± 0.000 | 1.2 ± 0.1 | 52.5 ± 5.2 |
| 3 | 56.0 ± 15.9 | 76.0 ± 15.9 | 76 ± 16 | 0.004 ± 0.001 | 3.7 ± 0.8 | 56.4 ± 11.8 |
| 5 | 59.0 ± 30.7 | 79.0 ± 30.7 | 79 ± 31 | 0.004 ± 0.002 | 6.3 ± 2.5 | 61.2 ± 23.8 |
| 10 | 46.0 ± 11.6 | 66.0 ± 11.6 | 132 ± 23 | 0.005 ± 0.001 | 10.6 ± 1.9 | 56.4 ± 9.9 |
| 20 | 60.0 ± 15.2 | 80.0 ± 15.2 | 240 ± 45 | 0.007 ± 0.001 | 25.7 ± 4.9 | 81.2 ± 15.4 |
| 50 | 59.0 ± 9.2 | 79.0 ± 9.2 | 553 ± 64 | 0.011 ± 0.001 | 63.4 ± 7.4 | 118.2 ± 13.7 |
| 100 | 68.0 ± 18.9 | 88.0 ± 18.9 | 1144 ± 245 | 0.021 ± 0.005 | 141.3 ± 30.3 | 202.3 ± 43.4 |
| 250 | 63.0 ± 12.9 | 83.0 ± 12.9 | 2656 ± 412 | 0.045 ± 0.007 | 333.1 ± 51.7 | 390.7 ± 60.7 |
| 500 | 55.0 ± 7.1 | 75.0 ± 7.1 | 4725 ± 445 | 0.079 ± 0.008 | 602.1 ± 56.8 | 654.1 ± 61.7 |
| 1000 | 55.0 ± 4.1 | 75.0 ± 4.1 | 9375 ± 510 | 0.150 ± 0.007 | 1204.1 ± 65.5 | 1256.2 ± 68.4 |
| all (1491) | 53.3 ± 10.3 | 73.3 ± 10.3 | 13713 ± 1921 | 0.227 ± 0.042 | 1755.5 ± 245.9 | 1806.3 ± 253.1 |

GPU: NVIDIA GeForce RTX 3080 Laptop GPU. Per-sample cost: 2.37 GFLOP per training sample for
the frozen encoder (decoder forward + backward only), 5.54 GFLOP for the trainable one and
16.06 GFLOP for the UNet, whose convolutions run at up to full 128×128 resolution; 1.87 GFLOP
per evaluated sample for A and B, 5.39 for C. Test checks dominate the FLOPs at small N
(644 images per check).

### Findings

- **The plain UNet (Experiment 3) gives the highest IoU at every N, in every seed**, ahead of
  both TerraMind models: 0.789 vs 0.747 (frozen) and 0.688 (trainable) with all images, and
  0.579 vs 0.402 / 0.395 with a single image. With 10 images it (0.687) is level with the
  frozen encoder at 100 (0.694), and with 250 images (0.753) it exceeds the frozen encoder with
  all 1,491. Its seed spread is small from N = 10 on (std ≤ 0.013).
- **Cost of the UNet.** It needs 6.8× more FLOPs per training sample than the frozen encoder;
  with all images 1,806 vs 284 TFLOP, and 0.227 vs 0.172 GPU hours.
- **The frozen pretrained encoder wins at every N ≥ 5 in every seed** (3/5 seeds at N = 1 and
  3), by +0.06 IoU at N = 3–5, +0.08–0.12 at N = 10–250 and +0.06 from N = 500 on
  (0.747 vs 0.688 with all images).
- **Data efficiency.** With 100 images the frozen encoder (0.694) already exceeds what the
  trainable encoder reaches with all 1,491 (0.688).
- **IoU rises with N for both encoders** and flattens for the frozen one from N = 500
  (0.729 → 0.740 → 0.747).
- **Cost.** The frozen encoder trains with 2.3× fewer FLOPs per sample; with all images it
  needs 0.172 vs 0.370 GPU hours and 284 vs 837 TFLOP. At N ≤ 100 both stop after a similar
  number of epochs and cost about the same.
- **Seed variance** is small for the frozen encoder from N = 3 on (std ≤ 0.023, ≤ 0.004 from
  N = 250) and larger for the trainable one (up to 0.057).

## Earlier protocols (archived)

Earlier grids are kept for reference and are not part of the report:

- `results/es_check1/main/`: the same early-stopping protocol, but with the test check after
  every epoch for N > 100. Because the learning rate is halved at every non-improving check,
  it decayed up to five times faster per epoch there, and IoU at N ≥ 250 was 0.02–0.06 lower
  (frozen encoder: 0.653 at N = 250, 0.723 with all images).
- `results/es_4checks/main/`: the same early-stopping protocol with patience defined as
  4 checks instead of 20 epochs (identical for N ≤ 100; at N > 100 runs stopped after
  10–22 epochs).
- `results/fixed_50ep/main/`: 4 conditions (with and without augmentation), 50 epochs, cosine
  learning rate 5e-3 → 1e-5, 196 runs. It showed that augmentation helps the frozen encoder
  consistently (+0.02 to +0.04 IoU) and not the trainable one, which is why the current
  experiment keeps only the augmented conditions. Best IoU there: 0.745 (frozen + aug, all images).
- `results/v1_10ep/`: the plan document's original schedule (10 epochs, constant lr 1e-4) with
  two diagnostic variants; see `results/v1_10ep/variants.md`.

## Layout

```
smallmines/          data.py (loading, normalisation, subsets, D4), model.py, train.py, metrics.py
scripts/             check_gpu.py, download_data.py, build_cache.py, run_experiments.py,
                     analyze.py, compare_variants.py
results/main/        current experiment: runs.csv, summary.csv, summary.md, subsets.json,
                     environment.json, figures/
results/es_check1/   archived early-stopping grid with a check after every epoch for N > 100
results/es_4checks/  archived early-stopping grid with 4-check patience
results/fixed_50ep/  archived 50-epoch grid (4 conditions)
results/v1_10ep/     archived 10-epoch grid and its variants
report/              final report (TerraMind_SmallMinesDS_Final_Report.pdf) and the Experiment 2 report
data/                (git-ignored) dataset, split CSVs, cache
```

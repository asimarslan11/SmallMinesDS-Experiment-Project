"""TerraMind v1 tiny encoder + shallow UNet decoder, built with TerraTorch; plus a plain UNet of
the same parameter count as the baseline of Experiment 3 (see UNet below).

Architecture (identical in every condition)
    encoder : terramind_v1_tiny, inputs S2L2A (10 bands) + S1RTC (2 bands), img_size 128,
              patch 16 -> 8x8 tokens of dim 192; the two modalities are fused inside the
              encoder (joint token sequence) and merged at the output by averaging.
    neck    : layers 2, 5, 8, 11 -> reshaped to 8x8 maps -> learned pyramid 32/16/8/4 px
    decoder : 4-block UNet decoder (smp) with channels 128/64/32/16 -> 64x64
    head    : 1x1 conv to 2 classes, bilinear x2 to 128x128

Normalisation layers on the decoder side are BatchNorm2d (TerraTorch/smp default) in the main
experiment; norm="groupnorm" swaps every one of them for GroupNorm (supplementary experiment).
"""
from __future__ import annotations

import warnings

import torch
from torch import nn

from .data import IMG_SIZE, S2_BANDS

ENCODER_LAYERS = [2, 5, 8, 11]
DECODER_CHANNELS = [128, 64, 32, 16]
NUM_CLASSES = 2
NORMS = ("batchnorm", "groupnorm")
GROUPNORM_GROUPS = 8  # every decoder-side channel count (16..192) is divisible by 8


def _swap_batchnorm_for_groupnorm(module: nn.Module) -> int:
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(GROUPNORM_GROUPS, child.num_features))
            n += 1
        else:
            n += _swap_batchnorm_for_groupnorm(child)
    return n


def build_model(pretrained: bool, seed: int, norm: str = "batchnorm") -> nn.Module:
    """Build the segmentation model for one condition.

    pretrained=True : TerraMind weights loaded, encoder frozen (requires_grad False, eval mode).
    pretrained=False: encoder randomly initialised from `seed`, trainable.
    The neck/decoder/head are re-initialised from a seed-derived generator *after* the build, so
    their initial weights are identical across all four conditions of the same seed.
    """
    from terratorch.models import EncoderDecoderFactory

    if norm not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")

    torch.manual_seed(seed)  # random-init encoder weights (overwritten when pretrained)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = EncoderDecoderFactory().build_model(
            task="segmentation",
            backbone="terramind_v1_tiny",
            backbone_pretrained=pretrained,
            backbone_modalities=["S2L2A", "S1RTC"],
            backbone_bands={"S2L2A": S2_BANDS},
            backbone_merge_method="mean",
            backbone_img_size=IMG_SIZE,
            necks=[
                {"name": "SelectIndices", "indices": ENCODER_LAYERS},
                {"name": "ReshapeTokensToImage", "remove_cls_token": False},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
            decoder="UNetDecoder",
            decoder_channels=DECODER_CHANNELS,
            num_classes=NUM_CLASSES,
            head_dropout=0.0,
            rescale=True,
        )

    decoder_side = (model.neck, model.decoder, model.head)
    if norm == "groupnorm":
        swapped = sum(_swap_batchnorm_for_groupnorm(part) for part in decoder_side)
        assert swapped > 0, "no BatchNorm2d layers found to replace"

    torch.manual_seed(30_000 + seed)
    for part in decoder_side:
        for m in part.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    if pretrained:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    return model


UNET_BASE = 30  # widths 30/60/120/240/480 -> 6.83 M parameters, within 1% of TerraMind + decoder (6.90 M)


class UNetOutput:
    """Mimics TerraTorch's ModelOutput so the training loop can use `.output` for both models."""

    def __init__(self, output: torch.Tensor):
        self.output = output


def _double_conv(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class _UNetEncoder(nn.Module):
    def __init__(self, c_in: int, widths: list[int]):
        super().__init__()
        self.stem = _double_conv(c_in, widths[0])
        self.downs = nn.ModuleList(
            nn.Sequential(nn.MaxPool2d(2), _double_conv(a, b)) for a, b in zip(widths[:-1], widths[1:]))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = [self.stem(x)]
        for down in self.downs:
            feats.append(down(feats[-1]))
        return feats


class _UNetDecoder(nn.Module):
    def __init__(self, widths: list[int], num_classes: int):
        super().__init__()
        rev = widths[::-1]
        self.ups = nn.ModuleList(nn.ConvTranspose2d(a, b, 2, stride=2) for a, b in zip(rev[:-1], rev[1:]))
        self.convs = nn.ModuleList(_double_conv(2 * b, b) for b in rev[1:])
        self.head = nn.Conv2d(widths[0], num_classes, 1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        x = feats[-1]
        for up, conv, skip in zip(self.ups, self.convs, feats[-2::-1]):
            x = conv(torch.cat([up(x), skip], dim=1))
        return self.head(x)


class UNet(nn.Module):
    """Plain UNet (Ronneberger et al. 2015) on the same 12 input channels, trained from scratch.

    Four 2x poolings take 128x128 to an 8x8 bottleneck, the same grid as TerraMind's tokens.
    Every conv is 3x3 + BatchNorm + ReLU; upsampling by transposed conv, skip by concatenation.
    """

    def __init__(self, base: int = UNET_BASE, c_in: int = len(S2_BANDS) + 2, num_classes: int = NUM_CLASSES):
        super().__init__()
        widths = [base * 2 ** i for i in range(5)]
        self.encoder = _UNetEncoder(c_in, widths)
        self.decoder = _UNetDecoder(widths, num_classes)

    def forward(self, x: dict[str, torch.Tensor]) -> UNetOutput:
        return UNetOutput(self.decoder(self.encoder(torch.cat([x["S2L2A"], x["S1RTC"]], dim=1))))


def build_unet(seed: int) -> nn.Module:
    """UNet with every weight initialised from `seed` (PyTorch default initialisation)."""
    torch.manual_seed(seed)
    return UNet()


def set_train_mode(model: nn.Module, pretrained: bool) -> None:
    model.train()
    if pretrained:
        model.encoder.eval()  # frozen encoder keeps its pretrained state


def decoder_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if not n.startswith("encoder.")]


def count_parameters(model: nn.Module) -> dict[str, int]:
    enc = sum(p.numel() for p in model.encoder.parameters())
    dec = sum(p.numel() for p in decoder_parameters(model))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"encoder": enc, "decoder": dec, "trainable": trainable}

"""TerraMind v1 tiny encoder + shallow UNet decoder, built with TerraTorch.

Architecture (identical in every condition)
    encoder : terramind_v1_tiny, inputs S2L2A (10 bands) + S1RTC (2 bands), img_size 128,
              patch 16 -> 8x8 tokens of dim 192; the two modalities are fused inside the
              encoder (joint token sequence) and merged at the output by averaging.
    neck    : layers 2, 5, 8, 11 -> reshaped to 8x8 maps -> learned pyramid 32/16/8/4 px
    decoder : 4-block UNet decoder (smp) with channels 128/64/32/16 -> 64x64
    head    : 1x1 conv to 2 classes, bilinear x2 to 128x128
"""
from __future__ import annotations

import warnings

import torch
from torch import nn

from .data import IMG_SIZE, S2_BANDS

ENCODER_LAYERS = [2, 5, 8, 11]
DECODER_CHANNELS = [128, 64, 32, 16]
NUM_CLASSES = 2


def build_model(pretrained: bool, seed: int) -> nn.Module:
    """Build the segmentation model for one condition.

    pretrained=True : TerraMind weights loaded, encoder frozen (requires_grad False, eval mode).
    pretrained=False: encoder randomly initialised from `seed`, trainable.
    The neck/decoder/head are re-initialised from a seed-derived generator *after* the build, so
    their initial weights are identical across all four conditions of the same seed.
    """
    from terratorch.models import EncoderDecoderFactory

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

    torch.manual_seed(30_000 + seed)
    for part in (model.neck, model.decoder, model.head):
        for m in part.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    if pretrained:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    return model


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

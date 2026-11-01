"""TerraMind v1 tiny encoder + shallow UNet decoder, built with TerraTorch; two other frozen
foundation-model encoders with the same decoder (DOFA base, SSL4EO-S12 ResNet50; see
build_foundation_model); and plain UNets as from-scratch baselines (see UNet below).

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

from .data import IMG_SIZE, S1_BANDS, S2_BANDS, S2_MEAN, S2_STD

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


# Plain UNet base widths: 30 -> 6.83 M params, within 1% of TerraMind + decoder (6.90 M), Experiment 3;
# 21 -> 3.35 M, half of that (Experiment 3 v2). Widths double at each of the 5 stages.
UNET_BASE = 30
UNET_HALF_BASE = 21


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


def build_unet(seed: int, base: int = UNET_BASE) -> nn.Module:
    """UNet with every weight initialised from `seed` (PyTorch default initialisation)."""
    torch.manual_seed(seed)
    return UNet(base=base)


_S2_MEAN_T = torch.tensor(S2_MEAN).view(1, -1, 1, 1)
_S2_STD_T = torch.tensor(S2_STD).view(1, -1, 1, 1)


def _dofa_input(x: dict[str, torch.Tensor]) -> torch.Tensor:
    # DOFA was pretrained on per-band standardised inputs: the cached z-scores are used as they are
    return torch.cat([x["S2L2A"], x["S1RTC"]], dim=1)


def _ssl4eo_s2_input(x: dict[str, torch.Tensor]) -> torch.Tensor:
    # SSL4EO-S12 MoCo weights expect S2 reflectance / 10000; undo the cached standardisation
    s2 = x["S2L2A"]
    return (s2 * _S2_STD_T.to(s2.device) + _S2_MEAN_T.to(s2.device)) / 10000.0


FOUNDATION_MODELS = {
    # name: (TerraTorch backbone, its band names, backbone kwargs, necks, input adapter)
    "dofa": ("dofa_base_patch16_224", S2_BANDS + S1_BANDS,
             {"backbone_img_size": IMG_SIZE, "backbone_out_indices": ENCODER_LAYERS},
             [{"name": "ReshapeTokensToImage", "remove_cls_token": True},
              {"name": "LearnedInterpolateToPyramidal"}], _dofa_input),
    "resnet50": ("ssl4eos12_resnet50_sentinel2_all_moco", S2_BANDS,
                 {"backbone_out_indices": [1, 2, 3, 4]}, [], _ssl4eo_s2_input),
}


class AdaptedModel(nn.Module):
    """A TerraTorch model fed from the modality dict through a model-specific input adapter."""

    def __init__(self, model: nn.Module, adapter):
        super().__init__()
        self.model = model
        self.adapter = adapter

    @property
    def encoder(self) -> nn.Module:
        return self.model.encoder

    def forward(self, x: dict[str, torch.Tensor]):
        return self.model(self.adapter(x))


def build_foundation_model(name: str, seed: int) -> nn.Module:
    """Frozen pretrained DOFA base (S2 + S1, 12 bands) or SSL4EO-S12 ResNet50 MoCo (S2, 10 bands)
    encoder with the same UNet decoder (128/64/32/16) and head as the TerraMind model.

    DOFA: layers 2/5/8/11 -> 8x8 maps (CLS token dropped) -> learned pyramid 32/16/8/4 px.
    ResNet50: its own stages at 32/16/8/4 px (strides 4-32). The decoder side is re-initialised
    from the same seed-derived generator as in build_model.
    """
    from terratorch.models import EncoderDecoderFactory

    backbone, bands, kwargs, necks, adapter = FOUNDATION_MODELS[name]
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = EncoderDecoderFactory().build_model(
            task="segmentation", backbone=backbone, backbone_pretrained=True,
            backbone_model_bands=bands, **kwargs, necks=necks or None,
            decoder="UNetDecoder", decoder_channels=DECODER_CHANNELS, num_classes=NUM_CLASSES,
            head_dropout=0.0, rescale=True,
        )
    torch.manual_seed(30_000 + seed)
    # without necks (ResNet50) TerraTorch's neck is a plain function, not a module
    for part in (model.neck, model.decoder, model.head):
        for m in part.modules() if isinstance(part, nn.Module) else ():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    return AdaptedModel(model, adapter)


def set_train_mode(model: nn.Module, pretrained: bool) -> None:
    model.train()
    if pretrained:
        model.encoder.eval()  # frozen encoder keeps its pretrained state


def decoder_parameters(model: nn.Module):
    enc = {id(p) for p in model.encoder.parameters()}
    return [p for p in model.parameters() if id(p) not in enc]


def count_parameters(model: nn.Module) -> dict[str, int]:
    enc = sum(p.numel() for p in model.encoder.parameters())
    dec = sum(p.numel() for p in decoder_parameters(model))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"encoder": enc, "decoder": dec, "trainable": trainable}

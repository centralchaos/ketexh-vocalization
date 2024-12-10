"""Enhanced VGG implementation for audio processing"""

from app.models.vgg import VGG, make_layers, cfgs
from typing import Any

def vgg11_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    """Enhanced VGG 11-layer model with batch normalization for audio processing"""
    model = VGG(make_layers(cfgs['A'], batch_norm=True), **kwargs)
    return model 
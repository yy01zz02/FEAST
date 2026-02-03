# Copyright (c) [2025] [FastVideo Team]
# SPDX-License-Identifier: [Apache License 2.0]
#
# Utility functions for loading Flux models.

import os
from pathlib import Path

import torch
from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel,
    FluxTransformerBlock,
    FluxSingleTransformerBlock,
)
from fastvideo.utils.logging_ import main_print


def get_no_split_modules(transformer):
    """Get the modules that should not be split for FSDP."""
    if isinstance(transformer, FluxTransformer2DModel):
        return (FluxTransformerBlock, FluxSingleTransformerBlock)
    else:
        raise ValueError(f"Unsupported transformer type: {type(transformer)}")

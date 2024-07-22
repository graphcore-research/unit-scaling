# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Unit-scaled versions of common torch.nn modules."""

# This all has to be done manually to keep mypy happy.
# Removing the `--no-implicit-reexport` option ought to fix this, but doesn't appear to.

from . import core, optim, parameter
from ._modules import (
    GELU,
    MHSA,
    MLP,
    CrossEntropyLoss,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    RMSNorm,
    SiLU,
    Softmax,
    TransformerDecoder,
    TransformerLayer,
    Trunk,
)
from .analysis import visualiser
from .parameter import MupType, Parameter

__all__ = [
    "CrossEntropyLoss",
    "Dropout",
    "Embedding",
    "GELU",
    "LayerNorm",
    "Linear",
    "MHSA",
    "MLP",
    "MupType",
    "Parameter",
    "RMSNorm",
    "SiLU",
    "Softmax",
    "TransformerDecoder",
    "TransformerLayer",
    "Trunk",
    "core",
    "optim",
    "parameter",
    "visualiser",
]

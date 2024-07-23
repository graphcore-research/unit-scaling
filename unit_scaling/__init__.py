# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Unit-scaled versions of common torch.nn modules."""

# This all has to be done manually to keep mypy happy.
# Removing the `--no-implicit-reexport` option ought to fix this, but doesn't appear to.

from . import core, functional, optim, parameter
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
    # Modules
    "CrossEntropyLoss",
    "Dropout",
    "Embedding",
    "GELU",
    "LayerNorm",
    "Linear",
    "MHSA",
    "MLP",
    "MupType",
    "RMSNorm",
    "SiLU",
    "Softmax",
    "TransformerDecoder",
    "TransformerLayer",
    "Trunk",
    # Modules
    "core",
    "functional",
    "optim",
    "parameter",
    # Functions
    "Parameter",
    "visualiser",
]

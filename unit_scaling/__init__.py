# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Unit-scaled versions of common torch.nn modules."""

# This all has to be done manually to keep mypy happy.
# Removing the `--no-implicit-reexport` option ought to fix this, but doesn't appear to.

from . import transforms
from ._modules import (
    GELU,
    MHSA,
    MLP,
    CrossEntropyLoss,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    Softmax,
    TransformerDecoder,
    TransformerLayer,
)

__all__ = [
    "GELU",
    "MHSA",
    "MLP",
    "CrossEntropyLoss",
    "Dropout",
    "Embedding",
    "LayerNorm",
    "Linear",
    "Softmax",
    "TransformerDecoder",
    "TransformerLayer",
]

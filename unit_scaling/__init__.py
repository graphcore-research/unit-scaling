# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Unit-scaled versions of common torch.nn modules."""

# This all has to be done manually to keep mypy happy.
# Removing the `--no-implicit-reexport` option ought to fix this, but doesn't appear to.

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
from ._version import __version__
from .analysis import visualiser

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
    "visualiser",
    "__version__",
]

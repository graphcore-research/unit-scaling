# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""A replacement for `torch.nn` with unit-scaled versions of various classes &
functions."""

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

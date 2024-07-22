# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Unit-scaled versions of common torch.nn modules."""

# This all has to be done manually to keep mypy happy.
# Removing the `--no-implicit-reexport` option ought to fix this, but doesn't appear to.

from . import optim
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
from ._parameter import MupType, Parameter, ParameterData, has_parameter_data
from .analysis import visualiser

__all__ = [
    "CrossEntropyLoss",
    "Dropout",
    "Embedding",
    "GELU",
    "has_parameter_data",
    "LayerNorm",
    "Linear",
    "MHSA",
    "MLP",
    "MupType",
    "Parameter",
    "ParameterData",
    "RMSNorm",
    "SiLU",
    "Softmax",
    "TransformerDecoder",
    "TransformerLayer",
    "Trunk",
    "optim",
    "visualiser",
]

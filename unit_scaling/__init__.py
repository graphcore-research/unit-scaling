# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Unit-scaled versions of common torch.nn modules."""

# This all has to be done manually to keep mypy happy.
# Removing the `--no-implicit-reexport` option ought to fix this, but doesn't appear to.

from . import core, functional, optim, parameter
from ._modules import (
    GELU,
    MHSA,
    MLP,
    Conv1d,
    CrossEntropyLoss,
    DepthModuleList,
    DepthSequential,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    LinearReadout,
    RMSNorm,
    SiLU,
    Softmax,
    TransformerDecoder,
    TransformerLayer,
)
from ._version import __version__
from .core.functional import transformer_residual_scaling_rule
from .parameter import MupType, Parameter

__all__ = [
    # Modules
    "Conv1d",
    "CrossEntropyLoss",
    "DepthModuleList",
    "DepthSequential",
    "Dropout",
    "Embedding",
    "GELU",
    "LayerNorm",
    "Linear",
    "LinearReadout",
    "MHSA",
    "MLP",
    "MupType",
    "RMSNorm",
    "SiLU",
    "Softmax",
    "TransformerDecoder",
    "TransformerLayer",
    # Modules
    "core",
    "functional",
    "optim",
    "parameter",
    # Functions
    "Parameter",
    "transformer_residual_scaling_rule",
    "__version__",
]

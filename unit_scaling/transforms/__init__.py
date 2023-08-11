# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Useful torch dynamo transforms of modules for the sake of numerics and unit
scaling."""

# This all has to be done manually to keep mypy happy.
# Removing the `--no-implicit-reexport` option ought to fix this, but doesn't appear to.

from ._compile import compile
from ._simulate_format import simulate_format, simulate_fp8
from ._track_scales import (
    Metrics,
    prune_non_float_tensors,
    prune_same_scale_tensors,
    prune_selected_nodes,
    track_scales,
)
from ._unit_scale import unit_scale

__all__ = [
    "Metrics",
    "compile",
    "prune_non_float_tensors",
    "prune_same_scale_tensors",
    "prune_selected_nodes",
    "simulate_format",
    "simulate_fp8",
    "track_scales",
    "unit_scale",
]

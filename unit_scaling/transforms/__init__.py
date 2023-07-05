# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Useful torch dynamo transforms of modules for the sake of numerics and unit
scaling."""

# This all has to be done manually to keep mypy happy.
# Removing the `--no-implicit-reexport` option ought to fix this, but doesn't appear to.

from ._simulate_format import simulate_format, simulate_fp8

__all__ = [
    "simulate_format",
    "simulate_fp8",
]

# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Unit-scaled versions of common `torch.nn` modules."""

from typing import Any, Callable, Optional

from torch import Tensor, nn

from . import functional as U
from .constraints import gmean
from .docs import inherit_docstring, unary_constraint_docstring


@inherit_docstring(
    short_description=(
        "Applies a **unit-scaled** linear transformation to the incoming data."
    ),
    add_args=[unary_constraint_docstring],
)
class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
        constraint: Optional[Callable[[float, float], float]] = gmean,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.constraint = constraint

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input: Tensor) -> Tensor:
        return U.linear(input, self.weight, self.bias, self.constraint)

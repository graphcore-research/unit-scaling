# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Unit-scaled versions of common `torch.nn` modules."""

from typing import Any, Optional

from torch import Tensor, nn

from . import functional as U
from .constraints import BinaryConstraint, gmean
from .docs import binary_constraint_docstring, format_docstring, inherit_docstring


@inherit_docstring(
    short_description="Applies a **unit-scaled** Gaussian Error Linear Units function:",
    add_args=[binary_constraint_docstring],
)
class GELU(nn.GELU):
    def __init__(
        self,
        approximate: str = "none",
        constraint: Optional[BinaryConstraint] = gmean,
    ) -> None:
        super().__init__(approximate)
        self.constraint = constraint

    def forward(self, input: Tensor) -> Tensor:
        return U.gelu(input, self.constraint)


@inherit_docstring(
    short_description=(
        "Applies a **unit-scaled** linear transformation to the incoming data."
    ),
    add_args=[binary_constraint_docstring],
)
class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
        constraint: Optional[BinaryConstraint] = gmean,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.constraint = constraint

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input: Tensor) -> Tensor:
        return U.linear(input, self.weight, self.bias, self.constraint)


@format_docstring(binary_constraint_docstring)
class MLP(nn.Module):
    """A **unit-scaled** implementation of an MLP layer.

    Args:
        hidden_size (int): the hidden dimension size of the input.
        act_fn (nn.Module): the activation function module.
            Defaults to `GELU()`.
        expansion_factor (int): the factor by which the MLP's intermediate size
            increases relative to `hidden_size`.
        {0}
    """

    def __init__(
        self,
        hidden_size: int,
        act_fn: nn.Module = GELU(),
        expansion_factor: int = 4,
        constraint: Optional[BinaryConstraint] = gmean,
    ) -> None:
        super().__init__()
        intermediate_size = hidden_size * expansion_factor
        self.linear_1 = Linear(hidden_size, intermediate_size, constraint=constraint)
        self.act_fn = act_fn
        self.linear_2 = Linear(intermediate_size, hidden_size, constraint=constraint)

    def forward(self, input: Tensor) -> Tensor:
        input = self.linear_1(input)
        input = self.act_fn(input)
        return self.linear_2(input)  # type: ignore

# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Unit-scaled versions of common `torch.nn` modules."""

from typing import Any, Optional

import einops
from torch import Tensor, nn

from . import functional as U
from .constraints import BinaryConstraint, VariadicConstraint, gmean
from .docs import (
    binary_constraint_docstring,
    format_docstring,
    inherit_docstring,
    variadic_constraint_docstring,
)


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
        "Applies a **unit-scaled** Softmax function to an n-dimensional input Tensor."
        "\n\n"
        "The standard softmax rescales values so that the elements of the"
        " n-dimensional output Tensor lie in the range [0,1] and sum to 1. Unit scaling"
        " multiplies by n, meaning the output Tensor lies in the range [0,n]."
        "\n\n"
        "The docs below are from the standard `nn.Softmax`"
        " implementation. Values therein (e.g. [0,1] ranges) should be adjusted"
        " accordingly."
    ),
    add_args=[binary_constraint_docstring],
)
class Softmax(nn.Softmax):
    def __init__(
        self,
        dim: Optional[int] = None,
        constraint: Optional[BinaryConstraint] = gmean,
    ) -> None:
        super().__init__(dim=dim)
        self.constraint = constraint

    def forward(self, input: Tensor) -> Tensor:
        return U.softmax(input, dim=self.dim, constraint=self.constraint)


@inherit_docstring(short_description="A **unit-scaled** implementation of Dropout.")
class Dropout(nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p, inplace)

    def forward(self, input: Tensor) -> Tensor:
        return U.dropout(input, self.p, self.training, self.inplace)


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


@inherit_docstring(
    short_description=(
        "Applies a **unit-scaled** Layer Normalization over a mini-batch of inputs."
    ),
)
class LayerNorm(nn.LayerNorm):
    def forward(self, input: Tensor) -> Tensor:
        return U.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )


@format_docstring(binary_constraint_docstring)
class MLP(nn.Module):
    """A **unit-scaled** implementation of an MLP layer.

    Args:
        hidden_size (int): the hidden dimension size of the input.
        act_fn (nn.Module): the activation function module. Defaults to `GELU()`.
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


@format_docstring(variadic_constraint_docstring)
class MHSA(nn.Module):
    """A **unit-scaled** implementation of a multi-head self-attention layer.

    Args:
        hidden_size (int): the hidden dimension size of the input.
        heads (int): the number of attention heads.
        dropout_p (float, optional): the probability of the post-softmax dropout.
            Defaults to 0.1.
        {0}
    """

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        dropout_p: float = 0.1,
        constraint: Optional[VariadicConstraint] = gmean,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dropout_p = dropout_p
        self.linear_qkv = Linear(
            hidden_size, 3 * hidden_size, bias=False, constraint=constraint
        )
        self.linear_o = Linear(
            hidden_size, hidden_size, bias=False, constraint=constraint
        )
        self.constraint = constraint

    def forward(self, input: Tensor) -> Tensor:
        qkv = self.linear_qkv(input)
        q, k, v = einops.rearrange(qkv, "b s (d z h) -> z b h s d", h=self.heads, z=3)
        qk = U.matmul(q, k.transpose(-1, -2), constraint=self.constraint)
        qk = U.softmax(qk, dim=-1, constraint=self.constraint)
        qk = U.dropout(qk, self.dropout_p, training=self.training)
        qkv = U.matmul(qk, v, constraint=self.constraint)
        qkv = einops.rearrange(qkv, "b h s d -> b s (h d)")
        return self.linear_o(qkv)  # type: ignore


class TransformerLayer(nn.Module):
    """A **unit-scaled** implementation of a PreNorm
    (see https://arxiv.org/abs/2002.04745) transformer layer.

    Args:
        hidden_size (int): the hidden dimension size of the input.
        heads (int): the number of attention heads.
        dropout_p (float, optional): the probability of the post-softmax dropout.
            Defaults to 0.1.
        act_fn (nn.Module): the activation function module. Defaults to `GELU()`.
        tau (float, optional): the weighting of the residual branch relative to the skip
            connection. Defaults to 0.2.
        {0}
    """

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        dropout_p: float = 0.1,
        act_fn: nn.Module = GELU(),
        tau: float = 0.2,
        constraint: Optional[VariadicConstraint] = gmean,
    ) -> None:
        super().__init__()
        self.dropout_p = dropout_p
        self.tau = tau
        self.mhsa_layer_norm = LayerNorm(hidden_size)
        self.mhsa = MHSA(hidden_size, heads, dropout_p, constraint=constraint)
        self.mlp_layer_norm = LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, act_fn, constraint=constraint)

    def forward(self, input: Tensor) -> Tensor:
        input, skip = U.residual_split(input, self.tau)
        input = self.mhsa_layer_norm(input)
        input = self.mhsa(input)
        input = U.dropout(input, self.dropout_p, self.training)
        input = U.residual_add(input, skip, self.tau)

        input, skip = U.residual_split(input, self.tau)
        input = self.mlp_layer_norm(input)
        input = self.mlp(input)
        input = U.dropout(input, self.dropout_p, self.training)
        return U.residual_add(input, skip, self.tau)

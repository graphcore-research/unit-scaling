# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Unit-scaled versions of common `torch.nn.functional` functions."""

import inspect
import sys
from typing import Any, Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, fx

from .constraints import BinaryConstraint, TernaryConstraint, gmean
from .docs import (
    binary_constraint_docstring,
    docstring_from,
    format_docstring,
    ternary_constraint_docstring,
)
from .scale import scale_bwd, scale_fwd


@format_docstring(binary_constraint_docstring)
def scale_elementwise(
    f: Callable[..., Tensor],
    output_scale: float,
    grad_input_scale: float,
    constraint: Optional[BinaryConstraint] = gmean,
) -> Callable[..., Tensor]:
    """Transforms an element-wise function into a scaled version.

    Args:
        f (Callable[..., Tensor]): the element-wise function to be scaled. Should take
            as its first input a `Tensor`, followed by `*args, **kwargs`.
        output_scale (float): the scale to be applied to the output
        grad_input_scale (float): the scale to be applied to the grad of the input
        {0}

    Returns:
        Callable[..., Tensor]: the scaled function
    """
    if constraint:
        output_scale = grad_input_scale = constraint(output_scale, grad_input_scale)

    def scaled_f(input: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        input = scale_bwd(input, grad_input_scale)
        output = f(input, *args, **kwargs)
        return scale_fwd(output, output_scale)

    return scaled_f


@docstring_from(
    F.gelu,
    short_description="Applies a **unit-scaled** GELU function.",
    add_args=[binary_constraint_docstring],
)
def gelu(
    input: Tensor,
    constraint: Optional[BinaryConstraint] = gmean,
) -> Tensor:
    output_scale = 0.588**-1
    grad_input_scale = 0.675**-1
    scaled_gelu = scale_elementwise(F.gelu, output_scale, grad_input_scale, constraint)
    return scaled_gelu(input)


@docstring_from(
    F.softmax,
    short_description="Applies a **unit-scaled** softmax function.",
    add_args=[binary_constraint_docstring],
)
def softmax(
    input: Tensor,
    dim: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    constraint: Optional[BinaryConstraint] = gmean,
) -> Tensor:
    dim_size = input.shape[dim] if dim is not None else input.numel()
    output_scale = dim_size / 1.31
    grad_input_scale = dim_size / 1.65
    scaled_softmax = scale_elementwise(
        F.softmax, output_scale, grad_input_scale, constraint
    )
    return scaled_softmax(input, dim=dim, dtype=dtype)


@docstring_from(
    F.dropout, short_description="Applies a **unit-scaled** dropout function."
)
def dropout(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    output_scale = grad_input_scale = (1 - p) ** 0.5
    scaled_dropout = scale_elementwise(
        F.dropout, output_scale, grad_input_scale, constraint=None
    )
    return scaled_dropout(input, p, training, inplace)


@docstring_from(
    torch.matmul,
    short_description="A **unit-scaled** matrix product of two tensors.",
    add_args=[ternary_constraint_docstring],
)
def matmul(
    left: Tensor,
    right: Tensor,
    constraint: Optional[TernaryConstraint] = gmean,
) -> Tensor:
    left_size = left.shape[-2]
    inner_size = left.shape[-1]
    right_size = right.shape[-1]

    output_scale = inner_size**-0.5
    left_grad_scale = right_size**-0.5
    right_grad_scale = left_size**-0.5

    if constraint:
        output_scale = left_grad_scale = right_grad_scale = constraint(
            output_scale, left_grad_scale, right_grad_scale
        )

    left = scale_bwd(left, left_grad_scale)
    right = scale_bwd(right, right_grad_scale)
    output = torch.matmul(left, right)
    return scale_fwd(output, output_scale)


@docstring_from(
    F.linear,
    short_description="Applies a **unit-scaled** linear transformation.",
    add_args=[binary_constraint_docstring],
)
def linear(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    constraint: Optional[BinaryConstraint] = gmean,
) -> Tensor:
    fan_out, fan_in = weight.shape
    batch_size = input.numel() // fan_in

    output_scale = fan_in**-0.5
    grad_input_scale = fan_out**-0.5
    grad_weight_scale = grad_bias_scale = batch_size**-0.5
    if constraint:
        output_scale = grad_input_scale = constraint(output_scale, grad_input_scale)

    input = scale_bwd(input, grad_input_scale)
    weight = scale_bwd(weight, grad_weight_scale)
    bias = scale_bwd(bias, grad_bias_scale) if bias is not None else None
    output = F.linear(input, weight, bias)
    return scale_fwd(output, output_scale)


# Wrap the public functions in this module so that fx tracing doesn't recurse
# into them
def _get_public_fns() -> List[str]:
    fns = []
    module = sys.modules[__name__]
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and not name.startswith("_"):
            fns.append(name)
    return fns


for f in _get_public_fns():
    fx.wrap(f)

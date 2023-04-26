# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Unit-scaled versions of common `torch.nn.functional` functions."""

from math import prod
from typing import Any, Callable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

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
    # Scale factors determined empirically, assuming unit scaled input & grad
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
    # Scale factors determined empirically, assuming unit-scaled & large dim_size
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


@docstring_from(
    F.layer_norm,
    short_description=(
        "Applies a **unit-scaled** Layer Normalization for last certain number of"
        " dimensions."
    ),
)
def layer_norm(
    input: Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    grad_weight_scale = grad_bias_scale = (
        prod(normalized_shape) / input.numel()
    ) ** 0.5
    if weight is not None:
        weight = scale_bwd(weight, grad_weight_scale)
    if bias is not None:
        bias = scale_bwd(bias, grad_bias_scale)
    return F.layer_norm(input, normalized_shape, weight, bias, eps)


def residual_split(residual: Tensor, tau: float = 0.2) -> Tuple[Tensor, Tensor]:
    """Splits a tensor into an `residual` and `skip` tensor, prior to being used
    in a residual layer, with a relative weighting `tau` applied to the residual branch.
    Should be used in conjunction with `residual_add`.

    This is necessary as unit scaling delays the residual branch scaling in the backward
    pass such that residual gradients are still unit-scaled. The need for a relative
    weighting between the two branches is a result of unit-scaling normalising the
    scales of the two branches, which in standard networks are typically not equal.

    Args:
        residual (Tensor): the tensor to which the residual layer is to be applied.
        tau (float, optional): the weighting of the residual branch relative to the skip
            connection. Defaults to 0.2.

    Returns:
        Tuple[Tensor, Tensor]: resulting tensors in the order: `residual, skip`.
    """
    skip = residual * (1 - tau) ** 0.5
    residual = scale_bwd(residual, tau**0.5)
    return residual, skip


def residual_add(residual: Tensor, skip: Tensor, tau: float = 0.2) -> Tensor:
    """Adds a residual connection and skip connection together, with a relative
    weighting `tau` applied to the residual branch. Should be used in conjunction with
    `residual_split`.

    Args:
        residual (Tensor): the tensor coming out of the residual connection.
        skip (Tensor): the tensor coming out of the skip connection.
        tau (float, optional): the weighting of the residual branch relative to the skip
            connection. Defaults to 0.2.

    Returns:
        Tensor: the result of the combined residual and skip tensors.
    """
    residual = scale_fwd(residual, tau**0.5)
    return residual + skip

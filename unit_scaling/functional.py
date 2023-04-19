# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Unit-scaled versions of common `torch.nn.functional` functions."""

from typing import Any, Callable, Optional

import numpy as np
import torch.nn.functional as F
from torch import Tensor

from .constraints import BinaryConstraint, gmean
from .docs import docstring_from, format_docstring, unary_constraint_docstring
from .scale import scale_bwd, scale_fwd


@format_docstring(unary_constraint_docstring)
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
    add_args=[unary_constraint_docstring],
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
    F.linear,
    short_description="Applies a **unit-scaled** linear transformation.",
    add_args=[unary_constraint_docstring],
)
def linear(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    constraint: Optional[BinaryConstraint] = gmean,
) -> Tensor:
    fan_out, fan_in = weight.shape
    batch_size = int(np.prod(input.shape[:-1]))

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

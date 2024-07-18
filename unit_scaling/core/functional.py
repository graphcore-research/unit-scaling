# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

"""Core functionality for implementing `unit_scaling.functional`."""

import math
from typing import Any, Callable, Optional

from torch import Tensor

from .._internal_utils import generate__all__
from ..constraints import apply_constraint
from ..docs import binary_constraint_docstring, format_docstring
from ..scale import scale_bwd, scale_fwd


@format_docstring(binary_constraint_docstring)
def scale_elementwise(
    f: Callable[..., Tensor],
    output_scale: float,
    grad_input_scale: float,
    constraint: Optional[str] = "to_output_scale",
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
    output_scale, grad_input_scale = apply_constraint(
        constraint, output_scale, grad_input_scale
    )

    def scaled_f(input: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        input = scale_bwd(input, grad_input_scale)
        output = f(input, *args, **kwargs)
        return scale_fwd(output, output_scale)

    return scaled_f


def logarithmic_interpolation(alpha: float, lower: float, upper: float) -> float:
    """Interpolate between lower and upper with logarithmic spacing (constant ratio).

    For example::

        logarithmic_interpolation(alpha=0.0, lower=1/1000, upper=1/10) == 1/1000
        logarithmic_interpolation(alpha=0.5, lower=1/1000, upper=1/10) == 1/100
        logarithmic_interpolation(alpha=1.0, lower=1/1000, upper=1/10) == 1/10

    Args:
        alpha (float): interpolation weight (0=lower, 1=upper)
        lower (float): lower limit (alpha=0), must be > 0
        upper (float): upper limit (alpha=1), must be > 0

    Returns:
        float: interpolated value
    """
    return math.exp(alpha * math.log(upper) + (1 - alpha) * math.log(lower))


__all__ = generate__all__(__name__)

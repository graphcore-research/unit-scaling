# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

"""Core functionality for implementing `unit_scaling.functional`."""

import math
from typing import Any, Callable, Optional, Tuple

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


def rms(
    x: Tensor,
    dims: Optional[Tuple[int, ...]] = None,
    keepdim: bool = False,
    eps: float = 0.0,
) -> Tensor:
    """Compute the RMS :math:`\\sqrt{\\mathrm{mean}(x^2) + \\epsilon}` of a tensor."""
    mean = x.float().pow(2).mean(dims, keepdim=keepdim)
    if eps:
        mean = mean + eps
    return mean.sqrt().to(x.dtype)


ResidualScalingFn = Callable[[int, int], float]


def transformer_residual_scaling_rule(
    residual_mult: float = 1.0, residual_attn_ratio: float = 1.0
) -> ResidualScalingFn:
    """Compute the residual tau ratios for the default transformer rule.

    For a transformer stack that starts with embedding, then alternates
    between attention and MLP layers, this rule ensures:

     - Every attention layer contributes the same scale.
     - Every MLP layer contributes the same scale.
     - The ratio of the average (variance) contribution of all attention
       and all MLP layers to the embedding layer is `residual_mult`.
     - The ratio of Attn to MLP contributions is `residual_attn_ratio`.

    If both hyperparameters are set to 1.0, the total contribution of
    embedding, attention and MLP layers are all equal.

    This scheme is described in Appendix G of the u-μP paper,

    Args:
        residual_mult (float, optional): contribution of residual layers
            (relative to an initial/embedding layer).
        residual_attn_ratio (float, optional): contribution of attn
            layers relative to FFN layers.

    Returns:
        :code:`fn(index, layers) -> tau` : a function for calculating tau
        at a given depth.
    """
    alpha_mlp = residual_mult * (2 / (1 + residual_attn_ratio**2)) ** 0.5
    alpha_attn = residual_attn_ratio * alpha_mlp

    def _tau(index: int, layers: int) -> float:
        n_attn = (index + 1) // 2
        n_mlp = index // 2
        tau = (alpha_attn if (index % 2) == 0 else alpha_mlp) / (
            layers / 2 + n_attn * alpha_attn**2 + n_mlp * alpha_mlp**2
        ) ** 0.5
        return tau  # type:ignore[no-any-return]

    return _tau


__all__ = generate__all__(__name__)

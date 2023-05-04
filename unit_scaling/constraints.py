# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Common scale-constraints used in unit-scaled operations."""

from math import pow, prod
from typing import Callable, Tuple, Union

from ._internal_utils import generate__all__

BinaryConstraint = Callable[[float, float], float]
TernaryConstraint = Callable[
    [float, float, float], Union[float, Tuple[float, float, float]]
]
VariadicConstraint = Callable[..., float]


def gmean(*scales: float) -> float:
    """Computes the geometric mean of the provided scales. Recommended for unit scaling.

    Args:
        scales: (*float): the group of constrained scales

    Returns:
        float: the geometric mean.
    """
    return pow(prod(scales), (1 / len(scales)))


def hmean(*scales: float) -> float:
    """Computes the harmonic mean of the provided scales. Used in Xavier/Glorot scaling.

    Args:
        scales: (*float): the group of constrained scales

    Returns:
        float: the harmonic mean.
    """
    return 1 / (sum(1 / s for s in scales) / len(scales))


def amean(*scales: float) -> float:
    """Computes the arithmetic mean of the provided scales.

    Args:
        scales: (*float): the group of constrained scales

    Returns:
        float: the arithmetic mean.
    """
    return sum(scales) / len(scales)


def to_output_scale(output_scale: float, *grad_input_scale: float) -> float:
    """Assumes an output scale is provided and any number of grad input scales:
    `(output_scale, *grad_input_scales)`. Selects only `output_scale` as the chosen
    scaling factor.

    Args:
        output_scale (float): the scale of the op's output
        grad_input_scales (*float): the scales of the op's input gradients

    Returns:
        float: equal to `output_scale`
    """
    return output_scale


def to_grad_input_scale(output_scale: float, grad_input_scale: float) -> float:
    """Assumes two provided scales: `(output_scale, grad_input_scale)`. Selects only
    `grad_input_scale` as the chosen scaling factor.

    Args:
        output_scale (float): the scale of the op's output
        grad_input_scale (float): the scale of the op's input gradient

    Returns:
        float: equal to `grad_input_scale`
    """
    return grad_input_scale


def to_left_grad_scale(
    output_scale: float, left_grad_scale: float, right_grad_scale: float
) -> float:
    """Assumes three provided scales:
    `(output_scale, left_grad_scale, right_grad_scale)`. Selects only `left_grad_scale`
    as the chosen scaling factor.

    Args:
        output_scale (float): the scale of the op's output
        left_grad_scale (float): the scale of the op's left input gradient
        right_grad_scale (float): the scale of the op's right input gradient

    Returns:
        float: equal to `left_grad_scale`
    """
    return left_grad_scale


def to_right_grad_scale(
    output_scale: float, left_grad_scale: float, right_grad_scale: float
) -> float:
    """Assumes three provided scales:
    `(output_scale, left_grad_scale, right_grad_scale)`. Selects only `right_grad_scale`
    as the chosen scaling factor.

    Args:
        output_scale (float): the scale of the op's output
        left_grad_scale (float): the scale of the op's left input gradient
        right_grad_scale (float): the scale of the op's right input gradient

    Returns:
        float: equal to `right_grad_scale`
    """
    return right_grad_scale


__all__ = generate__all__(__name__)

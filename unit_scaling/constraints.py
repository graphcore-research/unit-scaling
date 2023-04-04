# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Common scale-constraints used in unit-scaled operations."""

import numpy as np
from scipy import stats


def gmean(*scales: float) -> float:
    """Computes the geometric mean of the provided scales. Recommended for unit scaling.

    Args:
        scales: (*float): the group of constrained scales

    Returns:
        float: the geometric mean.
    """
    return stats.gmean(scales)  # type: ignore


def hmean(*scales: float) -> float:
    """Computes the harmonic mean of the provided scales. Used in Xavier/Glorot scaling.

    Args:
        scales: (*float): the group of constrained scales

    Returns:
        float: the harmonic mean.
    """
    return stats.hmean(scales)  # type: ignore


def amean(*scales: float) -> float:
    """Computes the arithmetic mean of the provided scales.

    Args:
        scales: (*float): the group of constrained scales

    Returns:
        float: the arithmetic mean.
    """
    return float(np.mean(scales))


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

# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Unit-scaled versions of common `torch.nn.functional` functions."""

from typing import Callable, Optional

import numpy as np
import torch.nn.functional as F
from torch import Tensor

from .constraints import gmean
from .docs import docstring_from, unary_constraint_docstring
from .scale import scale_bwd, scale_fwd


@docstring_from(
    F.linear,
    short_description="Applies a **unit-scaled** linear transformation.",
    add_args=[unary_constraint_docstring],
)
def linear(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    constraint: Optional[Callable[[float, float], float]] = gmean,
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

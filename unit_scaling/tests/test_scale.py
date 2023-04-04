# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch

from ..scale import scale_bwd, scale_fwd
from .helper import unit_backward, unit_normal


def test_scale_fwd() -> None:
    x = unit_normal(2**10)
    x_scaled = scale_fwd(x, 3.5)
    grad_in = unit_backward(x_scaled)

    assert torch.equal(x_scaled, x * 3.5)
    assert torch.equal(x.grad, grad_in)  # type: ignore


def test_scale_bwd() -> None:
    x = unit_normal(2**10)
    x_scaled = scale_bwd(x, 3.5)
    grad_in = unit_backward(x_scaled)

    assert torch.equal(x_scaled, x)
    assert torch.equal(x.grad, grad_in * 3.5)  # type: ignore

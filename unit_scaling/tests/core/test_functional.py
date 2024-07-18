# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import pytest
from torch import randn, tensor
from torch.testing import assert_close

from ...core.functional import rms, scale_elementwise
from ..helper import unit_backward


def test_scale_elementwise_no_constraint() -> None:
    input = randn(2**10, requires_grad=True)
    f = lambda x: x
    scaled_f = scale_elementwise(
        f, output_scale=2.5, grad_input_scale=0.5, constraint=None
    )
    output = scaled_f(input)
    unit_backward(output)

    assert output.std().detach() == pytest.approx(2.5, rel=0.1)
    assert input.grad.std().detach() == pytest.approx(0.5, rel=0.1)  # type: ignore


def test_scale_elementwise_for_output() -> None:
    input = randn(2**10, requires_grad=True)
    f = lambda x: x
    scaled_f = scale_elementwise(
        f, output_scale=2.5, grad_input_scale=0.5, constraint="to_output_scale"
    )
    output = scaled_f(input)
    unit_backward(output)

    assert output.std().detach() == pytest.approx(2.5, rel=0.1)
    assert input.grad.std().detach() == pytest.approx(2.5, rel=0.1)  # type: ignore


def test_scale_elementwise_for_grad_input() -> None:
    input = randn(2**10, requires_grad=True)
    f = lambda x: x
    scaled_f = scale_elementwise(
        f, output_scale=2.5, grad_input_scale=0.5, constraint="to_grad_input_scale"
    )
    output = scaled_f(input)
    unit_backward(output)

    assert output.std().detach() == pytest.approx(0.5, rel=0.1)
    assert input.grad.std().detach() == pytest.approx(0.5, rel=0.1)  # type: ignore


def test_rms() -> None:
    output = rms(-4 + 3 * randn(2**12))
    assert output.item() == pytest.approx(5, rel=0.1)

    x = tensor([[2, -2, -2, -2], [0, 2, 0, 0]]).float()
    output = rms(x, dims=(1,))
    assert_close(output, tensor([2.0, 1.0]))

    output = rms(tensor([0.0, 0.0, 0.0]), eps=1 / 16)
    assert output.item() == pytest.approx(1 / 4, rel=0.1)

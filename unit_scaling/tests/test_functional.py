# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest

from ..constraints import gmean, to_grad_input_scale, to_output_scale
from ..functional import gelu, linear, scale_elementwise
from .helper import (
    assert_not_unit_scaled,
    assert_unit_scaled,
    unit_backward,
    unit_normal,
)

# --- test scale_elementwise() ---


def test_scale_elementwise_no_constraint() -> None:
    input = unit_normal(2**10)
    f = lambda x: x
    scaled_f = scale_elementwise(
        f, output_scale=2.5, grad_input_scale=0.5, constraint=None
    )
    output = scaled_f(input)
    unit_backward(output)

    assert output.std().detach() == pytest.approx(2.5, rel=0.1)
    assert input.grad.std().detach() == pytest.approx(0.5, rel=0.1)  # type: ignore


def test_scale_elementwise_for_output() -> None:
    input = unit_normal(2**10)
    f = lambda x: x
    scaled_f = scale_elementwise(
        f, output_scale=2.5, grad_input_scale=0.5, constraint=to_output_scale
    )
    output = scaled_f(input)
    unit_backward(output)

    assert output.std().detach() == pytest.approx(2.5, rel=0.1)
    assert input.grad.std().detach() == pytest.approx(2.5, rel=0.1)  # type: ignore


def test_scale_elementwise_for_grad_input() -> None:
    input = unit_normal(2**10)
    f = lambda x: x
    scaled_f = scale_elementwise(
        f, output_scale=2.5, grad_input_scale=0.5, constraint=to_grad_input_scale
    )
    output = scaled_f(input)
    unit_backward(output)

    assert output.std().detach() == pytest.approx(0.5, rel=0.1)
    assert input.grad.std().detach() == pytest.approx(0.5, rel=0.1)  # type: ignore


# --- test gelu() ---


def test_gelu_no_constraint() -> None:
    input = unit_normal(2**10)
    output = gelu(input, constraint=None)
    unit_backward(output)

    assert_unit_scaled(input.grad, output)


def test_gelu_scale_for_output() -> None:
    input = unit_normal(2**10)
    output = gelu(input, constraint=to_output_scale)
    unit_backward(output)

    assert_unit_scaled(output)
    assert_not_unit_scaled(input.grad)


def test_gelu_scale_for_grad_input() -> None:
    input = unit_normal(2**10)
    output = gelu(input, constraint=to_grad_input_scale)
    unit_backward(output)

    assert_unit_scaled(input.grad)
    assert_not_unit_scaled(output)


# --- test linear() ---


def test_linear_no_constraint() -> None:
    input = unit_normal(2**8, 2**10)
    weight = unit_normal(2**12, 2**10)
    bias = unit_normal(2**12)
    output = linear(input, weight, bias, constraint=None)
    unit_backward(output)

    assert_unit_scaled(output, input.grad, weight.grad, bias.grad)


def test_linear_geo_mean() -> None:
    input = unit_normal(2**8, 2**10)
    weight = unit_normal(2**12, 2**10)
    bias = unit_normal(2**12)
    output = linear(input, weight, bias, constraint=gmean)
    unit_backward(output)

    assert_unit_scaled(weight.grad, bias.grad)
    assert_not_unit_scaled(output, input.grad)
    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)


def test_linear_scale_for_output() -> None:
    input = unit_normal(2**8, 2**10)
    weight = unit_normal(2**12, 2**10)
    bias = unit_normal(2**12)
    output = linear(input, weight, bias, constraint=to_output_scale)
    unit_backward(output)

    assert_unit_scaled(output, weight.grad, bias.grad)
    assert_not_unit_scaled(input.grad)


def test_linear_scale_for_grad_input() -> None:
    input = unit_normal(2**8, 2**10)
    weight = unit_normal(2**12, 2**10)
    bias = unit_normal(2**12)
    output = linear(input, weight, bias, constraint=to_grad_input_scale)
    unit_backward(output)

    assert_unit_scaled(input.grad, weight.grad, bias.grad)
    assert_not_unit_scaled(output)

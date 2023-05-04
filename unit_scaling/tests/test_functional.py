# Copyright (c) 2023 Graphcore Ltd. All rights reserved.


import pytest
from torch import Tensor, zeros

from ..constraints import (
    gmean,
    to_grad_input_scale,
    to_left_grad_scale,
    to_output_scale,
    to_right_grad_scale,
)
from ..functional import (
    dropout,
    gelu,
    layer_norm,
    linear,
    matmul,
    residual_add,
    residual_split,
    scale_elementwise,
    softmax,
)
from .helper import (
    assert_not_unit_scaled,
    assert_unit_scaled,
    unit_backward,
    unit_normal,
)


def retain_grad(t: Tensor) -> None:
    """Required as `torch.Tensor.retain_grad()` throws error with custom grad."""

    def set_tensor_grad(grad: Tensor) -> None:
        t.grad = grad

    t.register_hook(set_tensor_grad)  # type: ignore [no-untyped-call]


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


# --- test softmax() ---


def test_softmax_no_constraint() -> None:
    input = unit_normal(2**12)
    output = softmax(input, constraint=None)
    unit_backward(output)

    assert_unit_scaled(output, input.grad)


def test_softmax_scale_for_output() -> None:
    input = unit_normal(2**12)
    output = softmax(input, constraint=to_output_scale)
    unit_backward(output)

    assert_unit_scaled(output)
    assert_not_unit_scaled(input.grad)


def test_softmax_scale_for_grad_input() -> None:
    input = unit_normal(2**12)
    output = softmax(input, constraint=to_grad_input_scale)
    unit_backward(output)

    assert_unit_scaled(input.grad)
    assert_not_unit_scaled(output)


def test_softmax_dim() -> None:
    for dim in range(4):
        shape = [2, 2, 2, 2]
        shape[dim] = 2**12
        input = unit_normal(*shape)
        output = softmax(input, dim=dim, constraint=None)
        unit_backward(output)

        assert_unit_scaled(output, input.grad)


# --- test dropout() ---


def test_dropout() -> None:
    for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
        input = unit_normal(2**20)
        output = dropout(input, p)
        unit_backward(output)

        assert_unit_scaled(output, input.grad)


# --- test matmul() ---


def test_matmul_no_constraint() -> None:
    left = unit_normal(2**8, 2**10)
    right = unit_normal(2**10, 2**12)
    output = matmul(left, right, constraint=None)
    unit_backward(output)

    assert_unit_scaled(output, left.grad, right.grad)


def test_matmul_scale_for_output() -> None:
    left = unit_normal(2**8, 2**10)
    right = unit_normal(2**10, 2**12)
    output = matmul(left, right, constraint=to_output_scale)
    unit_backward(output)

    assert_unit_scaled(output)
    assert_not_unit_scaled(left.grad, right.grad)


def test_matmul_scale_for_grad_left() -> None:
    left = unit_normal(2**8, 2**10)
    right = unit_normal(2**10, 2**12)
    output = matmul(left, right, constraint=to_left_grad_scale)
    unit_backward(output)

    assert_unit_scaled(left.grad)
    assert_not_unit_scaled(output, right.grad)


def test_matmul_scale_for_grad_right() -> None:
    left = unit_normal(2**8, 2**10)
    right = unit_normal(2**10, 2**12)
    output = matmul(left, right, constraint=to_right_grad_scale)
    unit_backward(output)

    assert_unit_scaled(right.grad)
    assert_not_unit_scaled(output, left.grad)


# --- test linear() ---


def test_linear_no_constraint() -> None:
    input = unit_normal(2**8, 2**10)
    weight = unit_normal(2**12, 2**10)
    bias = zeros(2**12).requires_grad_()
    output = linear(input, weight, bias, constraint=None)
    unit_backward(output)

    assert_unit_scaled(output, input.grad, weight.grad, bias.grad)


def test_linear_geo_mean() -> None:
    input = unit_normal(2**8, 2**10)
    weight = unit_normal(2**12, 2**10)
    bias = zeros(2**12).requires_grad_()
    output = linear(input, weight, bias, constraint=gmean)
    unit_backward(output)

    assert_unit_scaled(weight.grad, bias.grad)
    assert_not_unit_scaled(output, input.grad)
    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)


def test_linear_scale_for_output() -> None:
    input = unit_normal(2**8, 2**10)
    weight = unit_normal(2**12, 2**10)
    bias = zeros(2**12).requires_grad_()
    output = linear(input, weight, bias, constraint=to_output_scale)
    unit_backward(output)

    assert_unit_scaled(output, weight.grad, bias.grad)
    assert_not_unit_scaled(input.grad)


def test_linear_scale_for_grad_input() -> None:
    input = unit_normal(2**8, 2**10)
    weight = unit_normal(2**12, 2**10)
    bias = zeros(2**12).requires_grad_()
    output = linear(input, weight, bias, constraint=to_grad_input_scale)
    unit_backward(output)

    assert_unit_scaled(input.grad, weight.grad, bias.grad)
    assert_not_unit_scaled(output)


# --- test layer_norm() ---


def test_layer_norm() -> None:
    input = unit_normal(2**8, 2**10)
    weight = unit_normal(2**10)
    bias = zeros(2**10).requires_grad_()
    output = layer_norm(input, (2**10,), weight, bias)
    unit_backward(output)

    assert_unit_scaled(output, input.grad, weight.grad, bias.grad)


# --- test residual() ---


def test_residual() -> None:
    for tau in [0.2, 0.5, 0.8]:
        input = unit_normal(2**10)
        residual, skip = residual_split(input, tau)
        residual = linear(residual, unit_normal(2**10, 2**10), bias=None)
        output = residual_add(residual, skip, tau)
        retain_grad(residual)
        retain_grad(skip)
        unit_backward(output)

        assert_unit_scaled(residual, output, residual.grad, skip.grad, input.grad)

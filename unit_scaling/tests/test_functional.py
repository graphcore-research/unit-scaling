# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch.nn.functional as F
from torch import Tensor, randint, randn, tensor, zeros

from ..functional import (
    add,
    conv1d,
    cross_entropy,
    dropout,
    embedding,
    gelu,
    layer_norm,
    linear,
    linear_readout,
    matmul,
    mse_loss,
    residual_add,
    residual_apply,
    residual_split,
    rms_norm,
    scaled_dot_product_attention,
    silu,
    silu_glu,
    softmax,
)
from .helper import (
    assert_not_unit_scaled,
    assert_scale,
    assert_unit_scaled,
    unit_backward,
)


def retain_grad(t: Tensor) -> None:
    """Required as `torch.Tensor.retain_grad()` throws error with custom grad."""

    def set_tensor_grad(grad: Tensor) -> None:
        t.grad = grad

    t.register_hook(set_tensor_grad)  # type: ignore [no-untyped-call]


# --- test gelu() ---


@pytest.mark.parametrize("mult", [1 / 4, 1, 4])
def test_gelu_no_constraint(mult: float) -> None:
    input = randn(2**10, requires_grad=True)
    output = gelu(input, mult=mult, constraint=None)
    unit_backward(output)

    assert_unit_scaled(input.grad, output)


def test_gelu_scale_for_output() -> None:
    input = randn(2**10, requires_grad=True)
    output = gelu(input, constraint="to_output_scale")
    unit_backward(output)

    assert_unit_scaled(output)
    assert_not_unit_scaled(input.grad)


def test_gelu_scale_for_grad_input() -> None:
    input = randn(2**10, requires_grad=True)
    output = gelu(input, constraint="to_grad_input_scale")
    unit_backward(output)

    assert_unit_scaled(input.grad)
    assert_not_unit_scaled(output)


# --- test silu() ---


@pytest.mark.parametrize("mult", [1 / 4, 1, 4])
def test_silu_no_constraint(mult: float) -> None:
    input = randn(2**10, requires_grad=True)
    output = silu(input, mult=mult, constraint=None)
    unit_backward(output)

    assert_unit_scaled(input.grad, output)


def test_silu_scale_for_output() -> None:
    input = randn(2**10, requires_grad=True)
    output = silu(input, mult=2, constraint="to_output_scale")
    unit_backward(output)

    assert_unit_scaled(output)
    assert_not_unit_scaled(input.grad)


def test_silu_scale_for_grad_input() -> None:
    input = randn(2**10, requires_grad=True)
    output = silu(input, constraint="to_grad_input_scale")
    unit_backward(output)

    assert_unit_scaled(input.grad)
    assert_not_unit_scaled(output)


# --- test silu() ---


@pytest.mark.parametrize("mult", [1 / 4, 1, 4])
def test_silu_glu(mult: float) -> None:
    input = randn(2**10, requires_grad=True)
    gate = randn(2**10, requires_grad=True)
    output = silu_glu(input, gate, mult=mult)
    unit_backward(output)

    # Scales are constrained, but forward and backward scaling
    # are similar enough that everything is nearly unit-scale
    assert_unit_scaled(input.grad, gate.grad, output)


# --- test softmax() ---

# In all these tests, use mult=1/4, as the approximation is quite bad at mult=1


def test_softmax_no_constraint() -> None:
    input = randn(2**12, requires_grad=True)
    output = softmax(input, dim=0, constraint=None, mult=1 / 4)
    unit_backward(output)

    assert_unit_scaled(output, input.grad, stat="rms")


def test_softmax_scale_for_output() -> None:
    input = randn(2**12, requires_grad=True)
    output = softmax(input, dim=0, constraint="to_output_scale", mult=1 / 4)
    unit_backward(output)

    assert_unit_scaled(output, stat="rms")
    assert_not_unit_scaled(input.grad, stat="rms")


def test_softmax_scale_for_grad_input() -> None:
    input = randn(2**12, requires_grad=True)
    output = softmax(input, dim=0, constraint="to_grad_input_scale", mult=1 / 4)
    unit_backward(output)

    assert_unit_scaled(input.grad, stat="rms")
    assert_not_unit_scaled(output, stat="rms")


def test_softmax_dim() -> None:
    for dim in range(4):
        shape = [2, 2, 2, 2]
        shape[dim] = 2**12
        input = randn(*shape, requires_grad=True)
        output = softmax(input, dim=dim, constraint=None, mult=1 / 4)
        unit_backward(output)

        assert_unit_scaled(output, input.grad, stat="rms")


# --- test dropout() ---


def test_dropout() -> None:
    for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
        input = randn(2**20, requires_grad=True)
        output = dropout(input, p)
        unit_backward(output)

        assert_unit_scaled(output, input.grad)

    with pytest.raises(ValueError):
        dropout(randn(2**20, requires_grad=True), 0.5, inplace=True)


# --- test matmul() ---


def test_matmul_no_constraint() -> None:
    left = randn(2**8, 2**10, requires_grad=True)
    right = randn(2**10, 2**12, requires_grad=True)
    output = matmul(left, right, constraint=None)
    unit_backward(output)

    assert_unit_scaled(output, left.grad, right.grad)


def test_matmul_scale_for_output() -> None:
    left = randn(2**8, 2**10, requires_grad=True)
    right = randn(2**10, 2**12, requires_grad=True)
    output = matmul(left, right, constraint="to_output_scale")
    unit_backward(output)

    assert_unit_scaled(output)
    assert_not_unit_scaled(left.grad, right.grad)


def test_matmul_scale_for_grad_left() -> None:
    left = randn(2**8, 2**10, requires_grad=True)
    right = randn(2**10, 2**12, requires_grad=True)
    output = matmul(left, right, constraint="to_left_grad_scale")
    unit_backward(output)

    assert_unit_scaled(left.grad)
    assert_not_unit_scaled(output, right.grad)


def test_matmul_scale_for_grad_right() -> None:
    left = randn(2**8, 2**10, requires_grad=True)
    right = randn(2**10, 2**12, requires_grad=True)
    output = matmul(left, right, constraint="to_right_grad_scale")
    unit_backward(output)

    assert_unit_scaled(right.grad)
    assert_not_unit_scaled(output, left.grad)


# --- test linear() ---


def test_linear_no_constraint() -> None:
    input = randn(2**8, 2**10, requires_grad=True)
    weight = randn(2**12, 2**10, requires_grad=True)
    bias = zeros(2**12).requires_grad_()
    output = linear(input, weight, bias, constraint=None)
    unit_backward(output)

    assert_unit_scaled(output, input.grad, weight.grad, bias.grad)


def test_linear_geo_mean() -> None:
    input = randn(2**8, 2**10, requires_grad=True)
    weight = randn(2**12, 2**10, requires_grad=True)
    bias = zeros(2**12).requires_grad_()
    output = linear(input, weight, bias, constraint="gmean")
    unit_backward(output)

    assert_unit_scaled(weight.grad, bias.grad)
    assert_not_unit_scaled(output, input.grad)
    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)


def test_linear_scale_for_output() -> None:
    input = randn(2**8, 2**10, requires_grad=True)
    weight = randn(2**12, 2**10, requires_grad=True)
    bias = zeros(2**12).requires_grad_()
    output = linear(input, weight, bias, constraint="to_output_scale")
    unit_backward(output)

    assert_unit_scaled(output, weight.grad, bias.grad)
    assert_not_unit_scaled(input.grad)


def test_linear_scale_for_grad_input() -> None:
    input = randn(2**8, 2**10, requires_grad=True)
    weight = randn(2**12, 2**10, requires_grad=True)
    bias = zeros(2**12).requires_grad_()
    output = linear(input, weight, bias, constraint="to_grad_input_scale")
    unit_backward(output)

    assert_unit_scaled(input.grad, weight.grad, bias.grad)
    assert_not_unit_scaled(output)


# --- test linear_readout() ---


def test_linear_readout() -> None:
    input = randn(2**8, 2**10, requires_grad=True)
    weight = randn(2**12, 2**10, requires_grad=True)
    bias = zeros(2**12).requires_grad_()
    output = linear_readout(input, weight, bias)
    unit_backward(output)

    assert_unit_scaled(weight.grad, bias.grad)  # constraint=None
    assert_scale(output, target=2**-5)  # 1/sqrt(fan_in)


# --- test conv1d() ---


def test_conv1d() -> None:
    batch_size = 2**6
    d_in = 2**6 * 3
    d_out = 2**6 * 5
    kernel_size = 11
    seq_len = 2**6 * 7
    input = randn(batch_size, d_in, seq_len, requires_grad=True)
    weight = randn(d_out, d_in, kernel_size, requires_grad=True)
    bias = zeros(d_out).requires_grad_()
    output = conv1d(input, weight, bias, constraint=None)
    unit_backward(output)

    assert_unit_scaled(output, input.grad, weight.grad, bias.grad)


def test_conv1d_stride() -> None:
    batch_size = 2**6
    d_in = 2**6 * 3
    d_out = 2**6 * 5
    kernel_size = 11
    seq_len = 2**6 * 7
    stride = 3

    input = randn(batch_size, d_in, seq_len, requires_grad=True)
    weight = randn(d_out, d_in, kernel_size, requires_grad=True)
    bias = zeros(d_out).requires_grad_()
    output = conv1d(input, weight, bias, stride=stride, constraint=None)
    unit_backward(output)

    assert_unit_scaled(output, input.grad, weight.grad, bias.grad)


def test_conv1d_padding() -> None:
    batch_size = 2**6
    d_in = 2**6 * 3
    d_out = 2**6 * 5
    kernel_size = 11
    seq_len = 2**6 * 7
    padding = 23  # If this is large enough wrt seq_len, test fails

    input = randn(batch_size, d_in, seq_len, requires_grad=True)
    weight = randn(d_out, d_in, kernel_size, requires_grad=True)
    bias = zeros(d_out).requires_grad_()
    output = conv1d(input, weight, bias, padding=padding, constraint=None)
    unit_backward(output)

    assert_unit_scaled(output, input.grad, weight.grad, bias.grad)


def test_conv1d_dilation() -> None:
    batch_size = 2**6
    d_in = 2**6 * 3
    d_out = 2**6 * 5
    kernel_size = 11
    seq_len = 2**6 * 7
    dilation = 8

    input = randn(batch_size, d_in, seq_len, requires_grad=True)
    weight = randn(d_out, d_in, kernel_size, requires_grad=True)
    bias = zeros(d_out).requires_grad_()
    output = conv1d(input, weight, bias, dilation=dilation, constraint=None)
    unit_backward(output)

    assert_unit_scaled(output, input.grad, weight.grad, bias.grad)


def test_conv1d_groups() -> None:
    batch_size = 2**6
    d_in = 2**6 * 3
    d_out = 2**6 * 5
    kernel_size = 11
    seq_len = 2**6 * 7
    groups = 32

    input = randn(batch_size, d_in, seq_len, requires_grad=True)
    weight = randn(d_out, d_in // groups, kernel_size, requires_grad=True)
    bias = zeros(d_out).requires_grad_()
    output = conv1d(input, weight, bias, groups=groups, constraint=None)
    unit_backward(output)

    assert_unit_scaled(output, input.grad, weight.grad, bias.grad)


# --- test layer_norm() ---


def test_layer_norm() -> None:
    input = randn(2**8, 2**10, requires_grad=True)
    weight = randn(2**10, requires_grad=True)
    bias = zeros(2**10).requires_grad_()
    output = layer_norm(input, (2**10,), weight, bias)
    unit_backward(output)

    assert_unit_scaled(output, input.grad, weight.grad, bias.grad)


def test_layer_norm_no_affine() -> None:
    input = randn(2**8, 2**10, requires_grad=True)
    output = layer_norm(input, (2**10,), None, None)
    unit_backward(output)

    assert_unit_scaled(output, input.grad)


# --- test rms_norm() ---


def test_rms_norm() -> None:
    input = randn(2**8, 2**10, requires_grad=True)
    weight = randn(2**10, requires_grad=True)
    output = rms_norm(input, (2**10,), weight)
    unit_backward(output)

    assert_unit_scaled(output, input.grad, weight.grad)


def test_rms_norm_no_affine() -> None:
    input = randn(2**8, 2**10, requires_grad=True)
    output = rms_norm(input, (2**10,), None)
    unit_backward(output)

    assert_unit_scaled(output, input.grad)


# --- test add() ---


def test_add_no_constraint() -> None:
    left = randn(2**8, 2**10, requires_grad=True)
    right = randn(2**8, 2**10, requires_grad=True)
    output = add(left, right, constraint=None)
    unit_backward(output)

    assert_unit_scaled(output, left.grad, right.grad)


def test_add_geo_mean() -> None:
    left = randn(2**8, 2**10, requires_grad=True)
    right = randn(2**8, 2**10, requires_grad=True)
    output = add(left, right, constraint="gmean")
    unit_backward(output)

    assert_not_unit_scaled(output, left.grad, right.grad)
    std = output.std().detach() * left.grad.std() * right.grad.std()  # type: ignore
    assert std == pytest.approx(1, abs=0.1)


def test_add_broadcast() -> None:
    left = tensor(5.0, requires_grad=True)
    right = randn(2**8, 2**10, requires_grad=True)
    output = add(left, right, constraint=None)
    unit_backward(output)

    assert left.grad is not None
    assert 0.0001 < left.grad.abs() < 5  # Reasonable bounds based on folded normal dist
    assert_unit_scaled(output, right.grad)

    left = randn(2**10, requires_grad=True)
    right = randn(2**8, 2**10, requires_grad=True)
    output = add(left, right, constraint=None)
    unit_backward(output)

    assert_unit_scaled(output, left.grad, right.grad)

    left = randn(2**8, 1, 2**10, requires_grad=True)
    right = randn(2**9, 2**10, requires_grad=True)
    output = add(left, right, constraint=None)
    unit_backward(output)

    assert_unit_scaled(output, left.grad, right.grad)


def test_add_scalar() -> None:
    left = 2.0
    right = randn(2**8, 2**10, requires_grad=True)
    output = add(left, right, constraint=None)
    unit_backward(output)

    assert_unit_scaled(output, right.grad)

    left = randn(2**8, 1, 2**10, requires_grad=True)  # type: ignore[assignment]
    right = -1.0  # type: ignore[assignment]
    output = add(left, right, constraint=None)
    unit_backward(output)

    assert_unit_scaled(output, left.grad)  # type: ignore[attr-defined]


# --- test residual() ---


def test_residual() -> None:
    for tau in [0.2, 1.0, 3.0]:
        input = randn(2**10, requires_grad=True)
        residual, skip = residual_split(input, tau)
        residual = linear(residual, randn(2**10, 2**10), bias=None)
        output = residual_add(residual, skip, tau)
        retain_grad(residual)
        retain_grad(skip)
        unit_backward(output)

        assert_unit_scaled(residual, output, residual.grad, skip.grad, input.grad)


def test_residual_apply() -> None:
    for tau in [0.2, 1.0, 3.0]:
        input = randn(2**10, requires_grad=True)
        weight = randn(2**10, 2**10, requires_grad=True)
        output = residual_apply(
            lambda x: linear(x, weight, bias=None, constraint=None), input, tau
        )
        unit_backward(output)

        assert_unit_scaled(output, input.grad, weight.grad)


# --- test embedding() ---


def test_embedding() -> None:
    batch_sz, seq_len, embedding_dim, num_embeddings = 2**4, 2**5, 2**6, 2**12
    input_idxs = randint(low=0, high=2**12, size=(batch_sz, seq_len))
    embedding_table = randn(num_embeddings, embedding_dim, requires_grad=True)
    output = embedding(input_idxs, embedding_table)
    unit_backward(output)

    assert_unit_scaled(output, embedding_table.grad)

    with pytest.raises(ValueError):
        embedding(input_idxs, embedding_table, scale_grad_by_freq=True)
    with pytest.raises(ValueError):
        embedding(input_idxs, embedding_table, sparse=True)


# --- test scaled_dot_product_attention() ---


def test_scaled_dot_product_attention() -> None:
    shape = 2**8, 2**6, 2**6
    q, k, v = (randn(*shape, requires_grad=True) for _ in range(3))
    output = scaled_dot_product_attention(q, k, v)
    unit_backward(output)

    assert_unit_scaled(output, v.grad)
    assert_scale(q.grad, k.grad, target=shape[1] ** -0.5)


# --- test cross_entropy() ---


def test_cross_entropy() -> None:
    num_tokens, vocab_sz = 2**12, 2**8
    for reduction in ["mean", "sum"]:
        for input_shape in [(vocab_sz,), (num_tokens, vocab_sz)]:
            input = randn(*input_shape, requires_grad=True)
            label_size = (input_shape[0],) if len(input_shape) == 2 else ()
            labels = randint(low=0, high=vocab_sz, size=label_size)
            loss = cross_entropy(input, labels, reduction=reduction)
            standard_loss = F.cross_entropy(input, labels, reduction=reduction)
            loss.backward()  # type: ignore [no-untyped-call]

            assert loss == standard_loss
            assert_unit_scaled(input.grad)

    input = randn(2**12, 2**8, requires_grad=True)
    labels = randint(low=0, high=vocab_sz, size=(num_tokens,))
    with pytest.raises(ValueError):
        cross_entropy(input, labels, weight=randn(vocab_sz))
    with pytest.raises(ValueError):
        cross_entropy(input, labels, label_smoothing=0.5)


# --- test mse_loss() ---


@pytest.mark.parametrize("reduction", ("mean", "sum"))
def test_mse_loss(reduction: str) -> None:
    input_a = randn(2, 2**10, requires_grad=True)
    input_b = randn(2, 2**10, requires_grad=True)
    loss = mse_loss(input_a, input_b, reduction=reduction)
    standard_loss = F.mse_loss(input_a, input_b, reduction=reduction)
    loss.backward()  # type: ignore [no-untyped-call]

    assert loss.item() == pytest.approx(standard_loss.item(), rel=1e-4)
    assert_unit_scaled(input_a.grad)
    assert_unit_scaled(input_b.grad)

    with pytest.raises(ValueError):
        mse_loss(zeros(2, 3), zeros(1, 3), reduction=reduction)

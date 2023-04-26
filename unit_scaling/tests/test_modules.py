# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch
from torch.optim import SGD

from ..modules import (
    GELU,
    MHSA,
    MLP,
    Dropout,
    LayerNorm,
    Linear,
    Softmax,
    TransformerLayer,
)
from .helper import (
    assert_non_zeros,
    assert_not_unit_scaled,
    assert_unit_scaled,
    assert_zeros,
    unit_backward,
    unit_normal,
)


def test_gelu() -> None:
    input = unit_normal(2**10)
    model = GELU()
    output = model(input)

    unit_backward(output)

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)


def test_softmax() -> None:
    input = unit_normal(2**14)
    model = Softmax()
    output = model(input)

    unit_backward(output)

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)


def test_dropout() -> None:
    input = unit_normal(2**12)
    model = Dropout()
    output = model(input)

    unit_backward(output)

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)


def test_linear() -> None:
    input = unit_normal(2**8, 2**10)
    model = Linear(2**10, 2**12)
    output = model(input)

    assert_unit_scaled(model.weight)
    assert_zeros(model.bias)
    assert output.shape == torch.Size([2**8, 2**12])

    unit_backward(output)
    SGD(model.parameters(), lr=1).step()

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)

    assert_not_unit_scaled(model.weight)
    assert_non_zeros(model.bias)


def test_layer_norm() -> None:
    input = unit_normal(2**8, 2**10)
    model = LayerNorm(2**10)
    output = model(input)

    assert output.shape == torch.Size([2**8, 2**10])

    unit_backward(output)
    SGD(model.parameters(), lr=1).step()

    assert_unit_scaled(output, input.grad, model.weight.grad, model.bias.grad)


def test_mlp() -> None:
    input = unit_normal(2**8, 2**10)
    model = MLP(2**10)
    output = model(input)

    assert_unit_scaled(model.linear_1.weight, model.linear_2.weight)
    assert_zeros(model.linear_1.bias, model.linear_2.bias)
    assert output.shape == torch.Size([2**8, 2**10])

    unit_backward(output)
    SGD(model.parameters(), lr=1).step()

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)

    assert_not_unit_scaled(model.linear_1.weight, model.linear_2.weight)
    assert_non_zeros(model.linear_1.bias, model.linear_2.bias)


def test_mhsa() -> None:
    b, s, d = 2**8, 2**6, 2**6
    input = unit_normal(b, s, d)
    model = MHSA(d, heads=8)
    output = model(input)

    assert_unit_scaled(model.linear_qkv.weight, model.linear_o.weight)
    assert output.shape == torch.Size([b, s, d])

    unit_backward(output)
    SGD(model.parameters(), lr=1).step()

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.5)

    assert_not_unit_scaled(model.linear_qkv.weight, model.linear_o.weight)


def test_transformer_layer() -> None:
    b, s, d = 2**8, 2**6, 2**6
    input = unit_normal(b, s, d)
    model = TransformerLayer(d, heads=8)
    output = model(input)

    assert output.shape == torch.Size([b, s, d])

    unit_backward(output)
    SGD(model.parameters(), lr=1).step()

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)

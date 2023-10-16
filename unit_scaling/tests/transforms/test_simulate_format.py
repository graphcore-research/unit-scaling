# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import logging

import torch
import torch.nn.functional as F
from pytest import LogCaptureFixture
from torch import Tensor, nn, randn

from ... import _modules as uu
from ... import functional as U
from ...transforms import simulate_fp8


def test_simulate_fp8_linear(caplog: LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    class Model(nn.Module):
        def __init__(self, d_in: int, d_out: int) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out)

        def forward(self, t: Tensor) -> Tensor:
            return self.linear(t).sum()  # type: ignore[no-any-return]

    input = randn(2**8, 2**9, requires_grad=True)
    model = Model(2**9, 2**10)
    output = model(input)
    output.backward()

    fp8_input = input.clone().detach().requires_grad_()
    fp8_model = simulate_fp8(model)
    fp8_output = fp8_model(fp8_input)
    fp8_output.backward()

    assert not torch.all(fp8_output == output)
    assert not torch.all(fp8_input.grad == input.grad)  # type: ignore
    assert not torch.all(
        fp8_model.linear.weight.grad == model.linear.weight.grad  # type: ignore
    )
    assert "quantising function" in caplog.text


def test_simulate_fp8_unit_scaled_linear() -> None:
    class Model(nn.Module):
        def __init__(self, d_in: int, d_out: int) -> None:
            super().__init__()
            self.linear = uu.Linear(d_in, d_out)

        def forward(self, t: Tensor) -> Tensor:
            return self.linear(t).sum()  # type: ignore[no-any-return]

    input = randn(2**8, 2**9, requires_grad=True)
    model = Model(2**9, 2**10)
    output = model(input)
    output.backward()

    fp8_input = input.clone().detach().requires_grad_()
    fp8_model = simulate_fp8(model)
    fp8_output = fp8_model(fp8_input)
    fp8_output.backward()

    assert not torch.all(fp8_output == output)
    assert not torch.all(fp8_input.grad == input.grad)  # type: ignore
    assert not torch.all(
        fp8_model.linear.weight.grad == model.linear.weight.grad  # type: ignore
    )


def test_simulate_fp8_attention() -> None:
    class Model(nn.Module):
        def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            return F.scaled_dot_product_attention(q, k, v).sum()

    inputs = list(randn(2**8, 2**8, requires_grad=True) for _ in range(3))
    model = Model()
    output = model(*inputs)
    output.backward()

    fp8_inputs = list(t.clone().detach().requires_grad_() for t in inputs)
    fp8_model = simulate_fp8(model)

    fp8_output = fp8_model(*fp8_inputs)
    fp8_output.backward()

    assert not torch.all(fp8_output == output)
    for fp8_input, input in zip(fp8_inputs, inputs):
        assert not torch.all(fp8_input.grad == input.grad)  # type: ignore


def test_simulate_fp8_unit_scaled_attention() -> None:
    class Model(nn.Module):
        def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            return U.scaled_dot_product_attention(q, k, v).sum()

    inputs = list(randn(2**8, 2**8, requires_grad=True) for _ in range(3))
    model = Model()
    output = model(*inputs)
    output.backward()

    fp8_inputs = list(t.clone().detach().requires_grad_() for t in inputs)
    fp8_model = simulate_fp8(model)

    fp8_output = fp8_model(*fp8_inputs)
    fp8_output.backward()

    assert not torch.all(fp8_output == output)
    for fp8_input, input in zip(fp8_inputs, inputs):
        assert not torch.all(fp8_input.grad == input.grad)  # type: ignore

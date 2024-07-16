# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import logging
import math
import re
from typing import Tuple

import torch
import torch.nn.functional as F
from pytest import LogCaptureFixture
from torch import Tensor, nn, randn

from ...transforms import simulate_fp8, unit_scale
from ..helper import assert_unit_scaled


def test_unit_scale(caplog: LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    def custom_gelu(x: Tensor) -> Tensor:
        inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        return 0.5 * x * (1.0 + torch.tanh(inner))

    class MLPLayer(nn.Module):  # pragma: no cover
        def __init__(
            self,
            hidden_size: int,
        ) -> None:
            super().__init__()
            self.layer_norm = nn.LayerNorm(hidden_size)
            self.l1 = nn.Linear(hidden_size, 4 * hidden_size)
            self.l2 = nn.Linear(4 * hidden_size, hidden_size)

        def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
            input = self.layer_norm(input)
            input = self.l1(input)
            input = custom_gelu(input)
            input = self.l2(input)
            input = F.dropout(input, 0.2)
            return input, input.sum()

    input = randn(2**6, 2**10, requires_grad=True)
    model = nn.Sequential(MLPLayer(2**10))
    model = unit_scale(model, replace={custom_gelu: F.gelu})
    output, loss = model(input)
    loss.backward()

    assert_unit_scaled(
        output,
        input.grad,
        model[0].layer_norm.weight.grad,
        model[0].l1.weight.grad,
        model[0].l2.weight.grad,
        abs=0.2,
    )

    expected_logs = [
        "unit scaling weight",
        "setting bias to zero",
        "unit scaling function",
        "replacing function",
        "unconstraining node",
    ]
    for log_msg in expected_logs:
        assert log_msg in caplog.text


def test_unit_scale_residual_add(caplog: LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    class MLPLayer(nn.Module):
        def __init__(
            self,
            hidden_size: int,
        ) -> None:
            super().__init__()
            self.l1 = nn.Linear(hidden_size, hidden_size)
            self.l2 = nn.Linear(hidden_size, hidden_size)

        def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:  # pragma: no cover
            skip = input
            input = input + 1
            input = self.l1(input)
            input = input + skip
            skip = input
            input += 1
            input = self.l2(input)
            input += skip
            return input, input.sum()

    input = randn(2**6, 2**10, requires_grad=True)
    model = MLPLayer(2**10)
    us_model = unit_scale(model)
    output, loss = us_model(input)
    loss.backward()

    expected_logs = [
        r"unit scaling function: (input_2)\n",
        r"unit scaling function: (input_4)\n",
        r"unit scaling function: (skip_1|input_3) \(residual-add, tau=0\.5\)",
        r"unit scaling function: (add_1|input_6) \(residual-add, tau=0\.5\)",
    ]

    for log_msg in expected_logs:
        assert re.search(log_msg, caplog.text)


def test_fp8_unit_scaling(caplog: LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    class Model(nn.Module):
        def __init__(self, d_in: int, d_out: int) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out)

        def forward(self, t: Tensor) -> Tensor:  # pragma: no cover
            return self.linear(t)  # type: ignore[no-any-return]

    input = randn(2**8, 2**9)
    model = Model(2**9, 2**10)
    model = simulate_fp8(model)
    model = unit_scale(model)
    model(input)

    expected_logs = [
        "moving unit scaling backend to precede quantisation backend",
        "running unit scaling backend",
        "running quantisation backend",
    ]
    for log_msg in expected_logs:
        assert log_msg in caplog.text

# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch
from torch.optim import SGD

from ..modules import GELU, MLP, Linear
from .helper import (
    assert_not_unit_scaled,
    assert_unit_scaled,
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


def test_linear() -> None:
    input = unit_normal(2**8, 2**10)
    model = Linear(2**10, 2**12)
    output = model(input)

    assert_unit_scaled(model.weight)
    assert torch.all(model.bias == 0)
    assert output.shape == torch.Size([2**8, 2**12])

    unit_backward(output)
    SGD(model.parameters(), lr=1).step()

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)

    assert_not_unit_scaled(model.weight)
    assert torch.any(model.bias != 0)


def test_mlp() -> None:
    input = unit_normal(2**8, 2**10)
    model = MLP(2**10)
    output = model(input)

    assert_unit_scaled(model.linear_1.weight, model.linear_2.weight)
    assert torch.all(model.linear_1.bias == 0)
    assert torch.all(model.linear_2.bias == 0)
    assert output.shape == torch.Size([2**8, 2**10])

    unit_backward(output)
    SGD(model.parameters(), lr=1).step()

    combined_std = output.std().detach() * input.grad.std()  # type: ignore
    assert combined_std == pytest.approx(1, abs=0.1)

    assert_not_unit_scaled(model.linear_1.weight, model.linear_2.weight)
    assert torch.any(model.linear_1.bias != 0)
    assert torch.any(model.linear_2.bias != 0)

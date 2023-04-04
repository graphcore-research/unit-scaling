# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch
from torch.optim import SGD

from ..modules import Linear
from .helper import (
    assert_not_unit_scaled,
    assert_unit_scaled,
    unit_backward,
    unit_normal,
)


def test_linear() -> None:
    input = unit_normal(2**8, 2**10)
    model = Linear(2**10, 2**12)
    output = model(input)

    assert_unit_scaled(model.weight)
    assert torch.all(model.bias == 0)
    assert output.shape == torch.Size([2**8, 2**12])

    unit_backward(output)
    SGD(model.parameters(), lr=1).step()

    assert_not_unit_scaled(model.weight)
    assert torch.any(model.bias != 0)

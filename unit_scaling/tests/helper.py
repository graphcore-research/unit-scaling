# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Optional

import pytest
import torch
from torch import Tensor


def unit_normal(*shape: int, requires_grad: bool = True) -> Tensor:
    """A normal distribution with `mean=0, std=1`. Requires grad by default.

    Args:
        *shapes (int): the shape dimensions of the tensor created.
        requires_grad (bool, optional): sets `_requires_grad()`. Defaults to True.

    Returns:
        Tensor: the unit normal tensor.
    """
    x = torch.normal(0, 1, shape)
    if requires_grad:
        x.requires_grad_()
    return x


def unit_backward(tensor: Tensor) -> Tensor:
    """Applies the `backward()` method with a unit normal tensor as input.

    Args:
        tensor (Tensor): tensor to have `backward()` applied.

    Returns:
        Tensor: the unit normal gradient tensor fed into `backward()`.
    """
    gradient = unit_normal(*tensor.shape, requires_grad=False)
    tensor.backward(gradient)  # type: ignore
    return gradient


def assert_unit_scaled(*tensors: Optional[Tensor]) -> None:
    for t in tensors:
        assert t is not None
        t = t.detach()
        approx_1 = pytest.approx(1, abs=0.1)
        assert t.std() == approx_1, f"std={t.std():.3f}, shape={list(t.shape)}"


def assert_not_unit_scaled(*tensors: Optional[Tensor]) -> None:
    for t in tensors:
        assert t is not None
        t = t.detach()
        approx_1 = pytest.approx(1, abs=0.1)
        assert t.std() != approx_1, f"std={t.std():.3f}, shape={list(t.shape)}"


def assert_zeros(*tensors: Optional[Tensor]) -> None:
    for t in tensors:
        assert t is not None
        t = t.detach()
        assert torch.all(t == 0)


def assert_non_zeros(*tensors: Optional[Tensor]) -> None:
    for t in tensors:
        assert t is not None
        t = t.detach()
        assert torch.any(t != 0)

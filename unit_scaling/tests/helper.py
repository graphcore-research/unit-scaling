# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Optional

import pytest
import torch
from torch import Tensor, randn


def unit_backward(tensor: Tensor) -> Tensor:
    """Applies the `backward()` method with a unit normal tensor as input.

    Args:
        tensor (Tensor): tensor to have `backward()` applied.

    Returns:
        Tensor: the unit normal gradient tensor fed into `backward()`.
    """
    gradient = randn(*tensor.shape)
    tensor.backward(gradient)  # type: ignore
    return gradient


def assert_scale(*tensors: Optional[Tensor], target: float, abs: float = 0.1) -> None:
    for t in tensors:
        assert t is not None
        t = t.detach()
        approx_target = pytest.approx(target, abs=abs)
        assert t.std() == approx_target, f"std={t.std():.3f}, shape={list(t.shape)}"


def assert_not_scale(
    *tensors: Optional[Tensor], target: float, abs: float = 0.1
) -> None:
    for t in tensors:
        assert t is not None
        t = t.detach()
        approx_target = pytest.approx(target, abs=abs)
        assert t.std() != approx_target, f"std={t.std():.3f}, shape={list(t.shape)}"


def assert_unit_scaled(*tensors: Optional[Tensor], abs: float = 0.1) -> None:
    return assert_scale(*tensors, target=1.0, abs=abs)


def assert_not_unit_scaled(*tensors: Optional[Tensor], abs: float = 0.1) -> None:
    return assert_not_scale(*tensors, target=1.0, abs=abs)


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

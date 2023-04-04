# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Operations to enable different scaling factors in the forward and backward passes."""

from typing import Tuple

import poptorch_experimental_addons as pea
import torch
from torch import Tensor

try:
    import poptorch

    _poptorch_available = True
except ModuleNotFoundError:  # pragma: no cover
    _poptorch_available = False


class _ScaledGrad(torch.autograd.Function):  # pragma: no cover
    """Enables a custom backward method which has a different scale to forward."""

    @staticmethod
    def forward(  # type:ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        X: Tensor,
        fwd_scale: float,
        bwd_scale: float,
    ) -> Tensor:
        ctx.save_for_backward(torch.tensor(bwd_scale, dtype=X.dtype))
        return fwd_scale * X

    @staticmethod
    def backward(  # type:ignore[override]
        ctx: torch.autograd.function.FunctionCtx, grad_Y: Tensor
    ) -> Tuple[Tensor, None, None]:
        (bwd_scale,) = ctx.saved_tensors  # type: ignore
        return bwd_scale * grad_Y, None, None


def _scale(
    t: Tensor, fwd_scale: float = 1.0, bwd_scale: float = 1.0
) -> Tensor:  # pragma: no cover
    """Given a tensor, applies a separate scale in the forward and backward pass."""

    if _poptorch_available and poptorch.isRunningOnIpu():
        return pea.autograd_proxy(t * fwd_scale, t * bwd_scale)  # type: ignore
    return _ScaledGrad.apply(t, fwd_scale, bwd_scale)  # type: ignore


def scale_fwd(input: Tensor, scale: float) -> Tensor:
    """Applies a scalar multiplication to a tensor in only the forward pass.

    Args:
        input (Tensor): the tensor to be scaled.
        scale (float): the scale factor applied to the tensor in the forward pass.

    Returns:
        Tensor: scaled in the forward pass, but with its original grad.
    """
    return _scale(input, fwd_scale=scale)


def scale_bwd(input: Tensor, scale: float) -> Tensor:
    """Applies a scalar multiplication to a tensor in only the backward pass.

    Args:
        input (Tensor): the tensor to be scaled.
        scale (float): the scale factor applied to the tensor in the backward pass.

    Returns:
        Tensor: unchanged in the forward pass, but with a scaled grad.
    """
    return _scale(input, bwd_scale=scale)

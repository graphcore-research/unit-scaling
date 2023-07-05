# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import math
from dataclasses import dataclass
from typing import Tuple, cast

import torch
from torch import Tensor

from ._internal_utils import generate__all__

Shape = Tuple[int, ...]


@dataclass
class Format:
    """Generic representation of a number format."""

    exponent_bits: int
    mantissa_bits: int

    @property
    def bits(self) -> int:
        """The number of bits used by the format."""
        return 1 + self.exponent_bits + self.mantissa_bits

    def count_bits(self, shape: Shape) -> int:
        """The number of bits used by a tensor of shape `shape` in the format."""
        return self.bits * math.prod(shape)

    def __str__(self) -> str:
        return f"E{self.exponent_bits}M{self.mantissa_bits}"


@dataclass
class FPFormat(Format):
    """Generic representation of a floating-point number format."""

    def __post_init__(self) -> None:
        assert self.exponent_bits >= 2, "FPFormat requires at least 2 exponent bits"

    def to_tuple(self) -> Tuple[int, int]:
        """Convert the format into a tuple of `(exponent_bits, mantissa_bits)`"""
        return (self.exponent_bits, self.mantissa_bits)

    @staticmethod
    def from_tuple(t: Tuple[int, int]) -> "FPFormat":
        """Given a tuple of `(exponent_bits, mantissa_bits)` returns the corresponding
        :class:`FPFormat`"""
        return FPFormat(*t)

    @property
    def max_absolute_value(self) -> float:
        """The maximum absolute value representable by the format."""
        max_exponent = 2 ** (self.exponent_bits - 1) - 1
        return cast(float, 2**max_exponent * (2 - 2**-self.mantissa_bits))

    @property
    def min_absolute_normal(self) -> float:
        """The minimum absolute normal value representable by the format."""
        min_exponent = 1 - 2 ** (self.exponent_bits - 1)
        return cast(float, 2**min_exponent)

    @property
    def min_absolute_subnormal(self) -> float:
        """The minimum absolute subnormal value representable by the format."""
        return self.min_absolute_normal * 2.0**-self.mantissa_bits

    def quantise_no_grad(self, x: Tensor) -> Tensor:
        """Non-differentiably quantise the given tensor in this format."""
        absmax = self.max_absolute_value
        downscale = 2.0 ** (127 - 2 ** (self.exponent_bits - 1))
        mask = torch.tensor(2 ** (23 - self.mantissa_bits) - 1, device=x.device)
        sr_mask = torch.randint(  # type: ignore[call-overload]
            0, mask + 1, x.shape, dtype=torch.int32, device=x.device
        )
        q = x.to(torch.float32)
        q = torch.clip(x, -absmax, absmax)
        q /= downscale
        q = ((q.view(torch.int32) + sr_mask) & ~mask).view(torch.float32)
        q *= downscale
        return q.to(x.dtype)

    def quantise(self, x: Tensor) -> Tensor:
        """Quantise the given tensor in the forward pass only."""

        class QuantiseForward(torch.autograd.Function):
            @staticmethod
            def forward(  # type:ignore[override]
                ctx: torch.autograd.function.FunctionCtx, x: Tensor
            ) -> Tensor:
                return self.quantise_no_grad(x)

            @staticmethod
            def backward(  # type:ignore[override]
                ctx: torch.autograd.function.FunctionCtx, grad_y: Tensor
            ) -> Tensor:
                return grad_y

        return QuantiseForward.apply(x)  # type: ignore

    def quantise_bwd(self, x: Tensor) -> Tensor:
        """Quantise the given tensor in the backward pass only."""

        class QuantiseBackward(torch.autograd.Function):
            @staticmethod
            def forward(  # type:ignore[override]
                ctx: torch.autograd.function.FunctionCtx, x: Tensor
            ) -> Tensor:
                return x

            @staticmethod
            def backward(  # type:ignore[override]
                ctx: torch.autograd.function.FunctionCtx, grad_y: Tensor
            ) -> Tensor:
                return self.quantise_no_grad(grad_y)

        return QuantiseBackward.apply(x)  # type: ignore


__all__ = generate__all__(__name__)

# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Classes for simulating (non-standard) number formats."""

from dataclasses import dataclass
from typing import Tuple, cast

import torch
from torch import Tensor

from ._internal_utils import generate__all__

Shape = Tuple[int, ...]


@dataclass
class FPFormat:
    """Generic representation of a floating-point number format."""

    exponent_bits: int
    mantissa_bits: int
    rounding: str = "stochastic"  # "stochastic|nearest"
    srbits: int = 0  # Number of bits for stochastic rounding, zero => use all bits

    def __post_init__(self) -> None:
        assert self.exponent_bits >= 2, "FPFormat requires at least 2 exponent bits"
        assert (
            self.srbits == 0 or self.rounding == "stochastic"
        ), "Nonzero srbits for non-stochastic rounding"
        if self.srbits == 0 and self.rounding == "stochastic":
            self.srbits = 23 - self.mantissa_bits

    @property
    def bits(self) -> int:
        """The number of bits used by the format."""
        return 1 + self.exponent_bits + self.mantissa_bits

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"E{self.exponent_bits}M{self.mantissa_bits}-"
            + dict(stochastic="SR", nearest="RN")[self.rounding]
        )

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

    def quantise(self, x: Tensor) -> Tensor:
        """Non-differentiably quantise the given tensor in this format."""
        absmax = self.max_absolute_value
        downscale = 2.0 ** (127 - 2 ** (self.exponent_bits - 1))
        mask = torch.tensor(2 ** (23 - self.mantissa_bits) - 1, device=x.device)
        if self.rounding == "stochastic":
            srbitsbar = 23 - self.mantissa_bits - self.srbits
            offset = (
                torch.randint(
                    0, 2**self.srbits, x.shape, dtype=torch.int32, device=x.device
                )
                << srbitsbar
            )
            # Correct for bias.  We can do this only for srbits < 23-mantissa_bits,
            # but it is only likely to matter when srbits is small.
            if srbitsbar > 0:
                offset += 1 << (srbitsbar - 1)

        elif self.rounding == "nearest":
            offset = mask // 2
        else:  # pragma: no cover
            raise ValueError(
                f'Unexpected FPFormat(rounding="{self.rounding}"),'
                ' expected "stochastic" or "nearest"'
            )
        q = x.to(torch.float32)
        q = torch.clip(x, -absmax, absmax)
        q /= downscale
        q = ((q.view(torch.int32) + offset) & ~mask).view(torch.float32)
        q *= downscale
        return q.to(x.dtype)

    def quantise_fwd(self, x: Tensor) -> Tensor:
        """Quantise the given tensor in the forward pass only."""

        class QuantiseForward(torch.autograd.Function):
            @staticmethod
            def forward(ctx: torch.autograd.function.FunctionCtx, x: Tensor) -> Tensor:
                return self.quantise(x)

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
            def forward(ctx: torch.autograd.function.FunctionCtx, x: Tensor) -> Tensor:
                return x

            @staticmethod
            def backward(  # type:ignore[override]
                ctx: torch.autograd.function.FunctionCtx, grad_y: Tensor
            ) -> Tensor:
                return self.quantise(grad_y)

        return QuantiseBackward.apply(x)  # type: ignore


def format_to_tuple(format: FPFormat) -> Tuple[int, int]:
    """Convert the format into a tuple of `(exponent_bits, mantissa_bits)`"""
    return (format.exponent_bits, format.mantissa_bits)


def tuple_to_format(t: Tuple[int, int]) -> FPFormat:
    """Given a tuple of `(exponent_bits, mantissa_bits)` returns the corresponding
    :class:`FPFormat`"""
    return FPFormat(*t)


__all__ = generate__all__(__name__)

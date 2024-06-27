# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import collections

import torch

from ..formats import FPFormat


def test_fp_format() -> None:
    fmt = FPFormat(2, 1)
    assert fmt.bits == 4
    assert str(fmt) == "E2M1-SR"
    assert fmt.max_absolute_value == 3
    assert fmt.min_absolute_normal == 0.5
    assert fmt.min_absolute_subnormal == 0.25
    assert set(fmt.quantise_fwd(torch.linspace(-4, 4, steps=100)).tolist()) == {
        sx for x in [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3] for sx in [x, -x]
    }
    assert set(
        FPFormat(3, 0).quantise_fwd(torch.linspace(-10, 10, steps=1000)).abs().tolist()
    ) == {0, 0.125, 0.25, 0.5, 1, 2, 4, 8}


def test_fp_format_rounding() -> None:
    n = 10000
    x = -1.35

    y_nearest = FPFormat(2, 1, rounding="nearest").quantise(torch.full((n,), x))
    assert collections.Counter(y_nearest.tolist()) == {-1.5: n}

    for srbits in (0, 13):
        srformat = FPFormat(2, 1, rounding="stochastic", srbits=srbits)
        y_stochastic = srformat.quantise(torch.full((n,), x))
        count = collections.Counter(y_stochastic.tolist())
        assert count.keys() == {-1.5, -1.0}
        expected_ratio = (1.35 - 1.0) / 0.5
        nearest_ratio = count[-1.5] / sum(count.values())
        std_x3 = 3 * (expected_ratio * (1 - expected_ratio) / n) ** 0.5
        assert expected_ratio - std_x3 < nearest_ratio < expected_ratio + std_x3


def test_fp_format_bwd() -> None:
    fmt = FPFormat(2, 1)
    x = torch.randn(100, requires_grad=True)
    y = fmt.quantise_bwd(x * 1)
    y.backward(torch.linspace(-4, 4, steps=100))  # type: ignore[no-untyped-call]
    assert x.grad is not None
    assert set(x.grad.tolist()) == {
        sx for x in [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3] for sx in [x, -x]
    }

# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest

from ..constraints import (
    amean,
    apply_constraint,
    gmean,
    hmean,
    to_grad_input_scale,
    to_output_scale,
)


def test_gmean() -> None:
    assert gmean(1.0) == 1.0
    assert gmean(8.0, 2.0) == 4.0
    assert gmean(12.75, 55.0, 0.001) == pytest.approx(0.8884322)


def test_hmean() -> None:
    assert hmean(1.0) == 1.0
    assert hmean(8.0, 2.0) == 3.2
    assert hmean(12.75, 55.0, 0.001) == pytest.approx(0.00299971)


def test_amean() -> None:
    assert amean(1.0) == 1.0
    assert amean(8.0, 2.0) == 5.0
    assert amean(12.75, 55.0, 0.001) == pytest.approx(22.583667)


def test_to_output_scale() -> None:
    assert to_output_scale(2, 3) == 2
    assert to_output_scale(2, 3, 4) == 2


def test_to_grad_input_scale() -> None:
    assert to_grad_input_scale(2, 3) == 3


def test_apply_constraint() -> None:
    assert apply_constraint("gmean", 1.0, 4.0, 2.0) == (2.0, 2.0, 2.0)
    assert apply_constraint("hmean", 8.0, 2.0) == (3.2, 3.2)
    with pytest.raises(ValueError):
        apply_constraint("invalid", 8.0, 2.0)

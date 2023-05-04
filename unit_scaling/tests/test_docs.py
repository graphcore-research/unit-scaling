# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest

from ..docs import _validate


def f(a, b: int, c="3", d: float = 4.0) -> str:  # type: ignore
    return f"{a} {b} {c} {d}"


def test_validate_no_args() -> None:
    def g() -> int:
        return 0

    valid_g = _validate(g)
    assert valid_g() == 0


def test_validate_positional_args() -> None:
    # Works with no unsupported args
    valid_f = _validate(f)
    assert valid_f(None, 2) == "None 2 3 4.0"
    assert valid_f(None, 2, "3", 4.5) == "None 2 3 4.5"

    # Works with some unsupported args
    valid_f = _validate(f, unsupported_args=["c", "d"])

    # Works if unsupported args are not present or equal default
    assert valid_f(None, 2) == "None 2 3 4.0"
    assert valid_f(None, 2, "3", 4.0) == "None 2 3 4.0"

    # Doesn't work if non-default unsupported args provided
    with pytest.raises(ValueError) as e:
        valid_f(None, 2, "3.4")
    assert "argument has not been implemented" in str(e.value)
    with pytest.raises(ValueError) as e:
        valid_f(None, 2, "3", 4.5)
    assert "argument has not been implemented" in str(e.value)


def test_validate_positional_kwargs() -> None:
    # Works with no unsupported args
    valid_f = _validate(f)
    assert valid_f(None, 2) == "None 2 3 4.0"
    assert valid_f(None, 2, c="3", d=4.5) == "None 2 3 4.5"

    # Works with some unsupported args
    valid_f = _validate(f, unsupported_args=["c", "d"])

    # Works if unsupported args are not present or equal default
    assert valid_f(None, 2) == "None 2 3 4.0"
    assert valid_f(None, 2, c="3")

    # Doesn't work if non-default unsupported args provided
    with pytest.raises(ValueError) as e:
        valid_f(None, 2, c="3.4")
    assert "argument has not been implemented" in str(e.value)
    with pytest.raises(ValueError) as e:
        valid_f(None, 2, d=4.5)
    assert "argument has not been implemented" in str(e.value)


def test_validate_invalid_arg() -> None:
    with pytest.raises(ValueError) as e:
        _validate(f, unsupported_args=["z"])
    assert "is not valid" in str(e.value)


def test_validate_no_default_arg() -> None:
    with pytest.raises(ValueError) as e:
        _validate(f, unsupported_args=["b"])
    assert "has no default value" in str(e.value)

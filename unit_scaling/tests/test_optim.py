# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from typing import Any, Dict, List, Optional, Type, cast

import pytest
import torch
from torch import nn, tensor, zeros
from torch.testing import assert_close

import unit_scaling as uu
import unit_scaling.functional as U

from ..optim import (
    SGD,
    Adam,
    AdamW,
    lr_scale_func_adam,
    lr_scale_func_sgd,
    scaled_parameters,
)


@pytest.mark.parametrize("opt_type", (Adam, AdamW, SGD))
def test_optim_optimizers(opt_type: Type[torch.optim.Optimizer]) -> None:
    torch.manual_seed(100)
    inputs = torch.randn(10, 16)
    outputs = torch.randn(10, 25)
    model = uu.Linear(16, 25)
    opt = opt_type(
        model.parameters(), lr=0.01, weight_decay=1e-6  # type:ignore[call-arg]
    )
    opt.zero_grad()
    loss = U.mse_loss(model(inputs), outputs)
    loss.backward()  # type:ignore[no-untyped-call]
    opt.step()
    assert U.mse_loss(model(inputs), outputs) < loss


@pytest.mark.parametrize("opt", ["adam", "sgd"])
@pytest.mark.parametrize("readout_constraint", [None, "to_output_scale"])
def test_scaled_parameters(opt: str, readout_constraint: Optional[str]) -> None:
    model = nn.Sequential(
        uu.Embedding(2**8, 2**4),
        uu.DepthSequential(*(uu.Linear(2**4, 2**4, bias=True) for _ in range(3))),
        uu.LinearReadout(
            2**4,
            2**10,
            bias=False,
            weight_mup_type="output",
            constraint=readout_constraint,
        ),
    )

    base_lr = 0.1
    base_wd = 0.001
    param_groups = scaled_parameters(
        model.parameters(),
        dict(sgd=lr_scale_func_sgd(readout_constraint), adam=lr_scale_func_adam)[opt],
        lr=base_lr,
        weight_decay=base_wd,
    )

    # Match parameters based on shapes, as their names have disappeared
    sqrt_d = 3**0.5
    shape_to_expected_lr = {
        (2**8, 2**4): base_lr / 2**2,  # embedding.weight
        (2**4, 2**4): base_lr / 2**2 / sqrt_d,  # stack.linear.weight
        (2**4,): base_lr / sqrt_d,  # stack.linear.bias
        (2**10, 2**4): base_lr,  # linear.weight (output)
    }
    if opt == "sgd" and readout_constraint == "to_output_scale":
        shape_to_expected_lr[(2**8, 2**4)] *= 2**4
        shape_to_expected_lr[(2**4, 2**4)] *= 2**4
        shape_to_expected_lr[(2**4,)] *= 2**4

    for shape, expected_lr in shape_to_expected_lr.items():
        for g in param_groups:
            assert isinstance(g, dict)
            (param,) = g["params"]
            if param.shape == shape:
                assert g["lr"] == pytest.approx(
                    expected_lr, rel=1e-3
                ), f"bad LR for param.shape={shape}"
                assert g["weight_decay"] == pytest.approx(
                    base_wd / expected_lr, rel=1e-3
                ), f"bad WD for param.shape={shape}"


def test_scaled_parameters_with_existing_groups() -> None:
    original_params = cast(
        List[Dict[str, Any]],
        [
            # Two parameters in this group, sharing a tensor LR, which must be cloned
            dict(
                params=[
                    uu.Parameter(zeros(1, 2**4), mup_type="weight"),
                    uu.Parameter(zeros(2, 2**6), mup_type="output"),
                ],
                lr=torch.tensor(0.3),
                weight_decay=0.05,
            ),
            # One parameter in this group, with no explicit LR or WD
            dict(
                params=[
                    uu.Parameter(
                        zeros(3, 2**6), mup_type="weight", mup_scaling_depth=5
                    ),
                ],
            ),
        ],
    )

    g0, g1, g2 = scaled_parameters(
        original_params, lr_scale_func_adam, lr=torch.tensor(0.02)
    )

    assert isinstance(g0, dict)
    assert g0["params"][0].shape == (1, 2**4)
    assert_close(g0["lr"], tensor(0.3 / 2**2))  # also checks it's still a Tensor
    assert_close(g0["weight_decay"], 0.05 * 2**2 / 0.3)

    assert isinstance(g1, dict)
    assert g1["params"][0].shape == (2, 2**6)
    assert_close(g1["lr"], tensor(0.3))
    assert_close(g1["weight_decay"], 0.05 / 0.3)

    assert isinstance(g2, dict)
    assert g2["params"][0].shape == (3, 2**6)
    assert_close(g2["lr"], tensor(0.02 / 2**3 / 5**0.5))
    assert g2["weight_decay"] == 0

    # ### Check error conditions ###

    # No lr, missing for group 1
    with pytest.raises(ValueError):
        params = scaled_parameters(original_params, lr_scale_func_adam)
    # No need for an lr when all groups have it explicitly
    params = scaled_parameters(original_params[:1], lr_scale_func_adam)
    assert len(params) == 2  # type:ignore[arg-type]

    # Non-unit-scaling parameters
    with pytest.raises(ValueError):
        params = scaled_parameters(
            original_params + [dict(params=[nn.Parameter(zeros(4, 2**4))])],
            lr_scale_func_adam,
            lr=0.1,
        )
    # Allow non-unit-scaling parameters
    params = scaled_parameters(
        original_params + [dict(params=[nn.Parameter(zeros(4, 2**4))])],
        lr_scale_func_adam,
        lr=0.1,
        allow_non_unit_scaling_params=True,
    )
    assert len(params) == 4  # type:ignore[arg-type]

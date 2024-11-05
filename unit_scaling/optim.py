# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

"""Optimizer wrappers that apply scaling rules for u-muP.

Provides :class:`Adam`, :class:`AdamW`, :class:`SGD` as out-of-the-box
optimizers.

Alternatively, :func:`scaled_parameters` provides finer control by
transforming a parameter group for any downstream optimizer, given a
function that defines the LR scaling rules.
"""

# mypy: disable-error-code="no-any-return"

from typing import Any, Callable, Optional, Union

import torch
from torch import Tensor
from torch.optim.optimizer import ParamsT

from ._internal_utils import generate__all__
from .docs import inherit_docstring
from .parameter import ParameterData, has_parameter_data


def lr_scale_for_depth(param: ParameterData) -> float:
    """Calculate the LR scaling factor for depth only."""
    if param.mup_scaling_depth is None:
        return 1
    return param.mup_scaling_depth**-0.5


def _get_fan_in(param: ParameterData) -> int:
    # Note: the "fan_in" of an embedding layer is the hidden (output) dimension
    if len(param.shape) == 1:
        return param.shape[0]
    if len(param.shape) == 2:
        return param.shape[1]
    if len(param.shape) == 3:
        return param.shape[1] * param.shape[2]
    raise ValueError(
        f"Cannot get fan_in of `ndim >= 4` param, shape={tuple(param.shape)}"
    )


def lr_scale_func_sgd(
    readout_constraint: Optional[str],
) -> Callable[[ParameterData], float]:
    """Calculate the LR scaling factor for :class:`torch.optim.SGD`."""

    if readout_constraint is None:
        # If there is no readout constraint we will have unit-scaled gradients and hence
        # unit-scaled weight updates. In this case the scaling rules are the same as
        # for Adam, which naturally has unit-scaled weight updates.
        return lr_scale_func_adam
    elif readout_constraint == "to_output_scale":

        def lr_scale_func_sgd_inner(param: ParameterData) -> float:
            scale = lr_scale_for_depth(param)

            if param.mup_type in ("bias", "norm"):
                return scale * param.shape[0]
            if param.mup_type == "weight":
                return scale * _get_fan_in(param) ** 0.5
            if param.mup_type == "output":
                return scale
            assert False, f"Unexpected mup_type {param.mup_type}"

        return lr_scale_func_sgd_inner
    else:
        assert False, f"Unhandled readout constraint: {readout_constraint}"


def lr_scale_func_adam(param: ParameterData) -> float:
    """Calculate the LR scaling factor for :class:`torch.optim.Adam`
    and :class:`torch.optim.AdamW`.
    """
    scale = lr_scale_for_depth(param)
    if param.mup_type in ("bias", "norm"):
        return scale
    if param.mup_type == "weight":
        return scale * _get_fan_in(param) ** -0.5
    if param.mup_type == "output":
        return scale
    assert False, f"Unexpected mup_type {param.mup_type}"


def scaled_parameters(
    params: ParamsT,
    lr_scale_func: Callable[[ParameterData], float],
    lr: Union[None, float, Tensor] = None,
    weight_decay: float = 0,
    independent_weight_decay: bool = True,
    allow_non_unit_scaling_params: bool = False,
) -> ParamsT:
    """Create optimizer-appropriate **lr-scaled** parameter groups.

    This method creates param_groups that apply the relevant scaling factors for u-muP
    models. For example::

        torch.optim.Adam(uu.optim.scaled_parameters(
            model.parameters(), uu.optim.adam_lr_scale_func, lr=1.0
        ))

    Args:
        params (ParamsT): an iterable of parameters of parameter groups, as passed to
            a torch optimizer.
        lr_scale_func (Callable): gets the optimizer-appropriate learning rate scale,
            based on a parameter tagged with `mup_type` and `mup_scaling_depth`. For
            example, :func:`lr_scale_func_sgd`.
        lr (float, optional): global learning rate (overridden by groups).
        weight_decay (float, optional): weight decay value (overridden by groups).
        independent_weight_decay (bool, optional): enable lr-independent weight decay,
            which performs an update per-step that does not depend on lr.
        allow_non_unit_scaling_params (bool, optional): by default, this method fails
            if passed any regular non-unit-scaled params; set to `True` to disable this
            check.

    Returns:
        ParamsT: for passing on to the optimizer.
    """

    result = []
    for entry in params:
        group = dict(params=[entry]) if isinstance(entry, Tensor) else entry.copy()
        group.setdefault("lr", lr)  # type: ignore[arg-type]
        group.setdefault("weight_decay", weight_decay)  # type: ignore[arg-type]
        if group["lr"] is None:
            raise ValueError(
                "scaled_params() requires lr to be provided,"
                " unless passing parameter groups which already have an lr"
            )
        for param in group["params"]:
            # Careful not to overwrite `lr` or `weight_decay`
            param_lr = group["lr"]
            if has_parameter_data(param):  # type: ignore[arg-type]
                if isinstance(param_lr, Tensor):
                    param_lr = param_lr.clone()
                param_lr *= lr_scale_func(param)  # type: ignore[operator]
            elif not allow_non_unit_scaling_params:
                raise ValueError(
                    "Non-unit-scaling parameter (no mup_type),"
                    f" shape {tuple(param.shape)}"
                )
            param_weight_decay = group["weight_decay"]
            if independent_weight_decay:
                # Note: only independent of peak LR, not of schedule
                param_weight_decay /= float(param_lr)  # type: ignore

            result.append(
                dict(
                    params=[param],
                    lr=param_lr,
                    weight_decay=param_weight_decay,
                    **{
                        k: v
                        for k, v in group.items()
                        if k not in ("params", "lr", "weight_decay")
                    },
                )
            )
    return result


@inherit_docstring(
    short_description="An **lr-scaled** version of :class:`torch.optim.SGD` for u-muP."
    "`readout_constraint` should match the `constraint` arg used in `LinearReadout`."
)
class SGD(torch.optim.SGD):

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        *args: Any,
        weight_decay: float = 0,
        independent_weight_decay: bool = True,
        allow_non_unit_scaling_params: bool = False,
        readout_constraint: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        params = scaled_parameters(
            params,
            lr_scale_func_sgd(readout_constraint),
            lr=lr,
            weight_decay=weight_decay,
            independent_weight_decay=independent_weight_decay,
            allow_non_unit_scaling_params=allow_non_unit_scaling_params,
        )
        # No need to forward {lr, weight_decay}, as each group has these specified
        super().__init__(params, *args, **kwargs)


@inherit_docstring(
    short_description="An **lr-scaled** version of :class:`torch.optim.Adam` for u-muP."
)
class Adam(torch.optim.Adam):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        *args: Any,
        weight_decay: float = 0,
        independent_weight_decay: bool = True,
        allow_non_unit_scaling_params: bool = False,
        **kwargs: Any,
    ) -> None:
        params = scaled_parameters(
            params,
            lr_scale_func_adam,
            lr=lr,
            weight_decay=weight_decay,
            independent_weight_decay=independent_weight_decay,
            allow_non_unit_scaling_params=allow_non_unit_scaling_params,
        )
        # No need to forward {lr, weight_decay}, as each group has these specified
        super().__init__(params, *args, **kwargs)


@inherit_docstring(
    short_description=(
        "An **lr-scaled** version of :class:`torch.optim.AdamW` for u-muP."
    )
)
class AdamW(torch.optim.AdamW):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        *args: Any,
        weight_decay: float = 0,
        independent_weight_decay: bool = True,
        allow_non_unit_scaling_params: bool = False,
        **kwargs: Any,
    ) -> None:
        params = scaled_parameters(
            params,
            lr_scale_func_adam,
            lr=lr,
            weight_decay=weight_decay,
            independent_weight_decay=independent_weight_decay,
            allow_non_unit_scaling_params=allow_non_unit_scaling_params,
        )
        # No need to forward {lr, weight_decay}, as each group has these specified
        super().__init__(params, *args, **kwargs)


__all__ = generate__all__(__name__)

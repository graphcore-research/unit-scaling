# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

"""Extends :class:`torch.nn.Parameter` with attributes for u-μP."""

# mypy: disable-error-code="attr-defined, method-assign, no-untyped-call"

from collections import OrderedDict
from typing import Any, Dict, Literal, Optional, Protocol, TypeGuard

import torch
from torch import Tensor, nn

MupType = Literal["weight", "bias", "norm", "output"]


class ParameterData(Protocol):
    """Extra fields for :class:`torch.nn.Parameter`, tagging u-μP metadata.

    Objects supporting this protocol should implicitly also support
    :class:`torch.nn.Parameter`.
    """

    mup_type: MupType
    mup_scaling_depth: Optional[int]
    shape: torch.Size  # repeated from nn.Parameter, for convenience


def has_parameter_data(parameter: nn.Parameter) -> TypeGuard[ParameterData]:
    """Check that the parameter supports the :class:`ParameterData` protocol."""
    return (
        getattr(parameter, "mup_type", None) in MupType.__args__
        and hasattr(parameter, "mup_scaling_depth")
        and isinstance(parameter.mup_scaling_depth, (type(None), int))
    )


def _parameter_deepcopy(self: nn.Parameter, memo: Dict[int, Any]) -> nn.Parameter:
    result: nn.Parameter = nn.Parameter.__deepcopy__(self, memo)
    result.mup_type = self.mup_type
    result.mup_scaling_depth = self.mup_scaling_depth
    return result


def _rebuild_parameter_with_state(*args: Any, **kwargs: Any) -> nn.Parameter:
    p: nn.Parameter = torch._utils._rebuild_parameter_with_state(*args, **kwargs)
    p.__deepcopy__ = _parameter_deepcopy.__get__(p)
    p.__reduce_ex__ = _parameter_reduce_ex.__get__(p)
    return p


def _parameter_reduce_ex(self: nn.Parameter, protocol: int) -> Any:
    # Based on `torch.nn.Parameter.__reduce_ex__`, filtering out the
    # dynamic methods __deepcopy__ and __reduce_ex__, as these
    # don't unpickle
    state = {
        k: v
        for k, v in torch._utils._get_obj_state(self).items()
        if k not in ["__deepcopy__", "__reduce_ex__"]
    }
    return (
        _rebuild_parameter_with_state,
        (self.data, self.requires_grad, OrderedDict(), state),
    )


def Parameter(
    data: Tensor, mup_type: MupType, mup_scaling_depth: Optional[int] = None
) -> nn.Parameter:
    """Construct a u-μP parameter object, an annotated :class:`torch.nn.Parameter`.

    The returned parameter also supports the :class:`ParameterData` protocol:

        p = uu.Parameter(torch.zeros(10), mup_type="weight")
        assert p.mup_type == "weight"
        assert p.mup_scaling_depth is None
    """
    p = nn.Parameter(data)
    p.mup_type = mup_type
    p.mup_scaling_depth = mup_scaling_depth
    p.__deepcopy__ = _parameter_deepcopy.__get__(p)
    p.__reduce_ex__ = _parameter_reduce_ex.__get__(p)
    # Note: cannot override __repr__ as it's __class__.__repr__
    return p

# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Unit-scaled versions of common `torch.nn.functional` functions."""

from __future__ import annotations  # required for docs to alias type annotations

import sys
from math import log, pi, prod
from types import FunctionType
from typing import Callable, Dict, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from ._internal_utils import generate__all__
from .constraints import apply_constraint
from .core.functional import logarithmic_interpolation, rms, scale_elementwise
from .docs import (
    binary_constraint_docstring,
    docstring_from,
    format_docstring,
    mult_docstring,
    ternary_constraint_docstring,
)
from .scale import scale_bwd, scale_fwd


def _get_broadcast_sizes(*args: Tensor) -> Tuple[int, ...]:
    """Returns the product of the dimensions added to each arg when broadcasting."""
    output_broadcast_shape = torch.broadcast_shapes(  # type: ignore [no-untyped-call]
        *(a.shape for a in args)
    )
    output_numel = output_broadcast_shape.numel()
    return tuple(output_numel // a.shape.numel() for a in args)


def _unscaled_gelu(x: Tensor, mult: float, approximate: str) -> Tensor:
    if mult == 1:
        return F.gelu(x, approximate=approximate)
    return F.gelu(x * mult, approximate=approximate) / mult


@docstring_from(
    F.gelu,
    short_description="Applies a **unit-scaled** GELU function.",
    add_args=[mult_docstring(), binary_constraint_docstring],
)
def gelu(
    input: Tensor,
    mult: float = 1.0,
    constraint: Optional[str] = "to_output_scale",
    approximate: str = "none",
) -> Tensor:
    # An empirical model of gelu output std given mult
    output_scale = logarithmic_interpolation(
        alpha=1 / (1 + 0.25 / mult**2),  # = sigmoid(log(4 * mult**2))
        lower=2**1,
        upper=(2 / (1 - 1 / pi)) ** 0.5,
    )
    grad_input_scale = logarithmic_interpolation(
        alpha=1 / (1 + 0.25 / mult**2),  # = sigmoid(log(4 * mult**2))
        lower=2**1,
        upper=2**0.5,
    )
    scaled_gelu = scale_elementwise(
        _unscaled_gelu, output_scale, grad_input_scale, constraint
    )
    return scaled_gelu(input, mult=mult, approximate=approximate)


def _unscaled_silu(x: Tensor, mult: float) -> Tensor:
    if mult == 1:
        return F.silu(x)
    return x * F.sigmoid(x * mult)


@docstring_from(
    F.silu,
    short_description="Applies a **unit-scaled** SiLU function.",
    add_args=[mult_docstring(), binary_constraint_docstring],
    unsupported_args=["inplace"],
)
def silu(
    input: Tensor,
    mult: float = 1.0,
    constraint: Optional[str] = "to_output_scale",
    inplace: bool = False,
) -> Tensor:
    # An empirical model of swish output std given mult
    output_scale = logarithmic_interpolation(
        alpha=1 / (1 + 0.25 / mult**2),  # = sigmoid(log(4 * mult**2))
        lower=2**1,
        upper=(2 / (1 - 1 / pi)) ** 0.5,
    )
    grad_input_scale = logarithmic_interpolation(
        alpha=1 / (1 + 1 / mult**2),  # = sigmoid(log(mult**2))
        lower=2**1,
        upper=2**0.5,
    )
    scaled_silu = scale_elementwise(
        _unscaled_silu, output_scale, grad_input_scale, constraint
    )
    return scaled_silu(input, mult=mult)


@format_docstring(mult_docstring())
def silu_glu(input: Tensor, gate: Tensor, mult: float = 1.0) -> Tensor:
    """Applies a **unit-scaled** gated linear unit for `input * silu(gate)`.

    .. math::
        \\text{{silu_glu}}(x, g) = x * g * \\sigma(g),
        \\text{{where }} \\sigma(g) \\text{{ is the logistic sigmoid.}}

    Args:
        input (Tensor): linear input
        gate (Tensor): gate (SiLU) input
        {0}

    Returns:
        Tensor: a scaled output, the same shape as `input`
    """
    # An empirical model of swish output std given mult
    scale = logarithmic_interpolation(
        alpha=1 / (1 + 1 / mult**2),  # = sigmoid(log(mult**2))
        lower=2**1,
        upper=2**0.5,
    )
    input = scale_bwd(input, scale)
    gate = scale_bwd(gate, scale)
    output = input * _unscaled_silu(gate, mult=mult)
    return scale_fwd(output, scale)


def _unscaled_softmax(
    x: Tensor, dim: int, dtype: Optional[torch.dtype], mult: float
) -> Tensor:
    return F.softmax(x * mult, dim=dim, dtype=dtype)


@docstring_from(
    F.softmax,
    short_description="Applies a **unit-scaled** softmax function.",
    add_args=[mult_docstring(), binary_constraint_docstring],
)
def softmax(
    input: Tensor,
    dim: int,
    dtype: Optional[torch.dtype] = None,
    constraint: Optional[str] = "to_output_scale",
    mult: float = 1.0,
) -> Tensor:
    dim_size = input.shape[dim]
    # Empirical model
    output_scale = logarithmic_interpolation(
        alpha=1 / (1 + 4 / mult**2),  # = sigmoid(log(mult**2 / 4))
        lower=dim_size,  # flat limit
        upper=dim_size**0.5,  # one-hot limit
    )
    grad_input_scale = logarithmic_interpolation(
        alpha=1 / (1 + 4 / mult**2),  # = sigmoid(log(mult**2 / 4))
        lower=dim_size / mult,  # flat limit
        upper=(dim_size / mult) ** 0.5,  # one-hot limit
    )
    scaled_softmax = scale_elementwise(
        _unscaled_softmax, output_scale, grad_input_scale, constraint
    )
    return scaled_softmax(input, dim=dim, dtype=dtype, mult=mult)


@docstring_from(
    F.dropout,
    short_description="Applies a **unit-scaled** dropout function.",
    unsupported_args=["inplace"],
)
def dropout(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    output_scale = grad_input_scale = (1 - p) ** 0.5
    scaled_dropout = scale_elementwise(
        F.dropout, output_scale, grad_input_scale, constraint=None
    )
    return scaled_dropout(input, p, training, inplace)


@docstring_from(
    torch.matmul,
    short_description="A **unit-scaled** matrix product of two tensors.",
    add_args=[ternary_constraint_docstring],
)
def matmul(
    left: Tensor,
    right: Tensor,
    constraint: Optional[str] = "to_output_scale",
) -> Tensor:
    left_size = left.shape[-2]
    inner_size = left.shape[-1]
    right_size = right.shape[-1]

    output_scale = inner_size**-0.5
    left_grad_scale = right_size**-0.5
    right_grad_scale = left_size**-0.5

    output_scale, left_grad_scale, right_grad_scale = apply_constraint(
        constraint, output_scale, left_grad_scale, right_grad_scale
    )

    left = scale_bwd(left, left_grad_scale)
    right = scale_bwd(right, right_grad_scale)
    output = torch.matmul(left, right)
    return scale_fwd(output, output_scale)


@docstring_from(
    F.linear,
    short_description="Applies a **unit-scaled** linear transformation.",
    add_args=[
        binary_constraint_docstring,
        "scale_power ((float, float, float), optional): scaling power"
        " for each of (output, grad(input), grad(weight|bias))",
    ],
)
def linear(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    constraint: Optional[str] = "to_output_scale",
    scale_power: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> Tensor:
    fan_out, fan_in = weight.shape
    batch_size = input.numel() // fan_in

    output_scale = 1 / fan_in ** scale_power[0]
    grad_input_scale = 1 / fan_out ** scale_power[1]
    grad_weight_scale = grad_bias_scale = 1 / batch_size ** scale_power[2]

    output_scale, grad_input_scale = apply_constraint(
        constraint, output_scale, grad_input_scale
    )

    input = scale_bwd(input, grad_input_scale)
    weight = scale_bwd(weight, grad_weight_scale)
    bias = scale_bwd(bias, grad_bias_scale) if bias is not None else None
    output = F.linear(input, weight, bias)
    return scale_fwd(output, output_scale)


@docstring_from(
    F.linear,
    short_description="Applies a **unit-scaled** linear transformation,"
    " for the final network output.",
    add_args=[binary_constraint_docstring],
)
def linear_readout(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    constraint: Optional[str] = None,
) -> Tensor:
    return linear(
        input, weight, bias, constraint=constraint, scale_power=(1.0, 0.5, 0.5)
    )


@docstring_from(
    F.conv1d,
    short_description="Applies a **unit-scaled** 1D convolution.",
    add_args=[
        binary_constraint_docstring,
        "scale_power ((float, float, float), optional): scaling power"
        " for each of (output, grad(input), grad(weight|bias))",
    ],
)
def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    constraint: Optional[str] = "to_output_scale",
    scale_power: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> Tensor:
    fan_out, fan_in, kernel_size = weight.shape
    seq_len = input.shape[-1]
    out_size = (seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    batch_size = out_size
    if len(input.shape) > 2:
        batch_size *= input.shape[:-2].numel()

    output_scale = 1 / (fan_in * kernel_size) ** scale_power[0]
    grad_input_scale = (stride * groups / (fan_out * kernel_size)) ** scale_power[1]
    grad_weight_scale = grad_bias_scale = 1 / batch_size ** scale_power[2]

    output_scale, grad_input_scale = apply_constraint(
        constraint, output_scale, grad_input_scale
    )

    input = scale_bwd(input, grad_input_scale)
    weight = scale_bwd(weight, grad_weight_scale)
    bias = scale_bwd(bias, grad_bias_scale) if bias is not None else None
    output = F.conv1d(input, weight, bias, stride, padding, dilation, groups)
    assert out_size == output.shape[-1]
    return scale_fwd(output, output_scale)


@docstring_from(
    F.layer_norm,
    short_description=(
        "Applies a **unit-scaled** Layer Normalization for last certain number"
        " of dimensions."
    ),
)
def layer_norm(
    input: Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    grad_weight_scale = grad_bias_scale = (
        prod(normalized_shape) / input.numel()
    ) ** 0.5
    if weight is not None:
        weight = scale_bwd(weight, grad_weight_scale)
    if bias is not None:
        bias = scale_bwd(bias, grad_bias_scale)
    return F.layer_norm(input, normalized_shape, weight, bias, eps)


def _unscaled_rms_norm(
    input: torch.Tensor,
    normalized_shape: Tuple[int, ...],
    weight: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    assert input.shape[-len(normalized_shape) :] == normalized_shape
    dims = tuple(range(-1, -1 - len(normalized_shape), -1))
    output = input / rms(input, dims=dims, keepdim=True, eps=eps)
    if weight is not None:
        output *= weight
    return output


def rms_norm(
    input: Tensor,
    normalized_shape: Tuple[int, ...],
    weight: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """Apply **unit-scaled** RMS Normalization for last certain number of dimensions.

    See :class:`~unit_scaling.RMSNorm` for details.
    """
    if weight is not None:
        scale = (prod(normalized_shape) / input.numel()) ** 0.5
        weight = scale_bwd(weight, scale)
    return _unscaled_rms_norm(input, normalized_shape, weight, eps=eps)


@docstring_from(
    torch.add,
    short_description="Applies a **unit-scaled** addition.",
    unsupported_args=["alpha"],
    add_args=[ternary_constraint_docstring],
)
def add(
    input: Union[Tensor, int, float],
    other: Union[Tensor, int, float],
    constraint: Optional[str] = "to_output_scale",
    alpha: int = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    # Adding a constant shouldn't change scale
    if isinstance(input, (int, float)) or isinstance(other, (int, float)):
        return torch.add(input, other, out=out)
    input_broadcast_size, other_broadcast_size = _get_broadcast_sizes(input, other)
    input_grad_scale = input_broadcast_size**-0.5
    other_grad_scale = other_broadcast_size**-0.5
    scalar_input = input.numel() == 1 or other.numel() == 1

    # If the input is a scalar the output std doesn't change, and hence we don't scale
    output_scale = 2**-0.5 if not scalar_input else 1.0

    output_scale, input_grad_scale, other_grad_scale = apply_constraint(
        constraint, output_scale, input_grad_scale, other_grad_scale
    )

    input = scale_bwd(input, input_grad_scale)
    other = scale_bwd(other, other_grad_scale)
    out = torch.add(input, other, out=out)
    return scale_fwd(out, output_scale)


def residual_split(input: Tensor, tau: float = 1.0) -> Tuple[Tensor, Tensor]:
    """Splits a tensor into an `residual` and `skip` tensor, prior to being used
    in a residual layer, with a relative weighting tau applied to the residual branch.
    Should be used in conjunction with :py:func:`unit_scaling.functional.residual_add`.

    This is necessary as unit scaling delays the residual branch scaling in the backward
    pass such that residual gradients are still unit-scaled.

    The need for a relative weighting between the two branches (tau) is a result of
    unit-scaling normalising the scales of the two branches. In non-unit-scaled models
    the two branches may have different scales, which can be beneficial to training.
    The tau factor allows unit scaling to behave as though the branches have different
    scales.

    Args:
        input (Tensor): the tensor to which the residual layer is to be applied.
        tau (float, optional): the ratio of scale of contributions of the residual
            branch to the skip connection. Values larger than one favor skip over
            residual. Defaults to 1 (equal contribution).

    Returns:
        Tuple[Tensor, Tensor]: resulting tensors in the order: `residual, skip`.
    """
    denom = (1 + tau**2) ** 0.5
    residual = scale_bwd(input, tau / denom)
    skip = scale_bwd(input, 1 / denom)
    return residual, skip


def residual_add(residual: Tensor, skip: Tensor, tau: float = 1.0) -> Tensor:
    """Adds a residual connection and skip connection together, with a relative
    weighting tau applied to the residual branch. Should be used in conjunction with
    :py:func:`unit_scaling.functional.residual_split`.

    Args:
        residual (Tensor): the tensor coming out of the residual connection.
        skip (Tensor): the tensor coming out of the skip connection.
        tau (float, optional): the ratio of scale of contributions of the residual
            branch to the skip connection. Larger values favor skip over residual.
            Defaults to 1 (equal contribution).

    Returns:
        Tensor: the result of the combined residual and skip tensors.
    """
    denom = (1 + tau**2) ** 0.5
    residual = scale_fwd(residual, tau / denom)
    skip = scale_fwd(skip, 1 / denom)
    return residual + skip


def residual_apply(
    fn: Callable[[Tensor], Tensor], input: Tensor, tau: float = 1.0
) -> Tensor:
    """Apply a weighted residual branch, maintaining unit scale.

    Combines :func:`residual_split` and :func:`residual_add` into a single function.

    Args:
        fn (Callable): the residual function to apply.
        input (Tensor): input tensor, also to use for the skip connection.
        tau (float, optional): the ratio of scale of contributions of the residual
            branch to the skip connection. Larger values favor skip over residual.
            Defaults to 1 (equal contribution).
    """
    residual, skip = residual_split(input, tau=tau)
    residual = fn(residual)
    return residual_add(residual, skip, tau=tau)


@docstring_from(
    F.embedding,
    short_description=(
        "A **unit-scaled** lookup table that looks up embeddings in a fixed dictionary"
        "and size."
    ),
    unsupported_args=["scale_grad_by_freq", "sparse"],
)
def embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    batch_size = prod(input.shape)
    weight = scale_bwd(weight, (weight.shape[0] / batch_size) ** 0.5)
    return F.embedding(
        input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
    )


@docstring_from(
    F.scaled_dot_product_attention,
    short_description=(
        "A **unit-scaled** dot-product attention function. Note that this will use"
        " whatever underlying PyTorch scaled_dot_product_attention implementation"
        " is available, so if flash attention is enabled it will be used here too."
        "\n\n"
        "Computes scaled dot product attention on query, key and value tensors,"
        " using an optional attention mask if passed, and applying dropout if a"
        " probability greater than 0.0 is specified."
        "\n\n"
        "Note that the scaling rule for causal attention will only be applied if"
        " is_causal is True, as an arbitrary mask does not identify causal versus"
        " non-causal."
    ),
    add_args=[mult_docstring()],
)
def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    mult: float = 1.0,
) -> Tensor:
    *_, seq_len, d_head = value.shape
    # Empirical model of attention output std given mult and seq_len
    scale = (1 - dropout_p) ** 0.5 / logarithmic_interpolation(
        alpha=1 / (1 + 4 * d_head / mult**2),  # = sigmoid(log(mult**2 / (4 * d_head)))
        lower=((log(seq_len) if is_causal else 1) / seq_len) ** 0.5,
        upper=1.0,
    )
    query, key, value = (scale_bwd(t, scale) for t in (query, key, value))
    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=mult / d_head,
    )
    return scale_fwd(out, scale)


@docstring_from(
    F.cross_entropy,
    short_description=(
        "Computes the **unit-scaled** cross entropy loss between input logits and"
        " target."
    ),
    unsupported_args=["weight", "size_average", "reduce", "label_smoothing"],
    add_args=[mult_docstring()],
)
def cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    mult: float = 1.0,
) -> Tensor:
    if len(input.shape) == 2:
        batch_size, vocab_size = input.shape
    elif len(input.shape) == 1:
        batch_size, vocab_size = 1, input.shape[0]
    else:
        assert False, (
            f"cross_entropy input shape is {input.shape}, but should be either"
            " (vocab_size,) or (batch_size, vocab_size)"
        )
    input = scale_bwd(input, vocab_size / (vocab_size - 1) ** 0.5)
    input = scale_fwd(input, mult)
    loss = F.cross_entropy(
        input,
        target,
        weight,
        size_average,
        ignore_index,
        reduce,
        reduction="sum",
        label_smoothing=label_smoothing,
    )
    if reduction == "mean":
        return scale_fwd(loss, 1 / batch_size)
    assert reduction == "sum"
    return loss


@docstring_from(
    F.mse_loss,
    short_description="Computes the **unit-scaled** element-wise mean squared error.",
    unsupported_args=["size_average", "reduce"],
)
def mse_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    if input.shape != target.shape:
        raise ValueError(
            "U.mse_loss requires input.shape == target.shape,"
            f" actual input.shape={tuple(input.shape)},"
            f" target.shape={tuple(target.shape)}"
        )
    grad_scale = 8**-0.5
    input = scale_bwd(input, grad_scale)
    target = scale_bwd(target, grad_scale)
    loss = F.mse_loss(input, target, size_average, reduce, reduction="sum")
    if reduction == "mean":
        return scale_fwd(loss, 1 / input.nelement())
    assert reduction == "sum"
    return loss


def _gen_torch_function_map() -> Dict[FunctionType, FunctionType]:
    torch_objects = {name: getattr(torch, name) for name in dir(torch)}
    torch_objects = {**torch_objects, **{name: getattr(F, name) for name in dir(F)}}
    current_module = sys.modules[__name__]
    function_map = {}
    for unit_fn_name in dir(current_module):
        unit_fn = getattr(current_module, unit_fn_name)
        if isinstance(unit_fn, FunctionType) and unit_fn_name in torch_objects:
            torch_fn = cast(FunctionType, torch_objects[unit_fn_name])
            function_map[torch_fn] = unit_fn
    return function_map


__all__ = generate__all__(__name__)


torch_map = _gen_torch_function_map()

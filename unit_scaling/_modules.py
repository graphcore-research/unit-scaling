# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Unit-scaled versions of common `torch.nn` modules."""

from __future__ import annotations  # required for docs to alias type annotations

from typing import Any, Iterable, List, Optional, Tuple, Union

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import functional as U
from ._internal_utils import generate__all__
from .core.functional import ResidualScalingFn, transformer_residual_scaling_rule
from .docs import (
    binary_constraint_docstring,
    format_docstring,
    inherit_docstring,
    mult_docstring,
)
from .parameter import MupType, Parameter, has_parameter_data


@inherit_docstring(
    short_description="Applies a **unit-scaled** Gaussian Error Linear Units function:",
    add_args=[mult_docstring(), binary_constraint_docstring],
)
class GELU(nn.GELU):
    def __init__(
        self,
        mult: float = 1.0,
        constraint: Optional[str] = "to_output_scale",
        approximate: str = "none",
    ) -> None:
        super().__init__(approximate=approximate)
        self.mult = mult
        self.constraint = constraint

    def forward(self, input: Tensor) -> Tensor:
        return U.gelu(
            input,
            mult=self.mult,
            approximate=self.approximate,
            constraint=self.constraint,
        )


@inherit_docstring(
    short_description="Applies a **unit-scaled** Sigmoid Linear Unit function:",
    add_args=[mult_docstring(), binary_constraint_docstring],
    unsupported_args=["inplace"],
)
class SiLU(nn.SiLU):
    def __init__(
        self,
        mult: float = 1.0,
        constraint: Optional[str] = "to_output_scale",
        inplace: bool = False,
    ) -> None:
        super().__init__(inplace=inplace)
        self.mult = mult
        self.constraint = constraint

    def forward(self, input: Tensor) -> Tensor:
        return U.silu(
            input,
            mult=self.mult,
            constraint=self.constraint,
            inplace=self.inplace,
        )


@inherit_docstring(
    short_description=(
        "Applies a **unit-scaled** Softmax function to an n-dimensional input Tensor."
        "\n\n"
        "The standard softmax rescales values so that the elements of the"
        " n-dimensional output Tensor lie in the range [0,1] and sum to 1. Unit scaling"
        " multiplies by n, meaning the output Tensor lies in the range [0,n]."
        "\n\n"
        "The documentation below is from the standard `nn.Softmax`"
        " implementation. Values there (for example [0,1] ranges) should be adjusted"
        " accordingly."
    ),
    add_args=[mult_docstring(), binary_constraint_docstring],
)
class Softmax(nn.Softmax):
    def __init__(
        self,
        dim: int,
        mult: float = 1.0,
        constraint: Optional[str] = "to_output_scale",
    ) -> None:
        super().__init__(dim=dim)
        self.mult = mult
        self.constraint = constraint

    def forward(self, input: Tensor) -> Tensor:
        return U.softmax(
            input, dim=self.dim, mult=self.mult, constraint=self.constraint
        )


@inherit_docstring(
    short_description="A **unit-scaled** implementation of Dropout.",
    unsupported_args=["inplace"],
)
class Dropout(nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p, inplace)

    def forward(self, input: Tensor) -> Tensor:
        return U.dropout(input, self.p, self.training, self.inplace)


@inherit_docstring(
    short_description=(
        "Applies a **unit-scaled** linear transformation to the incoming data."
        "\nNote that this layer sets :code:`bias=False` by default."
    ),
    add_args=[binary_constraint_docstring],
)
class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Any = None,
        dtype: Any = None,
        constraint: Optional[str] = "to_output_scale",
        weight_mup_type: MupType = "weight",
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.constraint = constraint
        self.weight = Parameter(self.weight.data, mup_type=weight_mup_type)
        if self.bias is not None:
            self.bias = Parameter(self.bias.data, mup_type="bias")

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input: Tensor) -> Tensor:
        return U.linear(input, self.weight, self.bias, self.constraint)


@inherit_docstring(
    short_description=(
        "Applies a **unit-scaled** linear transformation to the incoming data,"
        " scaled appropriately for the final network output."
        "\nNote that this layer sets :code:`bias=False` by default."
    ),
    add_args=[binary_constraint_docstring],
)
class LinearReadout(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Any = None,
        dtype: Any = None,
        constraint: Optional[str] = None,
        weight_mup_type: MupType = "output",
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            device,
            dtype,
            constraint=constraint,
            weight_mup_type=weight_mup_type,
        )

    def forward(self, input: Tensor) -> Tensor:
        return U.linear_readout(input, self.weight, self.bias, self.constraint)


@inherit_docstring(
    short_description=(
        "Applies a **unit-scaled** 1D convolution to the incoming data."
        "\nNote that this layer sets :code:`bias=False` by default."
        "We also require padding to be supplied as an integer, not a string."
    ),
    add_args=[binary_constraint_docstring],
)
class Conv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        device: Any = None,
        dtype: Any = None,
        constraint: Optional[str] = "to_output_scale",
        weight_mup_type: MupType = "weight",
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        assert isinstance(padding, int), "only `int` is supported for padding type"
        self.kernel_size = kernel_size  # type:ignore[assignment]
        self.stride = stride  # type:ignore[assignment]
        self.padding = padding  # type:ignore[assignment]
        self.dilation = dilation  # type:ignore[assignment]
        self.constraint = constraint
        self.weight = Parameter(self.weight.data, mup_type=weight_mup_type)
        if self.bias is not None:
            self.bias = Parameter(self.bias.data, mup_type="bias")

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input: Tensor) -> Tensor:
        if self.padding_mode != "zeros":
            input = F.pad(
                input, self._reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return U.conv1d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


@inherit_docstring(
    short_description=(
        "Applies a **unit-scaled** Layer Normalization over a mini-batch of inputs."
        "\nNote that this layer sets :code:`elementwise_affine=False` by default."
    ),
)
class LayerNorm(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = False,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        if self.weight is not None:
            self.weight = Parameter(self.weight.data, mup_type="norm")
        if self.bias is not None:
            self.bias = Parameter(self.bias.data, mup_type="bias")

    def forward(self, input: Tensor) -> Tensor:
        return U.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )


class RMSNorm(nn.Module):
    """Applies a **unit-scaled** RMS normalisation over trailing dimensions.

    This layer implements the operation as described in the paper
    `Root Mean Square Layer Normalization <https://arxiv.org/abs/1910.07467>`__.

    .. math::
        y = \\frac{x}{ \\sqrt{\\sum x^2 + \\epsilon}} * \\gamma

    Note that this layer sets :code:`elementwise_affine=False` by default.

    Args:
        normalized_shape (Tuple[int]): input shape, for an expected input tensor
          of shape `(*, *normalized_shape)`.
        elementwise_affine (bool): a boolean value that when set to True, this
          module has learnable per-element weight parameters initialized to ones.
          Default: False.
        eps (float): a value added to the denominator for numerical stability.
          Default: 1e-5.

    Attributes:
        weight: the learnable weights of the module of shape `normalized_shape` when
          elementwise_affine is set to True. The values are initialized to 1.
    """

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = False,
    ):
        super().__init__()
        self.normalized_shape = (
            (normalized_shape,)
            if isinstance(normalized_shape, int)
            else normalized_shape
        )
        self.eps = eps
        self.weight = (
            Parameter(torch.ones(normalized_shape), "norm")
            if elementwise_affine
            else None
        )

    def forward(self, input: Tensor) -> Tensor:
        return U.rms_norm(
            input,
            normalized_shape=self.normalized_shape,
            weight=self.weight,
            eps=self.eps,
        )


@inherit_docstring(
    short_description=(
        "A **unit-scaled** lookup table that looks up embeddings in a fixed dictionary"
        " and size."
    ),
    unsupported_args=["scale_grad_by_freq", "sparse"],
)
class Embedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        _freeze: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight,
            _freeze=_freeze,
            device=device,
            dtype=dtype,
        )
        self.weight = Parameter(self.weight.data, mup_type="weight")

    def forward(self, input: Tensor) -> Tensor:
        return U.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


@inherit_docstring(
    short_description=(
        "Computes a **unit-scaled** cross entropy loss between input logits and target."
    ),
    unsupported_args=["weight", "size_average", "reduce", "label_smoothing"],
    add_args=[mult_docstring()],
)
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        mult: float = 1.0,
        weight: Optional[Tensor] = None,
        size_average: Optional[bool] = None,
        ignore_index: int = -100,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        self.mult = mult

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return U.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
            mult=self.mult,
        )


class MLP(nn.Module):
    """A **unit-scaled** implementation of an MLP layer using SwiGLU.

    Args:
        hidden_size (int): the hidden dimension size of the input.
        expansion_factor (int): the factor by which the MLP's intermediate size
            increases relative to `hidden_size`.
    """

    def __init__(self, hidden_size: int, expansion_factor: int = 4) -> None:
        super().__init__()
        intermediate_size = hidden_size * expansion_factor
        # Note: constraint=None is safe here, because we know that the forward and
        # backward constraints are mirrored between {linear_1, linear_gate} and
        # linear_2.
        self.linear_1 = Linear(hidden_size, intermediate_size, constraint=None)
        self.linear_gate = Linear(hidden_size, intermediate_size, constraint=None)
        self.linear_2 = Linear(intermediate_size, hidden_size, constraint=None)

    def forward(self, input: Tensor) -> Tensor:
        z = U.silu_glu(self.linear_1(input), self.linear_gate(input))
        return self.linear_2(z)  # type:ignore[no-any-return]


@format_docstring(mult_docstring())
class MHSA(nn.Module):
    """A **unit-scaled** implementation of a multi-head self-attention layer.

    Args:
        hidden_size (int): the hidden dimension size of the input.
        heads (int): the number of attention heads.
        is_causal (bool): causal masking (for non-padded sequences).
        dropout_p (float, optional): the probability of the post-softmax dropout.
        {0}
    """

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        is_causal: bool,
        dropout_p: float = 0.0,
        mult: float = 1.0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.mult = mult
        self.linear_qkv = Linear(hidden_size, 3 * hidden_size)
        self.linear_o = Linear(hidden_size, hidden_size)

    def forward(self, input: Tensor) -> Tensor:
        q_k_v = self.linear_qkv(input)
        q, k, v = einops.rearrange(q_k_v, "b s (z h d) -> z b h s d", h=self.heads, z=3)
        qkv = U.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p, is_causal=self.is_causal, mult=self.mult
        )
        qkv = einops.rearrange(qkv, "b h s d -> b s (h d)")
        return self.linear_o(qkv)  # type: ignore


class TransformerLayer(nn.Module):
    """A **unit-scaled** implementation of a PreNorm
    (see https://arxiv.org/abs/2002.04745) transformer layer.

    Warning: using `constraint=None` here will likely give incorrect gradients.

    Args:
        hidden_size (int): the hidden dimension size of the input.
        heads (int): the number of attention heads.
        mhsa_tau (float): the weighting of the multi-head-self-attention
            branch relative to the skip connection.
        mlp_tau (float): the weighting of the MLP
            branch relative to the skip connection.
        is_causal (bool): causal masking (for non-padded sequences).
        dropout_p (float, optional): the probability of residual and post-softmax
            dropout.
    """

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        mhsa_tau: float,
        mlp_tau: float,
        is_causal: bool,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.dropout_p = dropout_p
        self.mhsa_tau = mhsa_tau
        self.mlp_tau = mlp_tau
        self.mhsa_norm = RMSNorm(hidden_size)
        self.mhsa = MHSA(hidden_size, heads, is_causal=is_causal, dropout_p=dropout_p)
        self.mlp_norm = RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size)

    def forward(self, input: Tensor) -> Tensor:
        input, skip = U.residual_split(input, tau=self.mhsa_tau)
        input = self.mhsa_norm(input)
        input = self.mhsa(input)
        input = U.dropout(input, self.dropout_p, self.training)
        input = U.residual_add(input, skip, tau=self.mhsa_tau)

        input, skip = U.residual_split(input, tau=self.mlp_tau)
        input = self.mlp_norm(input)
        input = self.mlp(input)
        input = U.dropout(input, self.dropout_p, self.training)
        return U.residual_add(input, skip, tau=self.mlp_tau)


@inherit_docstring(
    "A :class:`torch.nn.ModuleList` that automatically configures the depth"
    " for sake of scaling."
    "\nNote that this does not track depth changes caused by modifications"
    " after initial construction."
)
class DepthModuleList(nn.ModuleList):
    def __init__(self, modules: Iterable[nn.Module]) -> None:
        super().__init__(modules)
        for name, parameter in self.named_parameters():
            if not has_parameter_data(parameter):
                raise ValueError(
                    f"Parameter {name} is not a unit_scaling.Parameter."
                    " Is it from a regular nn.Module?"
                )
            assert parameter.mup_scaling_depth is None
            parameter.mup_scaling_depth = len(self)


@inherit_docstring(
    "A :class:`torch.nn.Sequential` that automatically configures the depth"
    " for sake of scaling."
    "\nNote that this does not track depth changes caused by modifications"
    " after initial construction."
)
class DepthSequential(nn.Sequential):
    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        for name, parameter in self.named_parameters():
            if not has_parameter_data(parameter):
                raise ValueError(
                    f"Parameter {name} is not a unit_scaling.Parameter."
                    " Is it from a regular nn.Module?"
                )
            assert parameter.mup_scaling_depth is None
            parameter.mup_scaling_depth = len(self)


class TransformerStack(DepthSequential):
    """A **unit-scaled** transformer stack, applying a residual scaling rule.

    See :code:`TransformerLayer` for arguments.

    Args:
        layers (int): number of transformer layers.
        residual_scaling (Callable[[int, int], float], optional): scheme for
            controlling residual weights in the transformer stack; see
            :func:`unit_scaling.core.functional.transformer_residual_scaling_rule`
            (default).
    """

    def __init__(
        self,
        layers: int,
        residual_scaling: ResidualScalingFn = transformer_residual_scaling_rule(),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *(
                TransformerLayer(
                    **kwargs,
                    mhsa_tau=residual_scaling(2 * i, 2 * layers),
                    mlp_tau=residual_scaling(2 * i + 1, 2 * layers),
                )
                for i in range(layers)
            )
        )


class TransformerDecoder(nn.Sequential):  # pragma: no cover
    """A **unit-scaled** implementation of a decoder-type transformer.

    Note: this class is currently just for demonstrating scaling and lacks key
    functionality (for example masking, positional embeddings, usage for
    inference).

    Args:
        hidden_size (int): the hidden dimension size of the input.
        vocab_size (int): the number of tokens in the vocabulary.
        layers (int): the number of transformer layers.
        heads (int): the number of attention heads.
        dropout_p (float, optional): the probability of embedding, residual and
            post-softmax dropout.
        residual_scaling (Callable[[int, int], float], optional): scheme for
            controlling residual weights in the transformer trunk; see
            :func:`unit_scaling.core.functional.transformer_residual_scaling_rule`
            (default).
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        layers: int,
        heads: int,
        dropout_p: float = 0.0,
        residual_scaling: ResidualScalingFn = transformer_residual_scaling_rule(),
    ) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, hidden_size)
        self.layers = TransformerStack(
            layers=layers,
            hidden_size=hidden_size,
            heads=heads,
            is_causal=True,
            dropout_p=dropout_p,
            residual_scaling=residual_scaling,
        )
        self.final_norm = RMSNorm(hidden_size)
        self.projection = LinearReadout(hidden_size, vocab_size)

    def loss(self, input_ids: Tensor) -> Tensor:
        logits = self(input_ids).float()
        return U.cross_entropy(
            logits[..., :-1, :].flatten(end_dim=-2), input_ids[..., 1:].flatten()
        )


__all__ = generate__all__(__name__)

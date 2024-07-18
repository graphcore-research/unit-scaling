# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Unit-scaled versions of common `torch.nn` modules."""

from __future__ import annotations  # required for docs to alias type annotations

from typing import Any, Optional, Tuple, Union

import einops
import torch
from torch import Tensor, nn

from . import functional as U
from ._internal_utils import generate__all__
from .docs import (
    binary_constraint_docstring,
    format_docstring,
    inherit_docstring,
    mult_docstring,
    variadic_constraint_docstring,
)


@inherit_docstring(
    short_description="Applies a **unit-scaled** Gaussian Error Linear Units function:",
    add_args=[mult_docstring(), binary_constraint_docstring],
    unsupported_args=["approximate"],
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
    ),
    add_args=[binary_constraint_docstring],
)
class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
        constraint: Optional[str] = "to_output_scale",
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.constraint = constraint

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input: Tensor) -> Tensor:
        return U.linear(input, self.weight, self.bias, self.constraint)


@inherit_docstring(
    short_description=(
        "Applies a **unit-scaled** Layer Normalization over a mini-batch of inputs."
    ),
)
class LayerNorm(nn.LayerNorm):
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

    Args:
        normalized_shape (Tuple[int]): input shape, for an expected input tensor
          of shape `(*, *normalized_shape)`.
        elementwise_affine (bool): a boolean value that when set to True, this
          module has learnable per-element weight parameters initialized to ones.
          Default: True.
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
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.normalized_shape = (
            (normalized_shape,)
            if isinstance(normalized_shape, int)
            else normalized_shape
        )
        self.eps = eps
        self.weight = (
            nn.Parameter(torch.ones(normalized_shape)) if elementwise_affine else None
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


@format_docstring(binary_constraint_docstring)
class MLP(nn.Module):
    """A **unit-scaled** implementation of an MLP layer.

    Args:
        hidden_size (int): the hidden dimension size of the input.
        act_fn (nn.Module): the activation function module. Defaults to `GELU()`.
        expansion_factor (int): the factor by which the MLP's intermediate size
            increases relative to `hidden_size`.
        {0}
    """

    def __init__(
        self,
        hidden_size: int,
        act_fn: nn.Module = GELU(),
        expansion_factor: int = 4,
        constraint: Optional[str] = "to_output_scale",
    ) -> None:
        super().__init__()
        intermediate_size = hidden_size * expansion_factor
        self.linear_1 = Linear(hidden_size, intermediate_size, constraint=constraint)
        self.act_fn = act_fn
        self.linear_2 = Linear(intermediate_size, hidden_size, constraint=constraint)

    def forward(self, input: Tensor) -> Tensor:
        input = self.linear_1(input)
        input = self.act_fn(input)
        return self.linear_2(input)  # type: ignore


@format_docstring(mult_docstring(), variadic_constraint_docstring)
class MHSA(nn.Module):
    """A **unit-scaled** implementation of a multi-head self-attention layer.

    Warning: using `constraint=None` here will likely give incorrect gradients.

    Args:
        hidden_size (int): the hidden dimension size of the input.
        heads (int): the number of attention heads.
        dropout_p (float, optional): the probability of the post-softmax dropout.
        {0}
        {1}
    """

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        dropout_p: float,
        constraint: Optional[str] = "to_output_scale",
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dropout_p = dropout_p
        self.linear_qkv = Linear(
            hidden_size, 3 * hidden_size, bias=False, constraint=constraint
        )
        self.linear_o = Linear(
            hidden_size, hidden_size, bias=False, constraint=constraint
        )
        self.constraint = constraint

    def forward(self, input: Tensor) -> Tensor:
        q_k_v = self.linear_qkv(input)
        q, k, v = einops.rearrange(q_k_v, "b s (z h d) -> z b h s d", h=self.heads, z=3)
        qk = U.matmul(q, k.transpose(-1, -2), constraint=self.constraint)
        qk = U.softmax(qk, dim=-1, constraint=self.constraint)
        qk = U.dropout(qk, self.dropout_p, training=self.training)
        qkv = U.matmul(qk, v, constraint=self.constraint)
        qkv = einops.rearrange(qkv, "b h s d -> b s (h d)")
        return self.linear_o(qkv)  # type: ignore


@format_docstring(variadic_constraint_docstring)
class TransformerLayer(nn.Module):
    """A **unit-scaled** implementation of a PreNorm
    (see https://arxiv.org/abs/2002.04745) transformer layer.

    Warning: using `constraint=None` here will likely give incorrect gradients.

    Args:
        hidden_size (int): the hidden dimension size of the input.
        heads (int): the number of attention heads.
        dropout_p (float, optional): the probability of the post-softmax dropout.
        act_fn (nn.Module): the activation function module. Defaults to `GELU()`.
        mhsa_tau (float, optional): the weighting of the multi-head-self-attention
            branch relative to the skip connection. Defaults to 0.01.
        mlp_tau (float, optional): the weighting of the MLP
            branch relative to the skip connection. Defaults to 0.5.
        {0}
    """

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        dropout_p: float,
        act_fn: nn.Module = GELU(),
        mhsa_tau: float = 0.01,
        mlp_tau: float = 0.5,
        constraint: Optional[str] = "to_output_scale",
    ) -> None:
        super().__init__()
        self.dropout_p = dropout_p
        self.mhsa_tau = mhsa_tau
        self.mlp_tau = mlp_tau
        self.mhsa_layer_norm = LayerNorm(hidden_size)
        self.mhsa = MHSA(hidden_size, heads, dropout_p, constraint=constraint)
        self.mlp_layer_norm = LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, act_fn, constraint=constraint)

    def forward(self, input: Tensor) -> Tensor:
        input, skip = U.residual_split(input, tau=self.mhsa_tau)
        input = self.mhsa_layer_norm(input)
        input = self.mhsa(input)
        input = U.dropout(input, self.dropout_p, self.training)
        input = U.residual_add(input, skip, tau=self.mhsa_tau)

        input, skip = U.residual_split(input, tau=self.mlp_tau)
        input = self.mlp_layer_norm(input)
        input = self.mlp(input)
        input = U.dropout(input, self.dropout_p, self.training)
        return U.residual_add(input, skip, tau=self.mlp_tau)


@format_docstring(variadic_constraint_docstring)
class TransformerDecoder(nn.Module):  # pragma: no cover
    """A **unit-scaled** implementation of a decoder-type transformer.

    Note: this class is currently just for demonstrating scaling and lacks key
    functionality (for example causal masking, positional embeddings, usage for
    inference).

    Warning: using `constraint=None` here will likely give incorrect gradients.

    Args:
        hidden_size (int): the hidden dimension size of the input.
        vocab_size (int): the number of tokens in the vocabulary.
        layers (int): the number of transformer layers.
        heads (int): the number of attention heads.
        dropout_p (float, optional): the probability of the post-softmax dropout.
        act_fn (nn.Module): the activation function module. Defaults to `GELU()`.
        mhsa_tau (float, optional): the weighting of the multi-head-self-attention
            branch relative to the skip connection. Defaults to 0.01.
        mlp_tau (float, optional): the weighting of the MLP
            branch relative to the skip connection. Defaults to 0.5.
        {0}
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        layers: int,
        heads: int,
        dropout_p: float,
        act_fn: nn.Module = GELU(),
        mhsa_tau: float = 0.01,
        mlp_tau: float = 0.5,
        constraint: Optional[str] = "to_output_scale",
    ) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, hidden_size)
        self.dropout_p = dropout_p
        self.initial_layer_norm = LayerNorm(hidden_size)
        self.transformer_layers = nn.Sequential(
            *(
                TransformerLayer(
                    hidden_size, heads, dropout_p, act_fn, mhsa_tau, mlp_tau, constraint
                )
                for _ in range(layers)
            )
        )
        self.final_layer_norm = LayerNorm(hidden_size)

    def forward(self, input_ids: Tensor, labels: Tensor) -> Tensor:
        input = self.embedding(input_ids)
        input = U.dropout(input, self.dropout_p, self.training)
        input = self.initial_layer_norm(input)
        input = self.transformer_layers(input)
        input = self.final_layer_norm(input)
        input = U.linear(input, self.embedding.weight, bias=None, constraint=None)
        return U.cross_entropy(input.flatten(end_dim=-2), labels.flatten())


__all__ = generate__all__(__name__)

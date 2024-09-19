# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import re

import torch

from .._modules import MHSA, MLP
from ..utils import analyse_module


def remove_scales(code: str) -> str:
    """Takes a code string containing scale annotations such as `(-> 1.0, <- 0.819)` and
    replaces them all with `(-> _, <- _)`."""
    return re.sub(r"\(-> \d+(\.\d+)?, <- \d+(\.\d+)?\)", "(-> _, <- _)", code)


def test_analyse_mlp() -> None:
    batch_size = 2**10
    hidden_size = 2**10
    input = torch.randn(batch_size, hidden_size).requires_grad_()
    backward = torch.randn(batch_size, hidden_size)

    annotated_code = analyse_module(
        MLP(hidden_size), input, backward, syntax_highlight=False
    )
    print(annotated_code)

    expected_code = """
def forward(self, input : Tensor) -> Tensor:
    input_1 = input;  (-> 1.0, <- 1.44)
    linear_1_weight = self.linear_1.weight;  (-> 1.0, <- 0.503)
    linear = U.linear(input_1, linear_1_weight, None, None);  (-> 1.0, <- 0.502)
    linear_gate_weight = self.linear_gate.weight;  (-> 1.0, <- 0.519)
    linear_1 = U.linear(input_1, linear_gate_weight, None, None);  (-> 1.0, <- 0.518)
    silu_glu = U.silu_glu(linear, linear_1);  (-> 1.0, <- 0.5)
    linear_2_weight = self.linear_2.weight;  (-> 1.0, <- 1.0)
    linear_2 = U.linear(silu_glu, linear_2_weight, None, None);  (-> 1.0, <- 1.0)
    return linear_2
    """.strip()  # noqa: E501

    assert remove_scales(annotated_code) == remove_scales(expected_code)


def test_analyse_mhsa() -> None:
    batch_size = 2**8
    seq_len = 2**6
    hidden_size = 2**6
    heads = 4
    input = torch.randn(batch_size, seq_len, hidden_size).requires_grad_()
    backward = torch.randn(batch_size, seq_len, hidden_size)

    annotated_code = analyse_module(
        MHSA(hidden_size, heads, is_causal=False, dropout_p=0.1),
        input,
        backward,
        syntax_highlight=False,
    )
    print(annotated_code)

    expected_code = """
def forward(self, input : Tensor) -> Tensor:
    input_1 = input;  (-> 1.0, <- 1.13)
    linear_qkv_weight = self.linear_qkv.weight;  (-> 1.01, <- 0.662)
    linear = U.linear(input_1, linear_qkv_weight, None, 'to_output_scale');  (-> 1.01, <- 0.633)
    rearrange = einops_einops_rearrange(linear, 'b s (z h d) -> z b h s d', h = 4, z = 3);  (-> 1.01, <- 0.633)
    getitem = rearrange[0];  (-> 1.0, <- 0.344)
    getitem_1 = rearrange[1];  (-> 1.0, <- 0.257)
    getitem_2 = rearrange[2];  (-> 1.02, <- 1.01)
    scaled_dot_product_attention = U.scaled_dot_product_attention(getitem, getitem_1, getitem_2, dropout_p = 0.1, is_causal = False, mult = 1.0);  (-> 1.04, <- 1.0)
    rearrange_1 = einops_einops_rearrange(scaled_dot_product_attention, 'b h s d -> b s (h d)');  (-> 1.04, <- 1.0)
    linear_o_weight = self.linear_o.weight;  (-> 1.0, <- 1.03)
    linear_1 = U.linear(rearrange_1, linear_o_weight, None, 'to_output_scale');  (-> 1.06, <- 1.0)
    return linear_1
    """.strip()  # noqa: E501

    assert remove_scales(annotated_code) == remove_scales(expected_code)

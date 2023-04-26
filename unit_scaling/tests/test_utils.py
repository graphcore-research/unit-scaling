# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import re

import torch

from ..modules import MHSA, MLP
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

    expected_code = """
def forward(self, input : torch.Tensor) -> torch.Tensor:
    input_1 = input;  (-> 1.0, <- 1.01)
    linear_1_weight = self.linear_1.weight;  (-> 1.0, <- 0.716)
    linear_1_bias = self.linear_1.bias;  (-> 0.0, <- 0.714)
    linear = U.linear(input_1, linear_1_weight, linear_1_bias, gmean);  (-> 0.707, <- 0.717)
    gelu = U.gelu(linear, gmean);  (-> 0.641, <- 0.708)
    linear_2_weight = self.linear_2.weight;  (-> 1.0, <- 0.691)
    linear_2_bias = self.linear_2.bias;  (-> 0.0, <- 0.998)
    linear_1 = U.linear(gelu, linear_2_weight, linear_2_bias, gmean);  (-> 0.973, <- 1.0)
    return linear_1
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
        MHSA(hidden_size, heads), input, backward, syntax_highlight=False
    )

    expected_code = """
def forward(self, input : torch.Tensor) -> torch.Tensor:
    input_1 = input;  (-> 1.0, <- 0.819)
    linear_qkv_weight = self.linear_qkv.weight;  (-> 1.01, <- 0.681)
    linear = U.linear(input_1, linear_qkv_weight, None, gmean);  (-> 0.766, <- 0.631)
    rearrange = einops_einops_rearrange(linear, 'b s (d z h) -> z b h s d', h = 4, z = 3);  (-> 0.766, <- 0.631)
    getitem = rearrange[0];  (-> 0.774, <- 0.463)
    getitem_1 = rearrange[1];  (-> 0.773, <- 0.34)
    getitem_2 = rearrange[2];  (-> 0.752, <- 0.929)
    transpose = getitem_1.transpose(-1, -2);  (-> 0.773, <- 0.34)
    matmul = U.matmul(getitem, transpose, constraint = gmean);  (-> 0.376, <- 0.344)
    softmax = U.softmax(matmul, dim = -1, constraint = gmean);  (-> 0.264, <- 0.477)
    dropout = U.dropout(softmax, 0.1);  (-> 0.34, <- 0.477)
    matmul_1 = U.matmul(dropout, getitem_2, constraint = gmean);  (-> 0.739, <- 1.0)
    rearrange_1 = einops_einops_rearrange(matmul_1, 'b h s d -> b s (h d)');  (-> 0.739, <- 1.0)
    linear_o_weight = self.linear_o.weight;  (-> 1.0, <- 0.73)
    linear_1 = U.linear(rearrange_1, linear_o_weight, None, gmean);  (-> 0.738, <- 1.0)
    return linear_1
    """.strip()  # noqa: E501

    assert remove_scales(annotated_code) == remove_scales(expected_code)

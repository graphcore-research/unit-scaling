# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch
import torch.nn as nn
from torch import Tensor

from ..utils import analyse_module


def test_analyse_module() -> None:
    class MLP(nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size * 4, hidden_size)

        def forward(self, x: Tensor) -> Tensor:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    hidden_size = 2**10
    dummy_input = torch.randn(hidden_size, hidden_size).requires_grad_()
    dummy_backward = torch.randn(hidden_size, hidden_size)

    annotated_code = analyse_module(
        MLP(hidden_size), dummy_input, dummy_backward, syntax_highlight=False
    )

    expected_code = """
def forward(self, x : torch.Tensor) -> torch.Tensor:  (-> 1.0, <- 0.236)
    fc1_weight = self.fc1.weight;  (-> 0.0181, <- 6.53)
    fc1_bias = self.fc1.bias;  (-> 0.018, <- 6.41)
    linear = torch._C._nn.linear(x, fc1_weight, fc1_bias);  (-> 0.578, <- 0.204)
    relu = torch.nn.functional.relu(linear, inplace = False);  (-> 0.337, <- 0.289)
    fc2_weight = self.fc2.weight;  (-> 0.00902, <- 13.1)
    fc2_bias = self.fc2.bias;  (-> 0.00897, <- 31.9)
    linear_1 = torch._C._nn.linear(relu, fc2_weight, fc2_bias);  (-> 0.238, <- 1.0)
    return linear_1
    """.strip()

    assert annotated_code == expected_code

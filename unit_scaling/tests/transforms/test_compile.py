# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...transforms import compile, unit_scale


def test_compile() -> None:
    class Module(nn.Module):  # pragma: no cover
        def __init__(
            self,
            hidden_size: int,
        ) -> None:
            super().__init__()
            self.layer_norm = nn.LayerNorm(hidden_size)
            self.l1 = nn.Linear(hidden_size, 4 * hidden_size)
            self.l2 = nn.Linear(4 * hidden_size, hidden_size)

        def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
            input = self.layer_norm(input)
            input = self.l1(input)
            input = F.gelu(input)
            input = self.l2(input)
            input = F.dropout(input, 0.2)
            return input, input.sum()

    mod = Module(2**6)
    x = torch.randn(2**3, 2**6)

    compile(mod)(x)
    compile(unit_scale(mod))(x)

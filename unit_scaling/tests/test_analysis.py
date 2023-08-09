# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pandas as pd
import torch.nn.functional as F
from torch import Tensor, nn, randn

from ..analysis import graph_to_dataframe, plot
from ..transforms import track_scales


def test_graph_to_dataframe() -> None:
    class Model(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.dim = dim
            self.linear = nn.Linear(dim, dim // 2)

        def forward(self, x: Tensor) -> Tensor:  # pragma: no covers
            y = F.relu(x)
            z = self.linear(y)
            return z.sum()  # type: ignore[no-any-return]

    b, dim = 2**4, 2**8
    input = randn(b, dim)
    model = Model(dim)
    model = track_scales(model)
    loss = model(input)
    loss.backward()

    graph = model.scales_graph()  # type: ignore[operator]
    df = graph_to_dataframe(graph)

    expected = pd.DataFrame.from_dict(
        {
            "layer": [
                "x",
                "x",
                "relu",
                "relu",
                "self_linear_weight",
                "self_linear_weight",
                "self_linear_bias",
                "self_linear_bias",
                "linear",
                "linear",
                "sum_1",
                "sum_1",
            ],
            "weight tensor": [
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
            ],
            "direction": [
                "fwd",
                "bwd",
                "fwd",
                "bwd",
                "fwd",
                "bwd",
                "fwd",
                "bwd",
                "fwd",
                "bwd",
                "fwd",
                "bwd",
            ],
            "tensor type": [
                "x",
                "grad_x",
                "x",
                "grad_x",
                "w",
                "grad_w",
                "w",
                "grad_w",
                "x",
                "grad_x",
                "x",
                "grad_x",
            ],
            "number of elements": [
                b * dim,
                b * dim,
                b * dim,
                b * dim,
                dim**2 // 2,
                dim**2 // 2,
                dim // 2,
                dim // 2,
                b * dim // 2,
                b * dim // 2,
                1,
                1,
            ],
        }
    )
    pd.testing.assert_frame_equal(expected, df[expected.columns])


def test_plot() -> None:
    class Model(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.dim = dim
            self.linear = nn.Linear(dim, dim // 2)

        def forward(self, x: Tensor) -> Tensor:  # pragma: no cover
            y = F.relu(x)
            z = self.linear(y)
            return z.sum()  # type: ignore[no-any-return]

    b, dim = 2**4, 2**8
    input = randn(b, dim)
    model = Model(dim)
    model = track_scales(model)
    loss = model(input)
    loss.backward()

    graph = model.scales_graph()  # type: ignore[operator]
    axes = plot(graph, "demo", xmin=2**-20, xmax=2**10)
    assert axes

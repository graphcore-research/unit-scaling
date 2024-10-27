# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import operator
from math import pi, sqrt
from typing import Any, Callable, Dict, Set, Tuple, Union

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn, randint, randn, randn_like
from torch.fx.graph import Graph
from torch.fx.node import Node

from ...transforms import (
    prune_non_float_tensors,
    prune_same_scale_tensors,
    prune_selected_nodes,
    track_scales,
)


def get_target_or_node_name(node: Node) -> Union[str, Callable[..., Any]]:
    return node.meta["clean_name"] if isinstance(node.target, str) else node.target


def get_targets(graph: Graph) -> Set[Union[str, Callable]]:  # type: ignore[type-arg]
    return set(get_target_or_node_name(node) for node in graph.nodes)


def get_target_map(
    graph: Graph,
) -> Dict[Union[str, Callable], Dict[str, Any]]:  # type: ignore[type-arg]
    return {get_target_or_node_name(node): node.meta for node in graph.nodes}


def test_track_scales() -> None:
    class Model(nn.Module):
        def forward(self, x: Tensor) -> Tensor:  # pragma: no cover
            x = F.relu(x)
            y = torch.ones_like(x, dtype=x.dtype)
            z = x + y
            return z.sum()

    model = Model()
    model = track_scales(model)
    assert len(model.scales_graph().nodes) == 0

    input = randn(2**4, 2**10)
    loss = model(input)

    graph = model.scales_graph()

    assert all("outputs_float_tensor" in n.meta for n in graph.nodes)
    meta_map = get_target_map(graph)

    assert "metrics" in meta_map["x"]
    assert meta_map["x"]["metrics"].bwd is None
    assert meta_map["x"]["metrics"].fwd.mean_abs == pytest.approx(
        sqrt(2 / pi), abs=0.01
    )
    assert meta_map["x"]["metrics"].fwd.abs_mean == pytest.approx(0, abs=0.01)
    assert meta_map["x"]["metrics"].fwd.std == pytest.approx(1, abs=0.01)
    assert meta_map["x"]["metrics"].fwd.numel == 2**14

    assert "metrics" in meta_map[F.relu]
    assert meta_map[F.relu]["metrics"].bwd is None
    assert (
        meta_map[F.relu]["metrics"].fwd.mean_abs
        == meta_map[F.relu]["metrics"].fwd.abs_mean
        == pytest.approx(sqrt(1 / (2 * pi)), abs=0.01)
    )
    assert meta_map[F.relu]["metrics"].fwd.std == pytest.approx(
        sqrt((1 - 1 / pi) / 2), abs=0.01
    )
    assert meta_map[F.relu]["metrics"].fwd.numel == 2**14

    assert "metrics" in meta_map[torch.ones_like]
    assert meta_map[torch.ones_like]["metrics"].bwd is None
    assert meta_map[torch.ones_like]["metrics"].fwd.mean_abs == 1.0
    assert meta_map[torch.ones_like]["metrics"].fwd.abs_mean == 1.0
    assert meta_map[torch.ones_like]["metrics"].fwd.std == 0.0
    assert meta_map[torch.ones_like]["metrics"].fwd.abs_max == 1.0
    assert meta_map[torch.ones_like]["metrics"].fwd.abs_min == 1.0
    assert meta_map[torch.ones_like]["metrics"].fwd.numel == 2**14

    assert "metrics" in meta_map[operator.add]
    assert "metrics" in meta_map["sum_1"]
    assert meta_map["sum_1"]["metrics"].fwd.numel == 1

    loss.backward()
    graph = model.scales_graph()

    meta_map = get_target_map(graph)

    assert meta_map["sum_1"]["metrics"].bwd is not None
    assert meta_map["sum_1"]["metrics"].bwd.numel == 1

    assert meta_map[operator.add]["metrics"].bwd is not None
    assert meta_map[operator.add]["metrics"].bwd.mean_abs == 1.0
    assert meta_map[operator.add]["metrics"].bwd.abs_mean == 1.0
    assert meta_map[operator.add]["metrics"].bwd.std == 0.0
    assert meta_map[operator.add]["metrics"].bwd.abs_max == 1.0
    assert meta_map[operator.add]["metrics"].bwd.abs_min == 1.0
    assert meta_map[operator.add]["metrics"].bwd.numel == 2**14

    assert meta_map["x"]["metrics"].bwd is not None
    assert meta_map["x"]["metrics"].bwd.std == pytest.approx(
        0.5, abs=0.01  # same as fwd pass except 0s are now 1s
    )


def test_prune_non_float_tensors() -> None:
    class Model(nn.Module):
        def __init__(self, emb_size: int, dim: int) -> None:
            super().__init__()
            self.emb = nn.Embedding(emb_size, dim)
            self.linear = nn.Linear(dim, dim)

        def forward(self, idxs: Tensor) -> Tuple[Tensor, Tensor]:  # pragma: no cover
            x = self.emb(idxs)
            scores = F.softmax(self.linear(x), dim=-1)
            top_idx = torch.argmax(scores, dim=-1)
            top_idx = torch.unsqueeze(top_idx, -1)
            top_score_x = torch.gather(x, -1, top_idx)
            top_score_x -= x.mean()
            return top_score_x, top_idx

    idxs = randint(0, 2**10, (2**3, 2**5))
    model = Model(2**10, 2**6)
    model = track_scales(model)
    model(idxs)

    graph = model.scales_graph()
    expected_targets = {
        "idxs",
        "self_modules_emb_parameters_weight",
        F.embedding,
        F.linear,
        "self_modules_linear_parameters_weight",
        "self_modules_linear_parameters_bias",
        F.softmax,
        torch.argmax,
        torch.unsqueeze,
        torch.gather,
        operator.isub,
        "mean",
        "output",
    }
    graph_targets = get_targets(graph)
    assert graph_targets == expected_targets

    graph = prune_non_float_tensors(graph)
    graph_targets = get_targets(graph)
    expected_targets -= {"idxs", torch.argmax, torch.unsqueeze}
    assert graph_targets == expected_targets


def test_prune_same_scale_tensors() -> None:
    class Model(nn.Module):
        def __init__(self, emb_size: int, dim: int) -> None:
            super().__init__()
            self.emb = nn.Embedding(emb_size, dim)
            self.linear = nn.Linear(dim, dim // 2)

        def forward(self, idxs: Tensor) -> Tuple[Tensor, Tensor]:  # pragma: no cover
            # idxs has 0 args -> shouldn't be pruned
            x = self.emb(idxs)  # emb has 1 float arg (weights) -> depends on tol
            _x = x.flatten(start_dim=0, end_dim=-1)  # 1 float, same scale -> prune
            x = _x.view(x.shape)  # 1 float arg, same scale -> prune
            y = self.linear(x)  # scale changes -> shouldn't be pruned
            scores = F.softmax(y, dim=-1)  # scale changes -> shouldn't be pruned
            top_idx = torch.argmax(scores, dim=-1)  # not float -> shouldn't be pruned
            top_idx = torch.unsqueeze(top_idx, -1)  # not float -> shouldn't be pruned
            top_score_x = torch.gather(x, -1, top_idx)  # small change -> depends on tol
            top_score_x += randn_like(top_score_x)  # 2 floats, same scale -> no prune
            return top_score_x, top_idx

    idxs = randint(0, 2**10, (2**3, 2**5))
    model = Model(2**10, 2**6)
    model = track_scales(model)
    model(idxs)

    graph = model.scales_graph()

    # Version-dependent, see https://github.com/graphcore-research/unit-scaling/pull/52
    var_lhs_flatten = "x"
    var_lhs_view = "x_1"
    expected_targets = {
        "idxs",
        "self_modules_emb_parameters_weight",
        F.embedding,
        var_lhs_flatten,
        var_lhs_view,
        "self_modules_linear_parameters_weight",
        "self_modules_linear_parameters_bias",
        F.linear,
        F.softmax,
        torch.argmax,
        torch.unsqueeze,
        torch.gather,
        randn_like,
        operator.iadd,
        "output",
    }
    graph_targets = get_targets(graph)
    assert graph_targets == expected_targets

    graph = prune_same_scale_tensors(graph)
    graph_targets = get_targets(graph)
    expected_targets -= {var_lhs_flatten, var_lhs_view}
    assert graph_targets == expected_targets

    graph = prune_same_scale_tensors(graph, rtol=2**-4)
    graph_targets = get_targets(graph)
    expected_targets -= {torch.gather, F.embedding}
    assert graph_targets == expected_targets


def test_prune_same_scale_tensors_with_grad() -> None:
    class Model(nn.Module):
        def forward(self, a: Tensor) -> Tensor:  # pragma: no cover
            b = a / 1.0  # same scale fwd & bwd
            c = b * 1.0  # same scale fwd, as b sums grads -> different scale bwd
            d = F.relu(c)  # different scale fwd & bwd
            e = b - d  # different scale fwd, same bwd
            f = e.sum()  # different scale fwd & bwd
            return f

    input = randn(2**6, 2**8)
    model = Model()
    model = track_scales(model)
    loss = model(input)

    graph = model.scales_graph()
    expected_targets = {
        "a",
        operator.truediv,
        operator.mul,
        F.relu,
        operator.sub,
        "f",
        "output",
    }
    graph_targets = get_targets(graph)
    assert graph_targets == expected_targets

    graph = prune_same_scale_tensors(graph)
    graph_targets = get_targets(graph)
    expected_targets -= {operator.truediv, operator.mul}
    assert graph_targets == expected_targets

    # The mul still has the same scale before & after in the fwd pass, but the same is
    # not true for its grads. It should no longer be pruned after `loss.backward`.
    loss.backward()
    graph = model.scales_graph()
    graph = prune_same_scale_tensors(graph)
    graph_targets = get_targets(graph)
    expected_targets.add(operator.mul)
    assert graph_targets == expected_targets


def test_prune_selected_nodes() -> None:
    class Model(nn.Module):
        def forward(self, x: Tensor) -> Tensor:  # pragma: no cover
            x = x + 1
            x = F.relu(x)
            x = torch.abs(x)
            return x.sum()

    input = randn(2**6, 2**8)
    model = Model()
    model = track_scales(model)
    model(input)

    graph = model.scales_graph()
    expected_targets = {
        "x",
        operator.add,
        F.relu,
        torch.abs,
        "sum_1",
        "output",
    }
    graph_targets = get_targets(graph)
    assert graph_targets == expected_targets

    graph = prune_selected_nodes(graph, targets=[torch.abs, F.relu])
    graph_targets = get_targets(graph)
    expected_targets -= {torch.abs, F.relu}
    assert graph_targets == expected_targets

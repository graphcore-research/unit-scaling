# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import logging
import math
import operator
from typing import Tuple

import torch
import torch.nn.functional as F
from pytest import LogCaptureFixture
from torch import Tensor, nn, randint, randn, randn_like

from ... import _modules as uu
from ... import functional as U
from ...transforms import (
    prune_non_float_tensors,
    prune_same_scale_tensors,
    prune_selected_nodes,
    simulate_fp8,
    track_scales,
    unit_scale,
)
from ..helper import assert_unit_scaled


def test_simulate_fp8_linear(caplog: LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    class Model(nn.Module):
        def __init__(self, d_in: int, d_out: int) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out)

        def forward(self, t: Tensor) -> Tensor:
            return self.linear(t).sum()  # type: ignore[no-any-return]

    input = randn(2**8, 2**9, requires_grad=True)
    model = Model(2**9, 2**10)
    output = model(input)
    output.backward()

    fp8_input = input.clone().detach().requires_grad_()
    fp8_model = simulate_fp8(model)
    fp8_output = fp8_model(fp8_input)
    fp8_output.backward()

    assert not torch.all(fp8_output == output)
    assert not torch.all(fp8_input.grad == input.grad)  # type: ignore
    assert not torch.all(
        fp8_model.linear.weight.grad == model.linear.weight.grad  # type: ignore
    )
    assert "quantising function" in caplog.text


def test_simulate_fp8_unit_scaled_linear() -> None:
    class Model(nn.Module):
        def __init__(self, d_in: int, d_out: int) -> None:
            super().__init__()
            self.linear = uu.Linear(d_in, d_out)

        def forward(self, t: Tensor) -> Tensor:
            return self.linear(t).sum()  # type: ignore[no-any-return]

    input = randn(2**8, 2**9, requires_grad=True)
    model = Model(2**9, 2**10)
    output = model(input)
    output.backward()

    fp8_input = input.clone().detach().requires_grad_()
    fp8_model = simulate_fp8(model)
    fp8_output = fp8_model(fp8_input)
    fp8_output.backward()

    assert not torch.all(fp8_output == output)
    assert not torch.all(fp8_input.grad == input.grad)  # type: ignore
    assert not torch.all(
        fp8_model.linear.weight.grad == model.linear.weight.grad  # type: ignore
    )


def test_simulate_fp8_attention() -> None:
    class Model(nn.Module):
        def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            return F.scaled_dot_product_attention(q, k, v).sum()  # type: ignore

    inputs = list(randn(2**8, 2**8, requires_grad=True) for _ in range(3))
    model = Model()
    output = model(*inputs)
    output.backward()

    fp8_inputs = list(t.clone().detach().requires_grad_() for t in inputs)
    fp8_model = simulate_fp8(model)

    fp8_output = fp8_model(*fp8_inputs)
    fp8_output.backward()

    assert not torch.all(fp8_output == output)
    for fp8_input, input in zip(fp8_inputs, inputs):
        assert not torch.all(fp8_input.grad == input.grad)  # type: ignore


def test_simulate_fp8_unit_scaled_attention() -> None:
    class Model(nn.Module):
        def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            return U.scaled_dot_product_attention(q, k, v).sum()  # type: ignore

    inputs = list(randn(2**8, 2**8, requires_grad=True) for _ in range(3))
    model = Model()
    output = model(*inputs)
    output.backward()

    fp8_inputs = list(t.clone().detach().requires_grad_() for t in inputs)
    fp8_model = simulate_fp8(model)

    fp8_output = fp8_model(*fp8_inputs)
    fp8_output.backward()

    assert not torch.all(fp8_output == output)
    for fp8_input, input in zip(fp8_inputs, inputs):
        assert not torch.all(fp8_input.grad == input.grad)  # type: ignore


def test_unit_scale(caplog: LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    def custom_gelu(x: Tensor) -> Tensor:
        inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        return 0.5 * x * (1.0 + torch.tanh(inner))

    class MLPLayer(nn.Module):  # pragma: no cover
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
            input = custom_gelu(input)
            input = self.l2(input)
            input = F.dropout(input, 0.2)
            return input, input.sum()

    input = randn(2**6, 2**10, requires_grad=True)
    model = MLPLayer(2**10)
    model = unit_scale(model, replace={custom_gelu: F.gelu})
    output, loss = model(input)
    loss.backward()

    assert_unit_scaled(
        output,
        input.grad,
        model.layer_norm.weight.grad,
        model.l1.weight.grad,
        model.l2.weight.grad,
        abs=0.2,
    )

    expected_logs = [
        "unit scaling weight",
        "setting bias to zero",
        "unit scaling function",
        "replacing function",
        "unconstraining node",
    ]
    for log_msg in expected_logs:
        assert log_msg in caplog.text


def test_unit_scale_residual_add(caplog: LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    class MLPLayer(nn.Module):
        def __init__(
            self,
            hidden_size: int,
        ) -> None:
            super().__init__()
            self.l1 = nn.Linear(hidden_size, hidden_size)
            self.l2 = nn.Linear(hidden_size, hidden_size)

        def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:  # pragma: no cover
            skip = input
            input = input + 1
            input = self.l1(input)
            input = input + skip
            skip = input
            input += 1
            input = self.l2(input)
            input += skip
            return input, input.sum()

    input = randn(2**6, 2**10, requires_grad=True)
    model = MLPLayer(2**10)
    us_model = unit_scale(model)
    output, loss = us_model(input)
    loss.backward()

    expected_logs = [
        "unit scaling function: add\n",
        "unit scaling function: iadd\n",
        "unit scaling function: iadd_1 (residual-add)",
        "unit scaling function: add_1 (residual-add)",
    ]
    print(caplog.text)
    for log_msg in expected_logs:
        assert log_msg in caplog.text


def test_fp8_unit_scaling(caplog: LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    class Model(nn.Module):
        def __init__(self, d_in: int, d_out: int) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out)

        def forward(self, t: Tensor) -> Tensor:  # pragma: no cover
            return self.linear(t)  # type: ignore[no-any-return]

    input = randn(2**8, 2**9)
    model = Model(2**9, 2**10)
    model = simulate_fp8(model)
    model = unit_scale(model)
    model(input)

    expected_logs = [
        "moving unit scaling backend to precede quantisation backend",
        "running unit scaling backend",
        "running quantisation backend",
    ]
    for log_msg in expected_logs:
        assert log_msg in caplog.text


def test_prune_non_float_tensors() -> None:
    class Model(nn.Module):
        def __init__(self, emb, dim) -> None:
            super().__init__()
            self.emb = nn.Embedding(emb, dim)
            self.linear = nn.Linear(dim, dim)

        def forward(self, idxs: Tensor) -> Tensor:
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

    # TODO: full targets list thing
    graph = model.scales_graph()
    expected_targets = {
        "idxs",
        "self_emb_weight",
        F.embedding,
        F.linear,
        "self_linear_weight",
        "self_linear_bias",
        F.softmax,
        torch.argmax,
        torch.unsqueeze,
        torch.gather,
        operator.isub,
        "mean",
        "output",
    }
    graph_targets = set(node.target for node in graph.nodes)
    assert graph_targets == expected_targets

    graph = prune_non_float_tensors(graph)
    graph_targets = set(node.target for node in graph.nodes)
    expected_targets -= {"idxs", torch.argmax, torch.unsqueeze}
    assert graph_targets == expected_targets


def test_prune_same_scale_tensors() -> None:
    class Model(nn.Module):
        def __init__(self, emb, dim) -> None:
            super().__init__()
            self.emb = nn.Embedding(emb, dim)
            self.linear = nn.Linear(dim, dim // 2)

        def forward(self, idxs: Tensor) -> Tensor:
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
    expected_targets = {
        "idxs",
        "self_emb_weight",
        F.embedding,
        "flatten",
        "view",
        "self_linear_weight",
        "self_linear_bias",
        F.linear,
        F.softmax,
        torch.argmax,
        torch.unsqueeze,
        torch.gather,
        randn_like,
        operator.iadd,
        "output",
    }
    graph_targets = set(node.target for node in graph.nodes)
    assert graph_targets == expected_targets

    graph = prune_same_scale_tensors(graph)
    graph_targets = set(node.target for node in graph.nodes)
    expected_targets -= {"flatten", "view"}
    assert graph_targets == expected_targets

    graph = prune_same_scale_tensors(graph, rel_tol=2**-4)
    graph_targets = set(node.target for node in graph.nodes)
    expected_targets -= {torch.gather, F.embedding}
    assert graph_targets == expected_targets


def test_prune_same_scale_tensors_with_grad() -> None:
    class Model(nn.Module):
        def forward(self, a: Tensor) -> Tensor:
            b = a / 1.0  # same scale fwd & bwd
            c = b * 1.0  # same scale fwd, as b sums grads -> different scale bwd
            d = F.relu(c)  # different scale fwd & bwd
            e = b - d  # different scale fwd, same bwd
            f = e.sum()  # different scale fwd & bwd
            return f

    input = randn(2**6, 2**8, requires_grad=True)
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
        "sum",
        "output",
    }
    graph_targets = set(node.target for node in graph.nodes)
    assert graph_targets == expected_targets

    graph = prune_same_scale_tensors(graph)
    graph_targets = set(node.target for node in graph.nodes)
    expected_targets -= {operator.truediv, operator.mul}
    assert graph_targets == expected_targets

    # The mul still has the same scale before & after in the fwd pass, but the same is
    # not true for its grads. It should no longer be pruned after `loss.backward`.
    loss.backward()
    graph = model.scales_graph()
    graph = prune_same_scale_tensors(graph)
    graph_targets = set(node.target for node in graph.nodes)
    expected_targets.add(operator.mul)
    assert graph_targets == expected_targets


def test_prune_selected_nodes() -> None:
    class Model(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
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
        "sum",
        "output",
    }
    graph_targets = set(node.target for node in graph.nodes)
    assert graph_targets == expected_targets

    graph = prune_selected_nodes(graph, targets=[torch.abs, F.relu])
    graph_targets = set(node.target for node in graph.nodes)
    expected_targets -= {torch.abs, F.relu}
    assert graph_targets == expected_targets


# TODO: refactor these tests into separate files
# then do track scales
# then check plotting works
# then write simple tests
# then go through code an tidy where appropriate

# TODO: when I sort out track scales eventually, it should require/make input have grad (do other transforms need this?)

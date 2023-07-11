# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import logging
from copy import deepcopy
from dataclasses import dataclass
from math import isclose, isinf
from operator import getitem
from types import MethodType
from typing import Any, Callable, Dict, Iterable, List, Tuple, TypeVar

import torch
from tabulate import tabulate
from torch import Tensor, nn
from torch.fx import Interpreter
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node

from .utils import Backend, apply_transform

logger = logging.getLogger(__name__)
M = TypeVar("M", bound=nn.Module)


class Metrics:
    @dataclass
    class DirectionMetrics:
        mean_abs: float
        abs_mean: float
        std: float
        abs_max: float
        abs_min: float
        numel: int

    def __init__(self, fwd_tensor: Tensor) -> None:
        self.fwd = self.from_tensor(fwd_tensor)
        self.bwd = None

    def set_bwd(self, bwd_tensor: Tensor) -> None:
        self.bwd = self.from_tensor(bwd_tensor)

    def __str__(self) -> str:
        return f"fwd: {self.fwd}, bwd: {self.bwd}"

    @staticmethod
    def from_tensor(t: Tensor) -> "Metrics.DirectionMetrics":
        abs_t = t.abs()
        return Metrics.DirectionMetrics(
            mean_abs=abs_t.mean().item(),
            abs_mean=t.mean().abs().item(),
            std=t.std().item(),
            abs_max=abs_t.max().item(),
            abs_min=abs_t.min().item(),
            numel=t.numel(),
        )

    @staticmethod
    def get_full_name(short_name: str) -> str:
        return {
            "mean_abs": "mean absolute value",
            "abs_mean": "absolute mean value",
            "std": "standard deviation",
            "abs_max": "absolute maximum",
            "abs_min": "absolute minimum",
            "numel": "number of elements",
        }[short_name]

    @staticmethod
    def names() -> List[str]:
        return list(Metrics.DirectionMetrics.__dataclass_fields__.keys())

    @staticmethod
    def full_names() -> List[str]:
        return [Metrics.get_full_name(m) for m in Metrics.names()]


def directions_same_scale(
    a: Metrics.DirectionMetrics, b: Metrics.DirectionMetrics, rel_tol=2**-16
):
    # We're just going to look at the mean_abs here
    return isclose(a.mean_abs, b.mean_abs, rel_tol=rel_tol)


def metrics_same_scale(a: Metrics, b: Metrics, rel_tol=2**-16) -> bool:
    # Both are None
    if a.bwd is None and b.bwd is None:
        return directions_same_scale(a.fwd, b.fwd, rel_tol)
    # Only one is None
    if a.bwd is None or b.bwd is None:
        return False
    # Neither is None
    return directions_same_scale(a.fwd, b.fwd, rel_tol) and directions_same_scale(
        a.bwd, b.bwd, rel_tol
    )


class _Track(torch.autograd.Function):
    @staticmethod
    def forward(  # type:ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        t: Tensor,
        node_meta: Dict[str, Any],
    ) -> Tensor:
        node_meta["metrics"] = Metrics(fwd_tensor=t)
        ctx.node_meta = node_meta  # type: ignore
        return t.clone()

    @staticmethod
    def backward(  # type:ignore[override]
        ctx: torch.autograd.function.FunctionCtx, t: Tensor
    ) -> Tuple[Tensor, None, None]:
        ctx.node_meta["metrics"].set_bwd(bwd_tensor=t)  # type: ignore
        return t.clone(), None, None


def _tabulate_graph_data(g: Graph) -> str:
    meta_ignore_fields = [
        "creation_timestamp",
        "stack_trace",
        "example_value",
        "tensor_dict",
        "grapharg",
    ]
    headers = ["opcode", "name", "target", "args", "kwargs", "users", "input_nodes"]
    meta_headers = []
    for n in g.nodes:
        for k in n.meta:
            if k not in meta_headers and k not in meta_ignore_fields:
                meta_headers.append(k)

    data = []
    for n in g.nodes:
        node_data = [
            n.op,
            n.name,
            n.target,
            n.args,
            n.kwargs,
            list(n.users.keys()),
            list(n._input_nodes.keys()),
        ]
        for h in meta_headers:
            if h in n.meta:
                node_data.append(n.meta[h])
            else:
                node_data.append(None)
        data.append(node_data)
    return tabulate(data, headers=headers + meta_headers, tablefmt="html")


def _add_tabular_html_display(g: Graph) -> None:
    """When `_repr_html_()` is called on `g` (e.g. if it's displayed in a Jupyter
    cell), return a tabulated view of the graph's data, including relevant metadata."""
    g._repr_html_ = MethodType(_tabulate_graph_data, g)


def _clean_node_name(name: str) -> str:
    return name.replace("l__self___", "").replace("l_", "").strip("_")


def _is_float_tensor(a: Any) -> bool:
    return isinstance(a, Tensor) and a.is_floating_point()


def _get_tracking_meta(n: Node, out: Any) -> Dict[str, Any]:
    return {
        "clean_name": _clean_node_name(n.name),
        "outputs_float_tensor": _is_float_tensor(out),
        "requires_grad": isinstance(out, nn.Parameter) and out.requires_grad,
    }


class _Tracker(Interpreter):
    def __init__(self, gm: GraphModule, graph_holder: List[Graph]):
        graph_holder[0] = gm.graph  # allows graph to be accessed from outside
        super().__init__(gm)

    def run_node(self, n: Node) -> Any:
        out = super().run_node(n)
        n.meta.update(_get_tracking_meta(n, out))
        if n.meta["outputs_float_tensor"]:
            logger.info("adding tracking to node: %s", n)
            out = _Track.apply(out, n.meta)
        return out


def scale_tracking_backend(graph_holder: List[Graph]) -> Backend:
    def inner_backend(
        gm: GraphModule, example_inputs: List[Tensor]
    ) -> Callable[..., Any]:
        _add_tabular_html_display(gm.graph)  # displays full info in notebooks
        return _Tracker(gm, graph_holder).run

    return inner_backend


def track_scales(module: M) -> M:
    graph_holder = [Graph()]
    tracking_module = apply_transform(module, scale_tracking_backend(graph_holder))

    def scales_graph(self) -> Graph:
        return graph_holder[0]

    tracking_module.scales_graph = MethodType(scales_graph, tracking_module)
    return tracking_module


def _prune(graph: Graph, node: Node, replacement_arg=None) -> None:
    for user in list(node.users):
        # output node's args are a tuple of tuples, so need special handling
        if user.name == "output":
            user.args = tuple(
                (tuple(None if a == node else a for a in outputs))
                for outputs in user.args
            )
        else:
            user.args = tuple(replacement_arg if a == node else a for a in user.args)
        user.kwargs = {
            k: replacement_arg if v == node else v for k, v in user.kwargs.items()
        }
    graph.erase_node(node)


def prune_non_float_tensors(graph: Graph) -> Graph:
    if "clean_name" not in list(graph.nodes)[0].meta:
        raise RuntimeError(
            "supplied graph must be a result of running"
            " `unit_scaling.transforms.track_scales` (and extracting"
            " the `scales_graph`). Arbitrary fx graphs lack metadata"
            " about the types of node outputs so cannot be pruned."
        )

    graph = deepcopy(graph)
    for n in graph.nodes:
        if n.name == "output":
            continue

        if not n.meta.get("outputs_float_tensor", False):
            float_tensor_args = [  # TODO: refactor as repeated
                a
                for a in n.args
                if isinstance(a, Node) and a.meta.get("outputs_float_tensor", False)
            ]
            a = float_tensor_args[0] if len(float_tensor_args) == 1 else None
            logger.info("pruning non-float node: %s", n)
            _prune(graph, n, replacement_arg=a)
    graph.lint()
    return graph


def prune_same_scale_tensors(graph: Graph, rel_tol=2**-16) -> Graph:
    graph = deepcopy(graph)
    for n in graph.nodes:
        if n.name == "output" or not n.meta.get("outputs_float_tensor", False):
            continue

        float_tensor_args = [
            a
            for a in n.args
            if isinstance(a, Node) and a.meta.get("outputs_float_tensor", False)
        ]
        if len(float_tensor_args) == 1:
            a = float_tensor_args[0]
            a_metrics = a.meta["metrics"]
            n_metrics = n.meta["metrics"]
            if metrics_same_scale(n_metrics, a_metrics, rel_tol):
                logger.info("pruning same-scale node: %s", n)
                _prune(graph, n, replacement_arg=a)
    return graph


def prune_selected_nodes(graph: Graph, targets: Iterable):
    for n in graph.nodes:
        if n.target in targets:
            logger.info("pruning non-float node: %s", n)
            _prune(graph, n)
    graph.lint()
    return graph

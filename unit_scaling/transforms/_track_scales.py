# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import logging
from copy import deepcopy
from dataclasses import dataclass
from math import isclose
from types import MethodType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar

import torch
from tabulate import tabulate
from torch import Tensor, nn
from torch.fx import Interpreter
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, Target

from .utils import apply_transform

logger = logging.getLogger(__name__)
M = TypeVar("M", bound=nn.Module)


class Metrics:
    """A set of metrics representing useful information about a tensor, in the forward
    and backward pass.

    :code:`metrics.fwd`, and optionally :code:`metrics.bwd`, are objects of type
    :code:`Data`, containing metrics about the
    tensor in the forward pass and its gradient in the backward pass.
    """

    @dataclass
    class Data:
        """Data representing key metrics for a tensor."""

        mean_abs: float
        abs_mean: float
        std: float
        abs_max: float
        abs_min: float
        numel: int

    def __init__(self, fwd_tensor: Tensor) -> None:
        self.fwd = self.from_tensor(fwd_tensor)
        self.bwd: Optional[Metrics.Data] = None

    def set_bwd(self, bwd_tensor: Tensor) -> None:
        self.bwd = self.from_tensor(bwd_tensor)

    def __str__(self) -> str:
        return f"fwd: {self.fwd}, bwd: {self.bwd}"  # pragma: no cover

    @staticmethod
    def from_tensor(t: Tensor) -> "Metrics.Data":
        abs_t = t.abs()
        return Metrics.Data(
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
        return list(Metrics.Data.__dataclass_fields__.keys())

    @staticmethod
    def full_names() -> List[str]:
        return [Metrics.get_full_name(m) for m in Metrics.names()]


def _directions_same_scale(
    a: Metrics.Data, b: Metrics.Data, rtol: float = 2**-16
) -> bool:
    # We're just going to look at the mean_abs here
    return isclose(a.mean_abs, b.mean_abs, rel_tol=rtol)


def _metrics_same_scale(a: Metrics, b: Metrics, rtol: float = 2**-16) -> bool:
    # Both are None
    if a.bwd is None and b.bwd is None:
        return _directions_same_scale(a.fwd, b.fwd, rtol)
    # Only one is None
    if a.bwd is None or b.bwd is None:  # pragma: no cover
        return False
    # Neither is None
    return _directions_same_scale(a.fwd, b.fwd, rtol) and _directions_same_scale(
        a.bwd, b.bwd, rtol
    )


def _make_input_tensors_require_grad(module: nn.Module) -> None:
    old_forward = module.forward

    def new_forward(*args: Any, **kwargs: Any) -> Any:
        for a in args + tuple(kwargs.values()):
            if _is_float_tensor(a):
                a.requires_grad_()
        return old_forward(*args, **kwargs)

    module.forward = new_forward


class ScaleTrackingAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
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


def _tabulate_graph_data(g: Graph) -> str:  # pragma: no cover
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
        ] + [n.meta.get(h) for h in meta_headers]
        data.append(node_data)
    return tabulate(data, headers=headers + meta_headers, tablefmt="html")


def _add_tabular_html_display(g: Graph) -> None:
    """When `_repr_html_()` is called on `g` (e.g. if it's displayed in a Jupyter
    cell), return a tabulated view of the graph's data, including relevant metadata."""
    g._repr_html_ = MethodType(_tabulate_graph_data, g)  # type: ignore[attr-defined]


def _clean_node_name(name: str) -> str:
    replace_strs = ["l__self___", "L__self___", "l_", "L_"]
    for r in replace_strs:
        name = name.replace(r, "")
    return name.strip("_")


def _is_float_tensor(a: Any) -> bool:
    return isinstance(a, Tensor) and a.is_floating_point()


def _get_tracking_meta(n: Node, out: Any) -> Dict[str, Any]:
    return {
        "clean_name": _clean_node_name(n.name),
        "outputs_float_tensor": _is_float_tensor(out),
        "requires_grad": isinstance(out, nn.Parameter) and out.requires_grad,
    }


class ScaleTrackingInterpreter(Interpreter):
    def __init__(self, gm: GraphModule) -> None:
        super().__init__(gm)

    def run_node(self, n: Node) -> Any:
        out = super().run_node(n)
        n.meta.update(_get_tracking_meta(n, out))
        if n.meta["outputs_float_tensor"]:
            logger.info("adding tracking to node: %s", n)
            out = ScaleTrackingAutogradFunction.apply(out, n.meta)  # type: ignore
        return out

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().run(*args, **kwargs)


class ScaleTrackingBackend:
    def __init__(self) -> None:
        self.graph = Graph()

    def __call__(
        self, gm: GraphModule, example_inputs: List[Tensor]
    ) -> Callable[..., Any]:
        _add_tabular_html_display(gm.graph)  # displays full info in notebooks
        self.graph = gm.graph  # allows graph to be accessed from outside
        return ScaleTrackingInterpreter(gm)


def _prune(graph: Graph, node: Node, replacement_arg: Optional[Node] = None) -> None:
    for user in list(node.users):
        # output node's args are a tuple of tuples, so need special handling
        if user.name == "output":
            user.args = tuple(
                (
                    (tuple(replacement_arg if o == node else o for o in out))
                    if isinstance(out, Iterable)
                    else (replacement_arg if out == node else out)
                )
                for out in user.args
            )
        else:
            user.args = tuple(replacement_arg if a == node else a for a in user.args)
        user.kwargs = {
            k: replacement_arg if v == node else v for k, v in user.kwargs.items()
        }
    graph.erase_node(node)


def _filter_float_tensors(args: List[Any]) -> List[Node]:
    return [
        a
        for a in args
        if isinstance(a, Node) and a.meta.get("outputs_float_tensor", False)
    ]


def track_scales(module: M) -> M:
    """Returns a version of the input module which tracks internal tensor metrics.

    When the :code:`forward()` and :code:`backward()` methods of the resulting module
    are called, internally various metrics (such as scale) are recorded for each
    intermediate tensor used. These can be accessed using an additional method
    :code:`module.scales_graph()` which is added to the module. The returned object
    is an instance of :external+pytorch:py:class:`torch.fx.Graph`, where each node
    representing a floating-point tensor now has a :code:`node.meta["metrics"]` object
    of type :class:`unit_scaling.transforms.Metrics` associated with it. Note that if
    :code:`forward()` or :code:`backward()` are not called, tensor metrics will not
    be available.

    The unit scaling library also provides a method to visualise FX Graphs,
    via the :func:`unit_scaling.analysis.plot` function. This is intended to be used
    as follows:

    .. code-block:: python

        from unit_scaling.transforms import track_scales
        from unit_scaling.analysis import plot

        inpt = ...
        model = ...

        model = track_scales(model)
        loss = model(inpt)
        loss.backward()

        graph = model.scales_graph()
        plot(graph)

    The :code:`inpt` tensor(s) provided to any model transformed by
    :code:`track_scales()` will automatically have :code:`inpt.requires_grad_()` set
    (this is required for full scale tracking in the backward pass), so the user need
    not specify this.

    :code:`track_scales()` can be used in conjunction with other graph transforms
    provided, but should always be the final transform in a chain. E.g.

    .. code-block:: python

        from unit_scaling.transforms import simulate_fp8, track_scales, unit_scale

        model = track_scales(unit_scale(simulate_fp8(model)))

    The full FX graph returned by this transform may contain more information than the
    user requires for the sake of analysis. For this reason the functions
    :func:`unit_scaling.transforms.prune_non_float_tensors` and
    :func:`unit_scaling.transforms.prune_same_scale_tensors` are provided, which in
    practice tend to limit the graph to only key tensors.

    Args:
        module (M): the input module to be tracked.

    Returns:
        M: a new version of the input module which tracks tensor metrics when used.
    """
    backend = ScaleTrackingBackend()
    tracking_module = apply_transform(module, backend)

    tracking_module.scales_graph = lambda: backend.graph
    _make_input_tensors_require_grad(tracking_module)
    return tracking_module  # type: ignore[no-any-return]


def prune_non_float_tensors(graph: Graph) -> Graph:
    """Given an FX Graph, prunes all nodes which don't output floating-point tensors.

    The supplied graph must have been generated via the :code:`module.scales_graph()`
    method, called on a module with :func:`unit_scaling.transforms.track_scales`
    applied. This is necessary as the scale-tracking process is what identifies which
    tensors have floating-point values. E.g.

    .. code-block:: python

        from unit_scaling.transforms import track_scales, prune_non_float_tensors

        inpt = ...
        model = ...

        model = track_scales(model)
        loss = model(inpt)
        loss.backward()

        graph = model.scales_graph()
        pruned_graph = prune_non_float_tensors(graph)

    Args:
        graph (Graph): the FX graph to be pruned.

    Returns:
        Graph: the pruned graph containing only nodes outputting floating-point tensors.
    """
    if "clean_name" not in list(graph.nodes)[0].meta:  # pragma: no cover
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
            float_tensor_args = _filter_float_tensors(n.args)
            a = float_tensor_args[0] if len(float_tensor_args) == 1 else None
            logger.info("pruning non-float node: %s", n)
            _prune(graph, n, replacement_arg=a)
    graph.lint()  # type: ignore[no-untyped-call]
    return graph


def prune_same_scale_tensors(graph: Graph, rtol: float = 2**-16) -> Graph:
    """Given an FX Graph, prunes all nodes with the same scale as the previous node.

    This is intended to remove non-informative nodes from the graph such as
    reshapes. Nodes with multiple floating-point tensors as inputs are never pruned.

    Certain operations (such as slices) may change the scale slightly, but
    negligibly—in this case we provide a tolerance parameter which can be used
    to specify the relative change that is deemed significant.

    The supplied graph must have been generated via the :code:`module.scales_graph()`
    method, called on a module with :func:`unit_scaling.transforms.track_scales`
    applied. E.g.

    .. code-block:: python

        from unit_scaling.transforms import track_scales, prune_same_scale_tensors

        inpt = ...
        model = ...

        model = track_scales(model)
        loss = model(inpt)
        loss.backward()

        graph = model.scales_graph()
        pruned_graph = prune_same_scale_tensors(graph)

    Args:
        graph (Graph): the FX graph to be pruned.
        rtol (float, optional): the relative tolerance for "same scale".
            Defaults to 2**-16.

    Returns:
        Graph: the pruned graph with nodes that don't change their input scale removed.
    """
    graph = deepcopy(graph)
    for n in graph.nodes:
        if n.name == "output" or not n.meta.get("outputs_float_tensor", False):
            continue

        float_tensor_args = _filter_float_tensors(n.args)
        if len(float_tensor_args) == 1:
            a = float_tensor_args[0]
            a_metrics = a.meta["metrics"]
            n_metrics = n.meta["metrics"]
            if _metrics_same_scale(n_metrics, a_metrics, rtol):
                logger.info("pruning same-scale node: %s", n)
                _prune(graph, n, replacement_arg=a)
    return graph


def prune_selected_nodes(graph: Graph, targets: Iterable[Target]) -> Graph:
    """Given an FX Graph, prunes all nodes with functions in the set of target nodes.

    Args:
        graph (Graph): the FX graph to prune.
        targets (Iterable[Target]): the functions which will not be represented by nodes
            once pruning is complete.

    Returns:
        Graph: the pruned graph.
    """
    for n in graph.nodes:
        if n.target in targets:
            logger.info("pruning node: %s", n)
            _prune(graph, n)
    graph.lint()  # type: ignore[no-untyped-call]
    return graph

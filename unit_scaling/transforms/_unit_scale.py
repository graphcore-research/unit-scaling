# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import logging
from inspect import signature
from operator import getitem
from types import BuiltinFunctionType
from typing import Any, Callable, Dict, List, Set, TypeVar

import torch
import torch._dynamo
import torch.nn.functional as F
from torch import Tensor, nn
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node

from .. import functional as U
from .._internal_utils import generate__all__
from .utils import Backend, apply_transform, replace_node_with_function

T = TypeVar("T")
logger = logging.getLogger(__name__)


def _add_dependency_meta(graph: Graph, recalculate: bool = False) -> None:
    def recurse(n: Node) -> Set[Node]:
        if "dependencies" in n.meta:
            return n.meta["dependencies"]  # type: ignore[no-any-return]
        deps = set(n.all_input_nodes)
        for parent in n.all_input_nodes:
            deps.update(recurse(parent))
        n.meta["dependencies"] = deps
        return deps

    if recalculate:
        for n in graph.nodes:
            if "dependencies" in n.meta:
                del n.meta["dependencies"]

    for n in graph.nodes:
        if n.op == "output":
            recurse(n)


def _is_add(n: Node) -> bool:
    return (
        n.op == "call_function"
        and isinstance(n.target, BuiltinFunctionType)
        and n.target.__name__ in ["add", "iadd"]
    )


def _is_self_attention(skip_node: Node, residual_node: Node) -> bool:
    # Identify all operations on the residual branch
    residual_fns = {residual_node.target}
    parents = [*residual_node.all_input_nodes]
    while parents:
        p = parents.pop()
        if p == skip_node:
            continue
        residual_fns.add(p.target)
        parents += p.all_input_nodes
    # Check if any of them are self-attention ops (softmax or fused self-attention)
    self_attention_fns = {
        F.scaled_dot_product_attention,
        U.scaled_dot_product_attention,
        F.softmax,
        U.softmax,
    }
    return bool(residual_fns.intersection(self_attention_fns))


def _unit_scale_residual(
    graph: Graph,
    add: Node,
    residual_arg_idx: int,
    is_self_attention: bool,
) -> None:
    residual, skip = add.args[residual_arg_idx], add.args[1 - residual_arg_idx]
    tau = 0.01 if is_self_attention else 0.5
    logger.info("unit scaling function: %s (residual-add, tau=%s)", add, tau)
    old_start_residuals = [
        u for u in skip.users if u is not add  # type: ignore[union-attr]
    ]
    with graph.inserting_after(skip):  # type: ignore[arg-type]
        split = graph.call_function(
            U.residual_split,
            args=(skip, tau),
            type_expr=getattr(skip, "type", None),
        )
    with graph.inserting_after(split):
        new_start_residual = graph.call_function(getitem, args=(split, 0))
    for old_start_residual in old_start_residuals:
        old_start_residual.replace_input_with(skip, new_start_residual)  # type: ignore
    with graph.inserting_after(split):
        new_skip = graph.call_function(getitem, args=(split, 1))
    replace_node_with_function(
        graph, add, U.residual_add, args=(residual, new_skip, tau)
    )


def _unconstrain_node(node: Node) -> None:
    if (
        node.op == "call_function"
        and callable(node.target)
        and not isinstance(node.target, BuiltinFunctionType)
        and "constraint" in signature(node.target).parameters
    ):
        logger.info("unconstraining node: %s", node)
        node.kwargs = dict(node.kwargs, constraint=None)


def unit_scaling_backend(
    replacement_map: Dict[Callable[..., Any], Callable[..., Any]] = dict()
) -> Backend:
    def inner_backend(gm: GraphModule, example_inputs: List[Tensor]) -> GraphModule:
        logger.info("running unit scaling backend")
        graph = gm.graph
        # Replace function nodes with those in `replacement_map` or with their
        # unit scaled equivalents
        for node in graph.nodes:
            if node.op == "call_function":
                if node.target in replacement_map:
                    target_fn = replacement_map[node.target]
                    logger.info(
                        "replacing function: %s with %s", node, target_fn.__name__
                    )
                    replace_node_with_function(graph, node, target_fn)
                elif node.target in U.torch_map:
                    target_fn = U.torch_map[node.target]
                    logger.info("unit scaling function: %s", node)
                    replace_node_with_function(graph, node, target_fn)

        # Add metadata denoting the dependencies of every node in the graph
        _add_dependency_meta(graph)

        # Go through and mark nodes which represent residual-adds
        residual_layer_number = 1
        for node in graph.nodes:
            if _is_add(node):
                is_residual_add = False
                if len(node.args) == 2:
                    l, r = node.args
                    if isinstance(l, Node) and isinstance(r, Node):
                        l_deps = l.meta.get("dependencies", list())
                        r_deps = r.meta.get("dependencies", list())
                        if l in r_deps or r in l_deps:
                            node.meta["residual_add"] = {
                                "residual_arg_idx": 1 if l in r_deps else 0,
                            }
                            residual_layer_number += 1
                            is_residual_add = True
                            skip_node, residual_node = (l, r) if l in r_deps else (r, l)
                            is_sa = _is_self_attention(skip_node, residual_node)
                            node.meta["residual_add"]["is_self_attention"] = is_sa
                # Regular adds are not picked up by the unit scaling sweep above as
                # the inbuilt + operation is handled differently when traced. It is
                # instead substituted for its unit scaled equivalent here.
                if not is_residual_add:
                    logger.info("unit scaling function: %s", node)
                    args = (*node.args, None)  # None denotes unconstrained
                    replace_node_with_function(graph, node, U.add, args=args)

        # Replace nodes marked as residual-adds with unit scaled equivalent
        for node in graph.nodes:
            residual_add = node.meta.get("residual_add", None)
            if residual_add is not None:
                _unit_scale_residual(graph, node, **residual_add)

        _add_dependency_meta(graph, recalculate=True)
        for node in graph.nodes:
            if node.target == U.residual_add:
                node.meta["has_residual_successor"] = True
                dependencies = node.meta.get("dependencies", [])
                for d in dependencies:
                    d.meta["has_residual_successor"] = True

        for node in graph.nodes:
            if "has_residual_successor" not in node.meta:
                _unconstrain_node(node)

        graph.lint()  # type: ignore[no-untyped-call]
        return GraphModule(gm, graph)

    return inner_backend


def _unit_init_weights(m: nn.Module) -> None:
    for name, mod in m.named_modules():
        if isinstance(mod, (nn.Linear, nn.Embedding)):
            with torch.no_grad():
                if isinstance(mod.weight, Tensor):
                    logger.info("unit scaling weight: %s.weight", name)
                    mod.weight /= mod.weight.std()


def _zero_init_biases(m: nn.Module) -> None:
    for name, mod in m.named_modules():
        if isinstance(mod, (nn.Linear, nn.Embedding)):
            with torch.no_grad():
                if hasattr(mod, "bias") and isinstance(mod.bias, Tensor):
                    logger.info("setting bias to zero: %s.bias", name)
                    mod.bias -= mod.bias


# Unit scaling should come before quantisation
def _order_backends(backends: List[Backend]) -> None:
    unit_scaling_backend_idx = -1
    quantisation_backend_idx = float("inf")
    for i, b in enumerate(backends):
        if "unit_scaling_backend" in b.__qualname__:
            unit_scaling_backend_idx = i
        if "quantisation_backend" in b.__qualname__:
            quantisation_backend_idx = i
    if unit_scaling_backend_idx > quantisation_backend_idx:
        logger.info("moving unit scaling backend to precede quantisation backend")
        u = backends.pop(unit_scaling_backend_idx)
        backends.insert(quantisation_backend_idx, u)  # type: ignore[arg-type]


M = TypeVar("M", bound=nn.Module)


def unit_scale(
    module: M, replace: Dict[Callable[..., Any], Callable[..., Any]] = dict()
) -> M:
    """**[Experimental]** Returns a unit-scaled version of the input model.

    Uses TorchDynamo to trace and transform the user-supplied module.
    This transformation identifies all :class:`torch.nn.functional` uses in the input
    module, and replaces them with their unit-scaled equivalents, should they exist.

    The tracing procedure automatically recurses into modules
    (whether defined in libraries, or by the user), identifying inner calls to any
    :class:`torch.nn.functional` operations, to build a graph of fundamental operations.
    Unit scaling is then applied as a transformation on top of this graph.

    This transformation proceeds in five stages:

    #. **Replacement of user-defined functions** according to the supplied `replace`
       dictionary.
    #. **Replacement of all functions with unit-scaled equivalents** defined in
       :mod:`unit_scaling.functional`.
    #. Identification & **replacement of all add operations that represent
       residual-adds**. The identification of residual connections is done via a
       dependency analysis on the graph. Residual-adds require special scaling compared
       with regular adds (see paper / User Guide for details).
    #. **Unconstraining of all operations after the final residual layer**. By default
       all unit scaled operations have their scaling factors constrained in the forward
       and backward pass to give valid gradients. This is not required in these final
       layers (see paper for proof), and hence we can unconstrain the operations to give
       better scaling.
    #. **Unit-scaling of all weights** and zero-initialisation of all biases.

    Note that by using TorchDynamo, `unit_scale()` is able to trace a much larger set of
    modules / operations than with previous PyTorch tracing approaches. This enables
    the process of unit scaling to be expressed as a generic graph transform that can be
    applied to arbitrary modules.

    Note that the current version of TorchDynamo (or :func:`torch.compile`, which is a
    wrapper around TorchDynamo) doesn't support nested transforms, so we implement our
    own system here. This makes it easy to nest transforms:

    .. code-block:: python

        from unit_scaling.transforms import compile, simulate_fp8, unit_scale

        module = compile(simulate_fp8(unit_scale(module)))

    However, these transforms are not interoperable with the standard
    :func:`torch.compile` interface.

    In some cases users may have a model definition that uses a custom implementation of
    a basic operation. In this case, `unit_scale()` can be told explicitly to substitute
    the layer for an equivalent, using the `replace` dictionary:

    .. code-block:: python

        import unit_scaling.functional as U
        from unit_scaling.transforms import unit_scale

        def new_gelu(x):
            ...

        class Model(nn.Module):
            def forward(x):
                ...
                x = new_gelu(x)
                ...

        model = unit_scale(Model(), replace={new_gelu: U.gelu})

    This can also be used to substitute a particular function for a user-defined
    unit-scaled function not provided by :mod:`unit_scaling.functional`.

    **Note:** `unit_scale()` is experimental and has not yet been widely tested on a
    range of models. The standard approach to unit scaling a model is still to
    manually substitute the layers/operations in a model with their unit-scaled
    equivalents. Having said this, `unit_scale()` is implemented in a sufficiently
    generic way that we anticipate many users will ultimately be able to rely on this
    graph transform alone.

    Args:
        module (nn.Module): the input module to be unit scaled.
        replace (Dict[Callable, Callable], optional): a dictionary where keys represent
            functions to be replaced by the corresponding value-functions. Note that
            these substitutions take priority over the standard unit scaling
            substitutions. Defaults to dict().

    Returns:
        nn.Module: the unit scaled module (with an independent copy of parameters)
    """
    unit_scaled_module = apply_transform(
        module,
        unit_scaling_backend(replace),
        non_recurse_functions=list(replace.keys()),
    )
    _order_backends(unit_scaled_module.backends)
    _unit_init_weights(unit_scaled_module)
    _zero_init_biases(unit_scaled_module)
    return unit_scaled_module  # type: ignore[no-any-return]


__all__ = generate__all__(__name__)

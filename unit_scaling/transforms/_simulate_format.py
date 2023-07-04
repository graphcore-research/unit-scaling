# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch._dynamo
import torch.nn.functional as F
from torch import Tensor, nn
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node

from .. import functional as U
from ..formats import FPFormat
from .utils import patch_to_expand_modules, replace_node_with_function

logger = logging.getLogger(__name__)

Backend = Callable[[GraphModule, List[Tensor]], Callable[..., Any]]

unit_scaled_functions = [getattr(U, f) for f in U.__all__]


# These functions currently have to be defined explicitly to make PyTorch happy
# Creating temporary wrapped functions doesn't work...
def _quantised_linear(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    fwd_format_tuple: Tuple[int, int],
    bwd_format_tuple: Tuple[int, int],
) -> Tensor:
    fwd_format = FPFormat.from_tuple(fwd_format_tuple)
    bwd_format = FPFormat.from_tuple(bwd_format_tuple)
    input = fwd_format.quantise(input)
    weight = fwd_format.quantise(weight)
    output = F.linear(input, weight, bias)
    return bwd_format.quantise_bwd(output)


def _quantised_u_linear(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    fwd_format_tuple: Tuple[int, int],
    bwd_format_tuple: Tuple[int, int],
    constraint: Optional[str] = "gmean",
) -> Tensor:
    fwd_format = FPFormat.from_tuple(fwd_format_tuple)
    bwd_format = FPFormat.from_tuple(bwd_format_tuple)
    input, weight = (fwd_format.quantise(t) for t in (input, weight))
    output = U.linear(input, weight, bias, constraint)
    return bwd_format.quantise_bwd(output)


def _quantised_scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    fwd_format_tuple: Tuple[int, int],
    bwd_format_tuple: Tuple[int, int],
    **kwargs: Any,
) -> Tensor:
    fwd_format = FPFormat.from_tuple(fwd_format_tuple)
    bwd_format = FPFormat.from_tuple(bwd_format_tuple)
    query, key, value = (fwd_format.quantise(t) for t in (query, key, value))
    output = F.scaled_dot_product_attention(query, key, value, **kwargs)
    return bwd_format.quantise_bwd(output)


def _quantised_u_scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    fwd_format_tuple: Tuple[int, int],
    bwd_format_tuple: Tuple[int, int],
    **kwargs: Any,
) -> Tensor:
    fwd_format = FPFormat.from_tuple(fwd_format_tuple)
    bwd_format = FPFormat.from_tuple(bwd_format_tuple)
    query, key, value = (fwd_format.quantise(t) for t in (query, key, value))
    output = U.scaled_dot_product_attention(query, key, value, **kwargs)
    return bwd_format.quantise_bwd(output)


_replacement_map: Dict[Callable[..., Any], Callable[..., Any]] = {
    F.linear: _quantised_linear,
    U.linear: _quantised_u_linear,
    F.scaled_dot_product_attention: _quantised_scaled_dot_product_attention,
    U.scaled_dot_product_attention: _quantised_u_scaled_dot_product_attention,
}


def _replace_with_quantised(
    graph: Graph,
    node: Node,
    fwd_format: FPFormat,
    bwd_format: FPFormat,
) -> None:
    # Ideally we'd pass the formats as kwargs, but it currently causes a torch fx bug.
    # This workaround will suffice for now...
    args = [*node.args]
    if len(node.args) == 2:  # pragma: no cover
        args.append(None)
    # Breaks when I pass in FPFormat objects, so convert to tuple and back
    args = args[:3] + [fwd_format.to_tuple(), bwd_format.to_tuple()] + args[3:]

    assert callable(node.target)
    quantised_fn = _replacement_map[node.target]
    logger.info("Quantising function: %s", node)
    replace_node_with_function(graph, node, quantised_fn, args=tuple(args))


def _quantisation_backend(fwd_format: FPFormat, bwd_format: FPFormat) -> Backend:
    def backend_fn(gm: GraphModule, example_inputs: List[Tensor]) -> GraphModule:
        logger.info("Running quantisation backend")
        graph = gm.graph
        for node in graph.nodes:
            if node.op == "call_function" and node.target in _replacement_map:
                _replace_with_quantised(graph, node, fwd_format, bwd_format)
        return GraphModule(gm, graph)  # type: ignore[no-any-return]

    return backend_fn


def simulate_format(
    model: nn.Module, fwd_format: FPFormat, bwd_format: FPFormat
) -> nn.Module:
    model = deepcopy(model)
    torch._dynamo.reset()  # type: ignore[no-untyped-call]
    quantised_model = torch._dynamo.optimize(  # type: ignore[no-untyped-call]
        backend=_quantisation_backend(fwd_format, bwd_format)
    )(model)
    quantised_model.forward = patch_to_expand_modules(
        quantised_model.forward, non_recurse_functions=unit_scaled_functions
    )
    return quantised_model  # type: ignore[no-any-return]


def simulate_fp8(model: nn.Module) -> nn.Module:
    return simulate_format(model, fwd_format=FPFormat(4, 3), bwd_format=FPFormat(5, 2))

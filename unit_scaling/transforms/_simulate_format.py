# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import torch.nn.functional as F
from torch import Tensor, nn
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node

from .. import functional as U
from .._internal_utils import generate__all__
from ..formats import FPFormat, format_to_tuple, tuple_to_format
from .utils import Backend, apply_transform, replace_node_with_function

logger = logging.getLogger(__name__)


# These functions currently have to be defined explicitly to make PyTorch happy
# Creating temporary wrapped functions doesn't work...
def _quantised_linear(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    fwd_format_tuple: Tuple[int, int],
    bwd_format_tuple: Tuple[int, int],
) -> Tensor:
    fwd_format = tuple_to_format(fwd_format_tuple)
    bwd_format = tuple_to_format(bwd_format_tuple)
    input = fwd_format.quantise_fwd(input)
    weight = fwd_format.quantise_fwd(weight)
    output = F.linear(input, weight, bias)
    return bwd_format.quantise_bwd(output)


def _quantised_u_linear(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    fwd_format_tuple: Tuple[int, int],
    bwd_format_tuple: Tuple[int, int],
    constraint: Optional[str] = "to_output_scale",
) -> Tensor:
    fwd_format = tuple_to_format(fwd_format_tuple)
    bwd_format = tuple_to_format(bwd_format_tuple)
    input, weight = (fwd_format.quantise_fwd(t) for t in (input, weight))
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
    fwd_format = tuple_to_format(fwd_format_tuple)
    bwd_format = tuple_to_format(bwd_format_tuple)
    query, key, value = (fwd_format.quantise_fwd(t) for t in (query, key, value))
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
    fwd_format = tuple_to_format(fwd_format_tuple)
    bwd_format = tuple_to_format(bwd_format_tuple)
    query, key, value = (fwd_format.quantise_fwd(t) for t in (query, key, value))
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
    args = (
        args[:3] + [format_to_tuple(fwd_format), format_to_tuple(bwd_format)] + args[3:]
    )

    assert callable(node.target)
    quantised_fn = _replacement_map[node.target]
    logger.info("quantising function: %s", node)
    replace_node_with_function(graph, node, quantised_fn, args=tuple(args))


def _quantisation_backend(fwd_format: FPFormat, bwd_format: FPFormat) -> Backend:
    def backend_fn(gm: GraphModule, example_inputs: List[Tensor]) -> GraphModule:
        logger.info("running quantisation backend")
        graph = gm.graph
        for node in graph.nodes:
            if node.op == "call_function" and node.target in _replacement_map:
                _replace_with_quantised(graph, node, fwd_format, bwd_format)
        graph.lint()  # type: ignore[no-untyped-call]
        return GraphModule(gm, graph)

    return backend_fn


M = TypeVar("M", bound=nn.Module)


def simulate_format(module: M, fwd_format: FPFormat, bwd_format: FPFormat) -> M:
    """**[Experimental]** Given a module, uses TorchDynamo to return a new module which
    simulates the effect of using the supplied formats for matmuls.

    Specifically, before each :func:`torch.nn.functional.linear` and
    :func:`torch.nn.functional.scaled_dot_product_attention` call, a quantisation op
    is inserted which simulates the effect of using the supplied `fwd_format`. This op
    reduces the range of values to that of the given format, and (stochastically) rounds
    values to only those representable by the format.

    The same is true for the backward pass, where an op is inserted to quantise to the
    `bwd_format`. Models which use modules that contain these functions internally
    (such as :class:`torch.nn.Linear`) will be inspected by TorchDynamo and have the
    correct quantisation ops inserted.

    If the equivalent unit-scaled functions from :mod:`unit_scaling.functional` are
    used in the module, these too will be quantised.

    Simulation of formats is run in FP32. Users should not expect speedups from using
    this method. The purpose is to simulate the numerical effects of running matmuls
    in various formats.

    Args:
        module (nn.Module): the module to be quantised
        fwd_format (FPFormat): the quantisation format to be used in the forward pass
            (activations and weights)
        bwd_format (FPFormat): the quantisation format to be used in the backward pass
            (gradients of activations and weights)

    Returns:
        nn.Module: a new module which when used, will run using the simulated formats.
    """
    return apply_transform(  # type: ignore[no-any-return]
        module, _quantisation_backend(fwd_format, bwd_format)
    )


def simulate_fp8(module: M) -> M:
    """**[Experimental]** Given a module, uses TorchDynamo to return a new module which
    simulates the effect of running matmuls in FP8. As is standard in the literature
    (Noune et al., 2022; Micikevicius et al., 2022), we use the FP8 E4 format in the
    forwards pass, and FP8 E5 in the backward pass.

    Specifically, before each :func:`torch.nn.functional.linear` and
    :func:`torch.nn.functional.scaled_dot_product_attention` call, a quantisation op
    is inserted which simulates the effect of using FP8. This op
    reduces the range of values to that of the format, and (stochastically) rounds
    values to only those representable by the format.

    The same is true for the backward pass.
    Models which use modules that contain these functions internally
    (such as :class:`torch.nn.Linear`) will be inspected by TorchDynamo and have the
    correct quantisation ops inserted.

    If the equivalent unit-scaled functions from :mod:`unit_scaling.functional` are
    used in the module, these too will be quantised.

    Simulation of formats is run in FP32. Users should not expect speedups from using
    this method. The purpose is to simulate the numerical effects of running matmuls
    in FP8.

    Args:
        module (nn.Module): the module to be quantised

    Returns:
        nn.Module: a new module which when used, will run with matmul inputs in FP8.
    """
    return simulate_format(module, fwd_format=FPFormat(4, 3), bwd_format=FPFormat(5, 2))


__all__ = generate__all__(__name__)

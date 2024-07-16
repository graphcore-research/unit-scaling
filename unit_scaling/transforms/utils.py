# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Utilities for working with transforms."""

import copy
import functools
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    no_type_check,
)
from unittest.mock import patch

import torch
import torch._dynamo
from torch import Tensor, nn
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node

from .. import functional as U
from .._internal_utils import generate__all__

T = TypeVar("T")

Backend = Callable[[GraphModule, List[Tensor]], Callable[..., Any]]

_unit_scaled_functions = [getattr(U, f) for f in U.__all__]


def torch_nn_modules_to_user_modules(mod: nn.Module) -> None:
    """
    Convert torch.nn.module classes to `trivial_subclass` versions.

    By default TorchDynamo doesn't recurse into :mod:`torch.nn` modules or
    :mod:`torch.nn.functional` functions when capturing the FX graph.

    This function makes `torch.nn` modules into user modules.

    To use this with a :class:`torch.nn.Module` the typical use case
    is to call `module = torch_nn_modules_to_user_modules(module)`.
    """

    for n, submod in mod.named_children():
        torch_nn_modules_to_user_modules(submod)

        # Mirroring the check at https://github.com/pytorch/pytorch/blob/34bce27f0d12bf7226b37dfe365660aad456701a/torch/_dynamo/variables/nn_module.py#L307 # noqa: E501
        if submod.__module__.startswith(("torch.nn.", "torch.ao.")):
            # Generate a new name, so e.g. torch.nn.modules.sparse.Embedding
            # becomes trivial_subclass_modules_sparse_Embedding
            modulename = submod.__module__
            modulename = modulename.replace("torch.nn.", "", 1)
            modulename = modulename.replace(".", "_")
            newtypename = "trivial_subclass_" + modulename + "_" + type(submod).__name__

            # Create a new type object deriving from type(submod)
            newmodtype = type(newtypename, (type(submod),), {})

            # Initialize and copy state using pickle
            newsubmod = newmodtype.__new__(newmodtype)  # type: ignore [call-overload]
            state = submod.__getstate__()  # type: ignore [no-untyped-call]
            newsubmod.__setstate__(state)

            # Update module in mod
            setattr(mod, n, newsubmod)


def patch_to_expand_modules(fn: Callable[..., T]) -> Callable[..., T]:
    """By default TorchDynamo doesn't recurse into :mod:`torch.nn` modules or
    :mod:`torch.nn.functional` functions when capturing the FX graph.
    Any function which is wrapped in
    :func:`torch._dynamo.optimise` (or :func:`torch.compile`) and is then passed to
    this function as `fn` will now automatically recurse into
    :mod:`torch.nn` modules or :mod:`torch.nn.functional` functions.

    In practice, to use this with a :class:`torch.nn.Module` the typical use case
    is to call `module = torch._dynamo.optimize(backend)(module)`, followed by
    `module.forward = patch_to_expand_modules(module.forward)`.

    This should be used in conjunction with :func:`torch_nn_modules_to_user_modules`

    Args:
        fn (Callable[..., T]): the function to be patched.

    Returns:
        Callable[..., T]: the new version of `fn` with patching applied.
    """

    def _patched_call_function(  # type: ignore[no-untyped-def]
        self,
        tx,
        args,
        kwargs,
    ):  # pragma: no cover
        # Removing the check in https://github.com/pytorch/pytorch/blob/72662bf05b3499ce96aae9183a489c78f0c44c84/torch/_dynamo/variables/functions.py#L335 # noqa: E501
        return super(
            torch._dynamo.variables.functions.UserMethodVariable, self
        ).call_function(tx, args, kwargs)

    @functools.wraps(fn)
    def new_fn(*args: Any, **kwargs: Any) -> Any:
        with patch(
            "torch._dynamo.variables.functions.UserMethodVariable.call_function",
            new=_patched_call_function,
        ):
            return fn(*args, **kwargs)

    return new_fn


def replace_node_with_function(
    graph: Graph,
    source: Node,
    target_fn: Callable[..., Any],
    args: Optional[Tuple[Any, ...]] = None,
    kwargs: Optional[Dict[Any, Any]] = None,
    keep_type_expr: bool = True,
) -> None:
    """Given a source node and its accompanying graph, remove the node and replace it
    with a new node that represents calling the target function.

    Args:
        graph (Graph): the graph in which the node is present.
        source (Node): the node to be replaced.
        target_fn (Callable[..., Any]): the function to be contained in the new node.
        args (Optional[Tuple[Any, ...]], optional): args of the new node.
            Defaults to None.
        kwargs (Optional[Dict[Any, Any]], optional): kwargs of the new node.
            Defaults to None.
        keep_type_expr (bool, optional): retain the type expression of the removed node.
            Defaults to True.
    """
    if args is None:
        args = source.args
    if kwargs is None:
        kwargs = source.kwargs
    type_expr = getattr(source, "type", None) if keep_type_expr else None
    with graph.inserting_after(source):
        new_node = graph.call_function(target_fn, args, kwargs, type_expr)
        source.replace_all_uses_with(new_node)
        graph.erase_node(source)


def _compose_backends(backends: Iterable[Backend]) -> Backend:
    def composite_backend(
        gm: GraphModule, example_inputs: List[Tensor]
    ) -> Callable[..., Any]:
        for b in backends:
            new_gm = b(gm, example_inputs)
            new_gm._param_name_to_source = getattr(  # type: ignore
                gm,
                "_param_name_to_source",
                None,
            )
            gm = new_gm  # type: ignore[assignment]
        return gm

    return composite_backend


M = TypeVar("M", bound=nn.Module)


@no_type_check
def apply_transform(
    module: M,
    backend: Backend,
    non_recurse_functions: List[Callable[..., Any]] = list(),
) -> M:
    """Applies a graph transformation to a module.

    The user-supplied :code:`backend` represents a transformation of a
    :class:`torch.fx.graph_module.GraphModule`. :code:`apply_transform()` uses
    :func:`torch._dynamo.optimize` to apply this transformation to the module,
    returning a new transformed module.

    Note that the current version of TorchDynamo (or :func:`torch.compile`, which is a
    wrapper around TorchDynamo) doesn't support nested transforms, so we implement our
    own system here. This makes it easy to nest transforms:

    .. code-block:: python

        module = apply_transform(apply_transform(module, backend_1), backend_2)

    However, it should be noted these transforms are not interoperable with the standard
    :func:`torch.compile` interface.

    This nesting system is implemented by moving the call to
    :func:`torch._dynamo.optimize` within the :code:`forward()` method of the module
    (though it is only executed on the first call to the module, or if a new transform
    is applied, the optimised call being cached thereafter). This differs from the
    standard approach used with :func:`torch._dynamo.optimize`, but enables this
    convenient nesting functionality.

    Args:
        _module (nn.Module): the module to be transformed.
        backend (Backend): the graph transformation to be applied.
        non_recurse_functions (Iterable[Callable[..., Any]], optional): functions which
            the user does not wish to be recursed into. Defaults to list().

    Returns:
        nn.Module: the transformed module.
    """
    module = copy.deepcopy(module)

    torch_nn_modules_to_user_modules(module)

    if not hasattr(module, "backends"):
        module.backends = []
    module.backends.append(backend)

    for v in non_recurse_functions:
        torch._dynamo.allow_in_graph(v)

    backend = _compose_backends(module.backends)

    def new_forward(*args: Any, **kwargs: Any) -> Any:
        if module.rerun_transform:
            torch._dynamo.reset()
            dynamo_module = torch._dynamo.optimize(backend)(module)
            module.dynamo_forward = patch_to_expand_modules(dynamo_module.forward)
            module.rerun_transform = False
        with patch.object(module, "forward", module.base_forward):
            return module.dynamo_forward(*args, **kwargs)

    module.rerun_transform = True
    module.base_forward = getattr(module, "base_forward", module.forward)
    module.forward = new_forward
    return module


__all__ = generate__all__(__name__)

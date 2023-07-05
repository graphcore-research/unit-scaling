# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import functools
from contextlib import contextmanager
from copy import copy
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, TypeVar
from unittest.mock import patch

import torch._dynamo
from torch import nn
from torch.fx.graph import Graph
from torch.fx.node import Node

from .._internal_utils import generate__all__

T = TypeVar("T")


def _get_patched_allowed_function_ids(
    non_recurse_functions: Iterable[Callable[..., Any]],
) -> Set[int]:
    allowed_function_ids = copy(torch._dynamo.allowed_functions._allowed_function_ids)
    for v in nn.modules.__dict__.values():
        if isinstance(v, type) and v not in nn.modules.loss.__dict__.values():
            i = id(v)
            if i in allowed_function_ids:
                allowed_function_ids.remove(i)
    for f in non_recurse_functions:
        allowed_function_ids.add(id(f))
    return allowed_function_ids  # type: ignore[no-any-return]


def _patched_call_function(self, tx, args, kwargs):  # type: ignore[no-untyped-def]
    if tx.output.is_root_tracer() and isinstance(
        self.obj, torch._dynamo.variables.NNModuleVariable
    ):
        module_attr = getattr(self.fn, "__module__", "")
        if (
            module_attr is not None
            and module_attr.startswith("torch.nn.modules.module")
            or self.is_constant
        ):
            return self.obj.call_method(  # type: ignore[no-untyped-call]
                tx, self.fn.__name__, args, kwargs, constant=self.is_constant
            ).add_options(self)
    return super(
        torch._dynamo.variables.functions.UserMethodVariable, self
    ).call_function(tx, args, kwargs)


@contextmanager
def _expand_modules_patch(non_recurse_functions):  # type: ignore[no-untyped-def]
    patcher_a = patch(
        "torch._dynamo.allowed_functions._allowed_function_ids",
        new=_get_patched_allowed_function_ids(non_recurse_functions),
    )
    patcher_b = patch(
        "torch._dynamo.variables.functions.UserMethodVariable.call_function",
        new=_patched_call_function,
    )
    try:
        yield (patcher_a.start(), patcher_b.start())
    finally:
        patcher_a.stop()
        patcher_b.stop()


def patch_to_expand_modules(
    fn: Callable[..., T], non_recurse_functions: Iterable[Callable[..., Any]] = ()
) -> Callable[..., T]:
    """By default torch.dynamo doesn't recurse into :mod:`torch.nn` modules or
    :mod:`torch.nn.functional` functions when capturing the FX graph.
    Any function which is wrapped in
    :func:`torch._dynamo.optimise` (or :func:`torch.compile`) and is then passed to
    this function as `fn` will now automatically recurse into
    :mod:`torch.nn` modules or :mod:`torch.nn.functional` functions.

    In practice, to use this with a :class:`torch.nn.Module` the typical use case
    is to call `module = torch._dynamo.optimize(backend)(module)`, followed by
    `module.forward = patch_to_expand_modules(module.forward)`.

    In addition, any functions the user *doesn't* wish to recurse into can be passed
    into `non_recurse_functions` and these will not be expanded.

    Args:
        fn (Callable[..., T]): the function to be patched.
        non_recurse_functions (Iterable[Callable[..., Any]], optional): functions which
            the user does not wish to be recursed into. Defaults to ().

    Returns:
        Callable[..., T]: the new version of `fn` with patching applied.
    """

    @functools.wraps(fn)
    def new_fn(*args: Any, **kwargs: Any) -> Any:
        with _expand_modules_patch(non_recurse_functions):
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


__all__ = generate__all__(__name__)

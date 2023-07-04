import functools
from contextlib import contextmanager
from copy import copy
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple
from unittest.mock import patch

import torch._dynamo
from torch import nn
from torch.fx.graph import Graph
from torch.fx.node import Node


def _get_patched_allowed_function_ids(
    non_recurse_functions: Iterable[Callable],
) -> Set[int]:
    allowed_function_ids = copy(torch._dynamo.allowed_functions._allowed_function_ids)
    for v in nn.modules.__dict__.values():
        if isinstance(v, type) and v not in nn.modules.loss.__dict__.values():
            i = id(v)
            if i in allowed_function_ids:
                allowed_function_ids.remove(i)
    for f in non_recurse_functions:
        allowed_function_ids.add(id(f))
    return allowed_function_ids


def _patched_call_function(self, tx, args, kwargs):
    if tx.output.is_root_tracer() and isinstance(
        self.obj, torch._dynamo.variables.NNModuleVariable
    ):
        module_attr = getattr(self.fn, "__module__", "")
        if (
            module_attr is not None
            and module_attr.startswith("torch.nn.modules.module")
            or self.is_constant
        ):
            return self.obj.call_method(
                tx, self.fn.__name__, args, kwargs, constant=self.is_constant
            ).add_options(self)
    return super(
        torch._dynamo.variables.functions.UserMethodVariable, self
    ).call_function(tx, args, kwargs)


@contextmanager
def expand_modules_patch(non_recurse_functions: Iterable[Callable]):
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
    fn: Callable, non_recurse_functions: Iterable[Callable] = ()
) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        with expand_modules_patch(non_recurse_functions):
            return fn(*args, **kwargs)

    return new_fn


def replace_node_with_function(
    graph: Graph,
    source: Node,
    target_fn: Callable,
    args: Optional[Tuple[Any, ...]] = None,
    kwargs: Optional[Dict[Any, Any]] = None,
    keep_type_expr: bool = True,
):
    if args is None:
        args = source.args
    if kwargs is None:
        kwargs = source.kwargs
    type_expr = getattr(source, "type", None) if keep_type_expr else None
    with graph.inserting_after(source):
        new_node = graph.call_function(target_fn, args, kwargs, type_expr)
        source.replace_all_uses_with(new_node)
        graph.erase_node(source)

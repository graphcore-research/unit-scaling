# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Utilities for working with transforms."""

import copy
import copyreg
import functools
from contextlib import contextmanager
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

# Check for torch < 2.2.   Note alphas are earlier than ".0"
pt21 = torch.__version__ >= "2.0" and torch.__version__ < "2.2alpha"


def deepcopy_with_intercept(obj: Any, interceptor: Callable[..., Any]) -> Any:
    """
    Make a deepcopy of ``param: obj``, intercepting every new constructor call.

    The standard python deepcopy traverses the input object, and for every
    object therein, calls a constructor like::

          __newobj__(type, *args)

    to create the new copy.  This function instead calls::

          interceptor(type, *args)

    If this were to act as a no-op, interceptor would just call::

          type.__new__(type, *args)

    We don't call the constructor ``type(*args)`` as the object's state may
    be later set by ``__setstate__``.

    Args:
        obj (Any): the object to be deep-copied.
        interceptor (Callable): replacement constructor

    Returns:
        Any: the deep-copied object.

    """
    old_reconstruct = copy._reconstruct  # type: ignore [attr-defined]

    def new_reconstruct(  # type: ignore [no-untyped-def]
        x,
        memo,
        func,
        args,
        state=None,
        listiter=None,
        dictiter=None,
        *,
        deepcopy=copy.deepcopy,
    ):
        if func == copyreg.__newobj__:  # type: ignore [attr-defined]
            func = interceptor
        return old_reconstruct(
            x, memo, func, args, state, listiter, dictiter, deepcopy=deepcopy
        )

    with patch("copy._reconstruct", new=new_reconstruct):
        return copy.deepcopy(obj)


def trivial_subclass(in_type: type) -> type:
    """
    Given a class type, make a subclass type which forwards all calls to the base.

    Useful to disable dynamo behaviors which match on stringy class names.
    """
    # TODO: this incredibly common pattern must be packaged somewhere....
    # A proper implementation would protect against generating the same name
    # for 'a_b.c' and 'a.b.c', most likely by '_' -> '__', '.' -> '_o_'

    name = (
        "trivial_subclass_"
        + in_type.__module__.replace(".", "_")
        + "_"
        + in_type.__name__
    )
    return type(name, (in_type,), {})


def torch_nn_modules_to_user_modules(mod: nn.Module) -> nn.Module:
    """
    Convert torch.nn.module classes to `trivial_subclass` versions.

    This will make dynamo inline them.
    """

    # Implementation note: This is not a dynamo pass, as it is merely
    # a data structure change.  Instead it uses a deepcopy, intercepting
    # the constructors.

    def intercept_ctor(ctor, *args):  # type: ignore [no-untyped-def]
        if isinstance(ctor, type) and ctor.__module__.startswith("torch.nn.modules"):
            ctor = trivial_subclass(ctor)
        return ctor.__new__(ctor, *args)

    mod = deepcopy_with_intercept(mod, intercept_ctor)
    assert isinstance(mod, nn.Module)
    return mod


def _torch_nn_module_functions_to_inline() -> Iterable[type]:
    for v in nn.modules.__dict__.values():
        if isinstance(v, type) and v not in nn.modules.loss.__dict__.values():
            yield v


if pt21:

    def _get_patched_allowed_function_ids(  # type: ignore [no-untyped-def]
        non_recurse_functions: Iterable[Callable[..., Any]],
    ):
        _af = torch._dynamo.allowed_functions  # type: ignore [attr-defined]
        allowed_function_ids = copy.copy(_af._allowed_function_ids)
        for v in _torch_nn_module_functions_to_inline():
            i = id(v)
            if i in allowed_function_ids:
                allowed_function_ids.remove(i)
        for f in non_recurse_functions:
            allowed_function_ids.add(id(f))
        return allowed_function_ids


def _patched_call_function(  # type: ignore[no-untyped-def]
    self,
    tx,
    args,
    kwargs,
):  # pragma: no cover
    if isinstance(self.obj, torch._dynamo.variables.NNModuleVariable):
        module_attr = getattr(self.fn, "__module__", "")
        if (
            module_attr is not None
            and module_attr.startswith("torch.nn.modules.module")
            or self.is_constant
        ):
            var = self.obj.call_method(
                tx, self.fn.__name__, args, kwargs, constant=self.is_constant
            )
            return var.add_options(self)  # type: ignore [no-untyped-call,attr-defined]
    return super(
        torch._dynamo.variables.functions.UserMethodVariable, self
    ).call_function(tx, args, kwargs)


if pt21 and hasattr(torch._dynamo, "trace_rules"):
    import torch._dynamo.trace_rules  # type: ignore [import]

    _uncached_get_torch_obj_rule_map = (
        torch._dynamo.trace_rules.get_torch_obj_rule_map.__wrapped__
    )


@contextmanager
def _expand_modules_patch(non_recurse_functions):  # type: ignore[no-untyped-def]
    if pt21:
        patcher_a = patch(
            "torch._dynamo.allowed_functions._allowed_function_ids",
            new=_get_patched_allowed_function_ids(non_recurse_functions),
        )
        patcher_b = patch(
            "torch._dynamo.variables.functions.UserMethodVariable.call_function",
            new=_patched_call_function,
        )
        with patcher_a, patcher_b:
            yield (patcher_a.start(), patcher_b.start())
    else:
        for v in non_recurse_functions:
            torch._dynamo.allow_in_graph(v)  # type: ignore [no-untyped-call]

        patcher_b = patch(
            "torch._dynamo.variables.functions.UserMethodVariable.call_function",
            new=_patched_call_function,
        )

        with patcher_b:
            yield patcher_b.start()


def patch_to_expand_modules(
    fn: Callable[..., T], non_recurse_functions: Iterable[Callable[..., Any]] = ()
) -> Callable[..., T]:
    """By default TorchDynamo doesn't recurse into :mod:`torch.nn` modules or
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
    module = torch_nn_modules_to_user_modules(module)

    if not hasattr(module, "backends"):
        module.backends = []
    module.backends.append(backend)
    if not hasattr(module, "non_recurse_functions"):
        module.non_recurse_functions = list(_unit_scaled_functions)
    module.non_recurse_functions += non_recurse_functions

    backend = _compose_backends(module.backends)

    def new_forward(*args: Any, **kwargs: Any) -> Any:
        if module.rerun_transform:
            torch._dynamo.reset()
            dynamo_module = torch._dynamo.optimize(backend)(module)
            module.dynamo_forward = patch_to_expand_modules(
                dynamo_module.forward, module.non_recurse_functions
            )
            module.rerun_transform = False
        with patch.object(module, "forward", module.base_forward):
            return module.dynamo_forward(*args, **kwargs)

    module.rerun_transform = True
    module.base_forward = getattr(module, "base_forward", module.forward)
    module.forward = new_forward
    return module


__all__ = generate__all__(__name__)

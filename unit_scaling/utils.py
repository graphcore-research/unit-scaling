# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Utility functions for developing unit-scaled models."""

import ast
import math
import re
import typing
from collections import OrderedDict
from dataclasses import dataclass
from types import FunctionType, ModuleType
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union, cast

import einops
import torch
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import PythonLexer
from torch import Tensor, fx, nn

from . import functional
from ._internal_utils import generate__all__


@dataclass
class ScalePair:
    """Dataclass containing a pair of scalars, intended to represent the standard
    deviation of an arbitrary tensor in the forward and backward passes."""

    forward: Optional[float] = None
    backward: Optional[float] = None

    def __str__(self) -> str:
        fwd = f"{self.forward:.3}" if self.forward is not None else "n/a"
        bwd = f"{self.backward:.3}" if self.backward is not None else "n/a"
        return f"(-> {fwd}, <- {bwd})"


ScaleDict = typing.OrderedDict[str, ScalePair]


class ScaleTracker(torch.autograd.Function):
    """Given a `nn.Tensor`, records its standard deviation in the forward and
    backward pass in the supplied `ScalePair`."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        t: Tensor,
        scale_tracker: ScalePair,
    ) -> Tensor:
        scale_tracker.forward = float(t.std())
        ctx.scale_tracker = scale_tracker  # type: ignore
        return t

    @staticmethod
    def backward(  # type:ignore[override]
        ctx: torch.autograd.function.FunctionCtx, t: Tensor
    ) -> Tuple[Tensor, None, None]:
        ctx.scale_tracker.backward = float(t.std())  # type: ignore
        return t, None, None

    @staticmethod
    def track(t: Tensor, scale_tracker: ScalePair) -> Tensor:
        # Add typing information to `apply()` method from `torch.autograd.Function`
        apply = cast(Callable[[Tensor, ScalePair], Tensor], ScaleTracker.apply)
        return apply(t, scale_tracker)


class ScaleTrackingInterpreter(fx.Interpreter):
    """Wraps an `fx.GraphModule` such than when executed it records the standard
    deviation of every intermediate `nn.Tensor` in the forward and backward pass.

    Args:
        module (fx.GraphModule): the module to be instrumented.
    """

    def __init__(self, module: fx.GraphModule):
        super().__init__(module)
        self.scales: typing.OrderedDict[str, ScalePair] = OrderedDict()

    def run_node(self, n: fx.Node) -> Any:
        out = super().run_node(n)
        if isinstance(out, Tensor) and out.is_floating_point():
            scale_pair = ScalePair()
            out = ScaleTracker.track(out, scale_pair)
            self.scales[n.name] = scale_pair
        return out

    def call_function(
        self, target: fx.node.Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        return super().call_function(target, args, kwargs)

    def placeholder(
        self,
        target: fx.node.Target,
        args: Tuple[fx.node.Argument, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        """To handle functions being passed as arguments (for example constraints) the
        tracer represents them as placeholder nodes. This method extracts the original
        function from the node, as stored in the `target_to_function` dict."""
        if isinstance(target, str) and target.startswith("function_placeholder__"):
            return self.module.graph._tracer_extras["target_to_function"][
                target
            ]  # pragma: no cover
        return super().placeholder(target, args, kwargs)


def _record_scales(
    fx_graph_module: fx.GraphModule,
    inputs: Tuple[Tensor, ...],
    backward: Optional[Tensor] = None,
) -> ScaleDict:
    """Given a `torch.fx.GraphModule`, and dummy tensors to feed into the forward and
    backward passes, returns a dictionary of the scales (standard deviations) of every
    intermediate tensor in the model (forward and backward pass).

    Args:
        fx_graph_module (fx.GraphModule): the module to record.
        input (Tuple[Tensor, ...]): fed into the forward pass for analysis.
        backward (Tensor, optional): fed into the output's `.backward()`  method for
            analysis. Defaults to `None`, equivalent to calling plain `.backward()`.

    Returns:
        ScaleDict: An ordered dictionary with `ScalePair`s for each intermediate tensor.
    """
    tracking_module = ScaleTrackingInterpreter(fx_graph_module)
    out = tracking_module.run(*inputs)
    out.backward(backward)
    return tracking_module.scales


def _annotate(code: str, scales: ScaleDict, syntax_highlight: bool) -> str:
    """Given a string representation of some code and an `ScaleDict` with accompanying
    scales, annotates the code to include the scales on the right-hand side."""

    function_placeholder_regex = r"function_placeholder__(\w+)"

    def is_function_placeholder_line(code_line: str) -> bool:
        return bool(re.search(f" = {function_placeholder_regex}$", code_line))

    def cleanup_function_signature(code_line: str) -> str:
        code_line = re.sub(f", {function_placeholder_regex}", "", code_line)
        inner_code_line = code_line.split("(", 1)[1]
        replacement = re.sub(r"_([a-zA-Z0-9_]+)_", r"\1", inner_code_line)
        return code_line.replace(inner_code_line, replacement)

    def annotate_line(code_line: str) -> str:
        if code_line.startswith("torch.fx._symbolic_trace.wrap"):
            return ""
        code_line = code_line.split(";")[0]
        if is_function_placeholder_line(code_line):  # pragma: no cover
            return ""
        words = code_line.strip().split(" ")
        if words:
            if words[0] in scales:
                return f"{code_line};  {scales[words[0]]}"
            elif words[0] == "def":
                parsed = ast.parse(code_line + "\n\t...").body[0]
                assert isinstance(parsed, ast.FunctionDef)
                arg_names = [arg.arg for arg in parsed.args.args]
                scale_strs = [str(scales[a]) for a in arg_names if a in scales]
                code_line = cleanup_function_signature(code_line)
                if scale_strs:
                    return f"{code_line}  {', '.join(scale_strs)}"  # pragma: no cover
                else:
                    return code_line
        return code_line

    def remove_empty_lines(code_lines: Iterator[str]) -> Iterator[str]:
        return (line for line in code_lines if line.strip())

    code_lines = map(annotate_line, code.splitlines())
    code = "\n".join(remove_empty_lines(code_lines)).strip()
    code = code.replace("unit_scaling_functional_", "U.")
    if syntax_highlight:
        return highlight(code, PythonLexer(), TerminalFormatter())  # pragma: no cover
    return code


class _DeepTracer(fx.Tracer):
    """Version of `torch.fx.Tracer` which recurses into all sub-modules (if specified).

    Args:
        recurse_modules (bool): toggles recursive behavour. Defaults to True.
        autowrap_modules (Tuple[ModuleType]): defaults to
            `(math, einops, U.functional)`,
            Python modules whose functions should be wrapped automatically
            without needing to use fx.wrap().
        autowrap_function (Tuple[Callable, ...]): defaults to `()`,
            Python functions that should be wrapped automatically without
            needing to use fx.wrap().
    """

    def __init__(
        self,
        recurse_modules: bool = True,
        autowrap_modules: Tuple[ModuleType, ...] = (math, einops, functional),
        autowrap_functions: Tuple[Callable[..., Any], ...] = (),
    ) -> None:
        super().__init__(
            autowrap_modules=autowrap_modules,  # type: ignore[arg-type]
            autowrap_functions=autowrap_functions,
        )
        self.recurse_modules = recurse_modules
        self.target_to_function: Dict[str, FunctionType] = {}
        self.function_to_node: Dict[FunctionType, fx.Node] = {}
        # Fixes: `TypeError: __annotations__ must be set to a dict object`
        if id(FunctionType) in self._autowrap_function_ids:
            self._autowrap_function_ids.remove(id(FunctionType))

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        return not self.recurse_modules

    def create_arg(self, a: Any) -> fx.node.Argument:
        """Replaces callable arguments with strings for tracing."""
        if isinstance(a, FunctionType):  # pragma: no cover
            node = self.function_to_node.get(a)
            if node is None:
                assert hasattr(
                    a, "__name__"
                ), f"can't create arg for unnamed function: {a}"
                name = getattr(a, "__name__")
                target = f"function_placeholder__{name}"
                node = self.create_node("placeholder", target, (), {}, name)
                self.target_to_function[target] = a
                self.function_to_node[a] = node
            return node
        return super().create_arg(a)

    def trace(
        self,
        root: Union[torch.nn.Module, Callable[..., Any]],
        concrete_args: Optional[Dict[str, Any]] = None,
    ) -> fx.Graph:
        """Adds the `target_to_function` dict to the graph so the interpreter can use it
        downstream."""
        graph = super().trace(root, concrete_args)
        if not hasattr(graph, "_tracer_extras") or graph._tracer_extras is None:
            graph._tracer_extras = {}
        graph._tracer_extras["target_to_function"] = self.target_to_function
        return graph


def analyse_module(
    module: nn.Module,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    backward: Optional[Tensor] = None,
    recurse_modules: bool = True,
    syntax_highlight: bool = True,
    autowrap_modules: Tuple[ModuleType, ...] = (math, einops, functional),
    autowrap_functions: Tuple[Callable[..., Any], ...] = (),
) -> str:
    """Given a `nn.Module` and dummy forward and backward tensors, generates code
    representing the module annotated with the scales (standard deviation) of each
    tensor in both forward and backward passes. Implemented using `torch.fx`.

    Args:
        module (nn.Module): the module to analyse.
        inputs (Union[Tensor, Tuple[Tensor, ...]]): fed into the forward pass for
            analysis.
        backward (Tensor, optional): fed into the output's `.backward()` method for
            analysis. Defaults to `None`, equivalent to calling plain `.backward()`.
        recurse_modules (bool, optional): toggles recursive behavour. Defaults to True.
        syntax_highlight (bool, optional): Defaults to True.
        autowrap_modules (Tuple[ModuleType]): defaults to
            `(math, einops, U.functional)`,
            Python modules whose functions should be wrapped automatically
            without needing to use fx.wrap().
        autowrap_function (Tuple[Callable, ...]): defaults to `()`,
            Python functions that should be wrapped automatically without
            needing to use fx.wrap().

    Returns:
        str:
            a code string representing the operations in the module with scale
            annotations for each tensor, reflecting their standard deviations in the
            forward and backward passes.

    Examples::

        >>> class MLP(nn.Module):
        >>>    def __init__(self, d):
        >>>        super().__init__()
        >>>        self.fc1 = nn.Linear(d, d * 4)
        >>>        self.relu = nn.ReLU()
        >>>        self.fc2 = nn.Linear(d * 4, d)

        >>>    def forward(self, x):
        >>>        x = self.fc1(x)
        >>>        x = self.relu(x)
        >>>        x = self.fc2(x)
        >>>        return x


        >>> hidden_size = 2**10
        >>> x = torch.randn(hidden_size, hidden_size).requires_grad_()
        >>> bwd = torch.randn(hidden_size, hidden_size)

        >>> code = analyse_module(MLP(hidden_size), x, bwd)
        >>> print(code)
        def forward(self, x):  (-> 1.0, <- 0.236)
            fc1_weight = self.fc1.weight;  (-> 0.018, <- 6.54)
            fc1_bias = self.fc1.bias;  (-> 0.0182, <- 6.51)
            linear = torch._C._nn.linear(x, fc1_weight, fc1_bias);  (-> 0.578, <- 0.204)
            relu = torch.nn.functional.relu(linear, inplace = False);  (-> 0.337, <- 0.288)
            fc2_weight = self.fc2.weight;  (-> 0.00902, <- 13.0)
            fc2_bias = self.fc2.bias;  (-> 0.00904, <- 31.6)
            linear_1 = torch._C._nn.linear(relu, fc2_weight, fc2_bias);  (-> 0.235, <- 0.999)
            return linear_1
    """  # noqa: E501
    tracer = _DeepTracer(recurse_modules, autowrap_modules, autowrap_functions)
    fx_graph = tracer.trace(module)
    fx_graph_module = fx.GraphModule(tracer.root, fx_graph)

    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    scales = _record_scales(fx_graph_module, inputs, backward)
    return _annotate(fx_graph_module.code, scales, syntax_highlight=syntax_highlight)


__all__ = generate__all__(__name__)

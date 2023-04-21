# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Utility functions for developing unit-scaled models."""

import ast
import typing
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import PythonLexer
from torch import Tensor, fx, nn


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
    backward pass in the supplied `Dict`."""

    @staticmethod
    def forward(  # type:ignore[override]
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
        if isinstance(out, Tensor):
            self.scales[n.name] = ScalePair()
            out = ScaleTracker.apply(out, self.scales[n.name])
        return out


def _record_scales(
    fx_graph_module: fx.GraphModule,
    dummy_input: Tensor,
    dummy_backward: Tensor,
) -> ScaleDict:
    """Given a `torch.fx.GraphModule`, and dummy tensors to feed into the forward and
    backward passes, returns a dictionary of the scales (standard deviations) of every
    intermediate tensor in the model (forward and backward pass).

    Args:
        fx_graph_module (fx.GraphModule): the module to record.
        dummy_input (Tensor): fed into the forward pass for analysis.
        dummy_backward (Tensor): fed into the output's `.backward()`  method for
            analysis.

    Returns:
        ScaleDict: An ordered dictionary with `ScalePair`s for each intermediate tensor.
    """
    tracking_module = ScaleTrackingInterpreter(fx_graph_module)
    out = tracking_module.run(dummy_input)
    out.backward(dummy_backward)
    return tracking_module.scales


def _annotate(code: str, scales: ScaleDict, syntax_highlight: bool) -> str:
    """Given a string representation of some code and an `ScaleDict` with accompanying
    scales, annotates the code to include the scales on the right-hand side."""

    def _annotate_line(code_line: str) -> str:
        code_line = code_line.split(";")[0]
        words = code_line.strip().split(" ")
        if words:
            if words[0] in scales:
                return f"{code_line};  {scales[words[0]]}"
            elif words[0] == "def":
                parsed = ast.parse(code_line + "\n\t...").body[0]
                assert isinstance(parsed, ast.FunctionDef)
                arg_names = [arg.arg for arg in parsed.args.args]
                scale_strs = [str(scales[a]) for a in arg_names if a in scales]
                return f"{code_line}  {', '.join(scale_strs)}"
        return code_line

    code = "\n".join(map(_annotate_line, code.splitlines())).strip()
    if syntax_highlight:
        return highlight(code, PythonLexer(), TerminalFormatter())  # pragma: no cover
    return code


class _DeepTracer(fx.Tracer):
    """Version of `torch.fx.Tracer` which recurses into all sub-modules (if specified).

    Args:
        recurse_modules (bool): toggles recursive behavour. Defaults to True.
    """

    def __init__(self, recurse_modules: bool = True) -> None:
        super().__init__()
        self.recurse_modules = recurse_modules

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        return not self.recurse_modules


def analyse_module(
    module: nn.Module,
    input: Tensor,
    backward: Tensor,
    recurse_modules: bool = True,
    syntax_highlight: bool = True,
) -> str:
    """Given a `nn.Module` and dummy forward and backward tensors, generates code
    representing the module annotated with the scales (standard deviation) of each
    tensor in both forward and backward passes. Implemented using `torch.fx`.

    Args:
        module (nn.Module): the module to analyse.
        input (Tensor): fed into the forward pass for analysis.
        backward (Tensor): fed into the output's `.backward()` method for analysis.
        recurse_modules (bool, optional): toggles recursive behavour. Defaults to True.
        syntax_highlight (bool, optional): Defaults to True.

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
    tracer = _DeepTracer(recurse_modules=recurse_modules)
    fx_graph = tracer.trace(module)
    fx_graph_module = fx.GraphModule(tracer.root, fx_graph)

    scales = _record_scales(fx_graph_module, input, backward)
    return _annotate(fx_graph_module.code, scales, syntax_highlight=syntax_highlight)

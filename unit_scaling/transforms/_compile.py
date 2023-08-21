# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import TypeVar

from torch import _TorchCompileInductorWrapper, nn

from .utils import apply_transform

M = TypeVar("M", bound=nn.Module)


def compile(module: M) -> M:
    """A transform that applies torch.compile to a module.

    Note that this is slightly different to calling :code:`torch.compile(module)`.

    The current version of :func:`torch.compile` doesn't allow for nested transforms, so
    the following is not supported:

    .. code-block:: python

        import torch

        from unit_scaling.transforms import unit_scale

        module = torch.compile(unit_scale(module))

    :mod:`unit_scaling.transforms` addresses this by introducing a range of composable
    transforms. This works by moving the call to
    :func:`torch._dynamo.optimize` within the :code:`forward()` method of the module
    and only executing it on the first call to the module, or if a new transform
    is applied, the optimised call being cached thereafter.

    The :func:`unit_scaling.transforms.compile` function is one such composable
    transform. This means that the following can be written:

    .. code-block:: python

        from unit_scaling.transforms import compile, unit_scale

        module = compile(unit_scale(module))

    This will successfully combine the two transforms in a single module. Note that
    the call to compile must still come last, as its underlying backend returns a
    standard :class:`torch.nn.Module` rather than a :class:`torch.fx.GraphModule`.

    Currently :func:`unit_scaling.transforms.compile` does not support the ops needed
    for the :func:`unit_scaling.transforms.simulate_fp8` transform, though this may
    change in future PyTorch releases.

    Modules implemented manually with unit-scaled layers (i.e. without the global
    :code:`unit_scale(module)` transform) can still use :func:`torch.compile` in the
    standard way.

    Args:
        module (M): the module to be compiled.

    Returns:
        M: the compiled module.
    """
    return apply_transform(  # type: ignore[no-any-return]
        module, _TorchCompileInductorWrapper("default", None, None)  # type: ignore
    )

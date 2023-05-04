# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from functools import partial
from typing import Any, Callable, List, Optional, Type, TypeVar

from docstring_parser.google import (
    DEFAULT_SECTIONS,
    GoogleParser,
    Section,
    SectionType,
    compose,
)

T = TypeVar("T")


def _get_docstring_from_target(
    source: T,
    target: Any,
    short_description: Optional[str] = None,
    add_args: Optional[List[str]] = None,
) -> T:
    """Takes the docstring from `target`, modifies it, and applies it to `source`."""

    # Make the parser aware of the Shape and Examples sections (standard in torch docs)
    parser_sections = DEFAULT_SECTIONS + [
        Section("Shape", "shape", SectionType.SINGULAR),
        Section("Examples:", "examples", SectionType.SINGULAR),
    ]
    parser = GoogleParser(sections=parser_sections)
    docstring = parser.parse(target.__doc__)
    docstring.short_description = short_description
    if add_args:
        for arg_str in add_args:
            # Parse the additional args strings and add them to the docstring object
            param = parser._build_meta(arg_str, "Args")
            docstring.meta.append(param)
    source.__doc__ = compose(docstring)  # docstring object to actual string
    return source


def inherit_docstring(
    short_description: Optional[str] = None,
    add_args: Optional[List[str]] = None,
) -> Callable[[Type[T]], Type[T]]:
    """Returns a decorator which causes the wrapped class to inherit its parent
    docstring, with the specified modifications applied.

    Args:
        short_description (Optional[str], optional): replaces the top one-line
            description in the parent docstring with the one supplied. Defaults to None.
        add_args (Optional[List[str]], optional): appends the supplied argument strings
            to the list of arguments. Defaults to None.

    Returns:
        Callable[[Type], Type]: the decorator used to wrap the child class.
    """

    def decorator(cls: Type[T]) -> Type[T]:
        parent = cls.mro()[1]
        return _get_docstring_from_target(
            source=cls,
            target=parent,
            short_description=short_description,
            add_args=add_args,
        )

    return decorator


def docstring_from(
    target: Any,
    short_description: Optional[str] = None,
    add_args: Optional[List[str]] = None,
) -> Callable[[T], T]:
    """Returns a decorator which causes the wrapped object to take the docstring from
    the target object, with the specified modifications applied.

    Args:
        target (Any): the object to take the docstring from
        short_description (Optional[str], optional): replaces the top one-line
            description in the parent docstring with the one supplied. Defaults to None.
        add_args (Optional[List[str]], optional): appends the supplied argument strings
            to the list of arguments. Defaults to None.

    Returns:
        Callable[[Callable], Callable]: the decorator used to wrap the child object.
    """
    return partial(
        _get_docstring_from_target,
        target=target,
        short_description=short_description,
        add_args=add_args,
    )


def format_docstring(*args: str) -> Callable[[T], T]:
    """Returns a decorator that applies `cls.__doc__.format(*args)` to the target class.

    Args:
        args: (*str): the arguments to be passed to the docstrings `.format()` method.

    Returns:
        Callable[[Type], Type]: a decorator to format the docstring.
    """

    def f(cls: T) -> T:
        if isinstance(cls.__doc__, str):
            cls.__doc__ = cls.__doc__.format(*args)
        return cls

    return f


binary_constraint_docstring = (
    "constraint (Optional[BinaryConstraint], optional): function which"
    "takes `output_scale` and `grad_input_scale` and returns a single"
    " 'constrained' scale (usually necessary for valid gradients). If `None` is"
    " provided, no constraint will be applied. Defaults to `gmean`."
)

ternary_constraint_docstring = (
    "constraint (Optional[Callable[[float, float, float], float]], optional): function"
    " which takes `output_scale`, `left_grad_scale` & `right_grad_scale` (in that"
    " order) and returns a single 'constrained' scale (usually necessary for valid"
    " gradients). If `None` is provided, no constraint will be applied. Defaults to"
    " `gmean`."
)

variadic_constraint_docstring = (
    "constraint (Optional[Callable[..., float]], optional): function which takes any"
    " number of output/grad-input scales and returns a single 'constrained'"
    " scale. If `None` is provided, no constraint will be applied. Defaults to `gmean`."
)

# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import inspect
from functools import wraps
from itertools import zip_longest
from typing import Any, Callable, Iterable, List, Optional, Type, TypeVar

from docstring_parser.google import (
    DEFAULT_SECTIONS,
    GoogleParser,
    Section,
    SectionType,
    compose,
)

T = TypeVar("T")


def _validate(
    f: Callable[..., T], unsupported_args: Iterable[str] = {}
) -> Callable[..., T]:
    """Wraps the supplied function in a check to ensure its arguments aren't in the
    unsupported args list. Unsupported args are by nature optional (they have
    a default value). It is assumed this default is valid, but all other values are
    invalid."""

    argspec = inspect.getfullargspec(f)

    # argspec.defaults is a tuple of default arguments. These may begin at an offset
    # relative to rgspec.args due to args without a default. To zip these properly the
    # lists are reversed, zipped, and un-reversed, with missing values filled with `...`
    rev_args = reversed(argspec.args)
    rev_defaults = reversed(argspec.defaults) if argspec.defaults else []
    rev_arg_default_pairs = list(zip_longest(rev_args, rev_defaults, fillvalue=...))
    default_kwargs = dict(reversed(rev_arg_default_pairs))

    for arg in unsupported_args:
        if arg not in default_kwargs:
            raise ValueError(f"unsupported arg '{arg}' is not valid.")
        if default_kwargs[arg] is ...:
            raise ValueError(f"unsupported arg '{arg}' has no default value")

    @wraps(f)
    def _validate_args_supported(*args: Any, **kwargs: Any) -> T:
        arg_values = dict(zip(argspec.args, args))
        full_kwargs = {**arg_values, **kwargs}
        for arg_name, arg_value in full_kwargs.items():
            if arg_name in unsupported_args:
                arg_default_value = default_kwargs[arg_name]
                if arg_value != arg_default_value:
                    raise ValueError(
                        f"Support for the '{arg_name}' argument has not been"
                        " implemented for the unit-scaling library."
                        " Please remove it or replace it with its default value."
                    )
        return f(*args, **kwargs)

    return _validate_args_supported


def _get_docstring_from_target(
    source: T,
    target: Any,
    short_description: Optional[str] = None,
    add_args: Optional[List[str]] = None,
    unsupported_args: Iterable[str] = {},
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
    if docstring.long_description:
        docstring.long_description += "\n"  # fixes "Args:" section merging

    for param in docstring.params:
        if param.arg_name in unsupported_args and param.description is not None:
            param.description = (
                "**[not supported by unit-scaling]** " + param.description
            )

    if add_args:
        for arg_str in add_args:
            # Parse the additional args strings and add them to the docstring object
            param_meta = parser._build_meta(arg_str, "Args")
            docstring.meta.append(param_meta)

    source.__doc__ = compose(docstring)  # docstring object to actual string
    return source


def inherit_docstring(
    short_description: Optional[str] = None,
    add_args: Optional[List[str]] = None,
    unsupported_args: Iterable[str] = {},
) -> Callable[[Type[T]], Type[T]]:
    """Returns a decorator which causes the wrapped class to inherit its parent
    docstring, with the specified modifications applied.

    Args:
        short_description (Optional[str], optional): Replaces the top one-line
            description in the parent docstring with the one supplied. Defaults to None.
        add_args (Optional[List[str]], optional): Appends the supplied argument strings
            to the list of arguments. Defaults to None.
        unsupported_args (Iterable[str]): A list of arguments which are not supported.
            Documentation is updated and runtime checks added to enforce this.

    Returns:
        Callable[[Type], Type]: The decorator used to wrap the child class.
    """

    def decorator(cls: Type[T]) -> Type[T]:
        parent = cls.mro()[1]
        source = _get_docstring_from_target(
            source=cls,
            target=parent,
            short_description=short_description,
            add_args=add_args,
            unsupported_args=unsupported_args,
        )
        source.__init__ = _validate(source.__init__, unsupported_args)  # type: ignore
        return source

    return decorator


def docstring_from(
    target: Callable[..., T],
    short_description: Optional[str] = None,
    add_args: Optional[List[str]] = None,
    unsupported_args: Iterable[str] = {},
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Returns a decorator which causes the wrapped object to take the docstring from
    the target object, with the specified modifications applied.

    Args:
        target (Any): The object to take the docstring from.
        short_description (Optional[str], optional): Replaces the top one-line
            description in the parent docstring with the one supplied. Defaults to None.
        add_args (Optional[List[str]], optional): Appends the supplied argument strings
            to the list of arguments. Defaults to None.
        unsupported_args (Iterable[str]): A list of arguments which are not supported.
            Documentation is updated and runtime checks added to enforce this.

    Returns:
        Callable[[Callable], Callable]: The decorator used to wrap the child object.
    """

    def decorator(source: Callable[..., T]) -> Callable[..., T]:
        source = _get_docstring_from_target(
            source=source,
            target=target,
            short_description=short_description,
            add_args=add_args,
            unsupported_args=unsupported_args,
        )
        return _validate(source, unsupported_args)

    return decorator


def format_docstring(*args: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Returns a decorator that applies `cls.__doc__.format(*args)` to the target class.

    Args:
        args: (*str): The arguments to be passed to the docstrings `.format()` method.

    Returns:
        Callable[[Type], Type]: A decorator to format the docstring.
    """

    def f(cls: T) -> T:
        if isinstance(cls.__doc__, str):
            cls.__doc__ = cls.__doc__.format(*args)
        return cls

    return f


binary_constraint_docstring = (
    "constraint (Optional[str], optional): The name of the constraint function to be"
    " applied to the outputs & input gradient. In this case, the constraint name must"
    " be one of:"
    " [None, 'gmean', 'hmean', 'amean', 'to_output_scale', 'to_grad_input_scale']"
    " (see `unit_scaling.constraints` for details on these constraint functions)."
    " Defaults to `gmean`."
)

ternary_constraint_docstring = (
    "constraint (Optional[str], optional): The name of the constraint function to be"
    " applied to the outputs & input gradients. In this case, the constraint name must"
    " be one of:"
    " [None, 'gmean', 'hmean', 'amean', 'to_output_scale', 'to_left_grad_scale',"
    " to_right_grad_scale]"
    " (see `unit_scaling.constraints` for details on these constraint functions)."
    " Defaults to `gmean`."
)

variadic_constraint_docstring = (
    "constraint (Optional[str], optional): The name of the constraint function to be"
    " applied to the outputs & input gradients. In this case, the constraint name must"
    " be one of:"
    " [None, 'gmean', 'hmean', 'amean', 'to_output_scale']"
    " (see `unit_scaling.constraints` for details on these constraint functions)."
    " Defaults to `gmean`."
)


def mult_docstring(name: str = "mult") -> str:
    return (
        f"{name} (float, optional): a multiplier to be applied to change the shape"
        " of a nonlinear function. Typically, high multipliers (> 1) correspond to a"
        " 'sharper' (low temperature) function, while low multipliers (< 1) correspond"
        " to a 'flatter' (high temperature) function."
    )

"""
Some helper function to format the error message.
"""

from typing import Any, Optional, Callable


def invalid_type(
    name: str,
    obj: Any,
    *,
    more_msg: str = "",
    expected: Any = None,
) -> str:
    """
    Invalid type string.

    Code inside:
    >>> return (
    ...     f"Unexpected type for '{name}': `{type(obj).__name__}`."
    ...     + (f"({more_msg})" if more_msg else "")
    ...     + (f"(Expected: {expected})" if expected else "")
    ... )
    """
    return (
        f"Unexpected type for '{name}': `{type(obj).__name__}`."
        + (f"({more_msg})" if more_msg else "")
        + (f"(Expected: {expected})" if expected else "")
    )


def invalid_value(
    name: str, obj: Any, *, more_msg: str = "", expected: Optional[str] = ""
) -> str:
    """
    Invalid value string.

    Code inside:
    >>> return (
    ...     f"Unexpected value for '{name}': `{obj}`."
    ...     + (f"({more_msg})" if more_msg else "")
    ...     + (f"(Expected: {expected})" if expected else "")
    ... )
    """
    return (
        f"Unexpected value for '{name}': `{obj}`."
        + (f"({more_msg})" if more_msg else "")
        + (f"(Expected: {expected})" if expected else "")
    )


def assert_fail(
    error_str_func: Optional[Callable[..., str]],
    *args,
    add_str: str = "",
    **kwargs,
) -> str:
    """
    Assert fail string.

    Code inside:
    >>> return (
    ...     "Assert failed: "
    ...     + (f"\\n{add_str}\\n" if add_str else "")
    ...     + (f"{error_str_func(*args, **kwargs)}" if error_str_func else "")
    ... )
    """
    return (
        "Assert failed: "
        + (f"\n{add_str}\n" if add_str else "")
        + (f"{error_str_func(*args, **kwargs)}" if error_str_func else "")
    )


def not_implemented(func_name: str, obj: Any = None) -> str:
    """
    Not implemented string.

    Code inside:
    >>> return f"Sorry, the function '{func_name}' is not complete" + (
    ...     f" for `{obj}`(type: `{type(obj).__name__}`)."
    ...     if obj is not None
    ...     else "."
    ... )
    """

    return f"Sorry, the function '{func_name}' is not complete" + (
        f" for `{obj}`(type: `{type(obj).__name__}`)."
        if obj is not None
        else "."
    )

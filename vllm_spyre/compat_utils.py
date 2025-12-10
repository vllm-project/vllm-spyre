import inspect
from dataclasses import fields
from typing import Callable


def dataclass_fields(cls: type) -> list[str]:
    return [f.name for f in fields(cls)]


def has_argument(func: Callable, param_name: str) -> bool:
    # Checks the signature of a method and returns true iff the method accepts
    # a parameter named `$param_name`. This is slow and should not be used in
    # the critical path.
    signature = inspect.signature(func)
    for param in signature.parameters.values():
        if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY) and param.name == param_name:
            return True
    return False

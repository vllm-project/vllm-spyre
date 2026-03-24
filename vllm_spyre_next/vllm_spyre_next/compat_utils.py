import inspect
from dataclasses import fields
from functools import lru_cache
from typing import Callable


def dataclass_fields(cls: type) -> list[str]:
    return [f.name for f in fields(cls)]


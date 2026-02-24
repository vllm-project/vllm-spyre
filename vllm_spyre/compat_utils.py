import inspect
from dataclasses import fields
from functools import lru_cache, wraps
from typing import Callable


def dataclass_fields(cls: type) -> list[str]:
    return [f.name for f in fields(cls)]


@lru_cache
def has_argument(func: Callable, param_name: str) -> bool:
    # Checks the signature of a method and returns true iff the method accepts
    # a parameter named `$param_name`.
    # `lru_cache` is used because inspect + for looping is pretty slow. This
    # should not be invoked in the critical path.
    signature = inspect.signature(func)
    for param in signature.parameters.values():
        if (
            param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and param.name == param_name
        ):
            return True
    return False


def patch_kv_cache_manager():
    print("Patching kv cache manager")

    from vllm.v1.core.kv_cache_coordinator import UnitaryKVCacheCoordinator
    from vllm.v1.core.kv_cache_utils import (
        BlockHash,
        KVCacheBlock,
    )

    wrapped = UnitaryKVCacheCoordinator.find_longest_cache_hit

    @wraps(UnitaryKVCacheCoordinator.find_longest_cache_hit)
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        max_cache_hit_length += 1

        hit_blocks, hit_tokens = wrapped(self, block_hashes, max_cache_hit_length)
        if max_cache_hit_length == hit_tokens:
            hit_tokens -= self.block_size
        return hit_blocks, hit_tokens

    UnitaryKVCacheCoordinator.find_longest_cache_hit = find_longest_cache_hit  # ty: ignore

"""Contains utilities for caching models (instantiated as vLLM endpoints) 
across test cases, to speed up test runtime."""

from typing import Optional

import pytest
from vllm import LLM
from vllm.v1.engine.core import EngineCore

from tests.llm_cache_utils import (EngineCache, LLMCache, RemoteOpenAIServer,
                                   RemoteOpenAIServerCache)
from tests.spyre_util import DecodeWarmupShapes

API_SERVER_CACHE = RemoteOpenAIServerCache()
LLM_CACHE = LLMCache()
ENGINE_CACHE = EngineCache()


def get_cached_llm(
    model: str,
    max_model_len: int,
    tensor_parallel_size: int,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    warmup_shapes: DecodeWarmupShapes | None = None,
    max_num_seqs: Optional[int] = None,
    use_cb: bool = False,
) -> LLM:
    # Clear other caches first
    API_SERVER_CACHE.clear()
    ENGINE_CACHE.clear()

    return LLM_CACHE.get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        backend=backend,
        monkeypatch=monkeypatch,
        warmup_shapes=warmup_shapes,
        max_num_seqs=max_num_seqs,
        use_cb=use_cb,
    )


def get_cached_api_server(model: str, server_args: list[str],
                          server_env: dict) -> RemoteOpenAIServer:
    # Clear other caches first
    LLM_CACHE.clear()
    ENGINE_CACHE.clear()

    return API_SERVER_CACHE.get_api_server(
        model=model,
        server_args=server_args,
        server_env=server_env,
    )


def clear_llm_caches():
    LLM_CACHE.clear()
    API_SERVER_CACHE.clear()
    ENGINE_CACHE.clear()


def print_llm_cache_info():
    print("\n----- LLM Cache info ----\n")
    print(f"vllm.LLM Cache hits: {LLM_CACHE._cache.hits} / "
          f"misses: {LLM_CACHE._cache.misses}")
    print(f"Runtime Server Cache hits: {API_SERVER_CACHE._cache.hits} / "
          f"misses: {API_SERVER_CACHE._cache.misses}")
    print(f"Engine Core Cache hits: {ENGINE_CACHE._cache.hits} / "
          f"misses: {ENGINE_CACHE._cache.misses}")
    print("\n-------------------------\n")


def get_cached_engine(
    model: str,
    max_model_len: int,
    max_num_seqs: int,
    available_blocks: int,
    backend: str,
    monkeypatch,
) -> EngineCore:
    # Clear other caches first
    LLM_CACHE.clear()
    API_SERVER_CACHE.clear()

    return ENGINE_CACHE.get_engine(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        available_blocks=available_blocks,
        backend=backend,
        monkeypatch=monkeypatch,
    )

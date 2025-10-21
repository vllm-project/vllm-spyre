"""Contains utilities for caching models (instantiated as vLLM endpoints)
across test cases, to speed up test runtime."""

from typing import Callable, Generic, Optional, TypeVar

import pytest
from llm_cache_util import force_engine_shutdown
from spyre_util import (DecodeWarmupShapes, ModelInfo, RemoteOpenAIServer,
                        patch_environment)
from vllm import LLM, EngineArgs
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor

from vllm_spyre.v1.sample.golden_token_injector import GoldenTokenInjector

T = TypeVar("T")

## class definitions ##########################################


class ModelCache(Generic[T]):

    def __init__(self, teardown_method: Callable[[T], None] | None = None):
        self._model: T | None = None
        self._runtime_config: dict | None = None
        self._past_runtime_configs: list[dict] = []

        self.hits = 0
        self.misses = 0

        if not teardown_method:
            self._teardown = lambda x: x.shutdown()
        else:
            self._teardown = teardown_method

    def maybe_get(self, runtime_config: dict) -> T | None:
        if runtime_config == self._runtime_config:
            self.hits += 1
            return self._model

        self.misses += 1

        print(f"\n\tModel cache miss for type [{self._type()}]")
        print(f"Requested config: {runtime_config}")
        print(f"Currently cached config {self._runtime_config}\n")

        return None

    def set(self, runtime_config: dict, model: T) -> T:
        assert runtime_config not in self._past_runtime_configs, (
            f"Runtime config {runtime_config} was previously cached for type "
            f"[{self._type()}], error in test ordering!")
        self._runtime_config = runtime_config
        self._past_runtime_configs.append(self._runtime_config)
        self._model = model

        return self._model

    def clear(self):
        if self._model:
            self._teardown(self._model)
            self._model = None
            self._runtime_config = None

    def _type(self) -> type | None:
        if hasattr(self, "__orig_class__"):
            return self.__orig_class__.__args__[0]
        return None


class LLMCache:
    """Caches a vllm.LLM for use in subsequent tests.

    Only a single LLM can be cached at a time, as AIUs don't support loading
    multiple models at once."""

    def __init__(self):
        self._cache: ModelCache[LLM] = ModelCache[LLM](
            teardown_method=lambda x: force_engine_shutdown(x))

    def get_cached_llm(
        self,
        model: str | ModelInfo,
        max_model_len: int,
        tensor_parallel_size: int,
        backend: str,
        monkeypatch: pytest.MonkeyPatch,
        warmup_shapes: DecodeWarmupShapes | None = None,
        max_num_seqs: Optional[int] = None,
        use_cb: bool = False,
    ) -> LLM:
        """Creates an LLM with the provided runtime configuration.

        If the last LLM created matches the config, then returns the cached LLM
        instead to reduce LLM instantiation overhead.
        """
        runtime_config = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            "backend": backend,
            "use_cb": use_cb,
        }
        if use_cb:
            runtime_config.update({
                "max_model_len": max_model_len,
                "max_num_seqs": max_num_seqs
            })
        else:
            runtime_config.update({"warmup_shapes": tuple(warmup_shapes)})

        # Always patch the environment so that it's consistent with the LLM
        patch_environment(use_cb, warmup_shapes, backend, monkeypatch)

        maybe_llm = self._cache.maybe_get(runtime_config)
        if maybe_llm:
            return maybe_llm
        self.clear()

        if isinstance(model, ModelInfo):
            revision = model.revision
            model_name = model.name
        else:
            revision = None
            model_name = model

        return self._cache.set(
            runtime_config,
            LLM(
                model=model_name,
                tokenizer=model_name,
                revision=revision,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                tensor_parallel_size=tensor_parallel_size,
                logits_processors=[GoldenTokenInjector],
            ),
        )

    def clear(self) -> None:
        self._cache.clear()


class EngineCache:
    """Cache for continuous batching engines"""

    def __init__(self):
        self._cache: ModelCache[EngineCore] = ModelCache[EngineCore]()

    def get_engine(
        self,
        model: str | ModelInfo,
        max_model_len: int,
        max_num_seqs: int,
        available_blocks: int,
        backend: str,
        monkeypatch,
    ) -> EngineCore:
        runtime_config = {
            "model": model,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "available_blocks": available_blocks,
        }

        # Always patch the environment so that it's consistent with the engine
        patch_environment(use_cb=True,
                          warmup_shapes=None,
                          backend=backend,
                          monkeypatch=monkeypatch)

        maybe_engine = self._cache.maybe_get(runtime_config)
        if maybe_engine:
            return maybe_engine
        self.clear()

        if isinstance(model, ModelInfo):
            revision = model.revision
            model_name = model.name
        else:
            revision = None
            model_name = model

        # 🌶️🌶️🌶️
        # Messing with the blocks and context length by either:
        # - setting context < 512 tokens
        # - setting available blocks != (context * batch size // 64)
        # can cause compilation failures on spyre hardware.

        # So we first create the engine and compile with valid configs,
        # then adjust these limits in the engine's scheduler for tests.

        # Setup the engine
        # Round max_num_seqs (batch size) to the next power of two for
        # Spyre compilation. This seems more robust and helps that all tests in
        # tests/e2e/test_spyre_cb_inference_steps.py pass on Spyre.
        max_num_seqs_compiled = 1 << (max_num_seqs - 1).bit_length()
        engine_args = EngineArgs(model=model_name,
                                 tokenizer=model_name,
                                 revision=revision,
                                 max_model_len=max(max_model_len, 512),
                                 max_num_seqs=max_num_seqs_compiled,
                                 num_gpu_blocks_override=None,
                                 logits_processors=[GoldenTokenInjector])
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)

        engine_core = EngineCore(vllm_config=vllm_config,
                                 executor_class=executor_class,
                                 log_stats=False)

        # Set scheduler configs for max_model_len and max_num_seqs to the
        # original values. They were changed for more robust compilation only.
        engine_core.scheduler.scheduler_config.max_model_len = max_model_len
        engine_core.scheduler.scheduler_config.max_num_seqs = max_num_seqs

        if available_blocks is not None:
            worker = engine_core.model_executor.driver_worker.worker
            # NB: We cannot create extra blocks after compilation
            assert worker.model_runner.n_blocks >= available_blocks, \
                "Cannot set available_blocks > (context * batch size // 64)"
            worker.model_runner.n_blocks = available_blocks

        return self._cache.set(
            runtime_config,
            engine_core,
        )

    def clear(self) -> None:
        self._cache.clear()


class RemoteOpenAIServerCache:

    def __init__(self):
        self._cache: ModelCache[RemoteOpenAIServer] = ModelCache[
            RemoteOpenAIServer]()

    def get_api_server(self, model: str | ModelInfo, server_args: list[str],
                       server_env: dict) -> RemoteOpenAIServer:
        """Get or create a new OpenAI server for a given model. and config"""
        runtime_config = {
            "model": model,
            "server_args": tuple(server_args),
            "server_env": server_env,
        }
        maybe_server = self._cache.maybe_get(runtime_config)
        if maybe_server:
            return maybe_server
        self.clear()

        return self._cache.set(
            runtime_config,
            RemoteOpenAIServer(model=model,
                               vllm_serve_args=server_args,
                               env_dict=server_env),
        )

    def clear(self) -> None:
        self._cache.clear()


## class definitions ##########################################

API_SERVER_CACHE = RemoteOpenAIServerCache()
LLM_CACHE = LLMCache()
ENGINE_CACHE = EngineCache()


def get_cached_llm(
    model: str | ModelInfo,
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

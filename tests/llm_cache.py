"""Contains utilities for caching models (instantiated as vLLM endpoints)
across test cases, to speed up test runtime."""

import functools
import os
from typing import Callable, Generic, TypeVar

import pytest
from llm_cache_util import force_engine_core_shutdown, force_engine_shutdown
from spyre_util import DecodeWarmupShapes, ModelInfo, RemoteOpenAIServer, patch_environment
from vllm import LLM, EngineArgs
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor
from vllm.forward_context import get_forward_context


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

        # All VLLM_DT env vars must be cleared between cache accesses to avoid polluting new models.
        # Cache any that are set at test collection time to put them back after clearing the cache
        self._cache_vllm_dt_env_vars()

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
            f"[{self._type()}], error in test ordering!"
        )
        self._runtime_config = runtime_config
        self._past_runtime_configs.append(self._runtime_config)
        self._model = model

        return self._model

    def clear(self):
        if self._model:
            self._teardown(self._model)
            self._model = None
            self._runtime_config = None
            self._reset_vllm_dt_env_vars()

    def _type(self) -> type | None:
        if hasattr(self, "__orig_class__"):
            return self.__orig_class__.__args__[0]
        return None

    def _get_vllm_dt_env_vars(self) -> dict[str, str]:
        return {k: v for k, v in os.environ.items() if k.startswith("VLLM_DT")}

    def _cache_vllm_dt_env_vars(self):
        self._preexisting_vllm_dt_env_vars = self._get_vllm_dt_env_vars()

    def _reset_vllm_dt_env_vars(self):
        # Clear all
        for k in self._get_vllm_dt_env_vars():
            os.environ.pop(k, None)
        # Set back
        for k, v in self._preexisting_vllm_dt_env_vars.items():
            os.environ[k] = v


class LLMCache:
    """Caches a vllm.LLM for use in subsequent tests.

    Only a single LLM can be cached at a time, as AIUs don't support loading
    multiple models at once."""

    def __init__(self):
        self._cache: ModelCache[LLM] = ModelCache[LLM](
            teardown_method=lambda x: force_engine_shutdown(x)
        )

    def get_cached_llm(
        self,
        model: str | ModelInfo,
        max_model_len: int,
        tensor_parallel_size: int,
        backend: str,
        monkeypatch: pytest.MonkeyPatch,
        warmup_shapes: DecodeWarmupShapes | None = None,
        max_num_seqs: int | None = None,
        use_cb: bool = False,
        use_pc: bool = False,
        max_num_batched_tokens: int | None = None,
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
            "use_pc": use_pc,
            "max_num_batched_tokens": max_num_batched_tokens,
        }
        if use_cb:
            runtime_config.update({"max_model_len": max_model_len, "max_num_seqs": max_num_seqs})
        else:
            runtime_config.update({"warmup_shapes": tuple(warmup_shapes)})

        # Always patch the environment so that it's consistent with the LLM
        # Use chunked prefill if max_num_batched_tokens is set
        use_chunked_prefill = bool(max_num_batched_tokens)
        if use_pc:
            assert use_chunked_prefill
        patch_environment(
            use_cb,
            warmup_shapes,
            backend,
            monkeypatch,
            use_chunked_prefill=use_chunked_prefill,
            max_num_batched_tokens=max_num_batched_tokens,
        )

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
                tokenizer_revision=revision,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                tensor_parallel_size=tensor_parallel_size,
                max_num_batched_tokens=max_num_batched_tokens,
                logits_processors=[GoldenTokenInjector],
                enable_prefix_caching=use_pc,
            ),
        )

    def clear(self) -> None:
        self._cache.clear()


class EngineCache:
    """Cache for continuous batching engines"""

    def __init__(self):
        self._cache: ModelCache[EngineCore] = ModelCache[EngineCore](
            teardown_method=lambda x: force_engine_core_shutdown(x)
        )

    def get_engine(
        self,
        model: str | ModelInfo,
        max_model_len: int,
        max_num_seqs: int,
        available_blocks: int,
        max_num_batched_tokens: int,
        use_pc: bool,
        backend: str,
        monkeypatch,
    ) -> EngineCore:
        runtime_config = {
            "model": model,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "available_blocks": available_blocks,
            "use_pc": use_pc,
            "max_num_batched_tokens": max_num_batched_tokens,
        }

        # Always patch the environment so that it's consistent with the engine
        if max_num_batched_tokens is not None and max_num_batched_tokens > 0:
            use_chunked_prefill = True
        else:
            use_chunked_prefill = False

        if use_pc:
            assert use_chunked_prefill
        patch_environment(
            use_cb=True,
            warmup_shapes=None,
            backend=backend,
            monkeypatch=monkeypatch,
            use_chunked_prefill=use_chunked_prefill,
        )

        maybe_engine = self._cache.maybe_get(runtime_config)
        if maybe_engine:
            if use_pc:
                # reset the blockpool across tests: this will erase any seen
                # prefixes and makes sure that the used block ids in each test
                # are independent of the test ordering.
                model_runner = maybe_engine.model_executor.driver_worker.worker.model_runner
                model_runner.block_pool = model_runner._make_block_pool()
                model_runner.kv_cache_manager = model_runner._make_kv_cache_manager()

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
        engine_args = EngineArgs(
            model=model_name,
            tokenizer=model_name,
            revision=revision,
            tokenizer_revision=revision,
            max_model_len=max(max_model_len, 512),
            max_num_seqs=max_num_seqs_compiled,
            num_gpu_blocks_override=None,
            logits_processors=[GoldenTokenInjector],
            max_num_batched_tokens=max_num_batched_tokens,
            enable_prefix_caching=use_pc,
        )
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)

        engine_core = EngineCore(
            vllm_config=vllm_config, executor_class=executor_class, log_stats=False
        )

        # Set scheduler configs for max_model_len and max_num_seqs to the
        # original values. They were changed for more robust compilation only.
        engine_core.scheduler.model_config.max_model_len = max_model_len
        engine_core.scheduler.scheduler_config.max_num_seqs = max_num_seqs

        if available_blocks is not None:
            worker = engine_core.model_executor.driver_worker.worker
            # NB: We cannot create extra blocks after compilation
            assert worker.model_runner.n_blocks >= available_blocks, (
                "Cannot set available_blocks > (context * batch size // 64)"
            )
            worker.model_runner.n_blocks = available_blocks
            # need to overwrite the block pool and kv cache manager if the
            # number of available blocks has changed
            worker.model_runner.block_pool = worker.model_runner._make_block_pool()
            worker.model_runner.kv_cache_manager = worker.model_runner._make_kv_cache_manager()

        # 🌶️🌶️🌶️
        # Set up a wrapper around the model that will store the forward context at each step.
        # This allows tests to make assertions on the SypreAttentionMetadata.
        self._wrap_model_forward(engine_core)

        return self._cache.set(
            runtime_config,
            engine_core,
        )

    def clear(self) -> None:
        self._cache.clear()

    @staticmethod
    def _wrap_model_forward(engine_core: EngineCore) -> None:
        # Set up the model to save the forward context at each step
        _model = engine_core.model_executor.driver_worker.worker.model_runner.model
        original_model_fwd = _model.forward

        engine_core.forward_context = None

        def __context_forward__(*args, **kwargs):
            engine_core.forward_context = get_forward_context()
            return original_model_fwd(*args, **kwargs)

        functools.update_wrapper(__context_forward__, _model.forward)
        _model.forward = __context_forward__


class RemoteOpenAIServerCache:
    def __init__(self):
        self._cache: ModelCache[RemoteOpenAIServer] = ModelCache[RemoteOpenAIServer]()

    def get_api_server(
        self, model: str | ModelInfo, server_args: list[str], server_env: dict
    ) -> RemoteOpenAIServer:
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
            RemoteOpenAIServer(model=model, vllm_serve_args=server_args, env_dict=server_env),
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
    max_num_seqs: int | None = None,
    use_cb: bool = False,
    max_num_batched_tokens: int | None = None,
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
        max_num_batched_tokens=max_num_batched_tokens,
    )


def get_cached_api_server(
    model: str, server_args: list[str], server_env: dict
) -> RemoteOpenAIServer:
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
    print(f"vllm.LLM Cache hits: {LLM_CACHE._cache.hits} / misses: {LLM_CACHE._cache.misses}")
    print(
        f"Runtime Server Cache hits: {API_SERVER_CACHE._cache.hits} / "
        f"misses: {API_SERVER_CACHE._cache.misses}"
    )
    print(
        f"Engine Core Cache hits: {ENGINE_CACHE._cache.hits} / misses: {ENGINE_CACHE._cache.misses}"
    )
    print("\n-------------------------\n")


def get_cached_engine(
    model: str,
    max_model_len: int,
    max_num_seqs: int,
    available_blocks: int,
    backend: str,
    monkeypatch,
    max_num_batched_tokens: int | None = None,
    use_pc: bool = False,
) -> EngineCore:
    # Clear other caches first
    LLM_CACHE.clear()
    API_SERVER_CACHE.clear()

    engine = ENGINE_CACHE.get_engine(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        available_blocks=available_blocks,
        max_num_batched_tokens=max_num_batched_tokens,
        use_pc=use_pc,
        backend=backend,
        monkeypatch=monkeypatch,
    )

    # TODO: clean up any remaining requests from previous failed tests
    return engine

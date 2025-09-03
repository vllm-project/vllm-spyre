"""Contains utilities for caching models (instantiated as vLLM endpoints) 
across test cases, to speed up test runtime."""

import os
import subprocess
import sys
import time
from concurrent.futures import Executor
from typing import Callable, Generic, Optional, TypeVar

import openai
import pytest
import requests
from vllm import LLM, EngineArgs
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser, get_open_port
from vllm.v1.engine.core import EngineCore

from tests.llm_cache_utils import force_engine_shutdown
from tests.spyre_util import DecodeWarmupShapes, patch_environment

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
        model: str,
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

        return self._cache.set(
            runtime_config,
            LLM(
                model=model,
                tokenizer=model,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                tensor_parallel_size=tensor_parallel_size,
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
        model: str,
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

        # Setup the engine
        engine_args = EngineArgs(
            model=model,
            tokenizer=model,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            num_gpu_blocks_override=available_blocks,
        )
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)
        return self._cache.set(
            runtime_config,
            EngineCore(vllm_config=vllm_config,
                       executor_class=executor_class,
                       log_stats=False),
        )

    def clear(self) -> None:
        self._cache.clear()


class RemoteOpenAIServer:
    """Subprocess wrapper that boots a vllm server with `vllm serve` for testing
    against"""

    DUMMY_API_KEY = "token-abc123"  # vLLM's OpenAI server does not need API key

    def __init__(
        self,
        model: str,
        vllm_serve_args: list[str],
        *,
        env_dict: Optional[dict[str, str]] = None,
        seed: Optional[int] = 0,
        auto_port: bool = True,
        max_wait_seconds: Optional[float] = None,
    ) -> None:
        # NB: This implementation does not ensure that the model is downloaded
        # before booting the server, it should be used with models already
        # cached on disk

        if auto_port:
            if "-p" in vllm_serve_args or "--port" in vllm_serve_args:
                raise ValueError("You have manually specified the port "
                                 "when `auto_port=True`.")

            # Don't mutate the input args
            vllm_serve_args = vllm_serve_args + [
                "--port", str(get_open_port())
            ]
        if seed is not None:
            if "--seed" in vllm_serve_args:
                raise ValueError("You have manually specified the seed "
                                 f"when `seed={seed}`.")

            vllm_serve_args = vllm_serve_args + ["--seed", str(seed)]

        parser = FlexibleArgumentParser(
            description="vLLM's remote OpenAI server.")
        parser = make_arg_parser(parser)
        args = parser.parse_args(["--model", model, *vllm_serve_args])
        self.host = str(args.host or "localhost")
        self.port = int(args.port)

        env = os.environ.copy()
        if env_dict is not None:
            env.update(env_dict)
        self.proc = subprocess.Popen(
            ["vllm", "serve", model, *vllm_serve_args],
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        max_wait_seconds = max_wait_seconds or 600
        self._wait_for_server(url=self.url_for("health"),
                              timeout=max_wait_seconds)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()

    def shutdown(self):
        self.proc.terminate()
        try:
            self.proc.wait(8)
        except subprocess.TimeoutExpired:
            # force kill if needed
            self.proc.kill()

    def _wait_for_server(self, *, url: str, timeout: float):
        # run health check
        start = time.time()
        while True:
            try:
                if requests.get(url).status_code == 200:
                    break
            except Exception:
                # this exception can only be raised by requests.get,
                # which means the server is not ready yet.
                # the stack trace is not useful, so we suppress it
                # by using `raise from None`.
                result = self.proc.poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from None

                time.sleep(0.5)
                if time.time() - start > timeout:
                    raise RuntimeError(
                        "Server failed to start in time.") from None

    @property
    def url_root(self) -> str:
        return f"http://{self.host}:{self.port}"

    def url_for(self, *parts: str) -> str:
        return self.url_root + "/" + "/".join(parts)

    def get_client(self, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600
        return openai.OpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
            max_retries=0,
            **kwargs,
        )

    def get_async_client(self, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600
        return openai.AsyncOpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
            max_retries=0,
            **kwargs,
        )


class RemoteOpenAIServerCache:

    def __init__(self):
        self._cache: ModelCache[RemoteOpenAIServer] = ModelCache[
            RemoteOpenAIServer]()

    def get_api_server(self, model: str, server_args: list[str],
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

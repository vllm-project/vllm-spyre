"""Contains utilities for caching models (instantiated as vLLM endpoints) 
across test cases, to speed up test runtime."""

import os
import subprocess
import sys
import time
from typing import Callable, Generic, NamedTuple, Optional, TypeVar

import openai
import pytest
import requests
from vllm import LLM, EngineArgs
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser, get_open_port
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor


def force_engine_shutdown(llm: LLM):
    """
    🌶️🌶️🌶️
    This hack is here because of an issue in vllm 0.9.2+ where a circular
    reference occurs in vllm.executor.ray_utils if ray is not installed. This
    circular reference holds a copy of the vllm config which contains a
    reference to the LLM, which means it can never be garbage collected.
    Since vllm.LLM relies on garbage collection to shut down its engine, the
    engine never shuts down. When running tensor parallel workloads, if the
    engine is never shut down then the TP worker processes are never killed.
    When the TP worker processes are held open, all future attempts to create a
    new engine will fail with an EADDRINUSE error.
    🌶️🌶️🌶️
    """
    llm.llm_engine.engine_core.shutdown()


def sort_tests_for_llm_caching(items: list) -> None:
    """Sorts a list of pytest cases based on the LLM parameterizations.
    
    This allows us to group tests together that use the same model and config,
    which means they can reuse the underlying LLM. Then we can cache the LLM
    across tests to save time.

    This is important because spinning up a new vLLM engine from scratch takes
    a decent amount of time, even with the torch compilation cache active. LLM
    creation dominates the runtime of our test suites.

    This sorts the `items` list in-place.
    """
    items.sort(key=SortKey.from_item)


DecodeWarmupShapes = list[tuple[int, int, int]]


class SortKey(NamedTuple):
    """Sort key that groups by runtime configuration.
    
    The order of attributes is important here and controls the test
    grouping.
    """
    cache_type: str  # None (empty str), online, llm, engine
    backend: str = ""
    model: str = ""
    tp_size: int = 1
    use_cb: bool = False
    max_model_len: int = 0
    max_num_seqs: int = 0
    num_blocks: int = 0
    warmup_shapes: DecodeWarmupShapes | None = None

    @staticmethod
    def from_item(item) -> 'SortKey':
        cache_type = SortKey._get_cache_type(item)
        if not cache_type:
            # Don't add any extra re-ordering logic for tests that won't utilize
            # the cache
            return SortKey(cache_type=cache_type)

        if not hasattr(item, "callspec"):
            # This isn't great- we probably want to cache but can't because the
            # test has no parameters at all
            return SortKey(cache_type="")

        use_cb = SortKey._uses_cb(item)
        if use_cb:
            sort_kwargs = {
                "max_model_len": SortKey._get_max_model_len(item),
                "max_num_seqs": SortKey._get_max_num_seqs(item),
            }
        else:
            sort_kwargs = {
                "warmup_shapes": SortKey._get_warmup_shapes(item),
            }

        return SortKey(cache_type=cache_type,
                       model=SortKey._get_model(item),
                       backend=SortKey._get_backend(item),
                       tp_size=SortKey._get_tp_size(item),
                       use_cb=SortKey._uses_cb(item),
                       num_blocks=SortKey._get_num_blocks(item),
                       **sort_kwargs)

    @staticmethod
    def _get_cache_type(item) -> str:
        # If not an e2e test then assume no cache
        if "e2e" not in item.listnames():
            return ""

        if "remote_openai_server" in item.fixturenames:
            # (Not actually caching these yet, but can in future)
            return "online"

        if "use_llm_cache" in item.fixturenames:
            return "llm"

        if "test_spyre_cb_scheduler_steps.py" in item.listnames():
            # Not currently cached and needs updating to fixture name
            # CB step tests require a raw engine for scheduler access
            return "engine"

        # Else shouldn't be using any cache
        return ""

    @staticmethod
    def _uses_cb(item) -> bool:
        """True if the test uses continuous batching, false for static batching.
        Checks for the pytest.mark.cb mark."""
        markers = {mark.name for mark in item.own_markers}
        return "cb" in markers

    @staticmethod
    def _get_max_model_len(item) -> int:
        params = item.callspec.params
        if "max_model_len" in params:
            SortKey._assert_param(isinstance(params["max_model_len"], int),
                                  "max_model_len must be an int", item)
            return params["max_model_len"]
        # Put `-1` to indicate that this couldn't be found
        return -1

    @staticmethod
    def _get_max_num_seqs(item) -> int:
        params = item.callspec.params
        if "max_num_seqs" in params:
            SortKey._assert_param(isinstance(params["max_num_seqs"], int),
                                  "max_num_seqs must be an int", item)
            return params["max_num_seqs"]
        # Put `-1` to indicate that this couldn't be found
        return -1

    @staticmethod
    def _get_warmup_shapes(item) -> list[tuple[int, int, int]]:
        key = "warmup_shapes"
        params = item.callspec.params
        if key in params:
            shapes = params[key]
            SortKey._assert_param(isinstance(shapes, list),
                                  "Warmup shape must be a list of tuples",
                                  item)
            SortKey._assert_param(isinstance(shapes[0], tuple),
                                  "Warmup shape must be a list of tuples",
                                  item)
            return params[key]
        # Use -1s to indicate that this couldn't be found
        return [
            (-1, -1, -1),
        ]

    @staticmethod
    def _get_tp_size(item) -> int:
        TP_KEYS = ["tp_size", "tensor_parallel_size", "tp"]
        params = item.callspec.params
        for key in TP_KEYS:
            if key in params:
                SortKey._assert_param(isinstance(params[key], int),
                                      "tp size must be an int", item)
                return params[key]
        # Assume no TP if not set
        return 1

    @staticmethod
    def _get_model(item) -> str:
        MODEL_KEYS = ["model", "model_name"]
        params = item.callspec.params
        for key in MODEL_KEYS:
            if key in params:
                SortKey._assert_param(isinstance(params[key], str),
                                      "model must be a string", item)
                return params[key]
        # No assumption about default model, we likely don't need an llm if this
        # isn't set
        return ""

    @staticmethod
    def _get_backend(item) -> str:
        if "backend" in item.callspec.params:
            backend = item.callspec.params["backend"]
            # if isinstance(backend, tuple) and len(backend) == 1:
            #     backend = backend[0]

            SortKey._assert_param(isinstance(backend, str),
                                  "backend must be a string.", item)
            return backend
        # If backend isn't given then this is likely a spyre-only test
        return "sendnn"

    @staticmethod
    def _get_num_blocks(item) -> int:
        if "available_blocks" in item.callspec.params:
            blocks = item.callspec.params["available_blocks"]
            SortKey._assert_param(isinstance(blocks, int),
                                  "available_blocks must be an int.", item)
            return blocks
        # Most tests don't use this param
        return 0

    @staticmethod
    def _assert_param(condition, message, item):
        assert condition, message + f"\n\n\tTest: {item.listnames()}"\
            f"\n\n\tParams: {item.callspec.params}"


T = TypeVar('T')


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
        assert runtime_config not in self._past_runtime_configs, \
            f"Runtime config {runtime_config} was previously cached for type " \
                f"[{self._type()}], error in test ordering!"
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
            "use_cb": use_cb
        }
        if use_cb:
            runtime_config.update({
                "max_model_len": max_model_len,
                "max_num_seqs": max_num_seqs
            })
        else:
            runtime_config.update({"warmup_shapes": tuple(warmup_shapes)})

        # Always patch the environment so that it's consistent with the LLM
        _patch_environment(use_cb, warmup_shapes, backend, monkeypatch)

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
            ))

    def clear(self) -> None:
        self._cache.clear()


class RemoteOpenAIServer:
    """Subprocess wrapper that boots a vllm server with `vllm serve` for testing
    against"""

    DUMMY_API_KEY = "token-abc123"  # vLLM's OpenAI server does not need API key

    def __init__(self,
                 model: str,
                 vllm_serve_args: list[str],
                 *,
                 env_dict: Optional[dict[str, str]] = None,
                 seed: Optional[int] = 0,
                 auto_port: bool = True,
                 max_wait_seconds: Optional[float] = None) -> None:
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
        self.host = str(args.host or 'localhost')
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
        return openai.AsyncOpenAI(base_url=self.url_for("v1"),
                                  api_key=self.DUMMY_API_KEY,
                                  max_retries=0,
                                  **kwargs)


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
                               env_dict=server_env))

    def clear(self) -> None:
        self._cache.clear()


class EngineCache:
    """Cache for continuous batching engines"""

    def __init__(self):
        self._cache: ModelCache[EngineCore] = ModelCache[EngineCore]()

    def get_engine(self, model: str, max_model_len: int, max_num_seqs: int,
                   available_blocks: int, backend: str,
                   monkeypatch) -> EngineCore:
        runtime_config = {
            "model": model,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "available_blocks": available_blocks,
        }

        # Always patch the environment so that it's consistent with the engine
        _patch_environment(use_cb=True,
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
                       log_stats=False))

    def clear(self) -> None:
        self._cache.clear()


def _patch_environment(use_cb: bool, warmup_shapes: DecodeWarmupShapes | None,
                       backend: str, monkeypatch):
    # Setup the environment correctly for the LLM

    # ---- For static batching ----
    if warmup_shapes:
        assert not use_cb, "Warmup shapes through environment variables have "\
            "been deprecated in continuous batching"

        warmup_prompt_length = [t[0] for t in warmup_shapes]
        warmup_new_tokens = [t[1] for t in warmup_shapes]
        warmup_batch_size = [t[2] for t in warmup_shapes]

        monkeypatch.setenv("VLLM_SPYRE_WARMUP_PROMPT_LENS",
                           ",".join(str(val) for val in warmup_prompt_length))
        monkeypatch.setenv("VLLM_SPYRE_WARMUP_NEW_TOKENS",
                           ",".join(str(val) for val in warmup_new_tokens))
        monkeypatch.setenv("VLLM_SPYRE_WARMUP_BATCH_SIZES",
                           ",".join(str(val) for val in warmup_batch_size))
    # --------------
    monkeypatch.setenv("VLLM_SPYRE_USE_CB", "1" if use_cb else "0")
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)


EmbeddingWarmupShapes = list[tuple[int, int]]

"""Contains utilities for caching models (instantiated as vLLM endpoints) 
across test cases, to speed up test runtime."""

import os
import subprocess
import sys
import time
from typing import NamedTuple, Optional

import openai
import pytest
import requests
from vllm import LLM, EngineArgs
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser, get_open_port
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor


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
    warmup_shapes: list[tuple[int, int, int]] | None = None

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
        """True if the test uses continuous batching, 
        false for static batching"""
        # Check for common param names
        CB_KEYS = ["cb", "use_cb"]
        params = item.callspec.params
        for key in CB_KEYS:
            if key in params:
                return bool(params[key])

        # Otherwise assume that the test uses CB
        return True

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
        WARMUP_KEYS = ["warmup_shape", "warmup_shapes"]
        params = item.callspec.params
        for key in WARMUP_KEYS:
            if key in params:
                shapes = params[key]
                if isinstance(shapes, tuple):
                    shapes = list(shapes)

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


class LLMCache:
    """Caches a vllm.LLM for use in subsequent tests.

    Only a single LLM can be cached at a time, as AIUs don't support loading
    multiple models at once."""

    def __init__(self):
        self._llm: LLM | None = None
        self._runtime_config: dict | None = None
        self._past_runtime_configs: list[dict] = []

        self.hits = 0
        self.misses = 0

    def get_cached_llm(
        self,
        model: str,
        max_model_len: int,
        tensor_parallel_size: int,
        backend: str,
        monkeypatch: pytest.MonkeyPatch,
        warmup_shapes: Optional[list[tuple[int, int, int]]] = None,
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

        if self._runtime_config and self._runtime_config == runtime_config:
            self.hits += 1
            return self._llm

        # cache miss
        self.misses += 1

        print("\n\n\n\n\t\t\tCACHE MISS!\n")
        print(runtime_config)
        print()
        print(self._runtime_config)
        print("\n\n\n\n")

        assert runtime_config not in self._past_runtime_configs, \
            f"Runtime config {runtime_config} was previously cached, error " \
                "in test ordering!"

        self._runtime_config = runtime_config
        self._past_runtime_configs.append(self._runtime_config)

        if self._llm:
            # Tear down the previous LLM before reconstruction
            # TODO: remove or refactor this
            from spyre_util import force_engine_shutdown
            force_engine_shutdown(self._llm)

        self._llm = LLM(
            model=model,
            tokenizer=model,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=tensor_parallel_size,
        )
        return self._llm

    def clear(self) -> None:
        if self._llm:
            from spyre_util import force_engine_shutdown
            force_engine_shutdown(self._llm)
            self._llm = None
            self._runtime_config = None


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
        self._api_server: RemoteOpenAIServer | None = None
        self._runtime_config: dict | None = None
        self._past_runtime_configs: list[dict] = []

        self.hits = 0
        self.misses = 0

    def get_api_server(self, model: str, server_args: list[str],
                       server_env: dict) -> RemoteOpenAIServer:
        """Get or create a new OpenAI server for a given model. and config"""
        runtime_config = {
            "model": model,
            "server_args": tuple(server_args),
            "server_env": server_env,
        }

        if self._runtime_config and self._runtime_config == runtime_config:
            self.hits += 1
            return self._api_server

        # cache miss
        self.misses += 1

        print("\n\n\n\n\t\t\tCACHE MISS!\n")
        print(runtime_config)
        print()
        print(self._runtime_config)
        print("\n\n\n\n")

        assert runtime_config not in self._past_runtime_configs, \
            f"Runtime config {runtime_config} was previously cached, error " \
                "in test ordering!"

        self._runtime_config = runtime_config
        self._past_runtime_configs.append(self._runtime_config)

        # Tear down old server before making a new one
        if self._api_server:
            self._api_server.shutdown()

        # Boot up new server
        self._api_server = RemoteOpenAIServer(model=model,
                                              vllm_serve_args=server_args,
                                              env_dict=server_env)

        return self._api_server

    def clear(self) -> None:
        if self._api_server:
            self._api_server.shutdown()
            self._api_server = None
            self._runtime_config = None


class EngineCache:
    """Cache for continuous batching engines"""

    def __init__(self):
        self._engine: EngineCore | None = None
        self._runtime_config: dict | None = None
        self._past_runtime_configs: list[dict] = []

        self.hits = 0
        self.misses = 0

    def get_engine(self, model: str, max_model_len: int, max_num_seqs: int,
                   available_blocks: int, backend: str,
                   monkeypatch) -> EngineCore:
        runtime_config = {
            "model": model,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "available_blocks": available_blocks,
        }

        # Patch the environment so it always matches the running engine
        _patch_environment(use_cb=True,
                           warmup_shapes=None,
                           backend=backend,
                           monkeypatch=monkeypatch)

        if self._runtime_config and self._runtime_config == runtime_config:
            self.hits += 1
            return self._engine

        # cache miss
        self.misses += 1

        print("\n\n\n\n\t\t\tCACHE MISS!\n")
        print(runtime_config)
        print()
        print(self._runtime_config)
        print("\n\n\n\n")

        assert runtime_config not in self._past_runtime_configs, \
            f"Runtime config {runtime_config} was previously cached, error " \
                "in test ordering!"

        self._runtime_config = runtime_config
        self._past_runtime_configs.append(self._runtime_config)

        # Tear down old engine
        if self._engine:
            self._engine.shutdown()

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
        self._engine = EngineCore(vllm_config=vllm_config,
                                  executor_class=executor_class,
                                  log_stats=False)

        return self._engine

    def clear(self) -> None:
        if self._engine:
            self._engine.shutdown()
            self._engine = None
            self._runtime_config = None


def _patch_environment(use_cb: bool,
                       warmup_shapes: Optional[list[tuple[int, int, int]]],
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

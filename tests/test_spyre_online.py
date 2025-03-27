import os
import subprocess
import sys
import time
from typing import Optional

import openai
import pytest
import requests
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser, get_open_port

from tests.spyre_util import get_spyre_backend_list, get_spyre_model_list


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
        max_wait_seconds = max_wait_seconds or 240
        self._wait_for_server(url=self.url_for("health"),
                              timeout=max_wait_seconds)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
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


@pytest.mark.parametrize("model", get_spyre_model_list())
# (prompt_length/new_tokens/batch_size)
@pytest.mark.parametrize("warmup_shape", [[
    (64, 20, 4),
]])
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize("vllm_version", ["V0", "V1"])
def test_openai_serving(model, warmup_shape, backend, vllm_version):
    """Test online serving using the `vllm serve` CLI"""

    # TODO: util or fixture-ize
    warmup_prompt_length = [t[0] for t in warmup_shape]
    warmup_new_tokens = [t[1] for t in warmup_shape]
    warmup_batch_size = [t[2] for t in warmup_shape]
    v1_flag = "1" if vllm_version == "V1" else "0"
    env_dict = {
        "VLLM_SPYRE_WARMUP_PROMPT_LENS":
        ','.join(str(val) for val in warmup_prompt_length),
        "VLLM_SPYRE_WARMUP_NEW_TOKENS":
        ','.join(str(val) for val in warmup_new_tokens),
        "VLLM_SPYRE_WARMUP_BATCH_SIZES":
        ','.join(str(val) for val in warmup_batch_size),
        "VLLM_SPYRE_DYNAMO_BACKEND":
        backend,
        "VLLM_USE_V1":
        v1_flag
    }

    with RemoteOpenAIServer(model, [], env_dict=env_dict) as server:
        # Run a few simple requests to make sure the server works.
        # This is not checking correctness of replies
        client = server.get_client()
        completion = client.completions.create(model=model,
                                               prompt="Hello World!",
                                               max_tokens=5,
                                               temperature=0.0)
        assert len(completion.choices) == 1
        assert len(completion.choices[0].text) > 0

        completion = client.completions.create(model=model,
                                               prompt="Hello World!",
                                               max_tokens=5,
                                               temperature=1.0,
                                               n=2)
        assert len(completion.choices) == 2
        assert len(completion.choices[0].text) > 0

        # Check some basic error handling as well. This is all done in one test
        # now to avoid server boot-up overhead to test each case.
        # To change this we'll need:
        # - A better way to share a server as a test fixture, or
        # - Much less overhead on server boot (e.g. cached compiled graphs)
        with pytest.raises(openai.APIError):
            # Prompt too long should raise
            long_prompt = "Hello " * 1000
            client.completions.create(model=model,
                                      prompt=long_prompt,
                                      max_tokens=500)

        # Short prompt under context length but requesting too many tokens for
        # the warmup shape should return an empty result
        completion = client.completions.create(model=model,
                                               prompt="Hello World!",
                                               max_tokens=25)

        assert len(completion.choices) == 1
        assert len(completion.choices[0].text) == 0
        assert completion.choices[0].finish_reason == "abort"

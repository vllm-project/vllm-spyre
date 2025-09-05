import inspect
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import openai
import pytest
import requests
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser, get_open_port
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import Request

EmbeddingWarmupShapes = list[tuple[int, int]]
DecodeWarmupShapes = list[tuple[int, int, int]]


def patch_environment(use_cb: bool, warmup_shapes: DecodeWarmupShapes | None,
                      backend: str, monkeypatch):
    # Setup the environment correctly for the LLM

    # ---- For static batching ----
    if warmup_shapes:
        assert not use_cb, ("Warmup shapes through environment variables have "
                            "been deprecated in continuous batching")

        patch_warmup_shapes(warmup_shapes, monkeypatch)

    # --------------
    monkeypatch.setenv("VLLM_SPYRE_USE_CB", "1" if use_cb else "0")
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)


def patch_warmup_shapes(warmup_shapes: DecodeWarmupShapes
                        | EmbeddingWarmupShapes, monkeypatch):
    warmup_prompt_length = [t[0] for t in warmup_shapes]
    warmup_batch_size = [t[-1] for t in warmup_shapes]

    monkeypatch.setenv('VLLM_SPYRE_WARMUP_PROMPT_LENS',
                       ','.join(str(val) for val in warmup_prompt_length))
    monkeypatch.setenv('VLLM_SPYRE_WARMUP_BATCH_SIZES',
                       ','.join(str(val) for val in warmup_batch_size))

    if all(len(s) == 3 for s in warmup_shapes):
        warmup_new_tokens = [t[1] for t in warmup_shapes]
        monkeypatch.setenv('VLLM_SPYRE_WARMUP_NEW_TOKENS',
                           ','.join(str(val) for val in warmup_new_tokens))


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


# get model directory path from env
# if unset, test model paths are assumed to be either hf hub names or absolute
# paths
def get_spyre_model_dir_path() -> Path:
    model_dir_path = os.environ.get("VLLM_SPYRE_TEST_MODEL_DIR", "")
    return Path(model_dir_path)


# get model backends from env or default to all and add pytest markers
def get_spyre_backend_list():
    user_backend_list = os.environ.get("VLLM_SPYRE_TEST_BACKEND_LIST",
                                       "eager,inductor,sendnn")

    backends = []
    for backend in user_backend_list.split(","):
        backend = backend.strip()
        marks = []
        if backend == "eager":
            marks = [pytest.mark.cpu]
        elif backend == "sendnn":
            marks = [pytest.mark.spyre]

        backends.append(pytest.param(backend, marks=marks, id=backend))
    return backends


# get model names from env, if not set then use default models for each type.
# Multiple models can be specified with a comma separated list in
# VLLM_SPYRE_TEST_MODEL_LIST
def get_spyre_model_list(isEmbeddings=False, isScoring=False):
    user_test_model_list = os.environ.get("VLLM_SPYRE_TEST_MODEL_LIST")
    if not user_test_model_list:
        return _default_test_models(isEmbeddings, isScoring)

    # User overridden model list
    spyre_model_dir_path = get_spyre_model_dir_path()
    if isEmbeddings:
        marks = [pytest.mark.embedding]
    elif isScoring:
        marks = [pytest.mark.scoring]
    else:
        marks = [pytest.mark.decoder]

    test_model_list = []
    for model in user_test_model_list.split(","):
        model_path = str(spyre_model_dir_path / model.strip())
        test_model_list.append(
            pytest.param(model_path, marks=marks, id=model.strip()))
    return test_model_list


def _default_test_models(isEmbeddings=False, isScoring=False):
    """Return the default set of test models as pytest parameterizations"""
    if isEmbeddings:
        model = "sentence-transformers/all-roberta-large-v1"
        return [pytest.param(model, marks=[pytest.mark.embedding], id=model)]

    if isScoring:
        model = "cross-encoder/stsb-roberta-large"
        return [pytest.param(model, marks=[pytest.mark.scoring], id=model)]

    # Decoders
    # We run tests for both the full-precision bf16 and fp8-quantized models,
    # but by default the `pytest.mark.quantized` marker is de-selected unless
    # the test command includes `-m quantized`.
    tinygranite = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
    tinygranite_fp8 = "ibm-ai-platform/micro-g3.3-8b-instruct-1b-FP8"
    params = [
        pytest.param(tinygranite, marks=[pytest.mark.decoder], id=tinygranite),
        pytest.param(tinygranite_fp8,
                     marks=[pytest.mark.decoder, pytest.mark.quantized],
                     id=tinygranite_fp8)
    ]
    return params


def create_text_prompt(model: str, min_token_length: int,
                       max_token_length: int) -> str:
    """Create a text prompt for the specified model that will tokenize to within
    the specified token length range."""
    tokenizer = AutoTokenizer.from_pretrained(model)
    pepper = "🌶️"
    pepper_tokens = len(tokenizer.encode(pepper, add_special_tokens=False))

    # Find a good starting number of peppers
    prompt = pepper * (min_token_length // pepper_tokens + 1)

    # And add more until we're over the minimum token length
    while len(tokenizer.encode(prompt)) <= min_token_length:
        prompt += pepper

    # Make sure this prompt is within the specified range
    assert min_token_length < len(tokenizer.encode(prompt)) < max_token_length

    return prompt


def create_seq_prompt(model: str, token_length: int) -> str:
    """Create a repeating sequential number prompt for the specified
    model that will tokenize to exactly the specified token length."""

    tokenizer = AutoTokenizer.from_pretrained(model)

    # 20-token pattern
    pattern = "0 1 2 3 4 5 6 7 8 9 "

    # Repeat to token_length
    repeat_count = (token_length // 20) + 1
    text_prompt = pattern * repeat_count

    # Tokenize and slice
    tokens = tokenizer.encode(text_prompt)[:token_length]

    # Assert exact token length
    assert len(tokens) == token_length, \
        f"Token length mismatch: {len(tokens)} != {token_length}"

    return tokenizer.decode(tokens)


def create_random_request(
    request_id: int,
    num_tokens: int,
    sampling_params: SamplingParams,
    from_model_vocab: bool = False,
    model: Optional[str] = None,
) -> Request | EngineCoreRequest:

    tokenizer = AutoTokenizer.from_pretrained(model)
    if from_model_vocab:
        assert model is not None, "Prompt requested to be generated from " \
        "model's vocabulary: need to provide model."

        valid_token_ids = sorted([
            v for v in tokenizer.vocab.values()
            if v not in tokenizer.all_special_ids
        ])
        prompt_token_ids = random.choices(valid_token_ids, k=num_tokens)
    else:
        # start with existing prompts and tokenize them
        prompts = get_longer_chicken_soup_prompts(1)
        tokenized_prompts = tokenizer(prompts)["input_ids"]
        prompt_token_ids = [p[:num_tokens] for p in tokenized_prompts][0]

        # make sure we get enough tokens from the prompts
        assert (len(prompt_token_ids) == num_tokens
                ), f"need {num_tokens} but got {len(prompt_token_ids)}"

    sig = inspect.signature(EngineCore.add_request)
    inputs_renamed = hasattr(EngineCoreRequest, 'mm_kwargs')
    if sig.parameters["request"].annotation == EngineCoreRequest:
        kwargs = {"mm_kwargs" if inputs_renamed else "mm_inputs": None}
        return EngineCoreRequest(
            request_id=str(request_id),
            prompt_token_ids=prompt_token_ids,
            mm_hashes=None,
            mm_placeholders=None,
            sampling_params=sampling_params,
            eos_token_id=None,
            arrival_time=0,
            lora_request=None,
            data_parallel_rank=None,
            pooling_params=None,
            cache_salt=None,
            **kwargs,
        )
    kwargs = {
        "multi_modal_kwargs" if inputs_renamed else "multi_modal_inputs": None
    }
    return Request(
        request_id=str(request_id),
        prompt_token_ids=prompt_token_ids,
        multi_modal_hashes=None,
        multi_modal_placeholders=None,
        sampling_params=sampling_params,
        eos_token_id=None,
        arrival_time=0,
        lora_request=None,
        pooling_params=None,
        cache_salt=None,
        **kwargs,
    )


def skip_unsupported_tp_size(size: int, backend: str):
    if backend in ["eager", "inductor"]:
        # Spyre cards aren't required for running TP on CPU backends
        # But it's really slow to run tp > 2
        if size > 2:
            pytest.skip("Skipping TP test on CPU with TP size > 2")
        return
    cards = int(os.getenv("AIU_WORLD_SIZE", "0"))
    if cards < size:
        pytest.skip(f"Cannot run TP size {size}: "
                    f"only {cards} cards are available")


def get_chicken_soup_prompts(num_prompts: int) -> list[str]:
    template = (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request. Be polite in your response to the"
        " user.\n\n### Instruction:\n{}\n\n### Response:")

    prompts = [
        template.format("Provide a list of instructions "
                        "for preparing chicken soup."),
        template.format("Provide me a list of things that I can do with my "
                        "new found wealth."),
        template.format(
            "how do I add multiple new columns in m for power query or \
                power bi?"),
        template.format("Convert char to string in Java."),
    ]

    if num_prompts > 4:
        prompts = prompts * (math.ceil(num_prompts / 4))

    return prompts[:num_prompts]


def get_longer_chicken_soup_prompts(num_prompts: int) -> list[str]:
    template = (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request. Be polite in your response to the"
        " user.\n\n### Instruction:\n{}\n\n### Response:")

    prompts = [
        template.format("Provide a list of instructions "
                        "for preparing chicken soup along with "
                        "rice curry to go with it so that "
                        "the flavor is amazing and make sure to follow the "
                        "recipe that my mum used to make during my "
                        "childhood so that I can relive my good "
                        "memories thanks"),
        template.format("Provide me a list of things that I can do with my "
                        "new found wealth which I have obtained through "
                        "nefarious activities including gambling "
                        "and betting on sports thanks"),
        template.format(
            "how do I add multiple new columns in m for power query or \
                power bi? Can you explain that to me like I'm 5 years old "
            "with thorough step by step explanation and covering all edge "
            "cases thanks"),
        template.format(
            "Convert char to string in Java "
            "and write unit tests for the same, making sure they all pass "
            "and we get amazing test coverage along with high level "
            "correctness so that the PR reviewers have an easy time "
            "reviewing the changes thanks"),
    ]

    if num_prompts > 4:
        prompts = prompts * (math.ceil(num_prompts / 4))

    return prompts[:num_prompts]

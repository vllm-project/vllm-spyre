import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import openai
import pytest
import requests
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.v1.engine.core import EngineCore

from vllm_spyre.platform import SpyrePlatform

try:
    # old
    from vllm.utils import FlexibleArgumentParser, get_open_port
except ImportError:
    # new
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.utils.network_utils import get_open_port

from vllm.v1.request import Request

EmbeddingWarmupShapes = list[tuple[int, int]]
DecodeWarmupShapes = list[tuple[int, int, int]]


def patch_environment(
    use_cb: bool,
    warmup_shapes: DecodeWarmupShapes | None,
    backend: str,
    monkeypatch,
    use_chunked_prefill: bool = False,
    max_num_batched_tokens: int | None = None,
):
    # Setup the environment correctly for the LLM

    # ---- For static batching ----
    if warmup_shapes:
        assert not use_cb, (
            "Warmup shapes through environment variables have "
            "been deprecated in continuous batching"
        )

        patch_warmup_shapes(warmup_shapes, monkeypatch)

    # --------------
    monkeypatch.setenv("VLLM_SPYRE_USE_CB", "1" if use_cb else "0")
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    monkeypatch.setenv("VLLM_SPYRE_USE_CHUNKED_PREFILL", "1" if use_chunked_prefill else "0")
    # NB: setting this env var explicitly is needed to set the desired value for
    # the chunk size in the case that granite 8b TP4 is detected
    if max_num_batched_tokens is not None:
        monkeypatch.setenv("VLLM_DT_CHUNK_LEN", str(max_num_batched_tokens))


def patch_warmup_shapes(warmup_shapes: DecodeWarmupShapes | EmbeddingWarmupShapes, monkeypatch):
    warmup_prompt_length = [t[0] for t in warmup_shapes]
    warmup_batch_size = [t[-1] for t in warmup_shapes]

    monkeypatch.setenv(
        "VLLM_SPYRE_WARMUP_PROMPT_LENS", ",".join(str(val) for val in warmup_prompt_length)
    )
    monkeypatch.setenv(
        "VLLM_SPYRE_WARMUP_BATCH_SIZES", ",".join(str(val) for val in warmup_batch_size)
    )

    if all(len(s) == 3 for s in warmup_shapes):
        warmup_new_tokens = [t[1] for t in warmup_shapes]
        monkeypatch.setenv(
            "VLLM_SPYRE_WARMUP_NEW_TOKENS", ",".join(str(val) for val in warmup_new_tokens)
        )


class ModelInfo(NamedTuple):
    name: str
    revision: str | None = None
    is_quantized: bool = False

    def __str__(self):
        return f"ModelInfo({self.name}@{self.revision})"


class RemoteOpenAIServer:
    """Subprocess wrapper that boots a vllm server with `vllm serve` for testing
    against"""

    DUMMY_API_KEY = "token-abc123"  # vLLM's OpenAI server does not need API key

    def __init__(
        self,
        model: str | ModelInfo,
        vllm_serve_args: list[str],
        *,
        env_dict: dict[str, str] | None = None,
        seed: int | None = 0,
        auto_port: bool = True,
        max_wait_seconds: float | None = None,
    ) -> None:
        # NB: This implementation does not ensure that the model is downloaded
        # before booting the server, it should be used with models already
        # cached on disk
        if isinstance(model, ModelInfo):
            if model.revision is not None:
                vllm_serve_args = vllm_serve_args + ["--revision", model.revision]
            model_name = model.name
        else:
            model_name = model

        if auto_port:
            if "-p" in vllm_serve_args or "--port" in vllm_serve_args:
                raise ValueError("You have manually specified the port when `auto_port=True`.")

            # Don't mutate the input args
            vllm_serve_args = vllm_serve_args + ["--port", str(get_open_port())]
        if seed is not None:
            if "--seed" in vllm_serve_args:
                raise ValueError(f"You have manually specified the seed when `seed={seed}`.")

            vllm_serve_args = vllm_serve_args + ["--seed", str(seed)]

        parser = FlexibleArgumentParser(description="vLLM's remote OpenAI server.")
        parser = make_arg_parser(parser)
        args = parser.parse_args(["--model", model_name, *vllm_serve_args])
        self.host = str(args.host or "localhost")
        self.port = int(args.port)

        env = os.environ.copy()
        if env_dict is not None:
            env.update(env_dict)
        self.proc = subprocess.Popen(
            ["vllm", "serve", model_name, *vllm_serve_args],
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        max_wait_seconds = max_wait_seconds or 600
        self._wait_for_server(url=self.url_for("health"), timeout=max_wait_seconds)

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
                    raise RuntimeError("Server failed to start in time.") from None

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


# add pytest markers to supported different backends
def get_spyre_backend_list():
    backend_list = ["eager", "inductor", "sendnn"]

    backends = []
    for backend in backend_list:
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
def get_spyre_model_list(isEmbeddings=False, isScoring=False, full_size_models=False):
    """Returns a list of pytest.params. The values are NamedTuples with a name
    and revision field."""
    user_test_model_list = os.environ.get("VLLM_SPYRE_TEST_MODEL_LIST")
    if not user_test_model_list:
        return _default_test_models(isEmbeddings, isScoring, full_size_models)

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
            pytest.param(ModelInfo(name=model_path), marks=marks, id=model.strip())
        )
    return test_model_list


def _default_test_models(isEmbeddings=False, isScoring=False, full_size_models=False):
    """Return the default set of test models as pytest parameterizations"""
    if isEmbeddings:
        model = ModelInfo(
            name="sentence-transformers/all-roberta-large-v1",
            revision="cf74d8acd4f198de950bf004b262e6accfed5d2c",
        )
        return [pytest.param(model, marks=[pytest.mark.embedding], id=model.name)]

    if isScoring:
        model = ModelInfo(
            name="cross-encoder/stsb-roberta-large",
            revision="2b12c2c0088918e76151fd5937b7bba986ef1f98",
        )
        return [pytest.param(model, marks=[pytest.mark.scoring], id=model.name)]

    # Decoders
    # We run tests for both the full-precision bf16 and fp8-quantized models,
    # but by default the `pytest.mark.quantized` marker is de-selected unless
    # the test command includes `-m quantized`.
    if not full_size_models:
        tinygranite = ModelInfo(
            name="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
            revision="6e9c6465a9d7e5e9fa35004a29f0c90befa7d23f",
        )
        tinygranite_fp8 = ModelInfo(
            name="ibm-ai-platform/micro-g3.3-8b-instruct-1b-FP8",
            revision="0dff8bacb968836dbbc7c2895c6d9ead0a05dc9e",
            is_quantized=True,
        )
        params = [
            pytest.param(tinygranite, marks=[pytest.mark.decoder], id=tinygranite.name),
            pytest.param(
                tinygranite_fp8,
                marks=[pytest.mark.decoder, pytest.mark.quantized],
                id=tinygranite_fp8.name,
            ),
        ]
        return params

    # Full-size decoders
    granite = ModelInfo(
        name="ibm-granite/granite-3.3-8b-instruct",
        revision="51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
    )
    granite_fp8 = ModelInfo(
        name="ibm-granite/granite-3.3-8b-instruct-FP8",
        revision="4b5990b8d402a75febe0086abbf1e490af494e3d",
    )
    params = [
        pytest.param(granite, marks=[pytest.mark.decoder], id=granite.name),
        pytest.param(
            granite_fp8, marks=[pytest.mark.decoder, pytest.mark.quantized], id=granite_fp8.name
        ),
    ]
    return params


def create_text_prompt(model: ModelInfo, min_token_length: int, max_token_length: int) -> str:
    """Create a text prompt for the specified model that will tokenize to within
    the specified token length range."""
    tokenizer = AutoTokenizer.from_pretrained(model.name, revision=model.revision)
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


def create_seq_prompt(model: ModelInfo, token_length: int) -> str:
    """Create a repeating sequential number prompt for the specified
    model that will tokenize to exactly the specified token length."""

    tokenizer = AutoTokenizer.from_pretrained(model.name, revision=model.revision)

    # 20-token pattern
    pattern = "0 1 2 3 4 5 6 7 8 9 "

    # Repeat to token_length
    repeat_count = (token_length // 20) + 1
    text_prompt = pattern * repeat_count

    # Tokenize and slice
    tokens = tokenizer.encode(text_prompt)[:token_length]

    # Assert exact token length
    assert len(tokens) == token_length, f"Token length mismatch: {len(tokens)} != {token_length}"

    return tokenizer.decode(tokens)


def create_random_request(
    request_id: int,
    num_tokens: int,
    sampling_params: SamplingParams,
    from_model_vocab: bool = False,
    model: ModelInfo | None = None,
    seed: int = None,
) -> Request:
    tokenizer = AutoTokenizer.from_pretrained(model.name, revision=model.revision)
    if from_model_vocab:
        assert model is not None, (
            "Prompt requested to be generated from model's vocabulary: need to provide model."
        )

        valid_token_ids = sorted(
            [v for v in tokenizer.vocab.values() if v not in tokenizer.all_special_ids]
        )
        if seed is not None:
            random.seed(seed)
        prompt_token_ids = random.choices(valid_token_ids, k=num_tokens)

    else:
        # start with existing prompts and tokenize them
        prompts = get_longer_chicken_soup_prompts(1)
        tokenized_prompts = tokenizer(prompts)["input_ids"]
        prompt_token_ids = [p[:num_tokens] for p in tokenized_prompts][0]

        # make sure we get enough tokens from the prompts
        assert len(prompt_token_ids) == num_tokens, (
            f"need {num_tokens} but got {len(prompt_token_ids)}"
        )

    return Request(
        request_id=str(request_id),
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        eos_token_id=None,
        arrival_time=0,
        lora_request=None,
        pooling_params=None,
        cache_salt=None,
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
        pytest.skip(f"Cannot run TP size {size}: only {cards} cards are available")


def get_chicken_soup_prompts(num_prompts: int) -> list[str]:
    template = (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request. Be polite in your response to the"
        " user.\n\n### Instruction:\n{}\n\n### Response:"
    )

    prompts = [
        template.format("Provide a list of instructions for preparing chicken soup."),
        template.format("Provide me a list of things that I can do with my new found wealth."),
        template.format(
            "how do I add multiple new columns in m for power query or \
                power bi?"
        ),
        template.format("Convert char to string in Java."),
    ]

    if num_prompts > 4:
        prompts = prompts * (math.ceil(num_prompts / 4))

    return prompts[:num_prompts]


def get_longer_chicken_soup_prompts(num_prompts: int) -> list[str]:
    template = (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request. Be polite in your response to the"
        " user.\n\n### Instruction:\n{}\n\n### Response:"
    )

    prompts = [
        template.format(
            "Provide a list of instructions "
            "for preparing chicken soup along with "
            "rice curry to go with it so that "
            "the flavor is amazing and make sure to follow the "
            "recipe that my mum used to make during my "
            "childhood so that I can relive my good "
            "memories thanks"
        ),
        template.format(
            "Provide me a list of things that I can do with my "
            "new found wealth which I have obtained through "
            "nefarious activities including gambling "
            "and betting on sports thanks"
        ),
        template.format(
            "how do I add multiple new columns in m for power query or \
                power bi? Can you explain that to me like I'm 5 years old "
            "with thorough step by step explanation and covering all edge "
            "cases thanks"
        ),
        template.format(
            "Convert char to string in Java "
            "and write unit tests for the same, making sure they all pass "
            "and we get amazing test coverage along with high level "
            "correctness so that the PR reviewers have an easy time "
            "reviewing the changes thanks"
        ),
    ]

    if num_prompts > 4:
        prompts = prompts * (math.ceil(num_prompts / 4))

    return prompts[:num_prompts]


def write_sample_model_config(tmp_path, data, filename="model_compile.log.json"):
    """Helper to write a sample model_compile.log.json in tmp_path."""
    config_path = tmp_path / filename
    config_path.write_text(json.dumps(data))
    return config_path


def get_block_tables(engine_core: EngineCore) -> tuple[dict[str, list[int]], dict[int, int]]:
    model_runner = engine_core.model_executor.driver_worker.worker.model_runner
    req_to_blocks = model_runner.kv_cache_manager.req_to_blocks

    block_tables = {
        req_id: [block.block_id for block in blocks] for req_id, blocks in req_to_blocks.items()
    }

    block_ref_count = {}

    for blocks in req_to_blocks.values():
        for block in blocks:
            block_ref_count[block.block_id] = block.ref_cnt

    return block_tables, block_ref_count


def verify_block_tables(engine_core: EngineCore, step_ref: dict[str, Any], disable_asserts: bool):
    block_tables, block_ref_count = get_block_tables(engine_core)

    if not disable_asserts:
        if "block_tables" in step_ref:
            assert step_ref["block_tables"] == block_tables, (
                f"Reference table {step_ref['block_tables']}, Actual table: {block_tables}"
            )

        if "block_ref_count" in step_ref:
            assert step_ref["block_ref_count"] == block_ref_count
    else:
        print(f"{block_tables=}")
        print(f"{block_ref_count=}")


def verify_slot_mappings(engine_core: EngineCore, step_ref: dict[str, Any], disable_asserts: bool):
    if "prefill_slot_mappings" not in step_ref:
        return
    prefill_slot_mappings = step_ref["prefill_slot_mappings"]

    # Slot mappings should be provided at "block level", convert here to token level.
    # e.g. [0, 1] expands to [0 ... 128]
    slot_mapping_tensor_list = []

    block_size = SpyrePlatform.get_block_size()

    # TODO: Need to map the keys (request ids) to correct order in tensor
    # (But we only ever prefill one request right now)
    for slot_mapping in prefill_slot_mappings.values():
        slot_mapping_tensor = torch.arange(block_size, dtype=torch.int64).repeat(len(slot_mapping))
        slot_mapping_tensor += (
            torch.tensor(slot_mapping, dtype=torch.int64)
            .repeat_interleave(block_size)
            .mul_(block_size)
        )
        slot_mapping_tensor_list.append(slot_mapping_tensor)  # TODO: slice?
    reference_slot_mapping = torch.stack(slot_mapping_tensor_list)

    actual_slot_mapping = engine_core.forward_context.attn_metadata.slot_mapping

    if not disable_asserts:
        if "block_tables" in step_ref:
            assert torch.equal(reference_slot_mapping, actual_slot_mapping), (
                f"Reference slot mapping {reference_slot_mapping}, "
                f"Actual slot mapping: {actual_slot_mapping}"
            )
    else:
        print(f"{actual_slot_mapping=}")
        print(f"{reference_slot_mapping=}")

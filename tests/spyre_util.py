import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import openai
import pytest
import requests
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser, get_open_port

DISABLE_ASSERTS = False  # used for debugging

ISCLOSE_REL_TOL_CPU = 0.1
ISCLOSE_REL_TOL_SPYRE = 0.35

VLLM_VERSIONS = [
    pytest.param("V0", marks=pytest.mark.v0, id="v0"),
    pytest.param("V1", marks=pytest.mark.v1, id="v1"),
]


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


def patch_warmup_shapes(warmup_shapes: list[tuple[int, int, int]],
                        monkeypatch):
    warmup_prompt_length = [t[0] for t in warmup_shapes]
    warmup_new_tokens = [t[1] for t in warmup_shapes]
    warmup_batch_size = [t[2] for t in warmup_shapes]

    monkeypatch.setenv('VLLM_SPYRE_WARMUP_PROMPT_LENS',
                       ','.join(str(val) for val in warmup_prompt_length))
    monkeypatch.setenv('VLLM_SPYRE_WARMUP_NEW_TOKENS',
                       ','.join(str(val) for val in warmup_new_tokens))
    monkeypatch.setenv('VLLM_SPYRE_WARMUP_BATCH_SIZES',
                       ','.join(str(val) for val in warmup_batch_size))


# vLLM / Spyre
def generate_spyre_vllm_output(model: str, prompts: list[str],
                               warmup_shapes: list[tuple[int, int, int]],
                               max_model_len: int, block_size: int,
                               sampling_params: Union[SamplingParams,
                                                      list[SamplingParams]],
                               tensor_parallel_size: int, backend: str,
                               vllm_version: str) -> list[dict[str, Any]]:

    warmup_prompt_length = [t[0] for t in warmup_shapes]
    warmup_new_tokens = [t[1] for t in warmup_shapes]
    warmup_batch_size = [t[2] for t in warmup_shapes]

    os.environ['VLLM_SPYRE_WARMUP_PROMPT_LENS'] = ','.join(
        str(val) for val in warmup_prompt_length)
    os.environ['VLLM_SPYRE_WARMUP_NEW_TOKENS'] = ','.join(
        str(val) for val in warmup_new_tokens)
    os.environ['VLLM_SPYRE_WARMUP_BATCH_SIZES'] = ','.join(
        str(val) for val in warmup_batch_size)
    os.environ['VLLM_SPYRE_DYNAMO_BACKEND'] = backend
    os.environ['VLLM_USE_V1'] = "1" if vllm_version == "V1" else "0"

    vllm_model = LLM(model=model,
                     tokenizer=model,
                     max_model_len=max_model_len,
                     block_size=block_size,
                     tensor_parallel_size=tensor_parallel_size)

    vllm_outputs = vllm_model.generate(prompts, sampling_params)

    results = []

    for req_output in vllm_outputs:
        result = {}
        result['text'] = req_output.outputs[0].text
        # TODO: Workaround for V1, if request does not fit in a warmup shape
        # token_ids may be filled with -1.
        token_ids = [t for t in req_output.outputs[0].token_ids if t >= 0]
        result['token_ids'] = tuple(token_ids)
        result['tokens'] = tuple([
            req_output.outputs[0].logprobs[i][t].decoded_token
            for i, t in enumerate(result['token_ids'])
        ])
        result['logprobs'] = tuple([
            req_output.outputs[0].logprobs[i][t].logprob
            for i, t in enumerate(result['token_ids'])
        ])
        results.append(result)

    return results


# Support for continuous batching
def generate_cb_spyre_vllm_output(
    model: str,
    prompts: list[str],
    max_model_len: int,
    block_size: int,
    sampling_params: Union[SamplingParams, list[SamplingParams]],
    tensor_parallel_size: int,
    backend: str,
    max_num_seqs: int,
    use_cb: int,
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    with monkeypatch.context() as m:

        m.setenv("VLLM_SPYRE_USE_CB", str(use_cb))
        m.setenv("VLLM_USE_V1", "1")
        m.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)

        vllm_model = LLM(
            model=model,
            tokenizer=model,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            block_size=block_size,
            tensor_parallel_size=tensor_parallel_size,
        )

        vllm_outputs = vllm_model.generate(prompts, sampling_params)
        results = []

        for req_output in vllm_outputs:
            result = {}
            result["text"] = req_output.outputs[0].text
            results.append(result)

        return results


# Hugging Face
def generate_hf_output(
        model: str, prompts: list[str],
        max_new_tokens: Union[int, list[int]]) -> list[dict[str, Any]]:

    if not isinstance(max_new_tokens, list):
        max_new_tokens = [max_new_tokens] * len(prompts)

    hf_model = AutoModelForCausalLM.from_pretrained(model)
    hf_tokenizer = AutoTokenizer.from_pretrained(model)

    results = []
    for prompt_index, prompt in enumerate(prompts):
        hf_input_tokens = hf_tokenizer(prompt, return_tensors="pt").input_ids
        hf_output = hf_model.generate(
            hf_input_tokens,
            do_sample=False,
            max_new_tokens=max_new_tokens[prompt_index],
            return_dict_in_generate=True,
            output_scores=True)

        # decode output tokens after first removing input tokens (prompt)
        hf_generated_text = hf_tokenizer.batch_decode(
            hf_output.sequences[:, len(hf_input_tokens[0]):])[0]
        hf_transition_scores = hf_model.compute_transition_scores(
            hf_output.sequences, hf_output.scores, normalize_logits=True)

        # return HF generated text, tokens, token ids and logprobs
        result = {}
        result['text'] = hf_generated_text
        result['token_ids'] = []
        result['tokens'] = []
        result['logprobs'] = []
        for tok_index, hf_logprob in enumerate(hf_transition_scores[0]):
            hf_token_id = hf_output.sequences[0][tok_index +
                                                 len(hf_input_tokens[0])]
            result['token_ids'].append(hf_token_id.item())
            result['tokens'].append(hf_tokenizer.decode(hf_token_id))
            result['logprobs'].append(hf_logprob.item())
        result['token_ids'] = tuple(result['token_ids'])
        result['tokens'] = tuple(result['tokens'])
        result['logprobs'] = tuple(result['logprobs'])
        results.append(result)

    return results


# compare results
def compare_results(model: str, prompts: list[str],
                    warmup_shapes: list[tuple[int, int,
                                              int]], tensor_parallel_size: int,
                    backend: str, vllm_results: list[dict[str, Any]],
                    hf_results: list[dict[str, Any]]):

    print(f"\nmodel:         {model:s}")
    print(f"warmup shapes: {warmup_shapes}")
    print(f"tp size:       {tensor_parallel_size}")
    print(f"backend:       {backend:s}")
    print(f"\n#prompts:      {len(prompts):d}")
    print(f"#HF results:   {len(hf_results):d}"
          f"{'' if len(hf_results) == len(prompts) else '  ERROR':s}")
    print(f"#vLLM results: {len(vllm_results):d}"
          f"{'' if len(vllm_results) == len(prompts) else '  ERROR':s}")
    print()

    assert DISABLE_ASSERTS or len(hf_results) == len(vllm_results)
    assert DISABLE_ASSERTS or len(hf_results) == len(prompts)

    for prompt_index, (prompt, hf_result, vllm_result) in enumerate(
            zip(prompts, hf_results, vllm_results)):
        err_msg = '' if hf_result['text'] == vllm_result['text'] else '  ERROR'
        print(f"\nprompt {prompt_index:3d}:    {repr(prompt):s}")
        print("generated:")
        print(f"        HF:    {repr(hf_result['text']):s}")
        print(f"        vLLM:  {repr(vllm_result['text']):s}{err_msg}")
        print()

        assert DISABLE_ASSERTS or backend == 'sendnn_decoder' or\
            hf_result['token_ids'] == vllm_result['token_ids']

        if len(hf_result['tokens']) > 0:
            print("   token id. token               logprob      "
                  "   token id. token               logprob")

            logprob_abs_diff_list = []
            logprob_rel_diff_list = []

            for hf_token, hf_token_id, hf_logprob, vllm_token,\
                 vllm_token_id, vllm_logprob in zip(
                    hf_result['tokens'], hf_result['token_ids'],
                    hf_result['logprobs'], vllm_result['tokens'],
                    vllm_result['token_ids'], vllm_result['logprobs']):
                logprob_abs_diff = math.fabs(hf_logprob - vllm_logprob)
                logprob_abs_diff_list.append(logprob_abs_diff)
                logprob_rel_diff = math.fabs(logprob_abs_diff / hf_logprob)
                logprob_rel_diff_list.append(logprob_rel_diff)

                print(
                    f"HF: {hf_token_id:8d} {repr(hf_token):14s} "
                    f"{hf_logprob:14f}  "
                    f"vLLM: {vllm_token_id:8d} {repr(vllm_token):14s} "
                    f"{vllm_logprob:14f}  ",
                    end='')

                if backend == 'sendnn_decoder':
                    rel_tol = ISCLOSE_REL_TOL_SPYRE
                else:
                    rel_tol = ISCLOSE_REL_TOL_CPU

                if hf_token_id != vllm_token_id:  # different tokens
                    if backend == 'sendnn_decoder' and math.isclose(
                            hf_logprob, vllm_logprob, rel_tol=rel_tol):
                        # probably still OK
                        print('DIVERGING')
                        break
                    else:
                        print('ERROR')
                        assert DISABLE_ASSERTS or False
                        break
                else:  # identical tokens
                    if math.isclose(hf_logprob, vllm_logprob, rel_tol=rel_tol):
                        print()
                    else:
                        print('ERROR')
                        assert DISABLE_ASSERTS or False
                        break

            print()
            print("logprob absolute differences: "
                  f"average={np.mean(logprob_abs_diff_list):f}  "
                  f"maximum={np.max(logprob_abs_diff_list):f}")
            print("logprob relative differences: "
                  f"average={np.mean(logprob_rel_diff_list):f}  "
                  f"maximum={np.max(logprob_rel_diff_list):f}")

        print()


# vLLM / Spyre
def spyre_vllm_embeddings(model: str, prompts: list[str],
                          warmup_shapes: list[tuple[int, int]],
                          max_model_len: int, block_size: int,
                          tensor_parallel_size: int, backend: str,
                          vllm_version: str) -> list[dict[str, Any]]:

    warmup_prompt_length = [t[0] for t in warmup_shapes]
    warmup_new_tokens = [0] * len(warmup_shapes)
    warmup_batch_size = [t[1] for t in warmup_shapes]

    os.environ['VLLM_SPYRE_WARMUP_PROMPT_LENS'] = ','.join(
        str(val) for val in warmup_prompt_length)
    os.environ['VLLM_SPYRE_WARMUP_NEW_TOKENS'] = ','.join(
        str(val) for val in warmup_new_tokens)
    os.environ['VLLM_SPYRE_WARMUP_BATCH_SIZES'] = ','.join(
        str(val) for val in warmup_batch_size)
    os.environ['VLLM_SPYRE_DYNAMO_BACKEND'] = backend
    os.environ['VLLM_USE_V1'] = "1" if vllm_version == "V1" else "0"

    vllm_model = LLM(model=model,
                     tokenizer=model,
                     max_model_len=max_model_len,
                     block_size=block_size,
                     tensor_parallel_size=tensor_parallel_size)

    vllm_outputs = vllm_model.embed(prompts)

    results = []
    for req_output in vllm_outputs:
        result = {}
        result["embeddings"] = req_output.outputs.embedding
        results.append(result)

    return results


# Hugging Face
def st_embeddings(model: str, prompts: list[str]) -> list[dict[str, Any]]:

    model = SentenceTransformer(model)

    results = []
    for prompt in prompts:
        embeddings = model.encode(prompt)

        # return ST generated embeddings
        result = {}
        result['embeddings'] = embeddings
        results.append(result)

    return results


# compare results
def compare_embedding_results(model: str, prompts: list[str],
                              warmup_shapes: list[tuple[int, int]],
                              tensor_parallel_size: int, backend: str,
                              vllm_results: list[dict[str, Any]],
                              hf_results: list[dict[str, Any]]):

    print(f"\nmodel:         {model:s}")
    print(f"warmup shapes: {warmup_shapes}")
    print(f"tp size:       {tensor_parallel_size}")
    print(f"backend:       {backend:s}")
    print(f"\n#prompts:      {len(prompts):d}")
    print(f"#HF results:   {len(hf_results):d}"
          f"{'' if len(hf_results) == len(prompts) else '  ERROR':s}")
    print(f"#vLLM results: {len(vllm_results):d}"
          f"{'' if len(vllm_results) == len(prompts) else '  ERROR':s}")
    print()

    assert DISABLE_ASSERTS or len(hf_results) == len(vllm_results)
    assert DISABLE_ASSERTS or len(hf_results) == len(prompts)

    for hf_result, vllm_result in zip(hf_results, vllm_results):

        sim = util.pytorch_cos_sim(hf_result["embeddings"],
                                   vllm_result["embeddings"])

        assert math.isclose(sim, 1.0, rel_tol=0.05)


# get model directory path from env, if not set then default to "models".
def get_spyre_model_dir_path() -> Path:
    model_dir_path = os.environ.get("VLLM_SPYRE_TEST_MODEL_DIR", "models")
    return Path(model_dir_path)


# get model backends from env or default to all and add pytest markers
def get_spyre_backend_list():
    user_backend_list = os.environ.get("VLLM_SPYRE_TEST_BACKEND_LIST",
                                       "eager,inductor,sendnn_decoder,sendnn")

    backends = []
    for backend in user_backend_list.split(","):
        backend = backend.strip()
        marks = []
        if backend == "eager":
            marks = [pytest.mark.cpu]
        elif backend == "sendnn_decoder":
            marks = [pytest.mark.spyre]

        backends.append(pytest.param(backend, marks=marks, id=backend))
    return backends


# get model names from env, if not set then default to "llama-194m"
# For multiple values:
# export SPYRE_TEST_MODEL_LIST="llama-194m,all-roberta-large-v1"
def get_spyre_model_list(isEmbeddings=False, quantization=None):
    spyre_model_dir_path = get_spyre_model_dir_path()

    if isEmbeddings:
        user_test_model_list = os.environ.get("VLLM_SPYRE_TEST_MODEL_LIST",
                                              "all-roberta-large-v1")
        marks = [pytest.mark.embedding]
    elif quantization == "gptq":
        user_test_model_list = os.environ.get("VLLM_SPYRE_TEST_MODEL_LIST",
                                              "granite-3.0-8b-instruct-gptq")
        marks = [pytest.mark.decoder, pytest.mark.quantized, pytest.mark.spyre]
    else:
        user_test_model_list = os.environ.get("VLLM_SPYRE_TEST_MODEL_LIST",
                                              "llama-194m")
        marks = [pytest.mark.decoder]

    test_model_list = []
    for model in user_test_model_list.split(","):
        model_path = str(spyre_model_dir_path / model.strip())
        test_model_list.append(
            pytest.param(model_path, marks=marks, id=model.strip()))
    return test_model_list


def create_text_prompt(model: str, min_tokens: int, max_tokens: int) -> str:
    """Create a text prompt for the specified model that will tokenize to within
    the specified token length range."""
    tokenizer = AutoTokenizer.from_pretrained(model)
    pepper = "🌶️"
    pepper_tokens = len(tokenizer.encode(pepper, add_special_tokens=False))

    # Find a good starting number of peppers
    prompt = pepper * (min_tokens // pepper_tokens + 1)

    # And add more until we're over the minimum token length
    while len(tokenizer.encode(prompt)) < min_tokens:
        prompt += pepper

    # Make sure this prompt is within the specified range
    assert min_tokens < len(tokenizer.encode(prompt)) < max_tokens

    return prompt

"""Contains utilities for generating and comparing output and
returning the results"""

import math
import os
from pathlib import Path
from typing import Any, Union

import numpy as np
import pytest
import torch
from hf_result_cache import HFResultCache
from llm_cache import LLM_CACHE, get_cached_llm
from sentence_transformers import SentenceTransformer, util
from spyre_util import DecodeWarmupShapes, EmbeddingWarmupShapes, ModelInfo
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer

DISABLE_ASSERTS = False  # used for debugging

ISCLOSE_ABS_TOL = float(os.environ.get("VLLM_SPYRE_TEST_ABS_TOL", "0.08"))
ISCLOSE_ABS_TOL_QUANTIZATION = float(os.environ.get("VLLM_SPYRE_TEST_QUANTIZED_ABS_TOL", "0.125"))

HF_RESULT_CACHE = HFResultCache()


# Hugging Face
def generate_hf_output(
    model: str | ModelInfo,
    prompts: Union[list[str], list[list[int]]],  # also accept token ids
    max_new_tokens: Union[int, list[int]],
    ignore_eos: bool = False,
    include_prompt: bool = False,
) -> list[dict[str, Any]]:
    """Loads and runs the model on cpu with transformers, caching the results.
    Returns cached results if any are found to avoid overhead."""

    if isinstance(model, ModelInfo):
        revision = model.revision
        model_name = model.name
    else:
        revision = None
        model_name = model

    if not isinstance(max_new_tokens, list):
        max_new_tokens = [max_new_tokens] * len(prompts)

    results = []
    for prompt, max_tokens in zip(prompts, max_new_tokens):
        results.append(HF_RESULT_CACHE.get_cached_result(model, prompt, max_tokens))

    if all(results):
        # Everything hit cache
        return results

    assert os.getenv("GITHUB_ACTIONS", "") != "true", (
        "HF results cache miss during Github Actions run. "
        "Please run tests locally with `-m 'cpu'` and check in the changes "
        "to hf_cache.json"
    )

    hf_model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if ignore_eos:
        hf_model.generation_config.eos_token_id = None

    for prompt_index, prompt in enumerate(prompts):
        if results[prompt_index]:
            # Already have cached result
            continue

        hf_input_tokens = (
            hf_tokenizer(prompt, return_tensors="pt").input_ids
            if isinstance(prompt[0], str)
            else torch.tensor([prompts[prompt_index]])
        )
        hf_output = hf_model.generate(
            hf_input_tokens,
            do_sample=False,
            max_new_tokens=max_new_tokens[prompt_index],
            return_dict_in_generate=True,
            output_scores=True,
        )

        # decode output tokens after first removing input tokens (prompt)
        hf_generated_text = hf_tokenizer.batch_decode(
            hf_output.sequences[:, len(hf_input_tokens[0]) :]
        )[0]
        hf_transition_scores = hf_model.compute_transition_scores(
            hf_output.sequences, hf_output.scores, normalize_logits=True
        )

        # return HF generated text, tokens, token ids and logprobs
        result = {}
        result["text"] = hf_generated_text
        result["token_ids"] = []
        result["tokens"] = []
        result["logprobs"] = []
        for tok_index, hf_logprob in enumerate(hf_transition_scores[0]):
            hf_token_id = hf_output.sequences[0][tok_index + len(hf_input_tokens[0])]
            result["token_ids"].append(hf_token_id.item())
            result["tokens"].append(hf_tokenizer.decode(hf_token_id))
            result["logprobs"].append(hf_logprob.item())
        result["token_ids"] = tuple(result["token_ids"])
        result["tokens"] = tuple(result["tokens"])
        result["logprobs"] = tuple(result["logprobs"])
        if include_prompt:
            result["prompt"] = prompt

        # Save and cache new result
        results[prompt_index] = result
        HF_RESULT_CACHE.add_to_cache(model, prompt, max_new_tokens[prompt_index], result)

    # Write back to the cache
    HF_RESULT_CACHE.write_cache()
    return results


# compare results
def compare_results(
    model: str | ModelInfo,
    tensor_parallel_size: int,
    backend: str,
    vllm_results: list[dict[str, Any]],
    hf_results: list[dict[str, Any]],
    prompts: list[str] | None = None,
):
    revision = None
    if isinstance(model, ModelInfo):
        revision = model.revision
        model = model.name

    if prompts is None:
        prompts = [""] * len(vllm_results)

    tokenizer = None
    # Decode the prompts if needed
    for idx in range(len(prompts)):
        prompt = prompts[idx]
        if not all(isinstance(t, int) for t in prompt):
            continue
        tokenizer = get_tokenizer(model, revision=revision) if tokenizer is None else tokenizer
        prompts[idx] = tokenizer.decode(prompt)

    print(f"\nmodel:         {model:s}")
    print(f"tp size:       {tensor_parallel_size}")
    print(f"backend:       {backend:s}")
    print(f"\n#prompts:      {len(prompts):d}")
    print(
        f"#HF results:   {len(hf_results):d}"
        f"{'' if len(hf_results) == len(prompts) else '  ERROR':s}"
    )
    print(
        f"#vLLM results: {len(vllm_results):d}"
        f"{'' if len(vllm_results) == len(prompts) else '  ERROR':s}"
    )
    print()

    assert DISABLE_ASSERTS or len(hf_results) == len(vllm_results)
    assert DISABLE_ASSERTS or len(hf_results) == len(prompts)

    for prompt_index, (prompt, hf_result, vllm_result) in enumerate(
        zip(prompts, hf_results, vllm_results)
    ):
        if "text" in vllm_result:
            err_msg = "" if hf_result["text"] == vllm_result["text"] else "  ERROR"
            print(f"\nprompt {prompt_index:3d}:    {repr(prompt):s}")
            print("generated:")
            print(f"        HF:    {repr(hf_result['text']):s}")
            print(f"        vLLM:  {repr(vllm_result['text']):s}{err_msg}")
            print()

        if isinstance(hf_result["token_ids"], list):
            hf_result["token_ids"] = tuple(hf_result["token_ids"])

        if len(hf_result["tokens"]) > 0:
            print(
                "   token id. token               logprob      "
                "   token id. token               logprob"
            )

            logprob_abs_diff_list = []
            logprob_rel_diff_list = []

            for i, (hf_token_id, hf_logprob, vllm_token_id, vllm_logprob) in enumerate(
                zip(
                    hf_result["token_ids"],
                    hf_result["logprobs"],
                    vllm_result["token_ids"],
                    vllm_result["logprobs"],
                )
            ):
                logprob_abs_diff = math.fabs(hf_logprob - vllm_logprob)
                logprob_abs_diff_list.append(logprob_abs_diff)
                logprob_rel_diff = math.fabs(
                    logprob_abs_diff / max(math.fabs(hf_logprob), math.fabs(vllm_logprob))
                )
                logprob_rel_diff_list.append(logprob_rel_diff)

                hf_token = repr(hf_result["tokens"][i]) if "tokens" in vllm_result else "-"
                vllm_token = repr(vllm_result["tokens"][i]) if "tokens" in vllm_result else "-"
                print(
                    f"HF: {hf_token_id:8d} {hf_token:14s} {hf_logprob:14f}  "
                    f"vLLM: {vllm_token_id:8d} {vllm_token:14s} "
                    f"{vllm_logprob:14f}  ",
                    end="",
                )

                if "FP8" in model:
                    # TODO: Improve this. For now our testing model can be
                    # solved with this logic
                    abs_tol = ISCLOSE_ABS_TOL_QUANTIZATION
                else:
                    abs_tol = ISCLOSE_ABS_TOL

                hf_token_prob = math.exp(hf_logprob)
                vllm_token_prob = math.exp(vllm_logprob)

                if hf_token_id != vllm_token_id:  # different tokens
                    if backend == "sendnn" and math.isclose(
                        hf_token_prob,
                        vllm_token_prob,
                        abs_tol=abs_tol,
                    ):
                        # probably still OK
                        print("DIVERGING")
                        break
                    else:
                        print("ERROR")
                        assert DISABLE_ASSERTS or False
                        break
                else:  # identical tokens
                    if math.isclose(
                        hf_token_prob,
                        vllm_token_prob,
                        abs_tol=abs_tol,
                    ):
                        print()
                    else:
                        prob_diff = abs(hf_token_prob - vllm_token_prob)
                        print(f"ERROR (prob_diff = {prob_diff * 100:.2f}%)")
                        assert DISABLE_ASSERTS or False
                        break

            print()
            print(
                "logprob absolute differences: "
                f"average={np.mean(logprob_abs_diff_list):f}  "
                f"maximum={np.max(logprob_abs_diff_list):f}"
            )
            print(
                "logprob relative differences: "
                f"average={np.mean(logprob_rel_diff_list):f}  "
                f"maximum={np.max(logprob_rel_diff_list):f}"
            )

        if hf_result["token_ids"] != vllm_result["token_ids"]:
            print(hf_result["token_ids"])
            print(vllm_result["token_ids"])
        assert (
            DISABLE_ASSERTS
            or backend == "sendnn"
            or hf_result["token_ids"] == vllm_result["token_ids"]
        ), f"Token ids differ: {hf_result['token_ids']} != {vllm_result['token_ids']}"

        print()


def check_output_against_hf(
    model: str | ModelInfo, backend, max_new_tokens, vllm_results, prompts
) -> None:
    hf_outputs = generate_hf_output(
        model=model,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        ignore_eos=True,
    )
    compare_results(
        model=model,
        tensor_parallel_size=1,
        backend=backend,
        vllm_results=vllm_results,
        hf_results=hf_outputs,
        prompts=prompts,
    )


# Hugging Face
def st_embeddings(model: str | ModelInfo, prompts: list[str]) -> list[dict[str, Any]]:
    if isinstance(model, ModelInfo):
        model = SentenceTransformer(model.name, revision=model.revision)
    else:
        model = SentenceTransformer(model)

    results = []
    for prompt in prompts:
        embeddings = model.encode(prompt)

        # return ST generated embeddings
        result = {}
        result["embeddings"] = embeddings
        results.append(result)

    return results


# compare results
def compare_embedding_results(
    model: str | ModelInfo,
    prompts: list[str],
    warmup_shapes: EmbeddingWarmupShapes,
    tensor_parallel_size: int,
    backend: str,
    vllm_results: list[dict[str, Any]],
    hf_results: list[dict[str, Any]],
):
    print(f"\nmodel:         {model}")
    print(f"warmup shapes: {warmup_shapes}")
    print(f"tp size:       {tensor_parallel_size}")
    print(f"backend:       {backend:s}")
    print(f"\n#prompts:      {len(prompts):d}")
    print(
        f"#HF results:   {len(hf_results):d}"
        f"{'' if len(hf_results) == len(prompts) else '  ERROR':s}"
    )
    print(
        f"#vLLM results: {len(vllm_results):d}"
        f"{'' if len(vllm_results) == len(prompts) else '  ERROR':s}"
    )
    print()

    assert DISABLE_ASSERTS or len(hf_results) == len(vllm_results)
    assert DISABLE_ASSERTS or len(hf_results) == len(prompts)

    for hf_result, vllm_result in zip(hf_results, vllm_results):
        sim = util.pytorch_cos_sim(hf_result["embeddings"], vllm_result["embeddings"])

        assert math.isclose(sim, 1.0, rel_tol=0.05)


# vLLM / Spyre
def spyre_vllm_embeddings(
    model: str | ModelInfo,
    prompts: list[str],
    max_model_len: int,
    tensor_parallel_size: int,
    backend: str,
) -> list[dict[str, Any]]:
    # NB: This doesn't use the same LLM caching as generate_spyre_vllm_output
    # There aren't as many embedding tests so it's not worth the effort atm to
    # cache

    # Clear any cached decoder model
    LLM_CACHE.clear()

    if isinstance(model, ModelInfo):
        revision = model.revision
        model_name = model.name
    else:
        revision = None
        model_name = model

    vllm_model = LLM(
        model=model_name,
        tokenizer=model_name,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        revision=revision,
        tokenizer_revision=revision,
    )

    vllm_outputs = vllm_model.embed(prompts)

    results = []
    for req_output in vllm_outputs:
        result = {}
        result["embeddings"] = req_output.outputs.embedding
        results.append(result)

    return results


def setup_golden_token(
    model: ModelInfo,
    sampling_params: Union[SamplingParams, list[SamplingParams]],
    hf_outputs: list[dict[str, Any]],
) -> list[SamplingParams]:
    abs_tol = ISCLOSE_ABS_TOL_QUANTIZATION if model.is_quantized else ISCLOSE_ABS_TOL

    if isinstance(sampling_params, SamplingParams):
        # golden tokens injection is per request, so we clone SamplingParams
        sampling_params = [sampling_params.clone() for _ in hf_outputs]

    assert len(sampling_params) == len(hf_outputs)
    for idx, (param, hf) in enumerate(zip(sampling_params, hf_outputs)):
        param.extra_args = {
            "golden_token_injector": {
                "expected_token_ids": hf["token_ids"],
                "expected_logprobs": hf["logprobs"],
                "error_threshold": abs_tol,
                "label": f"#{idx}",
            }
        }
    return sampling_params


# wrapper to be able to mark some tests as xfail for logprobs diffs but
# still run and report the comparison
def maybe_xfail(func):
    def wrapper(*args, **kwargs):
        model = kwargs["model"]
        use_cb = kwargs.get("use_cb", False)
        if "micro-g3.3-8b-instruct-1b" in model.name and model.is_quantized and not use_cb:
            try:
                func(*args, **kwargs)
            except AssertionError as e:
                print(e)
            pytest.xfail(
                "Micro model FP8 static-batch compilation may result in"
                " a model that fails quality checks"
            )
        else:
            func(*args, **kwargs)

    return wrapper


@maybe_xfail
def validate_vllm_vs_hf_output(
    model: ModelInfo,
    prompts: Union[list[str], list[list[int]]],
    max_model_len: int,
    max_new_tokens: Union[int, list[int]],
    sampling_params: Union[SamplingParams, list[SamplingParams]],
    tensor_parallel_size: int,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    warmup_shapes: DecodeWarmupShapes | None = None,
    max_num_seqs: int | None = None,
    use_cb: bool = False,
    use_golden_token=True,
    max_num_batched_tokens: int | None = None,
) -> None:
    hf_outputs = generate_hf_output(
        model=model,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        ignore_eos=True,
    )

    if use_golden_token:
        sampling_params = setup_golden_token(model, sampling_params, hf_outputs)

    vllm_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        max_model_len=max_model_len,
        sampling_params=sampling_params,
        tensor_parallel_size=tensor_parallel_size,
        backend=backend,
        monkeypatch=monkeypatch,
        warmup_shapes=warmup_shapes,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        use_cb=use_cb,
    )

    compare_results(
        model=model,
        tensor_parallel_size=1,
        backend=backend,
        vllm_results=vllm_results,
        hf_results=hf_outputs,
        prompts=prompts,
    )


# vLLM / Spyre
def generate_spyre_vllm_output(
    model: str | ModelInfo,
    prompts: Union[list[str], list[list[int]]],
    max_model_len: int,
    sampling_params: Union[SamplingParams, list[SamplingParams]],
    tensor_parallel_size: int,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    warmup_shapes: DecodeWarmupShapes | None = None,
    max_num_seqs: int | None = None,
    use_cb: bool = False,
    max_num_batched_tokens: int | None = None,
) -> list[dict[str, Any]]:
    # Allows to run multiprocess V1 engine without dumping meaningless logs at
    # shutdown engine this context.
    monkeypatch.setenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1")

    vllm_model = get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        backend=backend,
        monkeypatch=monkeypatch,
        warmup_shapes=warmup_shapes,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        use_cb=use_cb,
    )

    vllm_outputs = vllm_model.generate(prompts, sampling_params)
    results = []

    for req_output in vllm_outputs:
        result = extract_output(req_output)
        results.append(result)

    return results


def extract_output(req_output):
    """Extract text, token_ids, tokens, and logprobs from request output."""

    result = {}
    result["text"] = req_output.outputs[0].text

    # TODO: Workaround for V1, if request does not fit in a warmup shape
    # token_ids may be filled with -1.
    token_ids = [t for t in req_output.outputs[0].token_ids if t >= 0]
    result["token_ids"] = tuple(token_ids)
    result["tokens"] = tuple(
        req_output.outputs[0].logprobs[i][t].decoded_token for i, t in enumerate(token_ids)
    )
    result["logprobs"] = tuple(
        req_output.outputs[0].logprobs[i][t].logprob for i, t in enumerate(token_ids)
    )

    return result


def generate_cache_for_test_swap_decode_programs_for_cb(
    model: str | ModelInfo, prompts: list[str], parent_path: str
):
    """
    This function bakes the generation of prompts with long contexts. Which
    currently are used in the test
    `test_spyre_cb::test_swap_decode_programs_for_cb`.
    """

    # Generate
    assert len(prompts) == 4

    p8k = 8 * 1024
    p16k = 16 * 1024
    p32k = 32 * 1024

    import pickle

    hf_outputs = generate_hf_output(
        model=model,
        prompts=prompts[0:2],
        max_new_tokens=p8k,
        ignore_eos=True,
        include_prompt=True,
    )
    with open(Path(parent_path) / "prompts_8k_bs2.pickle", "wb") as f:
        f.write(pickle.dumps(hf_outputs))

    hf_outputs = generate_hf_output(
        model=model,
        prompts=[prompts[2]],
        max_new_tokens=p16k,
        ignore_eos=True,
        include_prompt=True,
    )
    with open(Path(parent_path) / "prompts_16k_bs1.pickle", "wb") as f:
        f.write(pickle.dumps(hf_outputs))

    hf_outputs = generate_hf_output(
        model=model,
        prompts=[prompts[3]],
        max_new_tokens=p32k,
        ignore_eos=True,
        include_prompt=True,
    )
    with open(Path(parent_path) / "prompts_32k_bs1.pickle", "wb") as f:
        f.write(pickle.dumps(hf_outputs))

"""Verification of continuous batching

Run `python -m pytest tests/e2e/test_spyre_cb.py`.
"""

import os
import sys
import tempfile

import pytest
from graph_compare_utils import (collect_graph_files, compare_graphs,
                                 get_aftu_graphs, get_model_path)
from spyre_util import (generate_spyre_vllm_output, get_chicken_soup_prompts,
                        get_spyre_model_list)
from vllm import SamplingParams


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", ["sendnn"])
@pytest.mark.parametrize("max_num_seqs", [2, 4],
                         ids=lambda val: f"max_num_seqs({val})")
def test_compare_graphs_cb(
    model: str,
    backend: str,
    max_num_seqs: int,
    monkeypatch: pytest.MonkeyPatch,
    runtime_xfail,
):
    """Test that the spyre worker correctly outputs
    continuous batches of requests by comparing to HF"""

    if max_num_seqs > 2 and backend == "sendnn":
        runtime_xfail("CB failures expected for batch size > 2")

    # AFTU
    max_model_len = 256
    model_path = get_model_path(model)

    inference_py_args = [
        sys.executable, "scripts/inference.py", "--architecture",
        "hf_configured", "--model_path", model_path, "--variant", model_path,
        "--tokenizer", model_path, "--unfuse_weights", "--model_source", "hf",
        "--device_type", "aiu", "--compile", "--default_dtype", "fp16",
        "--compile_dynamic", "--min_pad_length", "64", "--max_new_tokens", "5",
        "--batch_size",
        str(max_num_seqs), "--compile_dynamic_sendnn", "--attention_type=paged"
    ]

    extra_env = {
        "VLLM_DT_MAX_CONTEXT_LEN": str(max_model_len),
        "VLLM_DT_MAX_BATCH_SIZE": str(max_num_seqs)
    }
    aftu_graphs = get_aftu_graphs(inference_py_args, extra_env)

    ## VLLM
    prompts = get_chicken_soup_prompts(4)

    max_new_tokens = 20

    monkeypatch.setenv("DEE_DUMP_GRAPHS", "vllm_static")
    # Disable cache to produce the graphs
    monkeypatch.setenv("TORCH_SENDNN_CACHE_ENABLE", "0")
    vllm_sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=True)

    original_cwd = os.getcwd()
    try:
        # Change to temp dir to set the test clean
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            generate_spyre_vllm_output(model=model,
                                       prompts=prompts,
                                       max_model_len=max_model_len,
                                       block_size=256,
                                       sampling_params=vllm_sampling_params,
                                       tensor_parallel_size=1,
                                       backend=backend,
                                       max_num_seqs=max_num_seqs,
                                       use_cb=True,
                                       monkeypatch=monkeypatch)

            vllm_graphs = collect_graph_files(tmpdir)
    finally:
        # Restore in case of exception
        os.chdir(original_cwd)

    assert compare_graphs(vllm_graphs, aftu_graphs)


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("warmup_shape",
                         [(64, 5, 1), (64, 5, 2),
                          (64, 5, 4)])  # (prompt_length/new_tokens/batch_size)
@pytest.mark.parametrize("backend", ["sendnn"])
def test_compare_graphs_static_batching(
    model: str,
    warmup_shape: tuple[int, int, int],
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    # AFTU
    model_path = get_model_path(model)

    inference_py_args = [
        sys.executable,
        "scripts/inference.py",
        "--architecture",
        "hf_configured",
        "--model_path",
        model_path,
        "--variant",
        model_path,
        "--tokenizer",
        model_path,
        "--unfuse_weights",
        "--model_source",
        "hf",
        "--device_type",
        "aiu",
        "--compile",
        "--default_dtype",
        "fp16",
        "--compile_dynamic",
        "--min_pad_length",
        "64",
        "--max_new_tokens",
        str(warmup_shape[1]),
        "--batch_size",
        str(warmup_shape[2]),
    ]

    aftu_graphs = get_aftu_graphs(inference_py_args)

    ## VLLM
    prompts = get_chicken_soup_prompts(4)

    max_new_tokens = warmup_shape[1]

    monkeypatch.setenv("DEE_DUMP_GRAPHS", "vllm_static")
    # Disable cache to produce the graphs
    monkeypatch.setenv("TORCH_SENDNN_CACHE_ENABLE", "0")
    vllm_sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=True)

    original_cwd = os.getcwd()
    try:
        # Change to temp dir to set the test clean
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            generate_spyre_vllm_output(model=model,
                                       prompts=prompts,
                                       warmup_shapes=[warmup_shape],
                                       max_model_len=2048,
                                       block_size=2048,
                                       sampling_params=vllm_sampling_params,
                                       tensor_parallel_size=1,
                                       backend=backend,
                                       monkeypatch=monkeypatch)

            vllm_graphs = collect_graph_files(tmpdir)
    finally:
        # Restore in case of exception
        os.chdir(original_cwd)

    assert compare_graphs(vllm_graphs, aftu_graphs)

"""Test cases for models and runtime configurations that we expect to be 
supported on spyre hardware"""

from typing import NamedTuple

import pytest
from llm_cache import DecodeWarmupShapes
from spyre_util import (check_output_against_hf, generate_spyre_vllm_output,
                        get_chicken_soup_prompts)
from vllm import SamplingParams


class RuntimeConfiguration(NamedTuple):
    model: str
    continuous_batching: bool = False
    warmup_shapes: DecodeWarmupShapes = []
    max_model_len: int = 0
    max_num_seqs: int = 0
    tensor_parallel_size: int = 1


def get_decoder_configs():
    """List of test cases to run for text generation"""
    configs = [
        # Granite 3.3 8b
        # 8k context serial static batching
        RuntimeConfiguration(model="ibm-granite/granite-3.3-8b-instruct",
                             warmup_shapes=[(7168, 1024, 1)],
                             tensor_parallel_size=4),
        # 3k context parallel static batching
        RuntimeConfiguration(model="ibm-granite/granite-3.3-8b-instruct",
                             warmup_shapes=[(2048, 1024, 16)],
                             tensor_parallel_size=4),
        # 8k continuous batching
        RuntimeConfiguration(model="ibm-granite/granite-3.3-8b-instruct",
                             continuous_batching=True,
                             max_model_len=8192,
                             max_num_seqs=4,
                             tensor_parallel_size=4)
    ]

    return [pytest.param(config, id=f"config={config}") for config in configs]


@pytest.mark.supported_models
@pytest.mark.parametrize("runtime_configuration", get_decoder_configs())
def test_supported_decoders(runtime_configuration: RuntimeConfiguration,
                            monkeypatch):
    # Test that we can boot an LLM and run .generate() on a prompt

    vllm_sampling_params = SamplingParams(
        max_tokens=5,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=True)

    if runtime_configuration.continuous_batching:
        kwargs = {
            "max_model_len": runtime_configuration.max_model_len,
            "max_num_seqs": runtime_configuration.max_num_seqs
        }
    else:
        kwargs = {
            "warmup_shapes": runtime_configuration.warmup_shapes,
            # temp bug
            "max_model_len": 128 * 1024
        }

    prompts = get_chicken_soup_prompts(4)

    vllm_results = generate_spyre_vllm_output(
        model=runtime_configuration.model,
        prompts=prompts,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=runtime_configuration.tensor_parallel_size,
        backend="sendnn",
        monkeypatch=monkeypatch,
        **kwargs)
    check_output_against_hf(runtime_configuration.model, "sendnn", 5,
                            vllm_results, prompts)


def get_embedding_configs():
    """List of test cases to run for embedding models"""
    # granite-embeddings-125m
    # granite-embedding-278m-multilingual

    configs = [
        RuntimeConfiguration(
            model="ibm-granite/granite-embedding-125m-english",
            warmup_shapes=[(512, 1, 16)],
            tensor_parallel_size=4),
        RuntimeConfiguration(
            model="ibm-granite/granite-embedding-278m-multilingual",
            warmup_shapes=[(512, 1, 16)],
            tensor_parallel_size=4),
    ]

    return [pytest.param(config, id=f"config={config}") for config in configs]


@pytest.mark.supported_models
@pytest.mark.parametrize("runtime_configuration", get_embedding_configs())
def test_supported_embedding_models(
        runtime_configuration: RuntimeConfiguration, monkeypatch):
    # Test that we can boot an LLM and get embeddings
    # TODO
    pass

from vllm import LLM, RequestOutput, SamplingParams

import pytest


@pytest.mark.spyre
def test_basic_model_load():
    model = LLM("ibm-ai-platform/micro-g3.3-8b-instruct-1b", max_model_len=128, max_num_seqs=2)

    sampling_params = SamplingParams(max_tokens=5)
    output: list[RequestOutput] = model.generate(
        prompts="Hello World", sampling_params=sampling_params
    )

    assert len(output[0].outputs[0].text) > 0

import pytest
from vllm import LLM, SamplingParams


def test_spyre_backend_batch1_determinism(model: str, max_model_len: int,
                                          max_num_seqs: int, block_size: int,
                                          tensor_parallel_size: int,
                                          backend: str,
                                          monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    monkeypatch.setenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1")

    vllm_model = LLM(
        model=model,
        tokenizer=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        block_size=block_size,
        tensor_parallel_size=tensor_parallel_size,
    )

    prompt = "The capital of the United Kingdom is"
    params = SamplingParams(temperature=0.0, seed=8780, max_tokens=5)

    output1 = vllm_model.generate(prompt, params, request_id="1")
    output2 = vllm_model.generate(prompt, params, request_id="2")

    assert output1.outputs[0].text == output2.outputs[0].text
    assert "London" in output1.outputs[0].text


def test_spyre_dynamic_batch_isolation(model: str, max_model_len: int,
                                       max_num_seqs: int, block_size: int,
                                       tensor_parallel_size: int, backend: str,
                                       monkeypatch: pytest.MonkeyPatch):

    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    monkeypatch.setenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1")

    vllm_model = LLM(
        model=model,
        tokenizer=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        block_size=block_size,
        tensor_parallel_size=tensor_parallel_size,
    )

    prompts = [
        "Write a 100-word essay on artificial intelligence.",
        "The primary colors are", "Request this word: test test test test test"
    ]

    sampling_params = [
        SamplingParams(temperature=0.7, max_tokens=100, seed=42),
        SamplingParams(temperature=0.0, max_tokens=10, seed=42),
        SamplingParams(max_tokens=20, presence_penalty=2.0, seed=42)
    ]

    expected_out = []
    for prompt, param in zip(prompts, sampling_params):
        output = vllm_model.generate(prompt, param)
        expected_out.append(output[0])

    vllm_outputs = vllm_model.generate(prompts=prompts,
                                       sampling_params=sampling_params)

    # checks isolation
    assert expected_out[0].outputs[0].text == vllm_outputs[0].outputs[0].text
    assert expected_out[1].outputs[0].text == vllm_outputs[0].outputs[1].text
    assert expected_out[2].outputs[0].text == vllm_outputs[0].outputs[2].text

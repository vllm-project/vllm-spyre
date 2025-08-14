import asyncio

import pytest
from vllm import AsyncLLMEngine, EngineArgs, SamplingParams


@pytest.fixture(scope='module')
def engine() -> AsyncLLMEngine:
    engine_args = EngineArgs(
        model="ibm-granite/granite-3.2-8b-instruct",
        # device="spyre",
        max_model_len=1024,
        max_num_seqs=16,
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


@pytest.mark.asyncio
async def test_spyre_backend_batch1_determinism(engine: AsyncLLMEngine):

    prompt = "The capital of the United Kingdom is"
    params = SamplingParams(temperature=0.0, seed=8780, max_tokens=5)

    output1 = await engine.generate(prompt, params, request_id="1")
    output2 = await engine.generate(prompt, params, request_id="2")

    text1 = output1.outputs[0].text
    text2 = output2.outputs[0].text

    assert text1 == text2
    assert "London" in text1


@pytest.mark.asyncio
@pytest.mark.parametrize(["minimal", "limit", "spected_text"],
                         [50, 2, "red, blue and yellow"])
async def test_spyre_dynamic_batch_isolation(engine: AsyncLLMEngine,
                                             minimal: int, limit: int,
                                             spected_text: str):

    long_task = engine.generate(
        "Write a 100-word essay on artificial intelligence.",
        SamplingParams(temperature=0.7, max_tokens=100, seed=42),
        request_id="long_req",
    )

    deterministic_task = engine.generate(
        "The primary colors are",
        SamplingParams(temperature=0.0, max_tokens=10, seed=42),
        request_id="deterministic_req",
    )

    penalty_task = engine.generate(
        "Request this word: test test test test test",
        SamplingParams(max_tokens=20, presence_penalty=2.0),
        request_id="penalty_req",
    )

    # expected results
    expected_penalty_result = await asyncio.run(penalty_task)
    expected_long_res = await asyncio.run(long_task)
    expected_deterministic_result = await asyncio.run(deterministic_task)

    # async requests
    results = await asyncio.gather(
        long_task,
        deterministic_task,
        penalty_task,
        return_exceptions=True,
    )

    long_result, deterministic_result, penalty_result = results

    # checks that the requests finished
    assert not isinstance(long_result, Exception)
    assert not isinstance(deterministic_result, Exception)
    assert not isinstance(penalty_task, Exception)

    # checks isolation
    assert long_result.outputs[0].text == expected_long_res.outputs[0].text
    assert deterministic_result.outputs[
        0].text == expected_deterministic_result[0].text
    assert penalty_result.outputs[0].text == expected_penalty_result.outputs[
        0].text

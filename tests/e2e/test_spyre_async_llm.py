import asyncio
from contextlib import ExitStack

import pytest
from spyre_util import (get_chicken_soup_prompts, get_spyre_backend_list,
                        get_spyre_model_list)
from vllm import PromptType, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM


async def generate(
    engine: AsyncLLM | AsyncLLMEngine,
    request_id: str,
    prompt: PromptType,
    output_kind: RequestOutputKind,
    max_tokens: int,
    n: int = 1,
) -> tuple[int, str]:
    # Ensure generate doesn't complete too fast for cancellation test.
    await asyncio.sleep(0.2)

    count = 0
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        ignore_eos=True,
        output_kind=output_kind,
        seed=42,
        n=n,
    )
    async for out in engine.generate(request_id=request_id,
                                     prompt=prompt,
                                     sampling_params=sampling_params):

        num_tokens = sum(len(output.token_ids) for output in out.outputs)
        if output_kind == RequestOutputKind.DELTA:
            count += num_tokens
        else:
            count = num_tokens

        await asyncio.sleep(0.01)

    return count, request_id


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize("cb",
                         [pytest.param(1, marks=pytest.mark.cb, id="cb"), 0])
@pytest.mark.parametrize("warmup_shapes", [[
    (64, 20, 4),
]])
@pytest.mark.parametrize(
    "output_kind", [RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY])
@pytest.mark.asyncio
async def test_abort(
    model: str,
    backend: str,
    cb: int,
    warmup_shapes: list[list[int]],
    output_kind: RequestOutputKind,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test handling of cancelled requests"""
    with monkeypatch.context() as m, ExitStack() as after:
        m.setenv("VLLM_USE_V1", "1")
        m.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
        if cb == 1:
            m.setenv("VLLM_SPYRE_USE_CB", "1")
        else:
            warmup_prompt_length = [t[0] for t in warmup_shapes]
            warmup_new_tokens = [t[1] for t in warmup_shapes]
            warmup_batch_size = [t[2] for t in warmup_shapes]

            m.setenv('VLLM_SPYRE_WARMUP_PROMPT_LENS',
                     ','.join(str(val) for val in warmup_prompt_length))
            m.setenv('VLLM_SPYRE_WARMUP_NEW_TOKENS',
                     ','.join(str(val) for val in warmup_new_tokens))
            m.setenv('VLLM_SPYRE_WARMUP_BATCH_SIZES',
                     ','.join(str(val) for val in warmup_batch_size))

        # Async LLM API is a little different between v0 and V1
        engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=model,
                tokenizer=model,
                max_model_len=128,
                max_num_seqs=2,
                block_size=2048,
            ))
        has_unfinished_requests = \
            engine.output_processor.has_unfinished_requests
        after.callback(engine.shutdown)

        # Test structure here mirrors upstream vLLM test_abort:
        # https://github.com/vllm-project/vllm/blob/e6aab5de2999187c6cf0206f2d63ab6d7a0b6964/tests/v1/engine/test_async_llm.py#L160
        NUM_REQUESTS = 20
        NUM_EXPECTED_TOKENS = 20
        REQUEST_IDS_TO_ABORT = range(1, NUM_REQUESTS, 3)
        PARALLEL_SAMPLE_REQ_IDS = range(1, NUM_REQUESTS, 5)

        request_ids = [f"request-{i}" for i in range(NUM_REQUESTS)]

        # Create concurrent requests
        tasks: list[asyncio.Task] = []
        prompt = get_chicken_soup_prompts(1)[0]
        for idx, request_id in enumerate(request_ids):
            max_tokens = NUM_EXPECTED_TOKENS
            n = 3 if idx in PARALLEL_SAMPLE_REQ_IDS else 1
            tasks.append(
                asyncio.create_task(
                    generate(engine, request_id, prompt, output_kind,
                             max_tokens, n)))

        # Simulate cancellation from API server client disconnect
        for idx in REQUEST_IDS_TO_ABORT:
            tasks[idx].cancel()
            await asyncio.sleep(0.1)

        # Confirm that requests actually cancelled and that the other requests
        # are not impacted
        for idx, task in enumerate(tasks):
            if idx in REQUEST_IDS_TO_ABORT:
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                num_generated_tokens, request_id = await task
                n = 3 if idx in PARALLEL_SAMPLE_REQ_IDS else 1
                expected_tokens = NUM_EXPECTED_TOKENS * n
                assert num_generated_tokens == expected_tokens, (
                    f"{request_id} generated {num_generated_tokens} but "
                    f"expected {expected_tokens}")

        # Make sure all aborted requests were really aborted
        assert not has_unfinished_requests()

        # Confirm that the server is still up and functioning
        request_id = f"request-{REQUEST_IDS_TO_ABORT[0]}"
        task = asyncio.create_task(
            generate(engine, request_id, prompt, output_kind,
                     NUM_EXPECTED_TOKENS))
        num_generated_tokens, request_id = await task
        assert num_generated_tokens == NUM_EXPECTED_TOKENS
        assert not has_unfinished_requests()

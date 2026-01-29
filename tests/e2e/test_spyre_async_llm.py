import asyncio
from contextlib import ExitStack

import pytest
from spyre_util import DecodeWarmupShapes, ModelInfo, get_chicken_soup_prompts, patch_environment
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
    async for out in engine.generate(
        request_id=request_id, prompt=prompt, sampling_params=sampling_params
    ):
        num_tokens = sum(len(output.token_ids) for output in out.outputs)
        if output_kind == RequestOutputKind.DELTA:
            count += num_tokens
        else:
            count = num_tokens

        await asyncio.sleep(0.01)

    return count, request_id


@pytest.mark.parametrize("output_kind", [RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY])
@pytest.mark.asyncio
async def test_abort(
    model: ModelInfo,
    backend: str,
    mode: str,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    warmup_shapes: DecodeWarmupShapes,
    output_kind: RequestOutputKind,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test handling of cancelled requests"""
    with monkeypatch.context() as m, ExitStack() as after:
        patch_kwargs = (
            {
                "use_cb": True,
                "warmup_shapes": None,
                "use_chunked_prefill": mode in ["cp", "pc"],
            }
            if mode in ["cb", "cp", "pc"]
            else {
                "use_cb": False,
                "warmup_shapes": warmup_shapes,
            }
        )
        patch_environment(
            **patch_kwargs,
            backend=backend,
            monkeypatch=m,
        )

        # Async LLM API is a little different between v0 and V1
        engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=model.name,
                tokenizer=model.name,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens if mode in ["cp", "pc"] else None,
                enable_prefix_caching=mode == "pc",
                revision=model.revision,
                tokenizer_revision=model.revision,
            )
        )
        has_unfinished_requests = engine.output_processor.has_unfinished_requests
        after.callback(engine.shutdown)

        # Test structure here mirrors upstream vLLM test_abort:
        # https://github.com/vllm-project/vllm/blob/e6aab5de2999187c6cf0206f2d63ab6d7a0b6964/tests/v1/engine/test_async_llm.py#L160
        NUM_REQUESTS = 15
        NUM_EXPECTED_TOKENS = 5
        REQUEST_IDS_TO_ABORT = range(1, NUM_REQUESTS, 3)
        PARALLEL_SAMPLE_REQ_IDS = range(1, NUM_REQUESTS, 5)

        request_ids = [f"request-{i}" for i in range(NUM_REQUESTS)]

        # Create concurrent requests
        tasks: list[asyncio.Task] = []
        prompt = get_chicken_soup_prompts(1)[0]
        for idx, request_id in enumerate(request_ids):
            max_tokens = NUM_EXPECTED_TOKENS
            n = 2 if idx in PARALLEL_SAMPLE_REQ_IDS else 1
            tasks.append(
                asyncio.create_task(
                    generate(engine, request_id, prompt, output_kind, max_tokens, n)
                )
            )

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
                n = 2 if idx in PARALLEL_SAMPLE_REQ_IDS else 1
                expected_tokens = NUM_EXPECTED_TOKENS * n
                assert num_generated_tokens == expected_tokens, (
                    f"{request_id} generated {num_generated_tokens} but expected {expected_tokens}"
                )

        # Make sure all aborted requests were really aborted
        assert not has_unfinished_requests()

        # Confirm that the server is still up and functioning
        request_id = f"request-{REQUEST_IDS_TO_ABORT[0]}"
        task = asyncio.create_task(
            generate(engine, request_id, prompt, output_kind, NUM_EXPECTED_TOKENS)
        )
        num_generated_tokens, request_id = await task
        assert num_generated_tokens == NUM_EXPECTED_TOKENS
        assert not has_unfinished_requests()

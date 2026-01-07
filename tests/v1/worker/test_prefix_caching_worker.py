import pytest
from scheduling_utils import create_request_for_scheduler_test, random_prompt

from v1.worker.mock_model import InstrumentedModelRunner


@pytest.mark.cpu
@pytest.mark.worker
@pytest.mark.prefix_caching
def test_block_sharing_for_2_chunks(
    monkeypatch: pytest.MonkeyPatch,
):
    model = InstrumentedModelRunner.DEFAULT_TEST_MODEL
    pc_model_runner = InstrumentedModelRunner.build(monkeypatch)
    prompt = random_prompt(model=model, seed=0, length=192)

    request1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=2,
        prompt=prompt,
        use_golden_token_injection=False,
        generate_hf_results=False,
        block_hasher=pc_model_runner.request_block_hasher,
    )

    request2 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=2,
        prompt=prompt,
        use_golden_token_injection=False,
        generate_hf_results=False,
        block_hasher=pc_model_runner.request_block_hasher,
    )

    chunk_plan = pc_model_runner._plan_chunking(request1.request)

    assert chunk_plan.chunk_count == 2
    assert chunk_plan.padding_blocks == 1
    assert chunk_plan.usable_cache_blocks == 0
    assert chunk_plan.total_cache_blocks == 0

    kv_cache_manager = pc_model_runner.kv_cache_manager

    kv_cache_manager.allocate_new_blocks(request1.request.request_id, 192)
    kv_cache_manager.cache_blocks(request1.request, 192)
    kv_cache_manager.free(request1.request.request_id)

    chunk_plan = pc_model_runner._plan_chunking(request2.request)

    assert chunk_plan.chunk_count == 2
    assert chunk_plan.padding_blocks == 1
    assert chunk_plan.usable_cache_blocks == 1
    assert chunk_plan.total_cache_blocks == 3


@pytest.mark.cpu
@pytest.mark.worker
@pytest.mark.prefix_caching
def test_multi_chunk_partial_match_misaligned(
    monkeypatch: pytest.MonkeyPatch,
):
    """Scenario where two sequences are scheduled which share a common
    prefix. The second sequence shares 254 tokens with the first sequence,
    which is less than two chunks. We can therefore reuse only one chunk
    (254 < 2*128 = 256). This leads to recomputation of the third block.

    p1 = [AB|CD|EF]
    p2 = [AB|CX|EF]

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 384,  max tokens = 2, step joining = 0
            * 1: len = 384, max tokens = 2, step joining = 0
    """
    monkeypatch.setenv("VLLM_SPYRE_CP_INTERLEAVE_STEPS", "0")

    pc_model_runner = InstrumentedModelRunner.build(
        monkeypatch=monkeypatch,
        available_blocks=16,
    )

    # twice the same seed for a sequence of length 384
    # the first sequence shares the same prefix of length 384 tokens
    # the second sequence shares the same prefix of length 254 tokens
    # hence sequence 1 shares the first 254 tokens with sequence 0

    model = InstrumentedModelRunner.DEFAULT_TEST_MODEL
    prompt1 = random_prompt(model=model, seed=0, length=384)
    prompt2 = prompt1[0:254] + random_prompt(model=model, seed=0, length=384 - 254)

    request1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=2,
        prompt=prompt1,
        use_golden_token_injection=False,
        generate_hf_results=False,
        block_hasher=pc_model_runner.request_block_hasher,
    )

    request2 = create_request_for_scheduler_test(
        model=model,
        request_id=1,
        add_step=0,
        max_tokens=2,
        prompt=prompt2,
        use_golden_token_injection=False,
        generate_hf_results=False,
        block_hasher=pc_model_runner.request_block_hasher,
    )

    # Schedule chunk 0 of request 0
    model_runner_output_1 = pc_model_runner.execute_new_request(request1.request, 128)
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[1, 2]],
        slot_mappings=[[1, 2]],
    )

    assert model_runner_output_1.req_ids == ["0"]
    assert model_runner_output_1.sampled_token_ids == []
    assert model_runner_output_1.tkv == 384
    assert model_runner_output_1.n_free_blocks == 8
    assert model_runner_output_1.left_padding == {"0": 0}
    assert model_runner_output_1.prefix_cache_hit_len == {"0": 0}

    # Schedule chunk 1 of request 0
    model_runner_output_2 = pc_model_runner.execute_running_requests(
        req_ids=["0"],
        num_computed_tokens=[128],
        tokens_to_schedule=[128],
    )
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[1, 2, 3, 4]],
        slot_mappings=[[3, 4]],
    )

    assert model_runner_output_2.req_ids == ["0"]
    assert model_runner_output_2.sampled_token_ids == []
    assert model_runner_output_2.tkv == 384
    assert model_runner_output_2.n_free_blocks == 8
    assert model_runner_output_2.left_padding == {"0": 0}

    # Schedule chunk 2 of request 0
    model_runner_output_3 = pc_model_runner.execute_running_requests(
        req_ids=["0"],
        num_computed_tokens=[256],
        tokens_to_schedule=[128],
    )
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[1, 2, 3, 4, 5, 6]],
        slot_mappings=[[5, 6]],
    )

    assert model_runner_output_3.req_ids == ["0"]
    assert len(model_runner_output_3.sampled_token_ids) == 1
    assert model_runner_output_3.tkv == 384
    assert model_runner_output_3.n_free_blocks == 8
    assert model_runner_output_3.left_padding == {"0": 0}

    # Schedule chunk 0 of request 1
    model_runner_output_4 = pc_model_runner.execute_new_request(request2.request, 128)
    # chunk loaded from cache
    assert pc_model_runner.model.last_attn_metadata is None

    assert model_runner_output_4.req_ids == ["1"]
    assert model_runner_output_4.sampled_token_ids == []
    assert model_runner_output_4.tkv == 384
    assert model_runner_output_4.n_free_blocks == 1
    assert model_runner_output_4.left_padding == {"1": 0}
    assert model_runner_output_4.prefix_cache_hit_len == {"1": 128}

    # Schedule chunk 1 of request 1
    model_runner_output_5 = pc_model_runner.execute_running_requests(
        req_ids=["1"],
        num_computed_tokens=[128],
        tokens_to_schedule=[128],
    )
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[1, 2, 3, 7]],
        slot_mappings=[[0, 7]],
    )  # <-- HERE: the block table and slot mapping
    # point to different places when a block is recomputed

    assert model_runner_output_5.req_ids == ["1"]
    assert model_runner_output_5.sampled_token_ids == []
    assert model_runner_output_5.tkv == 384
    assert model_runner_output_5.n_free_blocks == 1
    assert model_runner_output_5.left_padding == {"1": 0}

    # Schedule chunk 2 of request 1
    model_runner_output_6 = pc_model_runner.execute_running_requests(
        req_ids=["1"],
        num_computed_tokens=[256],
        tokens_to_schedule=[128],
    )
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[1, 2, 3, 7, 8, 9]],
        slot_mappings=[[8, 9]],
    )

    assert model_runner_output_6.req_ids == ["1"]
    assert len(model_runner_output_6.sampled_token_ids) == 1
    assert model_runner_output_6.tkv == 384
    assert model_runner_output_6.n_free_blocks == 1
    assert model_runner_output_6.left_padding == {"1": 0}

    # Schedule decodes of requests 0 and 1
    model_runner_output_7 = pc_model_runner.execute_running_requests(
        req_ids=["1", "0"],
        num_computed_tokens=[384, 384],
        tokens_to_schedule=[1, 1],
    )
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[1, 2, 3, 4, 5, 6, 10], [1, 2, 3, 7, 8, 9, 11]],
        slot_mappings=[[10], [11]],
        slot_slice=slice(0, 1),
    )

    assert model_runner_output_7.req_ids == ["0", "1"]
    assert len(model_runner_output_7.sampled_token_ids) == 2
    assert model_runner_output_7.tkv == 385
    assert model_runner_output_7.n_free_blocks == 1
    assert model_runner_output_7.left_padding == {"0": 0, "1": 0}


@pytest.mark.cpu
@pytest.mark.worker
@pytest.mark.prefix_caching
def test_first_chunk_recomputation(
    monkeypatch: pytest.MonkeyPatch,
):
    """Scenario where two sequences are scheduled with 2 blocks
    each and a common 1 block prefix. Since chunk size is 4 times the block
    size, the first two blocks of the first chunk of each request will be
    padding blocks. In the second request, the third block of the chunk
    will be recomputed to prevent block table deduplication while the
    fourth block will be computed from scratch.

    p1 = [00AB]
    p2 = [00AC]

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 128,  max tokens = 2, step joining = 0
            * 1: len = 128, max tokens = 2, step joining = 0
    """
    monkeypatch.setenv("VLLM_SPYRE_CP_INTERLEAVE_STEPS", "0")

    pc_model_runner = InstrumentedModelRunner.build(
        monkeypatch=monkeypatch,
        max_num_batched_tokens=256,
        available_blocks=16,
    )

    model = InstrumentedModelRunner.DEFAULT_TEST_MODEL
    prompt1 = random_prompt(model=model, seed=0, length=128)
    prompt2 = prompt1[0:64] + random_prompt(model=model, seed=0, length=128 - 64)

    request1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=2,
        prompt=prompt1,
        use_golden_token_injection=False,
        generate_hf_results=False,
        block_hasher=pc_model_runner.request_block_hasher,
    )

    request2 = create_request_for_scheduler_test(
        model=model,
        request_id=1,
        add_step=0,
        max_tokens=2,
        prompt=prompt2,
        use_golden_token_injection=False,
        generate_hf_results=False,
        block_hasher=pc_model_runner.request_block_hasher,
    )

    # Schedule chunk 0 of request 0
    model_runner_output_1 = pc_model_runner.execute_new_request(request1.request, 128)
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[0, 0, 1, 2]],
        slot_mappings=[[0, 0, 1, 2]],
    )

    assert model_runner_output_1.req_ids == ["0"]
    assert len(model_runner_output_1.sampled_token_ids) == 1
    assert model_runner_output_1.tkv == 128
    assert model_runner_output_1.n_free_blocks == 12
    assert model_runner_output_1.left_padding == {"0": 128}
    assert model_runner_output_1.prefix_cache_hit_len == {}

    # Schedule chunk 0 of request 1
    model_runner_output_2 = pc_model_runner.execute_new_request(request2.request, 128)
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[0, 0, 1, 3]],
        slot_mappings=[[0, 0, 0, 3]],
    )
    # ^This block table and slot mapping is the crux of this test.
    # The padding blocks align with the slot mapping pointing to block
    # 0. The third block is a cache hit, but has to be recomputed
    # because we're in the last chunk with a prefix hit. The fourth
    # block is not a prefix hit and has to be computed from scratch
    # in a new block.

    assert model_runner_output_2.req_ids == ["1"]
    assert len(model_runner_output_2.sampled_token_ids) == 1
    assert model_runner_output_2.tkv == 128
    assert model_runner_output_2.n_free_blocks == 9
    assert model_runner_output_2.left_padding == {"1": 128}
    assert model_runner_output_2.prefix_cache_hit_len == {}

    # Schedule decodes of requests 0 and 1
    model_runner_output_3 = pc_model_runner.execute_running_requests(
        req_ids=["1", "0"],
        num_computed_tokens=[128, 128],
        tokens_to_schedule=[1, 1],
    )
    pc_model_runner.assert_block_tables_and_slot_mappings(
        block_tables=[[1, 2, 4], [1, 3, 5]],
        slot_mappings=[[4], [5]],
        slot_slice=slice(0, 1),
    )

    assert model_runner_output_3.req_ids == ["0", "1"]
    assert len(model_runner_output_3.sampled_token_ids) == 2
    assert model_runner_output_3.tkv == 129
    assert model_runner_output_3.n_free_blocks == 9
    assert model_runner_output_3.left_padding == {"0": 0, "1": 0}

from dataclasses import fields
from typing import Any, Optional

import pytest
import torch
from scheduling_utils import create_request_for_scheduler_test, random_prompt
from spyre_util import ModelInfo, patch_environment
from vllm import EngineArgs
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.outputs import ModelRunnerOutput, SamplerOutput
from vllm.v1.request import Request
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

import vllm_spyre.envs as envs_spyre
from vllm_spyre.model_executor.model_loader.spyre import SpyreAttentionMetadata
from vllm_spyre.platform import SpyrePlatform
from vllm_spyre.v1.worker.spyre_model_runner import ChunkedPrefillModelRunner


class MockContinuousBatchingFmsModel:

    def set_past_key_value_states(self, num_blocks) -> None:
        pass


class MockSpyreCausalLM:

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:

        try:
            ## Temporary backwards compatibility for 0.10.2
            from vllm.model_executor.layers.sampler import get_sampler
            self.sampler = get_sampler()
        except (ImportError, ModuleNotFoundError):
            self.sampler = Sampler()

        # boolean tensor of length batch size with indices:
        # True for unfinished sequences and
        # False for finished or padded sequences
        self.indices = None

        # number of right pads (relevant for continuous batching only)
        self.n_pads_right = 0

        self._mask_dtype = torch.float16 if \
            envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND == "sendnn" \
            else torch.float32

        self.model = MockContinuousBatchingFmsModel()

        self.vocab_size = vllm_config.model_config.get_vocab_size()

        self.last_input_ids: Optional[torch.Tensor] = None
        self.last_positions: Optional[torch.Tensor] = None
        self.last_masks: Optional[torch.Tensor] = None
        self.last_is_prompt: Optional[bool] = None
        self.last_attn_metadata: Optional[SpyreAttentionMetadata] = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        masks: torch.Tensor,
        is_prompt: bool,
    ) -> torch.Tensor:

        self.last_input_ids = input_ids
        self.last_positions = positions
        self.last_masks = masks
        self.last_is_prompt = is_prompt

        forward_context = get_forward_context()

        assert isinstance(forward_context.attn_metadata,
                          SpyreAttentionMetadata)
        self.last_attn_metadata = forward_context.attn_metadata

        batch_size = input_ids.shape[0]

        return torch.empty((batch_size, self.vocab_size),
                           dtype=torch.float32,
                           device=input_ids.device)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def get_mask_dtype(self) -> torch.dtype:
        return self._mask_dtype


class InstrumentedModelRunner(ChunkedPrefillModelRunner):

    def __init__(
        self,
        vllm_config: VllmConfig,
        is_driver_worker: bool,
        rank: int,
    ):
        super().__init__(vllm_config=vllm_config,
                         is_driver_worker=is_driver_worker,
                         rank=rank)

        self.model = MockSpyreCausalLM(vllm_config=vllm_config)

    @SpyrePlatform.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        **kwargs,
    ) -> ModelRunnerOutput:

        self.model.last_input_ids = None
        self.model.last_positions = None
        self.model.last_masks = None
        self.model.last_is_prompt = None
        self.model.last_attn_metadata = None

        return super().execute_model(scheduler_output, **kwargs)


@pytest.fixture
def pc_model_runner(
        monkeypatch: pytest.MonkeyPatch, model: ModelInfo, max_num_seqs: int,
        max_model_len: int, max_num_batched_tokens: int,
        available_blocks: Optional[int]) -> ChunkedPrefillModelRunner:
    """A fixture that returns a model runner configured for prefix caching."""

    patch_environment(use_cb=True,
                      warmup_shapes=None,
                      backend="eager",
                      monkeypatch=monkeypatch,
                      use_chunked_prefill=True,
                      max_num_batched_tokens=max_num_batched_tokens)

    engine_args = EngineArgs(model=model.name,
                             tokenizer=model.name,
                             revision=model.revision,
                             tokenizer_revision=model.revision,
                             max_model_len=max_model_len,
                             max_num_seqs=max_num_seqs,
                             num_gpu_blocks_override=available_blocks,
                             logits_processors=[],
                             max_num_batched_tokens=max_num_batched_tokens,
                             enable_prefix_caching=True)
    vllm_config = engine_args.create_engine_config()

    model_runner = InstrumentedModelRunner(
        vllm_config=vllm_config,
        is_driver_worker=True,
        rank=0,
    )

    model_runner.pre_warmup()
    model_runner.complete_warmup()

    yield model_runner


@pytest.mark.cpu
@pytest.mark.worker
@pytest.mark.prefix_caching
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [512])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_block_sharing_for_2_chunks(
        model: ModelInfo, max_num_seqs: int, max_model_len: int,
        max_num_batched_tokens: int, available_blocks: Optional[int],
        pc_model_runner: ChunkedPrefillModelRunner):

    prompt = random_prompt(model=model, seed=0, length=192)

    request1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=2,
        prompt=prompt,
        use_golden_token_injection=True,
        block_hasher=pc_model_runner.request_block_hasher)

    request2 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=2,
        prompt=prompt,
        use_golden_token_injection=True,
        block_hasher=pc_model_runner.request_block_hasher)

    (chunk_count, left_blocks, usable_blocks,
     n_hit) = pc_model_runner._plan_chunking(request1.request)

    assert chunk_count == 2
    assert left_blocks == 1
    assert usable_blocks == 0
    assert n_hit == 0

    kv_cache_manager = pc_model_runner.kv_cache_manager

    kv_cache_manager.allocate_new_blocks(request1.request.request_id, 192)
    kv_cache_manager.cache_blocks(request1.request, 192)
    kv_cache_manager.free(request1.request.request_id)

    (chunk_count, left_blocks, usable_blocks,
     n_hit) = pc_model_runner._plan_chunking(request2.request)

    assert chunk_count == 2
    assert left_blocks == 1
    assert usable_blocks == 1
    assert n_hit == 3


def _compat_sched_output_kwargs() -> dict[str, Any]:
    field_names = [field.name for field in fields(SchedulerOutput)]
    kwargs: dict[str, Any] = {}
    if "structured_output_request_ids" in field_names:
        kwargs["structured_output_request_ids"] = {}
    if "grammar_bitmask" in field_names:
        kwargs["grammar_bitmask"] = None
    return kwargs


def _schedule_new_request(request: Request,
                          tokens_to_schedule: int) -> SchedulerOutput:
    new_reqs = [NewRequestData.from_request(request=request, block_ids=[])]
    num_scheduled_tokens = {request.request_id: tokens_to_schedule}

    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=tokens_to_schedule,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        **_compat_sched_output_kwargs())


def _compat_request_data_kwargs() -> dict[str, Any]:
    field_names = [field.name for field in fields(CachedRequestData)]
    kwargs: dict[str, Any] = {}
    if "resumed_req_ids" in field_names:
        kwargs["resumed_req_ids"] = set()
    if "all_token_ids" in field_names:
        kwargs["all_token_ids"] = {}
    if "num_output_tokens" in field_names:
        kwargs["num_output_tokens"] = {}
    if "resumed_from_preemption" in field_names:
        kwargs["resumed_from_preemption"] = []
    return kwargs


def _schedule_running_requests(
    req_ids: list[int],
    num_computed_tokens: list[int],
    tokens_to_schedule: list[int],
) -> SchedulerOutput:

    cached_reqs = CachedRequestData(req_ids=req_ids,
                                    new_token_ids=[],
                                    new_block_ids=[],
                                    num_computed_tokens=num_computed_tokens,
                                    **_compat_request_data_kwargs())

    num_scheduled_tokens = {}
    total_num_scheduled_tokens = 0
    for req_id, tokens in zip(req_ids, tokens_to_schedule):
        num_scheduled_tokens[req_id] = tokens
        total_num_scheduled_tokens += tokens

    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_reqs,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        **_compat_sched_output_kwargs())


ALL_SLICE = slice(None)


def _assert_block_tables_and_slot_mappings(
        attn_metadata: SpyreAttentionMetadata,
        block_tables: list[list[int]],
        # just the first slot index divided by the block_size,
        # will be expanded until the 64th
        slot_mappings: list[list[int]],
        block_size: int = 64,
        slot_slice: slice = ALL_SLICE) -> None:

    expected_block_table = torch.tensor(block_tables)

    assert torch.equal(attn_metadata.block_table, expected_block_table)

    slot_mapping_tensor_list = []
    for slot_mapping in slot_mappings:
        slot_mapping_tensor = torch.arange(block_size,
                                           dtype=torch.int64).repeat(
                                               len(slot_mapping))
        slot_mapping_tensor += torch.tensor(
            slot_mapping,
            dtype=torch.int64).repeat_interleave(block_size).mul_(block_size)
        slot_mapping_tensor_list.append(slot_mapping_tensor[slot_slice])
    expected_slot_mapping = torch.stack(slot_mapping_tensor_list)

    assert torch.equal(attn_metadata.slot_mapping, expected_slot_mapping)


@pytest.mark.cpu
@pytest.mark.worker
@pytest.mark.prefix_caching
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [512])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [16])
def test_multi_chunk_partial_match_misaligned(
        model: ModelInfo, monkeypatch: pytest.MonkeyPatch, max_num_seqs: int,
        max_model_len: int, max_num_batched_tokens: int,
        available_blocks: Optional[int],
        pc_model_runner: ChunkedPrefillModelRunner):
    """ Scenario where two sequences are scheduled which share a common
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

    # twice the same seed for a sequence of length 384
    # the first sequence shares the same prefix of length 384 tokens
    # the second sequence shares the same prefix of length 254 tokens
    # hence sequence 1 shares the first 254 tokens with sequence 0

    prompt1 = random_prompt(model=model, seed=0, length=384)
    prompt2 = prompt1[0:254] + \
        random_prompt(model=model, seed=0, length=384 - 254)

    request1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=2,
        prompt=prompt1,
        use_golden_token_injection=False,
        block_hasher=pc_model_runner.request_block_hasher)

    request2 = create_request_for_scheduler_test(
        model=model,
        request_id=1,
        add_step=0,
        max_tokens=2,
        prompt=prompt2,
        use_golden_token_injection=False,
        block_hasher=pc_model_runner.request_block_hasher)

    # Schedule chunk 0 of request 0
    step_1_sched_output = _schedule_new_request(request1.request, 128)
    model_runner_output_1 = pc_model_runner.execute_model(step_1_sched_output)
    _assert_block_tables_and_slot_mappings(
        attn_metadata=pc_model_runner.model.last_attn_metadata,
        block_tables=[[1, 2]],
        slot_mappings=[[1, 2]])

    assert model_runner_output_1.req_ids == ['0']
    assert model_runner_output_1.sampled_token_ids == []
    assert model_runner_output_1.tkv == 384
    assert model_runner_output_1.n_free_blocks == 8
    assert model_runner_output_1.left_padding == {'0': 0}

    # Schedule chunk 1 of request 0
    step_2_sched_output = _schedule_running_requests(
        req_ids=['0'],
        num_computed_tokens=[128],
        tokens_to_schedule=[128],
    )
    model_runner_output_2 = pc_model_runner.execute_model(step_2_sched_output)
    _assert_block_tables_and_slot_mappings(
        attn_metadata=pc_model_runner.model.last_attn_metadata,
        block_tables=[[1, 2, 3, 4]],
        slot_mappings=[[3, 4]])

    assert model_runner_output_2.req_ids == ['0']
    assert model_runner_output_2.sampled_token_ids == []
    assert model_runner_output_2.tkv == 384
    assert model_runner_output_2.n_free_blocks == 8
    assert model_runner_output_2.left_padding == {'0': 0}

    # Schedule chunk 2 of request 0
    step_3_sched_output = _schedule_running_requests(
        req_ids=['0'],
        num_computed_tokens=[256],
        tokens_to_schedule=[128],
    )
    model_runner_output_3 = pc_model_runner.execute_model(step_3_sched_output)
    _assert_block_tables_and_slot_mappings(
        attn_metadata=pc_model_runner.model.last_attn_metadata,
        block_tables=[[1, 2, 3, 4, 5, 6]],
        slot_mappings=[[5, 6]])

    assert model_runner_output_3.req_ids == ['0']
    assert len(model_runner_output_3.sampled_token_ids) == 1
    assert model_runner_output_3.tkv == 384
    assert model_runner_output_3.n_free_blocks == 8
    assert model_runner_output_3.left_padding == {'0': 0}

    # Schedule chunk 0 of request 1
    step_4_sched_output = _schedule_new_request(request2.request, 128)
    model_runner_output_4 = pc_model_runner.execute_model(step_4_sched_output)
    # chunk loaded from cache
    assert pc_model_runner.model.last_attn_metadata is None

    assert model_runner_output_4.req_ids == ['1']
    assert model_runner_output_4.sampled_token_ids == []
    assert model_runner_output_4.tkv == 384
    assert model_runner_output_4.n_free_blocks == 1
    assert model_runner_output_4.left_padding == {'1': 0}

    # Schedule chunk 1 of request 1
    step_5_sched_output = _schedule_running_requests(
        req_ids=['1'],
        num_computed_tokens=[128],
        tokens_to_schedule=[128],
    )
    model_runner_output_5 = pc_model_runner.execute_model(step_5_sched_output)
    _assert_block_tables_and_slot_mappings(
        attn_metadata=pc_model_runner.model.last_attn_metadata,
        block_tables=[[1, 2, 3, 7]],
        slot_mappings=[[0, 7]])  # <-- HERE: the block table and slot mapping
    # point to different places when a block is recomputed

    assert model_runner_output_5.req_ids == ['1']
    assert model_runner_output_5.sampled_token_ids == []
    assert model_runner_output_5.tkv == 384
    assert model_runner_output_5.n_free_blocks == 1
    assert model_runner_output_5.left_padding == {'1': 0}

    # Schedule chunk 2 of request 1
    step_6_sched_output = _schedule_running_requests(
        req_ids=['1'],
        num_computed_tokens=[256],
        tokens_to_schedule=[128],
    )
    model_runner_output_6 = pc_model_runner.execute_model(step_6_sched_output)
    _assert_block_tables_and_slot_mappings(
        attn_metadata=pc_model_runner.model.last_attn_metadata,
        block_tables=[[1, 2, 3, 7, 8, 9]],
        slot_mappings=[[8, 9]])

    assert model_runner_output_6.req_ids == ['1']
    assert len(model_runner_output_6.sampled_token_ids) == 1
    assert model_runner_output_6.tkv == 384
    assert model_runner_output_6.n_free_blocks == 1
    assert model_runner_output_6.left_padding == {'1': 0}

    # Schedule decodes of requests 0 and 1
    step_7_sched_output = _schedule_running_requests(
        req_ids=['1', '0'],
        num_computed_tokens=[384, 384],
        tokens_to_schedule=[1, 1],
    )
    model_runner_output_7 = pc_model_runner.execute_model(step_7_sched_output)
    _assert_block_tables_and_slot_mappings(
        attn_metadata=pc_model_runner.model.last_attn_metadata,
        block_tables=[[1, 2, 3, 4, 5, 6, 10], [1, 2, 3, 7, 8, 9, 11]],
        slot_mappings=[[10], [11]],
        slot_slice=slice(0, 1))

    assert model_runner_output_7.req_ids == ['0', '1']
    assert len(model_runner_output_7.sampled_token_ids) == 2
    assert model_runner_output_7.tkv == 385
    assert model_runner_output_7.n_free_blocks == 1
    assert model_runner_output_7.left_padding == {'0': 0, '1': 0}

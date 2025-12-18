from typing import Optional

import pytest
import torch
from scheduling_utils import create_request_for_scheduler_test, random_prompt
from spyre_util import ModelInfo
from vllm import EngineArgs
from vllm.config import VllmConfig
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

import vllm_spyre.envs as envs_spyre
from vllm_spyre.v1.sample.golden_token_injector import GoldenTokenInjector
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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        masks: torch.Tensor,
        is_prompt: bool,
    ) -> torch.Tensor:

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


@pytest.fixture
def pc_model_runner(
        model: ModelInfo, max_num_seqs: int, max_model_len: int,
        max_num_batched_tokens: int,
        available_blocks: Optional[int]) -> ChunkedPrefillModelRunner:
    """A fixture that returns a model runner configured for prefix caching."""

    engine_args = EngineArgs(model=model.name,
                             tokenizer=model.name,
                             revision=model.revision,
                             tokenizer_revision=model.revision,
                             max_model_len=max_model_len,
                             max_num_seqs=max_num_seqs,
                             num_gpu_blocks_override=available_blocks,
                             logits_processors=[GoldenTokenInjector],
                             max_num_batched_tokens=max_num_batched_tokens,
                             enable_prefix_caching=True)
    vllm_config = engine_args.create_engine_config()

    mock_model = MockSpyreCausalLM(vllm_config=vllm_config)

    model_runner = ChunkedPrefillModelRunner(
        vllm_config=vllm_config,
        is_driver_worker=True,
        rank=0,
    )

    model_runner.model = mock_model

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

    num_cached_tokens = pc_model_runner._maybe_load_prefix_from_cache(
        request1.request)
    assert num_cached_tokens == 0

    kv_cache_manager = pc_model_runner.kv_cache_manager

    kv_cache_manager.allocate_new_blocks(request1.request.request_id, 192)
    kv_cache_manager.cache_blocks(request1.request, 192)
    kv_cache_manager.free(request1.request.request_id)

    num_cached_tokens = pc_model_runner._maybe_load_prefix_from_cache(
        request2.request)

    # it's 64 because only the first chunk has 1 block of padding
    assert num_cached_tokens == 64

    num_shared_blocks = kv_cache_manager.num_cached_block.get(
        request2.request.request_id, 0)

    # it's 3 because we touch all blocks due to the deduplication
    # of blocks in the last chunk
    assert num_shared_blocks == 3

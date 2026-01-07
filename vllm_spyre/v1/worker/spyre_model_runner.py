import math
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from logging import DEBUG
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Union, cast

import numpy
import torch
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import ClassifierPooler, Pooler
from vllm.sampling_params import SamplingType

try:
    # pre 0.11.1 compatibility
    from vllm.utils import get_hash_fn_by_name, is_pin_memory_available
except ImportError:
    from vllm.utils.platform_utils import is_pin_memory_available
    from vllm.utils.hashing import get_hash_fn_by_name

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock, get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.request import Request
from vllm.v1.sample.logits_processor import build_logitsprocs

import vllm_spyre.envs as envs_spyre
import vllm_spyre.utils as utils_spyre
from vllm_spyre.compat_utils import dataclass_fields, has_argument
from vllm_spyre.model_executor.model_loader.spyre import (
    BACKEND_LIST,
    SpyreAttentionMetadata,
    SpyreCausalLM,
)
from vllm_spyre.platform import SpyrePlatform
from vllm_spyre.utils import exact_div
from vllm_spyre.v1.sample.spyre_logits_processor import build_logitsprocs_for_cb

# yapf conflicts with ruff for this block
# yapf: disable
from vllm_spyre.v1.worker.spyre_input_batch import (BaseInputBatch,
                                                    BaseRequestState,
                                                    ChunkedPrefillRequestState,
                                                    PoolingInputBatch,
                                                    PoolingRequestState,
                                                    SamplingInputBatch,
                                                    SamplingRequestState)

# yapf: enable
if TYPE_CHECKING:
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
    from vllm.v1.sample.metadata import SamplingMetadata
else:
    SchedulerOutput = None
    NewRequestData = None
    SamplingMetadata = None

from vllm.tasks import SupportedTask
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

logger = init_logger(__name__)


@dataclass(frozen=True)
class ModelForwardInputs:
    input_tokens: torch.Tensor | None = None
    input_positions: torch.Tensor | None = None
    input_masks: torch.Tensor | None = None
    is_prompt: bool = False


@dataclass(frozen=True)
class PoolingForwardInputs(ModelForwardInputs):
    token_type_ids: torch.Tensor | None = None


@dataclass(frozen=True)
class SamplingForwardInputs(ModelForwardInputs):
    """
    Used by the SpyreModelRunner.
    """

    current_tkv_mask: torch.Tensor | None = None
    left_padded_prompt_mask: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    scale_indices: list[int] = field(default_factory=list)


@dataclass
class CBSpyreModelRunnerOutput(ModelRunnerOutput):
    # Add the current tkv and the number of free blocks to the output
    tkv: int = 0
    n_free_blocks: int = 0


@dataclass
class CPSpyreModelRunnerOutput(CBSpyreModelRunnerOutput):
    # Left padding of each request that was calculated by model runner
    left_padding: dict[str, int] = field(default_factory=dict)
    # Percentage of kv cache blocks allocated from our internal kv cache
    # management
    kv_cache_usage: float = 0.0
    # Prefix cache stats, set whenever prefills are happening
    prefix_cache_stats: PrefixCacheStats | None = None
    # In the case of prefix caching, we may have a much larger cached prefix
    # available than the number of scheduled tokens. In that case, the scheduler
    # needs to update its state to reflect the correct number of computed tokens
    prefix_cache_hit_len: dict[str, int] = field(default_factory=dict)


InputBatchT = TypeVar("InputBatchT", bound=BaseInputBatch)
RequestStateT = TypeVar("RequestStateT", bound=BaseRequestState)
ModelInputsT = TypeVar("ModelInputsT", bound=ModelForwardInputs)


class BaseSpyreModelRunner(ABC, Generic[InputBatchT, RequestStateT, ModelInputsT]):
    def __init__(
        self,
        vllm_config: VllmConfig,
        is_driver_worker: bool,
        rank: int,
    ):
        self.is_driver_worker = is_driver_worker
        self.rank = rank
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        self.pad_token_id = 0

        if self.model_config is not None:
            if self.model_config.hf_config is not None:
                self.pad_token_id = getattr(self.model_config.hf_config, "pad_token_id", None) or 0
            if self.model_config.get_sliding_window():
                logger.warning(
                    "Sliding window is not supported on Spyre. "
                    "The model will run without sliding window."
                )
            assert self.cache_config.block_size == self.model_config.max_model_len, (
                "cache_config.block_size must be set to model_config."
                "max_model_len to disable any paged attention ops in the base "
                "scheduler."
            )
        if vllm_config.device_config is None:
            self.device_config = DeviceConfig()
        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        # Lazy initialization: after load_model.
        self.model: nn.Module

        # Flag to be turned off after warmup is complete
        self.warmup_mode = True

        # Batch state
        self.input_batch = self.build_input_batch()

        # Requests
        self.requests: dict[str, RequestStateT] = {}

    @abstractmethod
    def build_input_batch(self) -> InputBatchT:
        raise NotImplementedError

    def get_model(self) -> nn.Module:
        return self.model

    @abstractmethod
    def load_model(self, prompt_lens: Iterable[int], num_decode_tokens: Iterable[int]) -> None:
        raise NotImplementedError

    def _prepare_pad_input_ids(
        self,
        input_ids_list: list[torch.Tensor],
        min_pad_length: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """left side padding implemented as
        in fms.utils.generation.pad_input_id"""
        max_len = max([min_pad_length] + [seq.size(0) for seq in input_ids_list])
        padded_input_ids_list = []
        mask_list = []
        position_ids_list = []
        for input_ids_i in input_ids_list:
            seq_len = input_ids_i.size(0)
            if max_len > seq_len:
                logger.info(
                    "Left padding request of length %d tokens to %d tokens.", seq_len, max_len
                )
            pads = (
                torch.ones(max_len - seq_len, dtype=torch.long, device=input_ids_i.device)
                * self.pad_token_id
            )
            non_pads = torch.ones(seq_len, dtype=torch.long, device=input_ids_i.device)

            pos_ids_seq = torch.arange(0, seq_len, dtype=torch.long, device=input_ids_i.device)

            # Setting this to 0, however if 0 is the eos, we will end up
            # truncating the output if using truncate_after_eos once this
            # workflow works for nested tensor, this can probably be removed
            padded_input_ids_list.append(torch.cat((pads, input_ids_i)))
            mask_list.append(torch.cat((torch.zeros_like(pads), non_pads)))
            position_ids_list.append(torch.cat((torch.zeros_like(pads), pos_ids_seq)))

        return padded_input_ids_list, mask_list, position_ids_list

    def get_kv_cache_spec(self) -> KVCacheSpec:
        """
        This method should generate the KVCache spec by parsing the kv cache
        format from each Attention module in the static forward context.

        In vLLM, this static forward context is populated by the base Attention
        class in the modeling code. Every attention layer populates an entry
        for itself in vllm_config.compilation_config.static_forward_context,
        which is a dictionary of layer_name -> layer for every attention layer.
        This allows the model runner to correctly create the kv cache spec for
        each layer.

        The spyre modeling code currently comes from `fms`, and does not
        integrate with vLLM's modeling classes, so we don't have access to any
        model-agnostic metadata about the attention layers. This just returns a
        dummy value for now.
        """
        # We do at least use the real size from the cache config.
        block_size = self.vllm_config.cache_config.block_size

        if "use_mla" in dataclass_fields(FullAttentionSpec):
            ## Temporary backwards compatibility for 0.10.2
            kwargs = {"use_mla": False}
        else:
            kwargs = {}

        attn_spec = FullAttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float16, **kwargs
        )
        return {"foo": attn_spec}

    def complete_warmup(self):
        """Turn off warmup mode once the warmup is complete"""
        self.warmup_mode = False

    def build_attn_metadata(self, _: ModelInputsT) -> SpyreAttentionMetadata:
        # TODO: probably sooner we will need a more sophisticated way to switch
        # build attention metadata based on model/attention. But for now, a
        # simple method override is good enough.
        return SpyreAttentionMetadata()

    @abstractmethod
    def update_states(self, scheduler_output: SchedulerOutput):
        raise NotImplementedError

    def _mark_input_tensors(self, model_input: ModelInputsT) -> None:
        """Yoinked from
        https://github.com/foundation-model-stack/aiu-fms-testing-utils/pull/13
        """
        if not self.warmup_mode:
            # Only mark tensors when we're warming up and compiling the graphs
            return

        # To produce like graphs during pre-fill, we mark the prefill
        # batch x seq as static, but relax this for decode for the seq
        if model_input.is_prompt:
            # we always want prefill to be static to produce same-like graph
            torch._dynamo.mark_static(model_input.input_tokens, 0)
            torch._dynamo.mark_static(model_input.input_tokens, 1)
            torch._dynamo.mark_static(model_input.input_masks, 0)
            torch._dynamo.mark_static(model_input.input_masks, 1)
            torch._dynamo.mark_static(model_input.input_masks, 2)
            torch._dynamo.mark_static(model_input.input_positions, 0)
            torch._dynamo.mark_static(model_input.input_positions, 1)
        else:
            # we always want the decode to be dynamic on sequence
            torch._dynamo.mark_dynamic(model_input.input_masks, 2)

    @SpyrePlatform.inference_mode()
    @abstractmethod
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        **kwargs,
    ) -> ModelRunnerOutput:
        raise NotImplementedError

    def _make_compatible_sampled_token_ids(
        self, sampled_token_ids: torch.Tensor
    ) -> list[list[int]] | list[numpy.ndarray]:
        """Some versions of vllm required a list of numpy arrays as output.
        This was ultimately rejected, see:
        https://github.com/vllm-project/vllm/pull/29121

        This can be removed once the *lower bound* of the vllm dependency is
        >= 0.12.0
        """
        if ModelRunnerOutput.__dataclass_fields__["sampled_token_ids"].type == list[numpy.ndarray]:
            sampled_token_ids = [x for x in sampled_token_ids.numpy()]
        else:
            sampled_token_ids = sampled_token_ids.tolist()
        return sampled_token_ids


class SpyreModelRunner(
    BaseSpyreModelRunner[SamplingInputBatch, SamplingRequestState, SamplingForwardInputs]
):
    def __init__(self, vllm_config: VllmConfig, is_driver_worker: bool, rank: int):
        super().__init__(vllm_config=vllm_config, is_driver_worker=is_driver_worker, rank=rank)

    def load_model(self, prompt_lens: Iterable[int], num_decode_tokens: Iterable[int]) -> None:
        max_pad_length = max(prompt_lens)
        max_decode_length = max(num_decode_tokens)
        self.model = SpyreCausalLM(
            vllm_config=self.vllm_config,
            max_prompt_length=max_pad_length,
            max_decode_length=max_decode_length,
            rank=self.rank,
        )

    def build_input_batch(self) -> SamplingInputBatch:
        # Define logits processors.

        custom_logitsprocs = self.vllm_config.model_config.logits_processors
        logits_processors = build_logitsprocs(
            vllm_config=self.vllm_config,
            device=self.device,
            is_pin_memory=self.pin_memory,
            is_pooling_model=False,
            custom_logitsprocs=custom_logitsprocs,
        )

        return SamplingInputBatch(
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.model_config.max_model_len,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            logitsprocs=logits_processors,
        )

    @property
    def vocab_size(self) -> int:
        return self.model.model.model.config.src_vocab_size

    def pad_input_ids(
        self,
        input_ids_list: list[torch.Tensor],
        min_pad_length: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padded_input_ids_list, mask_list, position_ids_list = self._prepare_pad_input_ids(
            input_ids_list, min_pad_length
        )

        input_ids = torch.stack(padded_input_ids_list)
        mask = torch.stack(mask_list).bool()
        # this is a causal mask for generation
        mask = (mask.unsqueeze(-1) == mask.unsqueeze(-2)).tril()
        mask = torch.where(mask.logical_not(), -torch.inf, 0.0)

        mask = mask.to(self.model.get_mask_dtype())
        position_ids = torch.stack(position_ids_list)

        return input_ids, position_ids, mask

    def get_sampling_metadata(self, _: bool) -> SamplingMetadata:
        return self.input_batch.sampling_metadata

    def get_req_id_to_index(self, _: bool) -> dict[str, int]:
        return self.input_batch.get_unpadded_output_indices()

    def no_prompt_logprob(self, _: bool) -> bool:
        return self.input_batch.no_prompt_logprob

    def get_num_prompt_logprobs(self) -> dict[str, int]:
        return self.input_batch.num_prompt_logprobs

    def update_states(self, scheduler_output: SchedulerOutput):
        # Update the states of the running/resumed requests.
        # Update input_batch's `token_ids_cpu`,
        # `num_tokens`. For continuous batching it cleans
        # finished requests from the batch
        #
        # NOTE: req_state.output_token_ids will be mutated when
        # PP will be enabled in the future
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state: SamplingRequestState = self.requests[req_id]

            # Update the cached states.
            num_computed_tokens = req_data.num_computed_tokens[i]
            req_state.num_computed_tokens = num_computed_tokens
            # The scheduler will send the sampled tokens back
            # when PP will be enabled in the future
            new_token_ids = req_data.new_token_ids[i] if len(req_data.new_token_ids) > 0 else []
            # Add the sampled token(s) from the previous step (if any).
            # This doesn't include "unverified" tokens like spec decode tokens.
            num_new_tokens = num_computed_tokens + len(new_token_ids) - req_state.num_tokens
            if num_new_tokens == 1:
                # Avoid slicing list in most common case.
                req_state.output_token_ids.append(new_token_ids[-1])
            elif num_new_tokens > 0:
                req_state.output_token_ids.extend(new_token_ids[-num_new_tokens:])

            req_index = self.input_batch.get_req_index(req_id)
            # Add new_token_ids to token_ids_cpu.
            # TODO: Update for spec decoding in the future
            start_token_index = num_computed_tokens
            end_token_index = num_computed_tokens + len(new_token_ids)
            self.input_batch.token_ids_cpu[req_index, start_token_index:end_token_index] = (
                new_token_ids
            )
            # Remove the entry for prompt_logprobs for this request,
            # if it exists
            self.input_batch.num_prompt_logprobs.pop(req_id, None)

        if scheduler_output.finished_req_ids:
            for req_id in scheduler_output.finished_req_ids:
                self.input_batch.remove_request(req_id)
                self.requests.pop(req_id, None)
                # TODO: Processing multiple removals at once can break alignment
                # of logitprocs. Refactor so that we can batch removals to the
                # `input_batch`
                self.input_batch.refresh_metadata()
        else:
            # Due to logits processor we need to refresh metadata at each step
            self.input_batch.refresh_metadata()

    def _get_prompt_logprobs_dict(
        self,
        logits: torch.Tensor,
        model_inputs: SamplingForwardInputs,
    ) -> dict[str, LogprobsTensors | None]:
        """Calculate prompt logprobs from hidden states.

        This currently only supports static batching, batch size 1
        """
        assert model_inputs.is_prompt is not None
        if self.no_prompt_logprob(model_inputs.is_prompt):
            return {}

        num_prompt_logprobs_dict = self.get_num_prompt_logprobs()

        # TODO: For chunked prefill, this will need to be updated to hold state
        # for prompt logprobs across multiple model iterations.
        # This assumes no chunked prefill for now
        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            logger.debug("Calculating prompt_logprobs for request %s", req_id)

            # Get metadata for this request.
            request = self.requests[req_id]
            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True
            )

            # No chunked prefill, so we always start at index 0, token 1.
            # (First token has no logprobs because there's no context)
            start_tok = 1
            num_logits = num_prompt_tokens - start_tok

            # Get the logits corresponding to this req's prompt tokens.
            req_idx = self.get_req_id_to_index(model_inputs.is_prompt)[req_id]
            logits = logits[req_idx]
            # The offset needs to account for the left padding that static
            # batching applies.
            # TODO: To support continuous batching the offset needs to be
            # calculated differently.
            offset = logits.shape[0] - num_prompt_tokens
            logits = logits[offset : offset + num_logits]

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.model.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks = self.model.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids
            )

            # To support chunked prefill, we will need to copy the chunks into
            # saved state at each iteration.
            # For now, we can just return the full tensors.
            logprobs_tensors = LogprobsTensors(
                logprob_token_ids=token_ids, logprobs=logprobs, selected_token_ranks=ranks
            )
            prompt_logprobs_dict[req_id] = logprobs_tensors

        return prompt_logprobs_dict

    def _prepare_prompt(self, _: list[NewRequestData]) -> SamplingForwardInputs:
        raise NotImplementedError

    def _prepare_decode(self, _: CachedRequestData) -> SamplingForwardInputs:
        raise NotImplementedError

    def prepare_model_input(self, scheduler_output: SchedulerOutput) -> SamplingForwardInputs:
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes. Also assuming that new sequences are prefills
        is_prompt = len(scheduler_output.scheduled_new_reqs) > 0

        # Prepare input tensors.
        if is_prompt:
            # Assert no running requests
            assert len(scheduler_output.scheduled_cached_reqs.req_ids) == 0

            return self._prepare_prompt(scheduler_output.scheduled_new_reqs)
        else:
            return self._prepare_decode(scheduler_output.scheduled_cached_reqs)

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        # vLLM models have methods available like `is_text_generation_model`
        # We don't use vLLM modeling code though :(
        # Default: assume text generation supported.
        # TODO: Actually detect what the model supports
        tasks.append("generate")

        return tuple(tasks)

    @SpyrePlatform.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        **kwargs,
    ) -> ModelRunnerOutput:
        t0 = time.time()

        self.update_states(scheduler_output)

        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOutput if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT

        model_input = self.prepare_model_input(scheduler_output)

        # Execute the model
        attn_metadata = self.build_attn_metadata(model_input)
        with set_forward_context(attn_metadata, self.vllm_config):
            logits = self.model(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                masks=model_input.input_masks,
                is_prompt=model_input.is_prompt,
            )

        is_prefill = cast(bool, model_input.is_prompt)

        # Sample the next token.
        output: SamplerOutput = self.model.sample(
            logits=logits,
            sampling_metadata=self.get_sampling_metadata(is_prefill),
        )
        t1 = time.time() - t0

        assert model_input.input_tokens is not None  # satisfy mypy
        batch_size = model_input.input_tokens.shape[0]
        step_type = "[prefill]" if is_prefill else "[decode]"
        logger.debug("t_token: %.2fms %s[batch size %d]", (t1 * 1000), step_type, batch_size)

        # Get mapping between requests ids to the index within the batch
        req_id_to_index = self.get_req_id_to_index(is_prefill)

        # Add the sampled token(s) to the request cache
        req_ids = (
            scheduler_output.scheduled_new_reqs
            if is_prefill
            else self.input_batch.sorted_requests_ids
        )
        sampled_ids = output.sampled_token_ids.tolist()
        for i, req in enumerate(req_ids):
            req_state = (
                self.requests[req.req_id] if not isinstance(req, str) else self.requests[req]
            )
            req_state.output_token_ids.extend(sampled_ids[i])

        prompt_logprobs_dicts = self._get_prompt_logprobs_dict(
            logits=logits, model_inputs=model_input
        )

        # Only return outputs from the driver worker
        if not self.is_driver_worker:
            return EMPTY_MODEL_RUNNER_OUTPUT

        sampled_token_ids = self._make_compatible_sampled_token_ids(output.sampled_token_ids)

        model_output = ModelRunnerOutput(
            req_ids=list(req_id_to_index.keys()),
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            logprobs=(output.logprobs_tensors.tolists() if output.logprobs_tensors else None),
            prompt_logprobs_dict=prompt_logprobs_dicts,
            pooler_output=[],
        )

        return model_output


class WarmupShapesMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vllm_config: VllmConfig = kwargs["vllm_config"]
        self.spyre_warmup_shapes = SpyrePlatform.get_warmup_shapes(vllm_config.scheduler_config)

    def _get_padded_batch_size(self, new_requests: list[NewRequestData]):
        # find warmup shape to be used for padding and batching
        applicable_spyre_warmup_shapes = [
            shape for shape in self.spyre_warmup_shapes if len(new_requests) <= shape["batch_size"]
        ]
        for request_data in new_requests:
            # retrieve initial (unpadded) tokens
            prompt_tokens = request_data.prompt_token_ids
            new_tokens = (
                request_data.sampling_params.max_tokens
                if request_data.sampling_params is not None
                else 0
            )

            updated_spyre_warmup_shapes = [
                shape
                for shape in applicable_spyre_warmup_shapes
                if len(prompt_tokens) <= shape["prompt_length"]
                and new_tokens <= shape["new_tokens"]
            ]
            applicable_spyre_warmup_shapes = updated_spyre_warmup_shapes

        assert applicable_spyre_warmup_shapes, (
            "No shapes available to run prefill batch. (This should not happen)"
        )

        # If multiple warmup shapes apply, the first one is selected.
        # For improving performance, the warmup shapes in scheduler_config
        # are ordered by "processing speed".
        min_pad_length_batch = applicable_spyre_warmup_shapes[0]["prompt_length"]
        padded_batch_size = applicable_spyre_warmup_shapes[0]["batch_size"]
        return padded_batch_size, min_pad_length_batch


class StaticBatchingSpyreModelRunner(WarmupShapesMixin, SpyreModelRunner):
    def __init__(
        self,
        vllm_config: VllmConfig,
        is_driver_worker: bool,
        rank: int,
    ):
        super().__init__(vllm_config=vllm_config, is_driver_worker=is_driver_worker, rank=rank)

        # position_ids of all the sequences in current batch
        self._position_ids: torch.Tensor = None
        # attention masks of all the sequences in current batch
        self._mask: torch.Tensor = None

    def _prepare_prompt(
        self,
        new_requests: list[NewRequestData],
    ) -> SamplingForwardInputs:
        assert len(new_requests) > 0
        input_token_list: list[torch.Tensor] = []
        padded_batch_size, min_pad_length_batch = self._get_padded_batch_size(new_requests)

        # Internal state is reset here.
        # We don't support continuous batching, so we know all previous requests
        # have finished decoding.
        self.input_batch.clear_requests()
        self.requests = {}

        # Build batch and prepare input_token1
        for request_data in new_requests:
            # retrieve initial (unpadded) tokens
            prompt_tokens = request_data.prompt_token_ids

            input_token_list.append(
                torch.tensor(prompt_tokens, dtype=torch.long, device=torch.device("cpu"))
            )

            # Add new requests to the cached states.
            req_id = request_data.req_id
            sampling_params = request_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            req_state = SamplingRequestState(
                req_id=req_id,
                prompt_token_ids=request_data.prompt_token_ids,
                sampling_params=sampling_params,
                generator=generator,
                output_token_ids=[],
                left_padding=0,
            )
            self.requests[req_id] = req_state
            self.input_batch.add_request(req_state)

        self.input_batch.padded_batch_size = padded_batch_size

        # Refresh sampling metadata after all request are added to the batch
        self.input_batch.refresh_metadata()

        # padding to compiled batch size
        while len(input_token_list) < padded_batch_size:
            input_token_list.append(
                torch.zeros(min_pad_length_batch, dtype=torch.long, device=torch.device("cpu"))
            )

        # get position ids and attention mask
        input_tokens, self._position_ids, self._mask = self.pad_input_ids(
            input_token_list, min_pad_length=min_pad_length_batch
        )

        model_input = SamplingForwardInputs(
            input_tokens=input_tokens,
            input_positions=self._position_ids,
            input_masks=self._mask,
            is_prompt=True,
        )

        self._mark_input_tensors(model_input)
        self.model.indices = self.input_batch.get_model_indices()

        return model_input

    def _prepare_decode(
        self,
        cached_request_data: CachedRequestData,
    ) -> SamplingForwardInputs:
        assert len(cached_request_data.req_ids) > 0
        input_tokens: list[list[int]] = [[0] for _ in range(self._position_ids.shape[0])]

        for req_id in cached_request_data.req_ids:
            # TODO: Will this always just be one token ID if there's no spec
            # or jump decoding?
            req_state: SamplingRequestState = self.requests[req_id]
            output_token_ids = req_state.output_token_ids
            generation_token = output_token_ids[-1]
            input_tokens[self.input_batch.req_id_to_index[req_id]] = [generation_token]

        # update position ids and attention mask
        self._update_position_ids()
        self._update_mask()

        input_tokens = torch.tensor(input_tokens, dtype=torch.long, device=self.device)
        model_input = SamplingForwardInputs(
            input_tokens=input_tokens,
            input_positions=self._position_ids,
            input_masks=self._mask,
            is_prompt=False,
        )
        self._mark_input_tensors(model_input)

        # TODO: Added here temporarily until we can remove dummy token
        # for batch_size=1. Once we can do that, we shall move it to
        # execute_model on SpyreModelRunner for both static and CB.
        self.model.indices = self.input_batch.get_model_indices()

        return model_input

    def _update_position_ids(self) -> None:
        """Updating the position ids of all sequences
        in a batch. Will be called in decoding phase"""

        self._position_ids = self._position_ids[:, -1] + 1
        self._position_ids = self._position_ids.unsqueeze(-1)

    def _update_mask(self) -> None:
        """Updating/extending the attention masks of all
        sequences in a batch. Will be called in decoding phase"""

        assert self._mask is not None
        masks = self._mask

        masks_new = []
        for mask in masks:
            # get the last row of the 3d mask
            mask_new = mask[-1:, :]

            # extend the mask one slot
            mask_new = torch.cat(
                (
                    mask_new,
                    torch.zeros(1, 1, dtype=mask_new.dtype, device=mask_new.device),
                ),
                dim=1,
            )
            masks_new.append(mask_new)

        self._mask = torch.stack(masks_new, dim=0)

    def _mark_input_tensors(self, model_input: SamplingForwardInputs) -> None:
        super()._mark_input_tensors(model_input=model_input)
        if not self.warmup_mode:
            # Only mark tensors when we're warming up and compiling the graphs
            return

        if not model_input.is_prompt:
            # here self.model.model is a StaticBatchingFmsModel
            for layer in self.model.model.past_key_value_states:
                for tensor in layer:
                    torch._dynamo.mark_static(tensor, 0)
                    # This used to be baked into the model's forward pass
                    torch._dynamo.mark_dynamic(tensor, 2)


class ContinuousBatchingSpyreModelRunner(SpyreModelRunner):
    def __init__(
        self,
        vllm_config: VllmConfig,
        is_driver_worker: bool,
        rank: int,
    ):
        super().__init__(vllm_config=vllm_config, is_driver_worker=is_driver_worker, rank=rank)

        self.block_size = SpyrePlatform.get_block_size()

        # max number of blocks needed (reserved) per request id
        self.req_ids2num_reserved_blocks: dict[str, int] = {}

        self.tkv: int = 0

        self._enable_prefix_caching = vllm_config.cache_config.enable_prefix_caching

        # TODO: Remove this once we can prefill and decode in the same step
        self.prefill_batch = SamplingInputBatch(
            # TODO: review this, currently we only support prefill for
            # `batch_size=1`
            max_num_reqs=1,
            max_model_len=vllm_config.model_config.max_model_len,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=vllm_config.model_config.get_vocab_size(),
        )

    @property
    def enable_prefix_caching(self):
        return self._enable_prefix_caching and not self.warmup_mode

    def pre_warmup(self) -> None:
        # Set the number of kv cache blocks to the minimal value of 2 which is
        # required for warmup. After the warmup, the number of blocks will be
        # set to the value returned by the Spyre compiler (see complete_warmup)
        # Note: Until this feature is supported by the compiler we have to set:
        # n_blocks_warmup = n_blocks_avail

        n_blocks_warmup = self.get_total_spyre_blocks()
        self._set_blocks(num_blocks=n_blocks_warmup)
        self.model.model.set_past_key_value_states(num_blocks=n_blocks_warmup)

        # Future code:

        # self._set_blocks(num_blocks=2)
        # self.model.model.set_past_key_value_states(num_blocks=2)

        # mark the num_blocks dimension dynamic for Spyre compiler for warmup
        # only, compiler will return the number of blocks it can accommodate.
        # (This is not yet supported by the compiler)
        # for layer in self.model.model.past_key_value_states:
        #     for tensor in layer:
        #         torch._dynamo.mark_dynamic(tensor, 0)

    def complete_warmup(self) -> None:
        super().complete_warmup()
        # get the number or pages from the actual Spyre card after the warmup
        # and set it accordingly in the model runner and for the kv cache size
        n_blocks_avail = self.get_total_spyre_blocks()
        self._set_blocks(num_blocks=n_blocks_avail)
        self.model.model.set_past_key_value_states(num_blocks=n_blocks_avail)

    def _set_blocks(self, num_blocks: int) -> None:
        # set number of available blocks, populate block_pool and
        # create the kv_cache_manager
        self.n_blocks = num_blocks - 1
        self.block_pool = self._make_block_pool()
        self.kv_cache_manager = self._make_kv_cache_manager()

    def _make_block_pool(self) -> BlockPool:
        kwargs = {}
        if has_argument(BlockPool, "hash_block_size"):
            kwargs["hash_block_size"] = self.block_size
        return BlockPool(
            num_gpu_blocks=self.n_blocks + 1,
            enable_caching=self.enable_prefix_caching,
            enable_kv_cache_events=False,
            **kwargs,
        )

    def _make_kv_cache_manager(self) -> FullAttentionManager:
        ## Temporary backwards compatibility for 0.10.2
        if "use_mla" in dataclass_fields(FullAttentionSpec):
            kwargs = {"use_mla": False}
        else:
            kwargs = {}

        self._attn_spec = FullAttentionSpec(
            block_size=self.block_size,
            # dummy values
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float16,
            **kwargs,
        )

        kv_cache_manager = FullAttentionManager(
            kv_cache_spec=self._attn_spec,
            block_pool=self.block_pool,
            # Currently don't support models with more than one
            # attention type, e.g. full and sliding window, so
            # there is only one group.
            kv_cache_group_id=0,
            # We don't support DCP
            # https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/#decode-context-parallel
            dcp_world_size=1,
        )
        return kv_cache_manager

    def _get_blocks(self, request_id: str) -> list[KVCacheBlock]:
        return self.kv_cache_manager.req_to_blocks[request_id]

    def get_total_spyre_blocks(self) -> int:
        """Returns the total number of KV cache blocks available for spyre.
        This currently returns the number of blocks required for a full-sized
        batch, which may be greater than the available memory.

        Until a correct available memory api is available, the number of blocks
        must be overridden with a known good value via
        cache_config.num_gpu_blocks_override
        """
        max_batch_size = self.scheduler_config.max_num_seqs
        max_model_len = self.model_config.max_model_len
        block_size = SpyrePlatform.get_block_size()
        min_req_num_blocks = max_model_len // block_size

        blocks_override = self.cache_config.num_gpu_blocks_override
        if blocks_override is not None and blocks_override > 0:
            num_blocks = blocks_override
        else:
            num_blocks = max_batch_size * min_req_num_blocks

        # Total number of blocks needs to be a multiple of the batch size
        # (spyre constraint) so round it down
        num_blocks = max_batch_size * (num_blocks // max_batch_size)

        if num_blocks < min_req_num_blocks:
            raise ValueError(
                f"Number of pages available on Spyre {num_blocks} is not "
                f"enough to serve the current model (need at least "
                f"{min_req_num_blocks} pages)."
            )

        max_concurrency = num_blocks * block_size / max_model_len
        backend = "Spyre" if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND == "sendnn" else "CPU"
        logger.info("%s KV cache size: %s tokens", backend, num_blocks * block_size)
        logger.info(
            "Maximum concurrency for %s tokens per request: %.2fx",
            str(max_model_len),
            max_concurrency,
        )

        return num_blocks

    def update_states(self, scheduler_output):
        super().update_states(scheduler_output)

        # TODO: move to kv cache manager
        # Continuous batching: free blocks
        for req_id in scheduler_output.finished_req_ids:
            if logger.isEnabledFor(DEBUG) and (blocks_to_free := self._get_blocks(req_id)):
                logger.debug("Freeing request id: %s", req_id)
                for block in blocks_to_free:
                    logger.debug("Freeing block with id: %s", block.block_id)
            self.req_ids2num_reserved_blocks.pop(req_id, None)
            self.kv_cache_manager.free(req_id)

    def _prepare_prompt(
        self,
        new_requests: list[NewRequestData],
    ) -> SamplingForwardInputs:
        # currently all prefills are of batch size 1
        assert len(new_requests) == 1

        request = new_requests[0]
        req_id = request.req_id
        prompt_token_ids = request.prompt_token_ids
        sampling_params = request.sampling_params
        is_new_batch = len(self.req_ids2num_reserved_blocks) == 0
        prompt_len = len(prompt_token_ids)

        # make sure that the current tkv of the decode batch is greater or
        # equal to the prompt length of the new joining sequence
        if not is_new_batch and prompt_len > self.tkv:
            # increasing the current tkv by a multiple of the block size
            tkv_offset = math.ceil((prompt_len - self.tkv) / self.block_size) * self.block_size
            if tkv_offset > 0:
                # Note: drawing explaining this optimization in more detail
                # can be found here (see page 3 in particular):
                # https://github.com/vllm-project/vllm-spyre/pull/340#issuecomment-3179337304
                logger.debug(
                    "Prefill optimization: Adding %d blocks per "
                    "sequence in the decode batch to prefill the current "
                    "sequence.",
                    tkv_offset // self.block_size,
                )
                self.tkv += tkv_offset

                # adding left pads to the requests in the current decode batch
                requests = self.requests.values()
                for req in requests:
                    if req.req_id != req_id:
                        req.left_padding += tkv_offset

        self.prefill_batch.clear_requests()

        # right padding to the next block boundary (ceil division)
        # -> prefills must to be multiples of the block size (Spyre constraint)
        n = prompt_len if is_new_batch else self.tkv
        right_padding_tkv = math.ceil(n / self.block_size) * self.block_size

        # set the tkv to the block padding if starting a new decode batch
        self.tkv = right_padding_tkv if is_new_batch else self.tkv

        # number of left pads to align the prefill sequence with the tkv of the
        # current decode batch (Spyre constraint)
        left_padding = self.tkv - prompt_len

        # optimization: cut out fully padded blocks on the left
        n_pad_blocks = (self.tkv - prompt_len) // self.block_size
        right_padding_tkv -= n_pad_blocks * self.block_size
        left_padding_tkv = self.tkv - n_pad_blocks * self.block_size
        if n_pad_blocks > 0:
            # Note: drawing explaining this optimization in more detail can
            # be found here (see page 2 in particular):
            # https://github.com/vllm-project/vllm-spyre/pull/340#issuecomment-3179337304
            logger.debug("Prefill reduced by %d blocks due to optimization.", n_pad_blocks)

        # Reserve the number of blocks that this new sequence requires in the
        # worst case (it might always stop early by producing the EOS token)
        new_tokens = sampling_params.max_tokens if sampling_params is not None else 0
        n = self.tkv + new_tokens - 1
        # subtract the padding blocks from the reserved blocks
        n_fully_padded_blocks = math.floor((self.tkv - len(prompt_token_ids)) / self.block_size)
        n_reserved_blocks = math.ceil(n / self.block_size) - n_fully_padded_blocks
        self.req_ids2num_reserved_blocks[req_id] = n_reserved_blocks

        # filling block table and slot mapping

        blocks = self.kv_cache_manager.allocate_new_blocks(req_id, right_padding_tkv)

        block_offsets = [block.block_id * self.block_size for block in blocks]
        slot_mapping = torch.arange(self.block_size, dtype=torch.int64).repeat(len(blocks))
        slot_mapping += torch.tensor(block_offsets, dtype=torch.int64).repeat_interleave(
            self.block_size
        )
        slot_mapping.unsqueeze_(0)

        # Add new request to the cached states.
        if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(sampling_params.seed)
        else:
            generator = None

        req_state = SamplingRequestState(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            generator=generator,
            output_token_ids=[],
            left_padding=left_padding,
        )
        self.requests[req_id] = req_state
        prefill_index = self.input_batch.add_request(req_state)
        self.prefill_batch.add_request(req_state)

        # set prefill index for logits processor
        for logitsproc in self.input_batch.logitsprocs_wrappers:
            logitsproc.set_prefill_index(prefill_index)

        # Refresh sampling metadata after all request are added to the batch
        self.input_batch.refresh_metadata()
        self.prefill_batch.refresh_metadata()

        self.model.indices = torch.ones(1, dtype=torch.bool, device="cpu")
        prompt_token_ids_tensor = torch.tensor(
            prompt_token_ids, dtype=torch.long, device=torch.device("cpu")
        )

        # get position ids and attention mask
        # applies left padding to ensure that the tkv of the new sequence
        # tkv_prefill aligns with tkv of current decode batch tkv_decode:
        # tkv_prefill % block_size = tkv_decode % block_size
        # and right padding to align with the next block boundary
        input_tokens, position_ids, mask = self.pad_input_ids(
            [prompt_token_ids_tensor],
            min_pad_left=left_padding_tkv,
            min_pad_right=right_padding_tkv,
        )
        mask = mask.unsqueeze(1).contiguous()

        # not needed for prefill
        current_tkv_mask = None
        # left padding info is stored in CachedRequestState of self.requests
        left_padded_prompt_mask = None
        # block table is only passed for decode
        block_table = None

        model_inputs = SamplingForwardInputs(
            input_tokens=input_tokens,
            input_positions=position_ids,
            input_masks=mask,
            current_tkv_mask=current_tkv_mask,
            left_padded_prompt_mask=left_padded_prompt_mask,
            block_table=block_table,
            slot_mapping=slot_mapping,
            is_prompt=True,
            # used only for quantized model
            scale_indices=[prefill_index],
        )

        self._mark_input_tensors(model_inputs)

        return model_inputs

    def _prepare_decode(
        self,
        cached_request_data: CachedRequestData,
    ) -> SamplingForwardInputs:
        assert len(cached_request_data.req_ids) > 0

        input_tokens = []
        input_positions = []
        block_table = []
        slot_mapping = []
        left_padded_prompt_mask = []

        assert len(self.input_batch.req_id_to_index) == len(cached_request_data.req_ids)
        # TODO(wallas): I think we can do better here, without sorting or
        # creating an intermediary dictionary
        cached_reqs_map = {req_id: i for i, req_id in enumerate(cached_request_data.req_ids)}
        req_ids = self.input_batch.sorted_requests_ids

        n_blocks = 0  # maximal number of blocks used by any seq in the batch
        for req_id in req_ids:
            # adding new blocks if needed
            if self.tkv % self.block_size == 0:
                req_state: SamplingRequestState = self.requests[req_id]
                # we want to allocate the block for 1 next token.
                # So we need to compute the number of tokens that the
                # kv_cache_manager knows about: the number of computed
                # tokens so far plus any intra-block padding.
                total_tokens = (
                    req_state.left_padding % self.block_size + req_state.num_computed_tokens + 1
                )
                blocks = self.kv_cache_manager.allocate_new_blocks(req_id, total_tokens)
                assert len(blocks) == 1
            n_blocks = max(n_blocks, len(self._get_blocks(req_id)))

        for req_id in req_ids:
            # TODO: Will this always just be one token ID if there's no spec
            # or jump decoding?

            req_state = self.requests[req_id]

            # filling block table with padding blocks to make it rectangular
            # Note: the padding block id 0 here is chosen arbitrarily, it can
            # be any allocated block id on the Sypre card (has to be in range
            # [0, self.n_blocks - 1]). Further, it also be a block id that holds
            # actual KV cache for another (or the same) sequence.
            blocks = self._get_blocks(req_id)
            left_padding_blocks = n_blocks - len(blocks)

            req_block_ids = [0] * left_padding_blocks + [block.block_id for block in blocks]
            block_table.append(req_block_ids)

            # slot_mapping for all blocks of sequence
            start_slot = block_table[-1][-1] * self.block_size
            offset = self.tkv % self.block_size
            slot = [start_slot + offset]
            slot_mapping.append(slot)

            # input token and position of the token generated in the last step
            generation_token = req_state.output_token_ids[-1]
            input_tokens.append([generation_token])
            seq_len = cached_request_data.num_computed_tokens[cached_reqs_map[req_id]]
            input_positions.append([seq_len])

            # retrieve left padding information stored during prefill and
            # updated when calling reduce_left_padding()
            left_padded_prompt_mask.append(req_state.left_padding)

        # update tkv
        self.tkv = self.tkv + 1

        # construct tensors from lists
        input_tokens = torch.tensor(input_tokens, dtype=torch.long, device=self.device)
        position_ids = torch.tensor(input_positions, dtype=torch.long, device=self.device)
        current_tkv_mask = torch.tensor([self.tkv] * len(input_tokens), dtype=torch.int64)
        left_padded_prompt_mask = torch.tensor(
            left_padded_prompt_mask, dtype=torch.long, device=self.device
        )
        block_table = torch.tensor(block_table, dtype=torch.int64)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64)
        self.model.indices = torch.ones(
            len(cached_request_data.req_ids), dtype=torch.bool, device="cpu"
        )

        # mask not needed during decode
        mask = None

        model_inputs = SamplingForwardInputs(
            input_tokens=input_tokens,
            input_positions=position_ids,
            input_masks=mask,
            current_tkv_mask=current_tkv_mask,
            left_padded_prompt_mask=left_padded_prompt_mask,
            block_table=block_table,
            slot_mapping=slot_mapping,
            is_prompt=False,
            scale_indices=self.input_batch.request_indices,
        )

        self._mark_input_tensors(model_inputs)

        return model_inputs

    def reduce_left_padding(self) -> None:
        """Optimizes the decode batch by removing entire columns that consist
        solely of left pads. This reduces unnecessary decode computation.

        Note: drawing explaining the optimization in more detail uploaded here:
        https://github.com/vllm-project/vllm-spyre/pull/131#issuecomment-3233440852
        """

        requests = self.requests.values()
        if len(self.requests) == 0:
            return

        min_left_pad = min([r.left_padding for r in requests])
        n_padded_blocks = min_left_pad // self.block_size
        offset = n_padded_blocks * self.block_size

        if offset > 0:
            logger.debug("Reducing left padding by %d blocks", n_padded_blocks)

            for req in requests:
                req.left_padding -= offset

        # update tkv
        self.tkv -= offset

    def pad_input_ids(
        self,
        input_ids_list: list[torch.Tensor],
        min_pad_left: int = 0,
        min_pad_right: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # left padding to align with tkv of current decode batch
        input_tokens_left, position_ids_left, mask_left = super().pad_input_ids(
            input_ids_list, min_pad_length=min_pad_left
        )

        # right padding to align with the next block boundary
        left_pad_len = input_tokens_left.shape[1]
        n_pads_right = min_pad_right - left_pad_len

        # set number of right pads for the next model forward pass:
        # need to be excluded before sampling tokens
        self.model.n_pads_right = n_pads_right

        if n_pads_right > 0:
            # apply right padding to input_tokens, position_ids and mask
            logger.info(
                "Right padding request of length %d tokens to %d tokens.",
                left_pad_len,
                min_pad_right,
            )

            input_tokens_right = torch.tensor(
                [[self.pad_token_id for i in range(n_pads_right)]],
                device=input_tokens_left.device,
                dtype=input_tokens_left.dtype,
            )
            input_tokens = torch.concat((input_tokens_left, input_tokens_right), dim=1)

            # Note: same output with i as padding for position ids
            pos_start = position_ids_left[0][-1] + 1
            position_ids_right = torch.tensor(
                [[0 for i in range(pos_start, pos_start + n_pads_right)]],
                device=position_ids_left.device,
                dtype=position_ids_left.dtype,
            )
            position_ids = torch.concat((position_ids_left, position_ids_right), dim=1)

            # pad left padded mask with -inf to the next block boundary
            mask = torch.nn.functional.pad(
                mask_left, (0, n_pads_right, 0, n_pads_right), value=-torch.inf
            )

            # lower triangle: 0.0, upper triangle -inf
            mask_pads = torch.zeros(n_pads_right, n_pads_right)
            mask_pads[~torch.tril(torch.ones(n_pads_right, n_pads_right)).bool()] = float("-inf")

            # insert triangular matrix for right pads
            mask[:, -n_pads_right:, -n_pads_right:] = mask_pads.unsqueeze(0)
        else:
            # no right padding needed
            input_tokens = input_tokens_left
            position_ids = position_ids_left
            mask = mask_left

        return input_tokens, position_ids, mask

    def build_attn_metadata(self, model_input: SamplingForwardInputs) -> SpyreAttentionMetadata:
        # TODO: probably we can remove some fields of the model input and
        # update only the SpyreAttentionMetadata

        return SpyreAttentionMetadata(
            slot_mapping=model_input.slot_mapping,
            current_tkv_mask=model_input.current_tkv_mask,
            left_padded_prompt_mask=model_input.left_padded_prompt_mask,
            block_table=model_input.block_table,
            scale_indices=torch.tensor(model_input.scale_indices, dtype=torch.int32),
            is_prefill=model_input.is_prompt,
        )

    def get_sampling_metadata(self, is_prefill: bool) -> SamplingMetadata:
        if is_prefill:
            sampling_data = self.prefill_batch.sampling_metadata
            sampling_data.logitsprocs = self.input_batch.logitsprocs
            return sampling_data
        else:
            return self.input_batch.sampling_metadata

    def get_req_id_to_index(self, is_prefill: bool) -> dict[str, int]:
        req_id_to_index = (
            self.prefill_batch.get_unpadded_output_indices()
            if is_prefill
            else self.input_batch.get_unpadded_output_indices()
        )

        return req_id_to_index

    def get_n_free_blocks(self) -> int:
        return self.n_blocks - sum(self.req_ids2num_reserved_blocks.values())

    def no_prompt_logprob(self, is_prefill: bool) -> bool:
        if is_prefill:
            return self.prefill_batch.no_prompt_logprob
        # If we're not running a prefill then this is a decode-only batch
        return True

    def get_num_prompt_logprobs(self) -> dict[str, int]:
        # Prompt logprobs will always be set on the prefill batch
        return self.prefill_batch.num_prompt_logprobs

    def prepare_model_input(self, scheduler_output: SchedulerOutput) -> SamplingForwardInputs:
        # remove left padding if applicable before next prefill/decode step
        self.reduce_left_padding()

        return super().prepare_model_input(scheduler_output)

    @SpyrePlatform.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        **kwargs,
    ) -> ModelRunnerOutput:
        output = super().execute_model(scheduler_output, **kwargs)

        return CBSpyreModelRunnerOutput(
            **asdict(output),
            tkv=self.tkv if scheduler_output.total_num_scheduled_tokens > 0 else 0,
            n_free_blocks=self.get_n_free_blocks(),
        )

    def _mark_input_tensors(self, model_input: SamplingForwardInputs) -> None:
        # Marking dimensions static/dynamic
        if model_input.is_prompt:
            # batch static (batch size 1)
            torch._dynamo.mark_static(model_input.input_tokens, 0)
            torch._dynamo.mark_static(model_input.slot_mapping, 0)
            torch._dynamo.mark_static(model_input.input_positions, 0)
            torch._dynamo.mark_static(model_input.input_masks, 0)

            # sequence dynamic
            torch._dynamo.mark_dynamic(model_input.input_tokens, 1)
            torch._dynamo.mark_dynamic(model_input.slot_mapping, 1)
            torch._dynamo.mark_dynamic(model_input.input_positions, 1)
            torch._dynamo.mark_dynamic(model_input.input_masks, 2)
            torch._dynamo.mark_dynamic(model_input.input_masks, 3)

        # decode
        else:
            # mask is no longer used here

            # batch dynamic
            torch._dynamo.mark_dynamic(model_input.input_tokens, 0)
            torch._dynamo.mark_dynamic(model_input.block_table, 0)
            torch._dynamo.mark_dynamic(model_input.slot_mapping, 0)
            torch._dynamo.mark_dynamic(model_input.input_positions, 0)
            torch._dynamo.mark_dynamic(model_input.current_tkv_mask, 0)
            torch._dynamo.mark_dynamic(model_input.left_padded_prompt_mask, 0)

            # sequence
            torch._dynamo.mark_static(model_input.input_tokens, 1)  # always 1
            torch._dynamo.mark_dynamic(model_input.block_table, 1)
            torch._dynamo.mark_static(model_input.slot_mapping, 1)  # always 1
            torch._dynamo.mark_static(model_input.input_positions, 1)  # always 1

    def build_input_batch(self) -> SamplingInputBatch:
        # Define logits processors.

        custom_logitsprocs = self.vllm_config.model_config.logits_processors

        batch_size = self.scheduler_config.max_num_seqs
        logits_processors = build_logitsprocs_for_cb(
            vllm_config=self.vllm_config,
            device=self.device,
            is_pin_memory=self.pin_memory,
            is_pooling_model=False,
            custom_logitsprocs=custom_logitsprocs,
            batch_size=batch_size,
        )

        return SamplingInputBatch(
            max_num_reqs=batch_size,
            max_model_len=self.model_config.max_model_len,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            logitsprocs=logits_processors,
        )


class PoolerAdapter(torch.nn.Module):
    def __init__(self, pooler: torch.nn.Module):
        super().__init__()
        self.pooler = pooler

    def forward(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        # Because we're using transformers to load the pooler
        # and classifier layers and the assumption there is that
        # we have a right padded batch, we need to split
        # and at the batch dimension.
        if isinstance(hidden_states, torch.Tensor):
            hidden_states = torch.split(hidden_states, pooling_metadata.prompt_lens.tolist())
        return [self.pooler(h.unsqueeze(dim=0)) for h in hidden_states]


def _cls(input: torch.Tensor) -> torch.Tensor:
    return input[:, 0]


class SpyrePoolingModelRunner(
    WarmupShapesMixin,
    BaseSpyreModelRunner[PoolingInputBatch, PoolingRequestState, PoolingForwardInputs],
):
    def __init__(
        self,
        vllm_config: VllmConfig,
        is_driver_worker: bool,
        rank: int,
    ):
        super().__init__(vllm_config=vllm_config, is_driver_worker=is_driver_worker, rank=rank)

        # position_ids of all the sequences in current batch
        self._position_ids: torch.Tensor = None
        self.use_token_type_ids = False

    def build_input_batch(self) -> PoolingInputBatch:
        return PoolingInputBatch(
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.model_config.max_model_len,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
        )

    def load_model(self, prompt_lens: Iterable[int], num_decode_tokens: Iterable[int]) -> None:
        task = self.model_config.task
        if task is None:
            # Task is being deprecated upstream because the models
            # support several tasks at once. But for now, here we need
            # to know the task to load the model with
            # AutoModelForSequenceClassification
            task = self.model_config._get_default_pooling_task(self.model_config.architectures)

        if task == "embed":
            self.model = AutoModel.from_pretrained(self.model_config.model)
        elif task == "classify":
            class_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_config.model
            )
            if hasattr(class_model, "bert"):
                self.model = class_model.bert
                self._pooler = PoolerAdapter(self.model.pooler)
            elif hasattr(class_model, "roberta"):
                self.model = class_model.roberta
                self._pooler = PoolerAdapter(_cls)
            else:
                raise ValueError(
                    f"Unsupported model {self.model_config.model}: Expected "
                    "Bert or Roberta for sequence classification"
                )
            self.classifier = class_model.classifier
        else:
            raise ValueError(f"Unsupported task {task}")

        # Disable pooler because in transformers it's
        # always run even tough we don't use the outputs
        # directly.
        self.model.pooler = None

        model_class_name = type(self.model).__name__
        self.is_roberta = "roberta" in model_class_name.lower()

        self.model.eval()
        torch.set_grad_enabled(False)
        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND in BACKEND_LIST:
            # Lazy import to avoid load torch_sendnn runtime before it is really
            # necessary. This solve issues of running forked tests that share
            # some resources from parent to children which can have problems
            # of caching even though the test run in isolated subprocesses.

            if SpyrePlatform.sendnn_configured():
                from torch_sendnn import torch_sendnn  # noqa: F401

            with utils_spyre.stagger_region(
                envs_spyre.VLLM_SPYRE_MAX_LOAD_PROCESSES, self.parallel_config.world_size, self.rank
            ):
                self.model = torch.compile(
                    self.model,
                    mode="default",
                    dynamic=False,
                    backend=envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND,
                )

        if task == "classify":
            tokenizer = AutoTokenizer.from_pretrained(self.model_config.model)
            output = tokenizer(text="foo", text_pair="bar")
            self.use_token_type_ids = "token_type_ids" in output
            if self.use_token_type_ids:
                self.sep_token_id = tokenizer.sep_token_id

        pooler_config = self.model_config.pooler_config

        if task == "embed":
            with set_current_vllm_config(self.vllm_config):
                self.pooler = Pooler.for_embed(pooler_config=pooler_config)
        elif task == "classify":
            with set_current_vllm_config(self.vllm_config):
                self.pooler = ClassifierPooler(
                    pooling=self._pooler,
                    classifier=self.classifier,
                    act_fn=ClassifierPooler.act_fn_for_cross_encoder(self.model_config),
                )

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    def pad_input_ids(
        self,
        input_ids_list: list[torch.Tensor],
        min_pad_length: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padded_input_ids_list, mask_list, position_ids_list = self._prepare_pad_input_ids(
            input_ids_list, min_pad_length
        )

        input_ids = torch.stack(padded_input_ids_list)
        mask = torch.stack(mask_list)
        position_ids = torch.stack(position_ids_list)

        return input_ids, position_ids, mask

    def update_states(self, scheduler_output: SchedulerOutput):
        assert len(scheduler_output.scheduled_cached_reqs.req_ids) == 0

        if scheduler_output.finished_req_ids:
            for req_id in scheduler_output.finished_req_ids:
                self.input_batch.remove_request(req_id)
                self.requests.pop(req_id, None)

    def _uncompress_token_types(self) -> list[list[int]]:
        pooling_metadata = self.input_batch.make_pooling_metadata()
        pooling_params = pooling_metadata.pooling_params

        token_type_id_requests = dict[int, Any]()
        for i, param in enumerate(pooling_params):
            if (
                param.extra_kwargs is not None
                and (token_types := param.extra_kwargs.get("compressed_token_type_ids")) is not None
            ):
                token_type_id_requests[i] = token_types

        if len(token_type_id_requests) == 0:
            return []

        seq_lens = pooling_metadata.prompt_lens
        token_type_ids = []

        for i, seq_len in enumerate(seq_lens):
            pos = token_type_id_requests.get(i, seq_len)
            ids = (torch.arange(seq_lens[i]) >= pos).int()
            token_type_ids.append(ids)

        return token_type_ids

    def _token_types(self, input_ids):
        if token_type_ids_lst := self._uncompress_token_types():
            token_type_ids = torch.zeros_like(input_ids)
            for i, token_types in enumerate(token_type_ids_lst):
                token_type_ids[i, -len(token_types) :] = token_types
            return token_type_ids
        else:
            locs = torch.where(input_ids == self.sep_token_id, 1, 0)
            return locs.cumsum(dim=1) - locs

    def _prepare_prompt(
        self,
        new_requests: list[NewRequestData],
    ) -> PoolingForwardInputs:
        assert len(new_requests) > 0
        input_token_list: list[torch.Tensor] = []
        padded_batch_size, min_pad_length_batch = self._get_padded_batch_size(new_requests)

        # Internal state is reset here.
        # We don't support continuous batching, so we know all previous requests
        # have finished decoding.
        self.input_batch.clear_requests()
        self.requests = {}

        # Build batch and prepare input_token1
        for request_data in new_requests:
            # retrieve initial (unpadded) tokens
            prompt_tokens = request_data.prompt_token_ids

            input_token_list.append(
                torch.tensor(prompt_tokens, dtype=torch.long, device=torch.device("cpu"))
            )

            # Add new requests to the cached states.
            req_id = request_data.req_id
            pooling_params = request_data.pooling_params
            assert pooling_params is not None

            req_state = PoolingRequestState(
                req_id=req_id,
                prompt_token_ids=request_data.prompt_token_ids,
                pooling_params=pooling_params,
            )
            self.requests[req_id] = req_state
            self.input_batch.add_request(req_state)

        self.input_batch.padded_batch_size = padded_batch_size

        # padding to compiled batch size
        while len(input_token_list) < padded_batch_size:
            input_token_list.append(
                torch.zeros(min_pad_length_batch, dtype=torch.long, device=torch.device("cpu"))
            )

        # get position ids and attention mask
        input_tokens, position_ids, mask = self.pad_input_ids(
            input_token_list, min_pad_length=min_pad_length_batch
        )

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = self._token_types(input_tokens)

        if self.is_roberta:
            position_ids += self.pad_token_id + 1
            position_ids *= mask

        model_input = PoolingForwardInputs(
            input_tokens=input_tokens,
            input_positions=position_ids,
            input_masks=mask,
            is_prompt=True,
            token_type_ids=token_type_ids,
        )

        self._mark_input_tensors(model_input)

        return model_input

    def prepare_model_input(self, scheduler_output: SchedulerOutput) -> PoolingForwardInputs:
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        # Also assuming that new sequences are prefills
        is_prompt = len(scheduler_output.scheduled_new_reqs) > 0

        # Prepare input tensors.
        assert is_prompt
        # Assert no running requests
        assert len(scheduler_output.scheduled_cached_reqs.req_ids) == 0
        return self._prepare_prompt(scheduler_output.scheduled_new_reqs)

    def _mark_input_tensors(self, model_input: PoolingForwardInputs) -> None:
        super()._mark_input_tensors(model_input=model_input)
        if not self.warmup_mode:
            # Only mark tensors when we're warming up and compiling the graphs
            return
        if self.use_token_type_ids:
            torch._dynamo.mark_static(model_input.token_type_ids, 0)
            torch._dynamo.mark_static(model_input.token_type_ids, 1)

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return tuple(self.pooler.get_supported_tasks())

    @SpyrePlatform.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        **kwargs,
    ) -> ModelRunnerOutput:
        t0 = time.time()

        self.update_states(scheduler_output)

        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOutput if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT

        model_input = self.prepare_model_input(scheduler_output)

        # Execute the model
        attn_metadata = self.build_attn_metadata(model_input)

        model_kwargs = {}
        if self.use_token_type_ids:
            model_kwargs["token_type_ids"] = model_input.token_type_ids
        with set_forward_context(attn_metadata, self.vllm_config):
            outputs = self.model(
                input_ids=model_input.input_tokens,
                position_ids=model_input.input_positions,
                attention_mask=model_input.input_masks,
                **model_kwargs,
            )

            hidden_states = outputs["last_hidden_state"]

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return EMPTY_MODEL_RUNNER_OUTPUT

        t1 = time.time() - t0
        logger.debug("t_batch: %.2fms", (t1 * 1000))

        pooling_metadata = self.input_batch.make_pooling_metadata()

        ## No partial prefill, hence we can use the prompt lens here
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens=pooling_metadata.prompt_lens, device=self.device
        )

        # prepare unpadded output for the pooler
        hidden_state_list: list[torch.Tensor] = []
        for hidden_state, prompt_len in zip(hidden_states, pooling_metadata.prompt_lens):
            # we're left padding
            hidden_state_list.append(hidden_state[-prompt_len:])

        raw_pooler_output = self.pooler(
            hidden_states=torch.cat(hidden_state_list), pooling_metadata=pooling_metadata
        )

        pooler_output: list[torch.Tensor | None] = []

        for raw_output in raw_pooler_output:
            pooler_output.append(raw_output.data.to("cpu"))

        model_output = ModelRunnerOutput(
            req_ids=self.input_batch.requests_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
        )
        return model_output


@dataclass
class ChunkedPrefillPlan:
    chunk_count: int
    padding_blocks: int
    usable_cache_blocks: int
    total_cache_blocks: int


class ChunkedPrefillModelRunner(ContinuousBatchingSpyreModelRunner):
    def __init__(
        self,
        vllm_config: VllmConfig,
        is_driver_worker: bool,
        rank: int,
    ):
        super().__init__(vllm_config=vllm_config, is_driver_worker=is_driver_worker, rank=rank)

        self.chunk_size = self.scheduler_config.max_num_batched_tokens
        self.chunk_blocks_count = self.chunk_size // self.block_size

        # For hybrid KV caches, the `alignment_tokens` arg needs to be set to
        # the lowest common multiple of kv cache block sizes. Currently we only
        # support homogeneous kv caches with a single block size though.
        self._alignment_token_kwargs = (
            {"alignment_tokens": self.block_size}
            if has_argument(FullAttentionManager.find_longest_cache_hit, "alignment_tokens")
            else {}
        )

        if vllm_config.cache_config.enable_prefix_caching:
            caching_hash_fn = get_hash_fn_by_name(vllm_config.cache_config.prefix_caching_hash_algo)
            init_none_hash(caching_hash_fn)

            self.request_block_hasher = get_request_block_hasher(self.block_size, caching_hash_fn)
        else:
            self.request_block_hasher = None

        self.prefix_cache_stats = None

    def _prepare_prompt(self, _):
        AssertionError("Should not call this method on chunked prefill implementation")

    def _prepare_chunked_prefill(self, req_id: str):
        """
        Cases / Scenarios for the chunked prefill with right padding.


        X    - Padding
        T    - Token
        O    - Right padding / unused slot
        |    - Blocks separator
        ||   - Chunks separator
        ...  - Trimmed sequence

        For the following "drawing" consider blocks of 4 tokens
        and chunks of 8 tokens

        # Case I

        Prompt fits in the chunk with left padding

        3 tokens
        1 block
        1 chunk
        4 left padding

        X X X X | T T T O ||

        Variation: Prompt fits in the chunk but no left padding needed

        T tokens
        2 block
        1 chunk
        0 left padding

        T T T T | T T T O ||

        ---
        # Case II

        Prompt is greater than chunk, and it contains left padding

        10 tokens
        4 blocks
        2 chunks
        4 left padding

        X X X X | T T T T || T T T T | T T O O ||

        # Case III

        No left padding and more than one chunk

        13 tokens
        4 blocks
        2 chunks
        0 left padding

        T T T T | T T T T || T T T T | T O O O ||

        NOTE: The goal of this "illustration" is to depict strategies to write
        code to create the chunks, not necessarily enumerate the possible
        scenarios. Of course there are interpretations where these cases
        overlap.

        """
        request = self.requests[req_id]
        assert isinstance(request, ChunkedPrefillRequestState)

        chunk_size = self.chunk_size
        left_padding = request.padding_blocks * self.block_size
        left_padded_prompt_mask = torch.tensor(
            [left_padding], dtype=torch.int64, device=self.device
        )

        num_computed_tokens = request.num_computed_tokens
        num_computed_blocks = exact_div(num_computed_tokens, self.block_size)

        if request.usable_blocks > num_computed_blocks:
            assert self.enable_prefix_caching, "prefix caching has to be enabled"
            # this will be an idle step
            return SamplingForwardInputs(
                is_prompt=True, left_padded_prompt_mask=left_padded_prompt_mask
            )

        # round up due to possible padding
        chunk_i = math.ceil(num_computed_tokens / chunk_size)

        # create block table tensor
        blocks = self._get_blocks(req_id)
        block_end = (chunk_i + 1) * self.chunk_blocks_count
        block_ids = [0] * request.padding_blocks + [block.block_id for block in blocks]
        block_table = torch.tensor(block_ids[:block_end], dtype=torch.int64).unsqueeze(0)

        # last chunk
        blocks_to_recompute = 0
        if request.total_hit_blocks > 0:
            if request.usable_blocks == 0:
                chunks_from_cache = 0
            else:
                chunks_from_cache = exact_div(
                    request.padding_blocks + request.usable_blocks, self.chunk_blocks_count
                )

            # When the current chunk has passed the number
            # of chunk loaded entirely from cache, the difference
            # between the blocks from cache and the allocated
            # blocks will the the number of blocks to recompute.
            if chunk_i == chunks_from_cache:
                blocks_to_recompute = request.total_hit_blocks - request.usable_blocks
                # For the first block we must also account for left-padding blocks:
                # blocks_to_recompute = total_hit_blocks - usable_blocks + padding_blocks
                if chunk_i == 0:
                    blocks_to_recompute += request.padding_blocks

        slot_mapping = []
        for i in range(self.chunk_blocks_count):
            block = block_table[0][-self.chunk_blocks_count + i].item()
            # if we're recomputing a cached block, set the slot
            # mapping to the padding block (0)
            block *= int(i >= blocks_to_recompute)
            slot_mapping += list(
                range(block * self.block_size, block * self.block_size + self.block_size)
            )
        slot_mapping = torch.tensor(slot_mapping, device=self.device, dtype=torch.int64).unsqueeze(
            0
        )

        prompt_token_ids = request.prompt_token_ids
        prompt_len = len(prompt_token_ids)
        if num_computed_tokens == 0:
            chunk_start = 0
            chunk_end = min(chunk_size - left_padding, prompt_len)
            chunk_left_offset = left_padding
        else:
            chunk_start = chunk_i * chunk_size - left_padding
            chunk_end = min(chunk_start + chunk_size, prompt_len)
            chunk_left_offset = 0

        input_tokens = torch.zeros(chunk_size, dtype=torch.int64, device=self.device)
        input_tokens_np = input_tokens.numpy()
        input_positions = torch.zeros(chunk_size, dtype=torch.int64, device=self.device)
        input_positions_np = input_positions.numpy()

        # Create tensors based on slice
        input_tokens_np[chunk_left_offset : chunk_left_offset + chunk_end - chunk_start] = (
            prompt_token_ids[chunk_start:chunk_end]
        )
        input_positions_np[chunk_left_offset : chunk_left_offset + chunk_end - chunk_start] = range(
            chunk_start, chunk_end
        )

        logger.debug(
            "Chunked prefill of request '%s' %d:%d of %d tokens",
            req_id,
            chunk_start,
            chunk_end,
            prompt_len,
        )

        input_tokens = input_tokens.unsqueeze(0).clone()
        input_positions = input_positions.unsqueeze(0).clone()

        # NOTE(wallas): Looks like we need to use multiple of blocks for prefill
        # so, later we use model.n_pads_right to get right logits.
        # In my naive mind this would be the `request_tkv` below,
        # but it gives me incorrect results
        #
        prefill_tkv = (chunk_i + 1) * chunk_size
        current_tkv_mask = torch.tensor([prefill_tkv], dtype=torch.int64, device=self.device)

        request_tkv = min(prefill_tkv, left_padding + prompt_len)

        # Trick padding:
        # In `model_executor/model_loader/spyre.py`: We have this line to
        # get the logits of a prefill:
        #
        #   logits = logits[self.indices, -self.n_pads_right - 1, :]
        #
        # So we have to match this padding with the tkv of this chunk
        # being prefilled.
        # `((request_tkv -1) % self.block_size) + 1`: gives us the tkv offset
        # removing all the left blocks. Reminder: this tkv must be in the range
        # [1, block_size]
        #
        # `self.block_size - `: will just flip the index to get it as
        # negative index
        self.model.n_pads_right = self.block_size - (((request_tkv - 1) % self.block_size) + 1)
        self.model.indices = torch.ones(1, dtype=torch.bool, device="cpu")

        model_inputs = SamplingForwardInputs(
            input_tokens=input_tokens,
            input_positions=input_positions,
            current_tkv_mask=current_tkv_mask,
            left_padded_prompt_mask=left_padded_prompt_mask,
            block_table=block_table,
            slot_mapping=slot_mapping,
            is_prompt=True,
            scale_indices=self.input_batch.request_indices,
        )

        self._mark_input_tensors(model_inputs)

        return model_inputs

    def _prepare_decode(
        self,
        cached_request_data: CachedRequestData,
    ) -> SamplingForwardInputs:
        assert len(cached_request_data.req_ids) > 0

        input_tokens = []
        input_positions = []
        block_table = []
        slot_mapping = []
        left_padded_prompt_mask = []
        tkv_mask = []

        assert len(self.input_batch.req_id_to_index) == len(cached_request_data.req_ids)
        req_ids = self.input_batch.sorted_requests_ids

        # maximal number of blocks used by any seq in the batch
        max_n_blocks = 0

        for req_id in req_ids:
            # adding new blocks if needed
            req_state = self.requests[req_id]
            if req_state.num_computed_tokens % self.block_size == 0:
                blocks = self.kv_cache_manager.allocate_new_blocks(
                    req_id, req_state.num_computed_tokens + 1
                )
                assert len(blocks) == 1
            max_n_blocks = max(max_n_blocks, len(self._get_blocks(req_id)))

        # We'll calculate tkv on the fly, it is the max num computed tokens
        # of a request since there is no tokens left padding, only for blocks
        tkv = 0
        for req_id in req_ids:
            # TODO: Will this always just be one token ID if there's no spec
            # or jump decoding?

            req_state = self.requests[req_id]

            # filling block table with padding blocks to make it rectangular
            # Note: the padding block id 0 here is chosen arbitrarily, it can
            # be any allocated block id on the Sypre card (has to be in range
            # [0, self.n_blocks - 1]). Further, it also be a block id that holds
            # actual KV cache for another (or the same) sequence.
            blocks = self._get_blocks(req_id)
            left_pad_blocks_count = max_n_blocks - len(blocks)
            block_ids = [0] * left_pad_blocks_count + [block.block_id for block in blocks]
            block_table.append(block_ids)

            # slot_mapping for all blocks of sequence
            start_slot = block_table[-1][-1] * self.block_size
            offset = req_state.num_computed_tokens % self.block_size
            slot = [start_slot + offset]
            slot_mapping.append(slot)

            # input token and position of the token generated in the last step
            generation_token = req_state.output_token_ids[-1]
            input_tokens.append([generation_token])
            input_positions.append([req_state.num_computed_tokens])

            # Calculate left padding on the fly
            left_padding = left_pad_blocks_count * self.block_size
            left_padded_prompt_mask.append(left_padding)

            req_tkv = left_padding + req_state.num_computed_tokens + 1
            tkv_mask.append(req_tkv)
            tkv = max(tkv, req_tkv)

        # update tkv
        self.tkv = tkv

        # construct tensors from lists
        input_tokens = torch.tensor(input_tokens, dtype=torch.long, device=self.device)
        position_ids = torch.tensor(input_positions, dtype=torch.long, device=self.device)
        current_tkv_mask = torch.tensor(tkv_mask, dtype=torch.int64)
        left_padded_prompt_mask = torch.tensor(
            left_padded_prompt_mask, dtype=torch.long, device=self.device
        )
        block_table = torch.tensor(block_table, dtype=torch.int64)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64)
        self.model.indices = torch.ones(
            len(cached_request_data.req_ids), dtype=torch.bool, device="cpu"
        )

        model_inputs = SamplingForwardInputs(
            input_tokens=input_tokens,
            input_positions=position_ids,
            current_tkv_mask=current_tkv_mask,
            left_padded_prompt_mask=left_padded_prompt_mask,
            block_table=block_table,
            slot_mapping=slot_mapping,
            is_prompt=False,
            scale_indices=self.input_batch.request_indices,
        )

        self._mark_input_tensors(model_inputs)

        return model_inputs

    def _plan_chunking(self, scheduler_request: Request) -> ChunkedPrefillPlan:
        prompt_len = len(scheduler_request.prompt_token_ids)

        chunk_size = self.chunk_size
        padded_prompt_len = math.ceil(prompt_len / self.block_size) * self.block_size
        chunk_count = math.ceil(prompt_len / chunk_size)

        left_padding = chunk_count * chunk_size - padded_prompt_len
        left_blocks = exact_div(left_padding, self.block_size)

        if self.enable_prefix_caching:
            computed_blocks: list[KVCacheBlock] = FullAttentionManager.find_longest_cache_hit(
                block_hashes=scheduler_request.block_hashes,
                max_length=prompt_len,
                kv_cache_group_ids=[0],
                block_pool=self.block_pool,
                kv_cache_spec=self._attn_spec,
                use_eagle=False,
                dcp_world_size=1,
                **self._alignment_token_kwargs,
            )[0]
            n_hit = len(computed_blocks)

            logger.debug("Prefix caching found: %d cached blocks", n_hit)

            full_chunks_with_cached_blocks = (left_blocks + n_hit) // self.chunk_blocks_count

            # the last chunk of the prompt must always be recomputed
            if full_chunks_with_cached_blocks == chunk_count:
                full_chunks_with_cached_blocks -= 1

            usable_blocks = max(
                0, (full_chunks_with_cached_blocks * self.chunk_blocks_count) - left_blocks
            )

            # blocks to compute from scratch or recompute in the last chunk
            blocks_to_compute = padded_prompt_len // self.block_size - usable_blocks
            logger.debug(
                "Prefix caching found: %d reusable blocks in cache. %d blocks will be (re)computed",
                usable_blocks,
                blocks_to_compute,
            )

            # Save all of the computed blocks and not only the usable
            # ones because we will make a dummy recomputation of computed
            # blocks in the last chunk to deduplicate the used blocks. So
            # although we will recompute, we'll still point the block table
            # to the cached blocks.
            self.block_pool.touch((computed_blocks,))
            self.kv_cache_manager.save_new_computed_blocks(
                scheduler_request.request_id, computed_blocks
            )
        else:
            usable_blocks = 0
            n_hit = 0

        return ChunkedPrefillPlan(
            chunk_count=chunk_count,
            padding_blocks=left_blocks,
            usable_cache_blocks=usable_blocks,
            total_cache_blocks=n_hit,
        )

    def add_new_request(self, request: NewRequestData):
        req_id = request.req_id
        prompt_token_ids = request.prompt_token_ids
        sampling_params = request.sampling_params
        is_new_batch = len(self.req_ids2num_reserved_blocks) == 0
        prompt_len = len(prompt_token_ids)

        self.prefill_batch.clear_requests()

        # set the new tkv to the prompt length if starting a new decode batch
        if is_new_batch:
            self.tkv = prompt_len

        # Reserve the number of blocks that this new sequence requires in the
        # worst case (it might always stop early by producing the EOS token)
        new_tokens = sampling_params.max_tokens if sampling_params is not None else 0
        total_tokens = prompt_len + new_tokens - 1
        # calculate the number of reserved blocks
        n_reserved_blocks = math.ceil(total_tokens / self.block_size)

        self.req_ids2num_reserved_blocks[req_id] = n_reserved_blocks

        scheduler_request = Request(
            request_id=req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=request.sampling_params,
            pooling_params=None,
            eos_token_id=None,
            block_hasher=self.request_block_hasher,
        )
        chunk_plan = self._plan_chunking(scheduler_request)
        num_cached_tokens = chunk_plan.usable_cache_blocks * self.block_size

        self.prefix_cache_stats = PrefixCacheStats(
            # We only support single-request chunked prefill so this is always 1
            requests=1,
            queries=prompt_len,
            hits=num_cached_tokens,
        )

        # allocate blocks
        self.kv_cache_manager.allocate_new_blocks(req_id, prompt_len)

        # Add new request to the cached states.
        if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(sampling_params.seed)
        else:
            generator = None

        req_state = ChunkedPrefillRequestState(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            generator=generator,
            output_token_ids=[],
            # We do not store it, we calculate it on the fly
            # to always use the optimizations of blocks
            # usage
            left_padding=0,
            scheduler_request=scheduler_request,
            chunk_count=chunk_plan.chunk_count,
            padding_blocks=chunk_plan.padding_blocks,
            usable_blocks=chunk_plan.usable_cache_blocks,
            total_hit_blocks=chunk_plan.total_cache_blocks,
        )

        self.requests[req_id] = req_state

        # Add only to prefill batch, it will be added later to the input batch
        # once if is fully prefilled
        self.prefill_batch.add_request(req_state)

    def _maybe_prepare_last_prefill(self, req_id: str, scheduler_output: SchedulerOutput) -> None:
        """In the last prefill we have to setup the batch to sample the
        first token.
        """
        # Check if it is last prefill
        request = self.requests[req_id]
        num_computed_tokens = request.num_computed_tokens
        prompt_len = len(request.prompt_token_ids)
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]

        if num_computed_tokens + num_scheduled_tokens < prompt_len:
            return

        # Last prefill: we might need to update the tkv
        req_n_blocks = math.ceil(prompt_len / self.block_size)
        cur_n_blocks = math.ceil(self.tkv / self.block_size)
        new_n_blocks = max(req_n_blocks, cur_n_blocks)
        assert new_n_blocks > 0
        base_n_tokens = (new_n_blocks - 1) * self.block_size
        req_tkv_new_block = base_n_tokens + (prompt_len - 1) % self.block_size + 1
        cur_tkv_new_block = base_n_tokens + (self.tkv - 1) % self.block_size + 1
        self.tkv = max(req_tkv_new_block, cur_tkv_new_block)

        # Last prefill we need to setup the logitsprocessors to sampling
        prefill_index = self.input_batch.add_request(request)
        for logitsproc in self.input_batch.logitsprocs_wrappers:
            logitsproc.set_prefill_index(prefill_index)

        # Refresh sampling metadata after all request are added to the batch
        self.input_batch.refresh_metadata()
        self.prefill_batch.refresh_metadata()

    def prepare_model_input(self, scheduler_output):
        is_prefill = False
        req_id: str = ""
        if len(scheduler_output.scheduled_new_reqs) == 1:
            # First prefill let's update cache
            assert len(scheduler_output.scheduled_cached_reqs.req_ids) == 0
            self.add_new_request(scheduler_output.scheduled_new_reqs[0])
            req_id = scheduler_output.scheduled_new_reqs[0].req_id
            is_prefill = True

        # NOTE: We assume that there's only one prefill at each step
        # and if it is a single request cached here and the number of
        # computed tokens is less than the length of the prompt then
        # it is still prefilling.
        if scheduler_output.scheduled_cached_reqs.num_reqs == 1:
            # Whether it's a prefill or not, should not have any request here
            assert len(scheduler_output.scheduled_new_reqs) == 0
            req_id = scheduler_output.scheduled_cached_reqs.req_ids[0]
            is_prefill = (
                len(self.requests[req_id].prompt_token_ids)
                > scheduler_output.scheduled_cached_reqs.num_computed_tokens[0]
            )

        # Prepare input tensors.
        if is_prefill:
            # All prefills are chunked
            # Get request id from new request or cached request
            model_inputs = self._prepare_chunked_prefill(req_id)
            self._maybe_prepare_last_prefill(req_id, scheduler_output)

            return model_inputs
        else:
            return self._prepare_decode(scheduler_output.scheduled_cached_reqs)

    def get_empty_output(self):
        return CPSpyreModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            num_nans_in_logits=None,
            tkv=0,
            n_free_blocks=self.get_n_free_blocks(),
            left_padding={},
            kv_cache_usage=self.get_kv_cache_usage(),
            prefix_cache_stats=None,
        )

    def check_incomplete_prefill(self, scheduler_output: SchedulerOutput):
        cached_reqs = scheduler_output.scheduled_cached_reqs
        new_reqs = scheduler_output.scheduled_new_reqs

        if cached_reqs.num_reqs != 1 and len(new_reqs) != 1:
            # Not a prefill
            return False

        # possible prefill
        req_id = new_reqs[0].req_id if len(new_reqs) == 1 else cached_reqs.req_ids[0]

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        if len(new_reqs) == 1:
            return num_scheduled_tokens < len(new_reqs[0].prompt_token_ids)
        else:
            req_state = self.requests[req_id]
            num_computed_tokens = cached_reqs.num_computed_tokens[0]
            return (num_computed_tokens + num_scheduled_tokens) < len(req_state.prompt_token_ids)

    def update_states(self, scheduler_output: SchedulerOutput):
        cached_reqs = scheduler_output.scheduled_cached_reqs

        # clear the prefix cache stats so that we only record them on the first
        # chunk of any prefill
        self.prefix_cache_stats = None

        if cached_reqs.num_reqs == 1:
            # NOTE: while prefilling the request is not yet in the
            # input batch, and update states try to access it there.
            req_id = cached_reqs.req_ids[0]
            req_state = self.requests[req_id]
            assert isinstance(req_state, ChunkedPrefillRequestState)
            num_computed_tokens = cached_reqs.num_computed_tokens[0]
            if num_computed_tokens < len(req_state.prompt_token_ids):
                # For now, if it is prefilling, we only need to update num of
                # computed tokens of the request
                req_state.num_computed_tokens = num_computed_tokens
                if self.enable_prefix_caching:
                    num_cached_blocks = self.kv_cache_manager.num_cached_block[req_id]
                    # if the number of cached tokens is larger or equal to the
                    # number of computed tokens, it means that during this call
                    # to execute_model we're just loading blocks from the KV
                    # cache and can't call `cache_blocks()`
                    if num_computed_tokens > num_cached_blocks * self.block_size:
                        self.kv_cache_manager.cache_blocks(
                            req_state.scheduler_request, num_computed_tokens
                        )
                # hide the prefill request from the super class
                scheduler_output.scheduled_cached_reqs = CachedRequestData.make_empty()
                super().update_states(scheduler_output)
                scheduler_output.scheduled_cached_reqs = cached_reqs
                return

        super().update_states(scheduler_output)

    @SpyrePlatform.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        **kwargs,
    ) -> ModelRunnerOutput:
        t0 = time.time()

        self.update_states(scheduler_output)

        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOutput if there's no work to do.
            return self.get_empty_output()

        model_input = self.prepare_model_input(scheduler_output)

        incomplete_prefill = False
        is_cached_chunk = False
        is_prefill = cast(bool, model_input.is_prompt)
        if is_prefill:
            incomplete_prefill = self.check_incomplete_prefill(scheduler_output)
            is_cached_chunk = model_input.input_tokens is None
            if is_cached_chunk:
                assert incomplete_prefill, "can't apply caching on the last chunked prefill"

        if not is_cached_chunk:
            # Execute the model
            attn_metadata = self.build_attn_metadata(model_input)
            with set_forward_context(attn_metadata, self.vllm_config):
                logits = self.model(
                    input_ids=model_input.input_tokens,
                    positions=model_input.input_positions,
                    masks=model_input.input_masks,
                    is_prompt=model_input.is_prompt,
                )

        # Get mapping between requests ids to the index within the batch
        req_id_to_index = self.get_req_id_to_index(is_prefill)

        # Prepare the left paddings to pass to the scheduler
        left_padded_prompt_mask = cast(torch.tensor, model_input.left_padded_prompt_mask)
        left_padding = {
            req_id: left_padded_prompt_mask[idx].item() for req_id, idx in req_id_to_index.items()
        }

        # TODO: dead code, this only works for SB with bs=1, either
        # fix it or remove it.
        # prompt_logprobs_dicts = self._get_prompt_logprobs_dict(
        #    logits=logits, model_inputs=model_input)
        # TODO: disable prefix caching for requests with prompt logprobs
        prompt_logprobs_dicts: dict[str, LogprobsTensors | None] = {}

        # If the prompt is being prefilled we don't have to sample
        # and generate a new token.
        if is_prefill and self.check_incomplete_prefill(scheduler_output):
            # Only return outputs from the driver worker
            if not self.is_driver_worker:
                return self.get_empty_output()

            t1 = time.time() - t0
            logger.debug("t_forward_pass: %.2fms [prefill single chunk][batch size 1]", (t1 * 1000))
            return CPSpyreModelRunnerOutput(
                req_ids=list(req_id_to_index.keys()),
                req_id_to_index=req_id_to_index,
                sampled_token_ids=[],
                logprobs=None,
                prompt_logprobs_dict=prompt_logprobs_dicts,
                pooler_output=[],
                tkv=self.tkv,
                n_free_blocks=self.get_n_free_blocks(),
                left_padding=left_padding,
                kv_cache_usage=self.get_kv_cache_usage(),
                prefix_cache_stats=self.prefix_cache_stats,
                prefix_cache_hit_len=self.get_prefix_cache_len(),
            )

        # Sample the next token.
        output: SamplerOutput = self.model.sample(
            logits=logits,
            sampling_metadata=self.get_sampling_metadata(is_prefill),
        )
        t1 = time.time() - t0
        batch_size = model_input.input_tokens.shape[0]
        step_type = "[prefill last chunk]" if is_prefill else "[decode]"
        logger.debug("t_token: %.2fms %s[batch size %d]", (t1 * 1000), step_type, batch_size)

        # Get the right batch, if this is the last chunk to conclude the
        # prefill, we'll generate a token and we should get from the prefill
        # batch because input_batch may have other request that are were
        # not processed at this step.
        batch = self.prefill_batch if is_prefill else self.input_batch

        # Add the sampled token(s) to the request cache
        req_ids = (
            [r.req_id for r in scheduler_output.scheduled_new_reqs]
            if len(scheduler_output.scheduled_new_reqs) > 0
            else batch.sorted_requests_ids
        )
        sampled_ids = output.sampled_token_ids.tolist()

        for i, req_id in enumerate(req_ids):
            req_state = self.requests[req_id]
            assert isinstance(req_state, ChunkedPrefillRequestState)
            assert req_state.scheduler_request is not None
            req_state.scheduler_request.append_output_token_ids(sampled_ids[i])
            if self.enable_prefix_caching:
                self.kv_cache_manager.cache_blocks(
                    req_state.scheduler_request, req_state.scheduler_request.num_tokens
                )
            req_state.output_token_ids.extend(sampled_ids[i])

        # Only return outputs from the driver worker
        if not self.is_driver_worker:
            return self.get_empty_output()

        sampled_token_ids = self._make_compatible_sampled_token_ids(output.sampled_token_ids)

        model_output = CPSpyreModelRunnerOutput(
            req_ids=list(req_id_to_index.keys()),
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            logprobs=(output.logprobs_tensors.tolists() if output.logprobs_tensors else None),
            prompt_logprobs_dict=prompt_logprobs_dicts,
            pooler_output=[],
            tkv=self.tkv,
            n_free_blocks=self.get_n_free_blocks(),
            left_padding=left_padding,
            kv_cache_usage=self.get_kv_cache_usage(),
            prefix_cache_stats=self.prefix_cache_stats,
        )

        return model_output

    def get_kv_cache_usage(self) -> float:
        return self.kv_cache_manager.block_pool.get_usage()

    def get_prefix_cache_len(self) -> dict[str, int]:
        """Get the prefix cache hit length for each prefilling request.
        This is in the number of usable cache tokens: Including the left padding
        this will always land at a chunk boundary.
        """
        result = {}
        for req_id in self.prefill_batch.requests_ids:
            request = self.requests[req_id]
            assert isinstance(request, ChunkedPrefillRequestState)
            result[req_id] = request.usable_blocks * self.block_size
        return result

    def _mark_input_tensors(self, model_input: SamplingForwardInputs) -> None:
        # Marking dimensions static/dynamic
        if model_input.is_prompt:
            # batch static (batch size 1)
            torch._dynamo.mark_static(model_input.input_tokens, 0)
            torch._dynamo.mark_static(model_input.slot_mapping, 0)
            torch._dynamo.mark_static(model_input.input_positions, 0)
            torch._dynamo.mark_static(model_input.block_table, 0)

            # sequence dynamic
            torch._dynamo.mark_dynamic(model_input.input_tokens, 1)
            torch._dynamo.mark_dynamic(model_input.slot_mapping, 1)
            torch._dynamo.mark_dynamic(model_input.input_positions, 1)
            torch._dynamo.mark_dynamic(model_input.block_table, 1)

        # decode
        else:
            # mask is no longer used here

            # batch dynamic
            torch._dynamo.mark_dynamic(model_input.input_tokens, 0)
            torch._dynamo.mark_dynamic(model_input.block_table, 0)
            torch._dynamo.mark_dynamic(model_input.slot_mapping, 0)
            torch._dynamo.mark_dynamic(model_input.input_positions, 0)
            torch._dynamo.mark_dynamic(model_input.current_tkv_mask, 0)
            torch._dynamo.mark_dynamic(model_input.left_padded_prompt_mask, 0)

            # sequence
            torch._dynamo.mark_static(model_input.input_tokens, 1)  # always 1
            torch._dynamo.mark_dynamic(model_input.block_table, 1)
            torch._dynamo.mark_static(model_input.slot_mapping, 1)  # always 1
            torch._dynamo.mark_static(model_input.input_positions, 1)  # always 1

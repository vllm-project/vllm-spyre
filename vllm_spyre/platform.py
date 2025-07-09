import sys

# When running this plugin on a Mac, we assume it's for local development
# purposes. However, due to a compatibility issue with vLLM, which overrides
# the Triton module with a placeholder, vLLM may fail to load on macOS. To
# mitigate this issue, we can safely remove the Triton module (if imported)
# and rely on PyTorch to handle the absence of Triton, ensuring fine execution
# in eager mode.
if sys.platform.startswith("darwin"):
    if sys.modules.get('triton'):
        del sys.modules['triton']

import operator
import os
from typing import TYPE_CHECKING, Optional, Union

import torch
from vllm.inputs import ProcessorInputs, PromptType
from vllm.logger import init_logger
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
else:
    ModelConfig = None
    VllmConfig = None
import vllm.envs as envs
from vllm.platforms import Platform, PlatformEnum

import vllm_spyre.envs as envs_spyre

logger = init_logger(__name__)


class SpyrePlatform(Platform):
    _enum = PlatformEnum.OOT

    # "spyre" device_name no longer worked due to https://github.com/vllm-project/vllm/pull/16464
    device_name: str = "cpu"
    device_type: str = "cpu"
    supported_quantization: list[str] = ["gptq"]
    _warmup_shapes: Optional[tuple[dict[str, int], ...]] = None
    _config: VllmConfig = None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "spyre"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        """
        Check if the current platform supports async output.
        """
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        cls._config = vllm_config
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        if scheduler_config.is_multi_step:
            raise NotImplementedError

        is_decoder = model_config.task == "generate"
        is_embedding = model_config.task == "embed"

        # v0 is only supported for embedding models, and embedding models must
        # be run on v0
        if is_embedding and envs.VLLM_USE_V1:
            raise ValueError("Embedding models are only supported on v0")
        elif is_decoder and not envs.VLLM_USE_V1:
            raise ValueError("Decoder models are only supported on v1")
        elif not is_decoder and not is_embedding:
            raise ValueError("Only the 'generate' and 'embed' tasks are "
                             "supported")

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = (
                f'vllm_spyre{".v1" if envs.VLLM_USE_V1 else ""}'\
                    '.worker.spyre_worker.SpyreWorker')

        if envs_spyre.VLLM_SPYRE_USE_CB and is_decoder:
            scheduler_config.scheduler_cls = "vllm_spyre.v1.core."\
                "scheduler.ContinuousBatchingSpyreScheduler"
            if envs_spyre.VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS:
                raise ValueError("Prompt logprobs not supported with " \
                "continuous batching")
        else:
            # Static batching or embedding model.
            # Override --max-num-seqs to the biggest warmup batch size
            # And override --max-model-len to the biggest warmup sequence
            cls._warmup_shapes = None
            max_model_len = model_config.max_model_len
            spyre_warmup_shapes = cls.get_warmup_shapes(
                scheduler_config, max_model_len)
            max_batch_size = 0
            max_seq_len = 0
            for shape in spyre_warmup_shapes:
                max_batch_size = max(max_batch_size, shape["batch_size"])
                max_seq_len = max(max_seq_len,
                                  shape["prompt_length"] + shape["new_tokens"])

            if (envs_spyre.VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS
                    and max_batch_size > 1):
                raise ValueError(
                    "Prompt logprobs only supported with batch size 1")

            model_config.max_model_len = max_seq_len
            scheduler_config.max_num_seqs = max_batch_size

            if is_decoder:
                scheduler_config.scheduler_cls = (
                    "vllm_spyre.v1.core.scheduler."\
                        "StaticBatchingSpyreScheduler")
            elif is_embedding:
                scheduler_config.scheduler_cls = (
                    "vllm_spyre.core.scheduler.SpyreScheduler")

        # To disable any paged attention ops in the base scheduler, we:
        # - Set the block size (in tokens) to the maximum sequence length
        #       so that the scheduler thinks an entire sequence will fit in
        #       one single block.
        # - Set the number of blocks to the maximum number of sequences, so
        #       the scheduler always thinks there's a block available
        # - Set `max_num_batched_tokens` to the size of a full batch of full
        #       length requests, so that the scheduler will always have token
        #       budget available to schedule a full batch
        if cache_config is not None:
            if envs.VLLM_USE_V1:
                # The V1 scheduler actually needs 2 blocks for each sequence...
                cache_config.num_gpu_blocks_override = \
                    scheduler_config.max_num_seqs * 2
            else:
                cache_config.num_gpu_blocks_override = \
                    scheduler_config.max_num_seqs

            cache_config.block_size = model_config.max_model_len
            scheduler_config.max_num_batched_tokens = (
                model_config.max_model_len * scheduler_config.max_num_seqs)

        logger.info(
            "Overriding configurations based on warmup shapes. "
            "max_model_len=%d, max_num_seqs=%d, block_size=%d, "
            "num_gpu_blocks_override=%d, max_num_batched_tokens=%d",
            model_config.max_model_len, scheduler_config.max_num_seqs,
            cache_config.block_size, cache_config.num_gpu_blocks_override,
            scheduler_config.max_num_batched_tokens)

        # set env vars for torch_sendnn to consume
        os.environ["VLLM_DT_MAX_CONTEXT_LEN"] = str(
            vllm_config.model_config.max_model_len)
        os.environ["VLLM_DT_MAX_BATCH_SIZE"] = str(
            vllm_config.scheduler_config.max_num_seqs)

    @classmethod
    def use_all_gather(cls) -> bool:
        """
        Whether to use allgather in LogitsProcessor to gather the logits.
        """
        return True

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on Spyre.")
        return False

    @classmethod
    def inference_mode(cls):
        """
        Spyre does not support `torch.inference_mode`. 
        This allows to fall back to `torch.no_grad` when inference mode is set.
        """
        return torch.no_grad()

    @classmethod
    def get_warmup_shapes(
            cls,
            scheduler_config,
            max_model_len: int = sys.maxsize) -> tuple[dict[str, int], ...]:
        if cls._warmup_shapes is not None:
            return cls._warmup_shapes
        # load warmup shapes and sort by "speed"
        wup_prompt_lens = envs_spyre.VLLM_SPYRE_WARMUP_PROMPT_LENS or []
        if not all(pl % 64 == 0 for pl in wup_prompt_lens):
            raise RuntimeError(
                "All values in VLLM_SPYRE_WARMUP_PROMPT_LENS must be multiples "
                "of 64.")

        wup_batch_sizes = envs_spyre.VLLM_SPYRE_WARMUP_BATCH_SIZES or []
        if len(wup_prompt_lens) != len(wup_batch_sizes):
            raise RuntimeError(
                "The lists in VLLM_SPYRE_WARMUP_PROMPT_LENS and "
                "VLLM_SPYRE_WARMUP_BATCH_SIZES must have equal length")
        if scheduler_config.runner_type == "pooling":
            wup_new_tokens = [0] * len(wup_prompt_lens)
        else:
            wup_new_tokens = envs_spyre.VLLM_SPYRE_WARMUP_NEW_TOKENS or []
            if len(wup_new_tokens) != len(wup_prompt_lens):
                raise RuntimeError(
                    "The lists in VLLM_SPYRE_WARMUP_PROMPT_LENS and "
                    "VLLM_SPYRE_WARMUP_NEW_TOKENS must have equal length")

        logger.info("VLLM_SPYRE_WARMUP_PROMPT_LENS = %s", wup_prompt_lens)
        logger.info("VLLM_SPYRE_WARMUP_NEW_TOKENS = %s", wup_new_tokens)
        logger.info("VLLM_SPYRE_WARMUP_BATCH_SIZES = %s", wup_batch_sizes)

        cls._warmup_shapes = tuple(
            sorted([{
                'prompt_length': pl,
                'new_tokens': nt,
                'batch_size': bs
            } for pl, nt, bs in zip(wup_prompt_lens, wup_new_tokens,
                                    wup_batch_sizes)],
                   key=operator.itemgetter('batch_size', 'prompt_length')))

        for shape in cls._warmup_shapes:
            max_seq_len = shape["prompt_length"] + shape["new_tokens"]
            if max_seq_len > max_model_len:
                raise RuntimeError(
                    f"Warmup shape [{shape['batch_size']},"
                    f" {shape['prompt_length']}, {shape['new_tokens']}]"
                    f" results in a maximum sequence length of "
                    f"{max_seq_len} which is longer that what the model "
                    f"supports ({max_model_len})")
        return cls._warmup_shapes

    @classmethod
    def supports_v1(cls, model_config: ModelConfig) -> bool:
        """Returns whether the current platform can support v1 for the supplied
        model configuration.
        """
        # We don't have an embedding runner for v1 yet
        return model_config.task != "embed"

    @classmethod
    def validate_request(
        cls,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        processed_inputs: Optional[ProcessorInputs] = None,
    ) -> None:
        """Raises if this request is unsupported on this platform"""
        if isinstance(params, PoolingParams):
            # Only validating generation requests for now
            return None

        if (params.prompt_logprobs is not None
                and not envs_spyre.VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS):
            raise ValueError("Prompt logprobs must be enabled with "
                             "`VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS=1`")

        if isinstance(prompt, dict) and "prompt_token_ids" in prompt:
            prompt_len = len(prompt["prompt_token_ids"])
        elif processed_inputs is not None:
            if "encoder" in processed_inputs:
                raise ValueError("Encoder-decoder models not supported ")
            prompt_len = len(processed_inputs["prompt_token_ids"])
        else:
            # We need a prompt length to do any validation here
            return

        max_tokens = 0
        if params is not None and params.max_tokens is not None:
            max_tokens = params.max_tokens

        if envs_spyre.VLLM_SPYRE_USE_CB:
            # For continuous batching, check if the request is within the max
            # context length. This needs to take the padded prompt length
            # into account.

            # ceil division to pad to next block boundary
            n = prompt_len
            d = 64  # hardcoded AIU Spyre block size
            prompt_padding_len = ((n + d - 1) // d) * d
            if (prompt_padding_len + max_tokens
                    > cls._config.scheduler_config.max_model_len):
                raise ValueError(
                    "Could not add request: prompt length is "
                    f"{prompt_len} tokens, which gets padded to "
                    f"{prompt_padding_len} tokens, maximum number of output "
                    f"tokens is {max_tokens} tokens, but max model context "
                    f"length is {cls._config.scheduler_config.max_model_len}.")
        else:
            # For non-continuous batching, check if the request matches a warmup
            # shape
            assert cls._warmup_shapes is not None, "Warmup shapes must be set"
            if len(
                    cls._get_matching_warmup_shapes(
                        prompt_len=prompt_len,
                        max_tokens=max_tokens,
                        warmup_shapes=cls._warmup_shapes)) == 0:
                raise ValueError(
                    "No applicable warmup shape exists for "
                    f"combination of prompt length ({prompt_len} tokens) "
                    "and maximum number of output tokens to be "
                    f"generated ({max_tokens} tokens)")

    @classmethod
    def _get_matching_warmup_shapes(
            cls, prompt_len: int, max_tokens: int,
            warmup_shapes: tuple[dict[str, int], ...]) -> list[dict[str, int]]:
        """Return the subset of shapes that match this request"""
        return [
            shape for shape in warmup_shapes
            if prompt_len <= shape['prompt_length']
            and max_tokens <= shape['new_tokens']
        ]

    @classmethod
    def get_max_output_tokens(self, prompt_len: int) -> int:
        """Return the size of biggest ```new_tokens``` of the \
            warmup shapes that fits the prompt length"""
        max_new_tokens = 1
        if self._warmup_shapes is None:
            return max_new_tokens
        for shape in self._warmup_shapes:
            if prompt_len <= shape['prompt_length']:
                max_new_tokens = max(max_new_tokens, shape['new_tokens'])

        return max_new_tokens
"""A Spyre worker class."""
import json
import os
import platform
import time
from typing import Optional, Union, cast

import torch
import torch.distributed as dist
import vllm.envs as envs
from huggingface_hub import hf_hub_download
from vllm.config import VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase as WorkerBaseV1
from vllm.worker.worker_base import WorkerBase

import vllm_spyre.envs as envs_spyre
import vllm_spyre.perf_metrics as perf_metrics
from vllm_spyre.model_executor.model_loader import spyre_setup
from vllm_spyre.platform import SpyrePlatform
from vllm_spyre.v1.worker.spyre_model_runner import (
    ContinuousBatchingSpyreModelRunner, StaticBatchingSpyreModelRunner)

logger = init_logger(__name__)


class SpyreWorker(WorkerBaseV1):
    """A worker class that executes the model on a group of Spyre cores.
    """

    def get_kv_cache_spec(self) -> KVCacheSpec:
        """Get specifications for KV cache implementation.

        These specs are used to:
        - build the kv_cache_configs that are then passed to
            initialize_from_config() on this instance
        - determine the number of available kv_cache_blocks, see
            SpyreWorker.determine_available_memory
        """
        return self.model_runner.get_kv_cache_spec()

    def compile_or_warm_up_model(self) -> None:
        """Prepare model for execution through compilation/warmup."""
        # TO DO: implement warmup for continuous batching
        if envs_spyre.VLLM_SPYRE_USE_CB:
            self._warmup_spyre_dynamic_size(self.restricted_tokens)
            return

        wup_prompt_lens, wup_new_tokens = zip(
            *[(s["prompt_length"], s["new_tokens"])
              for s in self.spyre_warmup_shapes])

        logger.info(
            "Start warming up %d different "
            "prompt/decode/batchsize-shape combinations.", len(wup_new_tokens))
        all_warmup_start_t = time.time()
        for i, (prompt_len, num_decode_tokens, batch_size) in enumerate([
            (s["prompt_length"], s["new_tokens"], s["batch_size"])
                for s in self.spyre_warmup_shapes
        ]):
            if self.model_config.task != "embed":
                # TODO: remove if spyre supports
                # lower number of output tokens
                assert num_decode_tokens >= 3, (
                    "VLLM_SPYRE_WARMUP_NEW_TOKENS must be "
                    "at least 3 (spyre requirement).")
            # warmup individual combination
            logger.info(
                "Warmup %d/%d prompt/decode/batchsize-shape "
                "combinations...", i + 1, len(wup_new_tokens))
            logger.info(
                "Warming up for prompt length %d, decoding %d tokens with "
                "batch size %d", prompt_len, num_decode_tokens, batch_size)
            self._warmup_spyre_fixed_size(prompt_len, num_decode_tokens,
                                          self.restricted_tokens, batch_size)
        all_warmup_end_t = time.time()
        all_warmup_total_t = all_warmup_end_t - all_warmup_start_t
        self.perf_metrics.log("total warmup time", all_warmup_total_t)
        # No more perf metric are captured (so far) after warmup, cleanup now.
        del self.perf_metrics
        logger.info(
            "All warmups for %d different prompt/decode/batchsize-shape "
            "combinations finished. Total warmup time %.3fs.",
            len(wup_new_tokens), all_warmup_total_t)
        self.model_runner.complete_warmup()

    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        # TODO: Implement something!
        return

    def determine_available_memory(self) -> int:
        """Return available device memory in bytes.

        This is used in conjunction with the result from `get_kv_cache_spec`
        to determine the number of KV cache blocks that can fit on the device.

        The number of available blocks is calculated as:
            available_memory / page_size / # of layers
        where the page size and number of layers come from the kv cache spec.

        The number of device blocks (called "gpu blocks" in most places) can
        also be overridden by `--num-gpu-blocks-override`, which is set under
        `vllm_config.cache_config.num_gpu_blocks_override`.
        """
        # Currently we override vllm_config.cache_config.num_gpu_blocks_override
        # in platform.py, so this value is only used by vllm to check that the
        # number of gpu blocks will fit in available memory.
        # Since we also return dummy values for the kv cache spec, this check is
        # meaningless and we can just return a large value to ensure vllm does
        # not raise a validation error.
        # TODO: Return the real available device memory when we implement real
        # kv-caching.
        return 1 << 64

    def initialize_from_config(self,
                               kv_cache_configs: list[KVCacheConfig]) -> None:
        """Construct the KV cache from the provided configs.
        Currently, we do not support paged attention or kv caching"""
        pass

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        WorkerBase.__init__(self, vllm_config=vllm_config)
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        self.perf_metrics = perf_metrics.create_perf_metric_logger(rank)
        if self.parallel_config and is_driver_worker:
            assert rank % self.parallel_config.tensor_parallel_size == 0, \
                   "Driver worker should be rank 0 of tensor parallel group."
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()
        self.model_runner: \
            Union[StaticBatchingSpyreModelRunner,
                  ContinuousBatchingSpyreModelRunner]
        if self.model_config.task == "embed":
            raise NotImplementedError
        else:
            if envs_spyre.VLLM_SPYRE_USE_CB:
                self.model_runner = ContinuousBatchingSpyreModelRunner(
                    self.vllm_config, self.is_driver_worker)
            else:
                self.model_runner = StaticBatchingSpyreModelRunner(
                    self.vllm_config, self.is_driver_worker)
                self.spyre_warmup_shapes = SpyrePlatform.get_warmup_shapes(
                    self.vllm_config.scheduler_config)
        self._env_initialized = False
        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            activities = [torch.profiler.ProfilerActivity.CPU]

            if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND == "sendnn_decoder":
                from torch_sendnn import torch_sendnn
                torch.utils.rename_privateuse1_backend("aiu")
                torch._register_device_module("aiu",
                                              torch_sendnn.sendnn_backend)
                torch.utils.generate_methods_for_privateuse1_backend()
                activities.append(torch.profiler.ProfilerActivity.PrivateUse1)

            self.profiler = torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
            print(
                "[SpyreWorker] Profiling enabled. Traces will be saved to: %s",
                torch_profiler_trace_dir)
        else:
            self.profiler = None

    def init_distributed_environment(self) -> None:
        """Initialize the distributed environment."""

        torch._C._distributed_c10d._register_process_group(
            "default", dist.group.WORLD)

        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND in [
                "sendnn", "sendnn_decoder"
        ]:
            spyre_setup.spyre_dist_setup(
                rank=self.rank,
                world_size=self.parallel_config.world_size,
                verbose=True)

        # A small all_reduce for warmup.
        torch.distributed.all_reduce(torch.zeros(1).cpu())

    def init_device(self) -> None:

        if platform.machine() == "s390x":
            from torch.serialization import LoadEndianness
            torch.serialization.set_default_load_endianness(
                LoadEndianness.LITTLE)

        if not self._env_initialized:

            init_distributed_environment(
                world_size=self.parallel_config.world_size,
                rank=self.rank,
                distributed_init_method="env://",
                backend="gloo",
            )

            if self.parallel_config.world_size > 1:
                self.init_distributed_environment()
            elif envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND in [
                    "sendnn", "sendnn_decoder"
            ]:
                spyre_setup.spyre_setup()

            ensure_model_parallel_initialized(
                self.parallel_config.tensor_parallel_size,
                self.parallel_config.pipeline_parallel_size,
            )

            self._env_initialized = True

        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        assert self._env_initialized

        is_local = os.path.isdir(self.model_config.model)
        if is_local:
            cf_file = os.path.join(self.model_config.model, 'config.json')
        else:
            cf_file = hf_hub_download(repo_id=self.model_config.model,
                                      revision=self.model_config.revision,
                                      filename="config.json")
        with open(cf_file, 'rb') as f:
            config = json.load(f)

        restricted_tokens = []
        if tok := config.get("bos_token_id") is not None:
            restricted_tokens.append(int(tok))
        if tok := config.get("eos_token_id") is not None:
            restricted_tokens.append(int(tok))

        self.restricted_tokens = restricted_tokens

        logger.info("load model...")
        # TODO: check additionally if the Spyre card has enough memory
        # for all requested model warmups
        # printing env variables for debugging purposes
        load_model_start_t = time.time()

        if envs_spyre.VLLM_SPYRE_USE_CB:
            # unused for continuous batching: set here to use same API
            wup_prompt_lens, wup_new_tokens = (0, ), (0, )
        else:
            wup_prompt_lens, wup_new_tokens = zip(
                *[(s["prompt_length"], s["new_tokens"])
                  for s in self.spyre_warmup_shapes])

        self.model_runner.load_model(prompt_lens=wup_prompt_lens,
                                     num_decode_tokens=wup_new_tokens)

        load_model_end_t = time.time()
        load_model_total_t = load_model_end_t - load_model_start_t
        self.perf_metrics.log("load model time",
                              load_model_total_t,
                              model=self.model_config.model)
        logger.info("load model took %.3fs", load_model_total_t)

    def _warmup_spyre_dynamic_size(self, special_token_ids):

        warmup_start_t = time.time()

        # satisfy mypy
        model_runner: ContinuousBatchingSpyreModelRunner = \
            cast(ContinuousBatchingSpyreModelRunner, self.model_runner)

        vocab_size = model_runner.vocab_size

        valid_token_ids = [
            i for i in range(1, vocab_size) if i not in set(special_token_ids)
        ]

        # Convert to tensor for sampling
        valid_token_ids_tensor = torch.tensor(valid_token_ids,
                                              dtype=torch.long,
                                              device=torch.device("cpu"))
        batch_size = 2
        prompt_len = 42
        num_decode_tokens = 2

        # Sample from the valid token ids
        warmup_tokens_tensor = valid_token_ids_tensor[torch.randint(
            0, len(valid_token_ids_tensor), (batch_size, prompt_len))]

        dummy_requests = [
            NewRequestData(
                req_id="warmup-%d" % (i),
                prompt_token_ids=warmup_tokens_tensor[i].tolist(),
                mm_inputs=[],
                mm_hashes=[],
                mm_positions=[],
                sampling_params=SamplingParams(max_tokens=num_decode_tokens),
                block_ids=[0],  # not actually used
                num_computed_tokens=0,
                lora_request=None,
            ) for i in range(batch_size)
        ]

        for i, req in enumerate(dummy_requests):
            scheduler_output = SchedulerOutput(
                scheduled_new_reqs=[req],
                scheduled_cached_reqs=[],
                num_scheduled_tokens={req.req_id: prompt_len},
                total_num_scheduled_tokens=prompt_len,
                scheduled_spec_decode_tokens={},
                scheduled_encoder_inputs={},
                num_common_prefix_blocks=0,
                finished_req_ids=set(),
                free_encoder_input_ids=[],
                structured_output_request_ids={},
                grammar_bitmask=None,
            )
            logger.info("Warmup prefill %d/%d...", i + 1, batch_size)
            self.execute_model(scheduler_output)

        # one decode iteration across both sequences
        cached_requests = [
            CachedRequestData(
                req_id=req.req_id,
                resumed_from_preemption=False,
                new_token_ids=[
                    valid_token_ids_tensor[torch.randint(
                        0, len(valid_token_ids_tensor), (1, )).item()]
                ],  # placeholder token
                new_block_ids=req.block_ids,
                num_computed_tokens=prompt_len,
            ) for req in dummy_requests
        ]

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached_requests,
            num_scheduled_tokens={f"warmup-{i}": 1
                                  for i in range(batch_size)},
            total_num_scheduled_tokens=batch_size,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=0,
            finished_req_ids=set(),
            free_encoder_input_ids=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )
        logger.info("Warmup decode 1/1...")
        self.execute_model(scheduler_output)

        # Needed to clean up the data of model runner
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=[],
            num_scheduled_tokens={},
            # NOTE: this means no work to do
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=0,
            # The requests to be removed
            finished_req_ids=set([r.req_id for r in dummy_requests]),
            free_encoder_input_ids=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )
        self.execute_model(scheduler_output)

        self.model_runner.tkv = 0  # type: ignore[union-attr]

        # update lazyhandle (once)
        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND == "sendnn_decoder":
            from torch_sendnn import torch_sendnn
            ul_start_time = time.time()
            torch_sendnn.update_lazyhandle()
            ul_stop_time = time.time()
            logger.info("update_lazyhandle() done (duration: %.3fs)",
                        ul_stop_time - ul_start_time)

        warmup_end_t = time.time()
        warmup_total_t = warmup_end_t - warmup_start_t
        logger.info("Warmup finished.")
        logger.info("Warmup took %.3fs", warmup_total_t)

    def _warmup_spyre_fixed_size(self, prompt_len, num_decode_tokens,
                                 special_token_ids, batch_size):

        warmup_start_t = time.time()
        # NOTE(ngl): empty tensor causes spyre to hang, so using
        # randint without 0 and the eos and bos token

        # Create a list of valid values between 1 (inclusive) and vocab
        # size (exclusive) by excluding the eos and bos token ids
        # (in special_token_ids)
        vocab_size = self.model_runner.vocab_size
        valid_token_ids = [
            i for i in range(1, vocab_size) if i not in set(special_token_ids)
        ]
        # Convert to tensor for sampling
        valid_token_ids_tensor = torch.tensor(valid_token_ids,
                                              dtype=torch.long,
                                              device=torch.device("cpu"))

        # Sample from the valid token ids
        warmup_tokens_tensor = valid_token_ids_tensor[torch.randint(
            0, len(valid_token_ids_tensor), (batch_size, prompt_len))]

        extra_kwargs = {}
        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND not in [
                "sendnn", "sendnn_decoder"
        ]:
            # Bug in 2.3.1 fixed in 2.4.1 for SDPA flash cpu
            # impl when padding too much
            extra_kwargs["attn_algorithm"] = "math"

        # Set up dummy requests for prefill steps
        dummy_requests = [
            NewRequestData(
                req_id="warmup",
                prompt_token_ids=warmup_tokens_tensor[i].tolist(),
                mm_inputs=[],
                mm_hashes=[],
                mm_positions=[],
                sampling_params=SamplingParams(max_tokens=num_decode_tokens),
                block_ids=[0],
                num_computed_tokens=0,
                lora_request=None,
            ) for i in range(batch_size)
        ]

        # Set up dummy cached_requests for decode steps
        cached_requests = [
            CachedRequestData(
                req_id=req.req_id,
                resumed_from_preemption=False,
                new_token_ids=[
                    valid_token_ids_tensor[torch.randint(
                        0, len(valid_token_ids_tensor), (1, )).item()]
                ],  # placeholder token
                new_block_ids=req.block_ids,
                num_computed_tokens=req.num_computed_tokens,
            ) for req in dummy_requests
        ]

        # Set up scheduler_output for execute_model
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=dummy_requests,
            scheduled_cached_reqs=[],
            num_scheduled_tokens={i: prompt_len
                                  for i in range(batch_size)},
            total_num_scheduled_tokens=sum(prompt_len
                                           for _ in range(batch_size)),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=0,
            finished_req_ids=set(),
            free_encoder_input_ids=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )

        # First full forward pass
        logger.info("Warmup forward pass 1/2...")
        self._warmup_model_forward_pass(scheduler_output, dummy_requests,
                                        cached_requests, num_decode_tokens)
        self.perf_metrics.log("warmup 1 time",
                              time.time() - warmup_start_t,
                              batch_size=batch_size,
                              max_tokens=num_decode_tokens,
                              prompt_len=prompt_len)

        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND == "sendnn_decoder":
            from torch_sendnn import torch_sendnn
            ul_start_time = time.time()
            torch_sendnn.update_lazyhandle()
            ul_stop_time = time.time()
            ul_total_t = ul_stop_time - ul_start_time
            logger.info("update_lazyhandle() done (duration: %.3fs)",
                        ul_total_t)
            self.perf_metrics.log("update_lazyhandle() time",
                                  ul_total_t,
                                  batch_size=batch_size,
                                  max_tokens=num_decode_tokens,
                                  prompt_len=prompt_len)

        # Second full forward pass
        logger.info("Warmup forward pass 2/2...")
        warmup2_start_t = time.time()
        self._warmup_model_forward_pass(scheduler_output, dummy_requests,
                                        cached_requests, num_decode_tokens)

        warmup_end_t = time.time()
        warmup_total_t = warmup_end_t - warmup_start_t
        self.perf_metrics.log("warmup 2 time",
                              time.time() - warmup2_start_t,
                              batch_size=batch_size,
                              max_tokens=num_decode_tokens,
                              prompt_len=prompt_len)
        logger.info("Warmup finished.")
        logger.info(
            "Warmup took %.3fs (for prompt length %d and max output tokens %d)",
            warmup_total_t, prompt_len, num_decode_tokens)

    def _warmup_model_forward_pass(
        self,
        scheduler_output: SchedulerOutput,
        requests: list[NewRequestData],
        cached_requests: list[CachedRequestData],
        num_decode_tokens,
    ):
        """Handle a complete forward pass"""
        scheduler_output.scheduled_new_reqs = requests
        scheduler_output.scheduled_cached_reqs = []
        self.execute_model(scheduler_output)  # Prefill

        # Switch to cached requests to trigger decoding steps
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = cached_requests
        for _ in range(num_decode_tokens - 1):
            self.execute_model(scheduler_output)

    def profile(self, is_start=True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()

    @property
    def do_metadata_broadcast(self) -> bool:
        return True

    @property
    def kv_cache(self) -> Optional[list[list[torch.Tensor]]]:
        return None

    @SpyrePlatform.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        output = self.model_runner.execute_model(scheduler_output)
        return output if self.is_driver_worker else None

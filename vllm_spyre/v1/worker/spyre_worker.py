"""A Spyre worker class."""
import json
import os
import platform
import time
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from huggingface_hub import hf_hub_download
from vllm.config import VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.core.scheduler import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase as WorkerBaseV1
from vllm.worker.worker_base import WorkerBase

import vllm_spyre.envs as envs_spyre
from vllm_spyre.model_executor.model_loader import spyre_setup
from vllm_spyre.platform import SpyrePlatform
from vllm_spyre.v1.worker.spyre_model_runner import SpyreModelRunner


class SpyreWorker(WorkerBaseV1):
    """A worker class that executes the model on a group of Spyre cores.
    """

    def get_kv_cache_spec(self) -> KVCacheSpec:
        """Get specifications for KV cache implementation."""
        return {
            "foo":
            FullAttentionSpec(block_size=10,
                              num_kv_heads=1,
                              head_size=1,
                              dtype=torch.float16)
        }

    def compile_or_warm_up_model(self) -> None:
        """Prepare model for execution through compilation/warmup."""
        spyre_warmup_shapes = current_platform.get_warmup_shapes()
        wup_prompt_lens, wup_new_tokens = zip(*[(s["prompt_length"],
                                                 s["new_tokens"])
                                                for s in spyre_warmup_shapes])

        print(f"[SpyreWorker] Start warming up "
              f"{len(wup_new_tokens)} "
              f"different prompt/decode/batchsize-shape combinations.")
        all_warmup_start_t = time.time()
        for i, (prompt_len, num_decode_tokens, batch_size) in enumerate([
            (s["prompt_length"], s["new_tokens"], s["batch_size"])
                for s in spyre_warmup_shapes
        ]):
            if self.model_config.task != "embed":
                # TODO: remove if spyre supports
                # lower number of output tokens
                assert num_decode_tokens >= 3, (
                    "VLLM_SPYRE_WARMUP_NEW_TOKENS must be "
                    "at least 2 (spyre requirement).")
            # warmup individual combination
            print(f"[SpyreWorker] Warmup {i+1}/"
                  f"{len(wup_new_tokens)} "
                  f"prompt/decode/batchsize-shape combinations...")
            print(f"[SpyreWorker] Warming up for prompt length {prompt_len}, "
                  f"decoding {num_decode_tokens} tokens with batch "
                  f"size {batch_size}")

            num_scheduled_tokens: dict = {}
            total_num_scheduled_tokens: int = 0
            dummy_requests: list = []
            for i in range(batch_size):
                dummy_requests.append(
                    NewRequestData(
                        req_id=f"warmup-{i}",
                        prompt_token_ids=[1] * prompt_len,
                        prompt="test",
                        mm_inputs=[],
                        mm_hashes=[],
                        mm_positions=[],
                        sampling_params=SamplingParams(
                            max_tokens=num_decode_tokens),
                        block_ids=[0],
                        num_computed_tokens=0,
                        lora_request=None,
                    ))
                num_scheduled_tokens[i] = prompt_len
                total_num_scheduled_tokens += num_scheduled_tokens[i]

            scheduler_output = SchedulerOutput(
                scheduled_new_reqs=dummy_requests,
                scheduled_cached_reqs=[],
                num_scheduled_tokens=num_scheduled_tokens,
                total_num_scheduled_tokens=total_num_scheduled_tokens,
                scheduled_spec_decode_tokens={},
                scheduled_encoder_inputs={},
                num_common_prefix_blocks=0,
                finished_req_ids=set(),
                free_encoder_input_ids=[],
            )

            # Use execute_model for warm up
            self.execute_model(scheduler_output)

            if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND == "sendnn_decoder":
                from torch_sendnn import torch_sendnn
                ul_start_time = time.time()
                torch_sendnn.update_lazyhandle()
                ul_stop_time = time.time()
                ul_total_t = ul_stop_time - ul_start_time
                print(f"update_lazyhandle() done (duration: {ul_total_t}s)")

        all_warmup_end_t = time.time()
        all_warmup_total_t = all_warmup_end_t - all_warmup_start_t
        print(f"[SpyreWorker] All warmups for "
              f"{len(wup_new_tokens)} different "
              f"prompt/decode/batchsize-shape combinations finished. "
              f"Total warmup time {all_warmup_total_t}s.")

    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        # TODO: Implement something!
        return

    def determine_available_memory(self) -> int:
        # TODO: figure out what to do based on determine_num_available_blocks
        return 10 * 1024 * 1024

    def initialize_from_config(self,
                               kv_cache_configs: List[KVCacheConfig]) -> None:
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
        if self.parallel_config and is_driver_worker:
            assert rank % self.parallel_config.tensor_parallel_size == 0, \
                   "Driver worker should be rank 0 of tensor parallel group."
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        if self.model_config.task == "embed":
            raise NotImplementedError
        else:
            self.model_runner = SpyreModelRunner(self.model_config,
                                                 self.parallel_config,
                                                 self.scheduler_config,
                                                 self.device_config,
                                                 self.is_driver_worker)
        self._env_initialized = False

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
                spyre_setup.spyre_setup(rank=0, world_size=1, verbose=True)

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

        print("[SpyreWorker] load model...")
        # TODO: check additionally if the Spyre card has enough memory
        # for all requested model warmups
        # printing env variables for debugging purposes
        load_model_start_t = time.time()
        spyre_warmup_shapes = current_platform.get_warmup_shapes()
        wup_prompt_lens, wup_new_tokens = zip(*[(s["prompt_length"],
                                                 s["new_tokens"])
                                                for s in spyre_warmup_shapes])

        self.model_runner.load_model(prompt_lens=wup_prompt_lens,
                                     num_decode_tokens=wup_new_tokens)

        load_model_end_t = time.time()
        load_model_total_t = load_model_end_t - load_model_start_t
        print(f"\tload model took {load_model_total_t}s")

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks.

        Swapping is not yet supported, so always return num_cpu_blocks=0.

        We configure num_gpu_blocks to be equal to max_num_seqs.
        """
        # Set the number of GPU blocks to be the same as the maximum number of
        # sequences that can be processed in a single batch. This is equivalent
        # to schedule without PagedAttention.
        num_gpu_blocks = self.scheduler_config.max_num_seqs

        # Swap not yet supported with Spyre backend.
        num_cpu_blocks = 0

        return num_gpu_blocks, num_cpu_blocks

    def get_cache_block_size_bytes(self) -> int:
        """Determine the size in bytes of a cache block.

        This is required for speculative decoding; it is not yet implemented.
        """
        raise NotImplementedError

    @property
    def do_metadata_broadcast(self) -> bool:
        return True

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return None

    @SpyrePlatform.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        output = self.model_runner.execute_model(scheduler_output)
        return output if self.is_driver_worker else None

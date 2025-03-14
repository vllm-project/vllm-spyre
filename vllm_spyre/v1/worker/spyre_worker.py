"""A Spyre worker class."""
import json
import os
import platform
import time
from typing import List, Optional

import torch
import torch.distributed as dist
from huggingface_hub import hf_hub_download
from vllm.config import VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.v1.core.scheduler import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
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
            self._warmup_spyre_fixed_size(prompt_len, num_decode_tokens,
                                          self.restricted_tokens, batch_size)
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
                               kv_cache_configs: List[KVCacheConfig]) -> None:
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
            self.model_runner = SpyreModelRunner(self.vllm_config,
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

    def _warmup_spyre_fixed_size(self, prompt_len, num_decode_tokens,
                                 special_token_ids, batch_size):
        # TODO See if we can use `self.execute_model` instead for the warmup
        # It's slightly risky to implement different forward pass logic here,
        # which can go out of sync with the real forward pass and cause problems
        # for torch.compile

        # warmup the model
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

        print(f"[SpyreWorker] warmup for prompt length "
              f"{prompt_len} and max output tokens {num_decode_tokens}.")

        # 1. trace
        print("[SpyreWorker] warmup 1/2...")
        # TODO: torch_sendnn.CleanGraph() should be necessary?
        # warmup 1st forward pass
        self._warmup_model_forward_pass(warmup_tokens_tensor,
                                        valid_token_ids_tensor, prompt_len,
                                        num_decode_tokens, batch_size,
                                        extra_kwargs)

        # 2. compile
        print("[SpyreWorker] warmup 2/2...")
        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND == "sendnn_decoder":
            from torch_sendnn import torch_sendnn
            ul_start_time = time.time()
            torch_sendnn.update_lazyhandle()
            ul_stop_time = time.time()
            ul_total_t = ul_stop_time - ul_start_time
            print(f"update_lazyhandle() done (duration: {ul_total_t}s)")

        # warmup 2nd forward pass
        self._warmup_model_forward_pass(warmup_tokens_tensor,
                                        valid_token_ids_tensor, prompt_len,
                                        num_decode_tokens, batch_size,
                                        extra_kwargs)

        warmup_end_t = time.time()
        warmup_total_t = warmup_end_t - warmup_start_t
        print("[SpyreWorker] ... warmup finished.")
        print(f"\twarmup took {warmup_total_t}s (for prompt length"
              f"{prompt_len} and max output tokens {num_decode_tokens})")

    def _warmup_model_forward_pass(self, warmup_tokens_tensor,
                                   valid_token_ids_tensor, prompt_len,
                                   num_decode_tokens, batch_size,
                                   extra_kwargs):
        # padding warmup tokens to obtain the
        # corresponding position ids and mask
        warmup_tokens_pad, self.model_runner._position_ids, \
        self.model_runner._mask = self.model_runner.pad_input_ids(
            warmup_tokens_tensor, min_pad_length=prompt_len)

        logits, past_key_value_states = self.model_runner._raw_model_forward(
            warmup_tokens_pad,
            position_ids=self.model_runner._position_ids,
            mask=self.model_runner._mask,
            past_key_value_states=None,
            use_cache=True,
            only_last_token=True,
            **extra_kwargs)
        # decoding
        for i in range(num_decode_tokens - 1):
            # sampling next input token from vocab without bos and eos tokens
            decode_tokens = valid_token_ids_tensor[torch.randint(
                0, len(valid_token_ids_tensor), (batch_size, 1))]

            # update mask and position_ids
            self.model_runner._update_mask()
            self.model_runner._update_position_ids()

            if past_key_value_states is not None:
                for layer in past_key_value_states:
                    for tensor in layer:
                        torch._dynamo.mark_dynamic(tensor, 2)

            logits, past_key_value_states = self.model_runner.\
                _raw_model_forward(
                decode_tokens,
                position_ids=self.model_runner._position_ids,
                mask=self.model_runner._mask,
                past_key_value_states=past_key_value_states,
                use_cache=True,
                only_last_token=True,
                **extra_kwargs)

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

import pytest
import torch
from spyre_util import REFERENCE_MODELS, patch_environment
from v1.worker.mock_model import InstrumentedModelRunner
from vllm import EngineArgs

class FP8InstrumentedModelRunner(InstrumentedModelRunner):
    """Extended model runner with FP8 support for testing _set_scale_for_fp8."""

    DEFAULT_TEST_MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b-FP8"

    def __init__(self, vllm_config, is_driver_worker: bool, rank: int):
        super().__init__(vllm_config, is_driver_worker, rank)

        # Set up FP8 KV cache with ScaledTensor
        num_layers = 3
        block_size = 64
        num_kv_heads = 8
        head_dim = 128

        self.model.model.kv_cache_specs = {
            "num_layers": num_layers,
            "block_size": block_size,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        }

        self.model.model.is_fp8_model = True

        # Import and bind the actual methods from ContinuousBatchingFmsModel
        from typing import cast

        from vllm.forward_context import get_forward_context

        from vllm_spyre.model_executor.model_loader.spyre import (
            ContinuousBatchingFmsModel,
        )

        self.model.model.set_past_key_value_states = (
            ContinuousBatchingFmsModel.set_past_key_value_states.__get__(
                self.model.model, type(self.model.model)
            )
        )
        self.model.model._set_scale_for_fp8 = (
            ContinuousBatchingFmsModel._set_scale_for_fp8.__get__(
                self.model.model, type(self.model.model)
            )
        )
        self.model.model._update_scale_for_fp8 = (
            ContinuousBatchingFmsModel._update_scale_for_fp8.__get__(
                self.model.model, type(self.model.model)
            )
        )

        # Bind the actual ContinuousBatchingFmsModel.forward method
        self.model.model.forward = ContinuousBatchingFmsModel.forward.__get__(
            self.model.model, type(self.model.model)
        )

        # Mock the inner self.model call to return dummy logits
        def mock_inner_model_call(
            input_ids,
            position_ids,
            mask,
            past_key_value_states,
            use_cache,
            last_n_tokens,
            current_tkv_mask,
            left_padded_prompt_mask,
            block_table,
            slot_mapping,
            **extra_kwargs,
        ):
            # Return dummy logits and the past_key_value_states
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            vocab_size = vllm_config.model_config.get_vocab_size()
            logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32)
            return logits, past_key_value_states

        self.model.model.model = mock_inner_model_call

    @classmethod
    def build_fp8(
        cls,
        monkeypatch: pytest.MonkeyPatch,
        model_name: str = DEFAULT_TEST_MODEL,
        use_chunked_prefill: bool = False,
        max_num_seqs: int = 4,
        max_model_len: int = 512,
        max_num_batched_tokens: int = 128,
    ):
        """Build an FP8 model runner for testing."""
        patch_environment(
            use_cb=True,
            warmup_shapes=None,
            backend="eager",
            monkeypatch=monkeypatch,
            use_chunked_prefill=use_chunked_prefill,
            max_num_batched_tokens=max_num_batched_tokens,
        )

        model = REFERENCE_MODELS[model_name]

        engine_args = EngineArgs(
            model=model.name,
            tokenizer=model.name,
            revision=model.revision,
            tokenizer_revision=model.revision,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            quantization="compressed-tensors",
            max_num_batched_tokens=max_num_batched_tokens,
            enable_prefix_caching=False,
        )
        vllm_config = engine_args.create_engine_config()

        model_runner = cls(
            vllm_config=vllm_config,
            is_driver_worker=True,
            rank=0,
        )
        model_runner.pre_warmup()
        model_runner.complete_warmup()
        return model_runner


# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _set_scale_for_fp8 function in ContinuousBatchingFmsModel."""

import pytest
import torch
from unittest.mock import patch
from vllm import EngineArgs

from vllm_spyre.model_executor.model_loader.spyre import SpyreAttentionMetadata
from spyre_util import patch_environment, REFERENCE_MODELS
from v1.worker.mock_model import InstrumentedModelRunner

try:
    from fms_mo.aiu_addons.fp8.fp8_utils import ScaledTensor
    FP8_AVAILABLE = True
except ImportError:
    FP8_AVAILABLE = False


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
        
        # Import and bind the actual set_past_key_value_states method from ContinuousBatchingFmsModel
        from vllm_spyre.model_executor.model_loader.spyre import ContinuousBatchingFmsModel
        self.model.model.set_past_key_value_states = ContinuousBatchingFmsModel.set_past_key_value_states.__get__(
            self.model.model, type(self.model.model)
        )
        # Import and bind the actual _set_scale_for_fp8 method from ContinuousBatchingFmsModel
        from vllm_spyre.model_executor.model_loader.spyre import ContinuousBatchingFmsModel
        self.model.model._set_scale_for_fp8 = ContinuousBatchingFmsModel._set_scale_for_fp8.__get__(
            self.model.model, type(self.model.model)
        )
    
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


@pytest.mark.cpu
@pytest.mark.worker
@pytest.mark.skipif(not FP8_AVAILABLE, reason="FP8 support not available")
@pytest.mark.parametrize("use_chunked_prefill", [True, False])
def test_set_scale_for_fp8_prefill_single_request(monkeypatch, use_chunked_prefill):
    """Test _set_scale_for_fp8 for prefill with a single request."""
    monkeypatch.setenv("VLLM_SPYRE_USE_CHUNKED_PREFILL", str(use_chunked_prefill))
    
    model_runner = FP8InstrumentedModelRunner.build_fp8(
        monkeypatch=monkeypatch,
        use_chunked_prefill=use_chunked_prefill,
    )
    
    # Create attention metadata for prefill with single request
    attn_metadata = SpyreAttentionMetadata(
        slot_mapping=torch.zeros(1, 64, dtype=torch.int64),
        current_tkv_mask=torch.ones(1, dtype=torch.bool),
        left_padded_prompt_mask=torch.zeros(1, dtype=torch.bool),
        block_table=torch.zeros(1, 4, dtype=torch.int64),
        is_prefill=True,
        scale_indices=torch.tensor([0], dtype=torch.int64),
    )
    
    # Call the function
    model_runner.model.model._set_scale_for_fp8(attn_metadata)
    
    # Verify scales are set correctly
    for layer_idx, (k, v) in enumerate(model_runner.model.model.past_key_value_states):
        assert k._scale.shape[0] == 1, f"Layer {layer_idx}: Expected scale shape (1,), got {k._scale.shape}"
        assert v._scale.shape[0] == 1, f"Layer {layer_idx}: Expected scale shape (1,), got {v._scale.shape}"
        assert torch.allclose(k._scale, torch.ones(1, dtype=torch.float32))
        assert torch.allclose(v._scale, torch.ones(1, dtype=torch.float32))
        
        if use_chunked_prefill:
            assert k._scaled is True
            assert v._scaled is True
        else:
            assert k._scaled is False
            assert v._scaled is False


@pytest.mark.cpu
@pytest.mark.worker
@pytest.mark.skipif(not FP8_AVAILABLE, reason="FP8 support not available")
@pytest.mark.parametrize("use_chunked_prefill", [True, False])
def test_set_scale_for_fp8_decode_single_request(monkeypatch, use_chunked_prefill):
    """Test _set_scale_for_fp8 for decode with a single request (bs=1)."""
    monkeypatch.setenv("VLLM_SPYRE_USE_CHUNKED_PREFILL", str(use_chunked_prefill))
    
    model_runner = FP8InstrumentedModelRunner.build_fp8(
        monkeypatch=monkeypatch,
        use_chunked_prefill=use_chunked_prefill,
    )
    
    # Create attention metadata for decode with single request
    attn_metadata = SpyreAttentionMetadata(
        slot_mapping=torch.zeros(1, 1, dtype=torch.int64),
        current_tkv_mask=torch.ones(1, dtype=torch.bool),
        left_padded_prompt_mask=torch.zeros(1, dtype=torch.bool),
        block_table=torch.zeros(1, 4, dtype=torch.int64),
        is_prefill=False,
        scale_indices=torch.tensor([0], dtype=torch.int64),
    )
    
    # Call the function
    model_runner.model.model._set_scale_for_fp8(attn_metadata)
    
    # Verify scales are set correctly
    for layer_idx, (k, v) in enumerate(model_runner.model.model.past_key_value_states):
        if use_chunked_prefill:
            # Static scaling for chunked prefill
            assert k._scale.shape[0] == 2, f"Layer {layer_idx}: Expected scale shape (2,), got {k._scale.shape}"
            assert v._scale.shape[0] == 2, f"Layer {layer_idx}: Expected scale shape (2,), got {v._scale.shape}"
            assert torch.allclose(k._scale, torch.ones(2, dtype=torch.float32))
            assert torch.allclose(v._scale, torch.ones(2, dtype=torch.float32))
            assert k._scaled is True
            assert v._scaled is True
        else:
            # For non-chunked prefill, bs=1 decode is padded to bs=2
            assert k._scale.shape[0] == 2, f"Layer {layer_idx}: Expected scale shape (2,), got {k._scale.shape}"
            assert v._scale.shape[0] == 2, f"Layer {layer_idx}: Expected scale shape (2,), got {v._scale.shape}"


@pytest.mark.cpu
@pytest.mark.worker
@pytest.mark.skipif(not FP8_AVAILABLE, reason="FP8 support not available")
@pytest.mark.parametrize("use_chunked_prefill", [True, False])
@pytest.mark.parametrize("batch_size", [2, 3, 4])
def test_set_scale_for_fp8_decode_multiple_requests(monkeypatch, use_chunked_prefill, batch_size):
    """Test _set_scale_for_fp8 for decode with multiple requests."""
    monkeypatch.setenv("VLLM_SPYRE_USE_CHUNKED_PREFILL", str(use_chunked_prefill))
    
    model_runner = FP8InstrumentedModelRunner.build_fp8(
        monkeypatch=monkeypatch,
        use_chunked_prefill=use_chunked_prefill,
        max_num_seqs=batch_size,
    )
    
    # Create attention metadata for decode with multiple requests
    scale_indices = torch.arange(batch_size, dtype=torch.int64)
    attn_metadata = SpyreAttentionMetadata(
        slot_mapping=torch.zeros(batch_size, 1, dtype=torch.int64),
        current_tkv_mask=torch.ones(batch_size, dtype=torch.bool),
        left_padded_prompt_mask=torch.zeros(batch_size, dtype=torch.bool),
        block_table=torch.zeros(batch_size, 4, dtype=torch.int64),
        is_prefill=False,
        scale_indices=scale_indices,
    )
    
    # Call the function
    model_runner.model.model._set_scale_for_fp8(attn_metadata)
    
    # Verify scales are set correctly
    for layer_idx, (k, v) in enumerate(model_runner.model.model.past_key_value_states):
        if use_chunked_prefill:
            # Static scaling for chunked prefill
            assert k._scale.shape[0] == batch_size, f"Layer {layer_idx}: Expected scale shape ({batch_size},), got {k._scale.shape}"
            assert v._scale.shape[0] == batch_size, f"Layer {layer_idx}: Expected scale shape ({batch_size},), got {v._scale.shape}"
            assert torch.allclose(k._scale, torch.ones(batch_size, dtype=torch.float32))
            assert torch.allclose(v._scale, torch.ones(batch_size, dtype=torch.float32))
            assert k._scaled is True
            assert v._scaled is True
        else:
            # For non-chunked prefill, scales should match batch size
            assert k._scale.shape[0] == batch_size, f"Layer {layer_idx}: Expected scale shape ({batch_size},), got {k._scale.shape}"
            assert v._scale.shape[0] == batch_size, f"Layer {layer_idx}: Expected scale shape ({batch_size},), got {v._scale.shape}"


@pytest.mark.cpu
@pytest.mark.worker
@pytest.mark.skipif(not FP8_AVAILABLE, reason="FP8 support not available")
def test_set_scale_for_fp8_scale_persistence(monkeypatch):
    """Test that scales are properly persisted in current_kv_scales for non-chunked prefill."""
    monkeypatch.setenv("VLLM_SPYRE_USE_CHUNKED_PREFILL", "False")
    
    model_runner = FP8InstrumentedModelRunner.build_fp8(
        monkeypatch=monkeypatch,
        use_chunked_prefill=False,
    )
    
    # Prefill for request 0
    attn_metadata_prefill = SpyreAttentionMetadata(
        slot_mapping=torch.zeros(1, 64, dtype=torch.int64),
        current_tkv_mask=torch.ones(1, dtype=torch.bool),
        left_padded_prompt_mask=torch.zeros(1, dtype=torch.bool),
        block_table=torch.zeros(1, 4, dtype=torch.int64),
        is_prefill=True,
        scale_indices=torch.tensor([0], dtype=torch.int64),
    )
    
    model_runner.model.model._set_scale_for_fp8(attn_metadata_prefill)
    
    # Verify prefill sets scale to 1.0 and stores in current_kv_scales
    for layer_idx in range(len(model_runner.model.model.past_key_value_states)):
        k_scale = model_runner.model.model.current_kv_scales[layer_idx][0][0]
        v_scale = model_runner.model.model.current_kv_scales[layer_idx][1][0]
        assert torch.allclose(k_scale, torch.ones(1, dtype=torch.float32))
        assert torch.allclose(v_scale, torch.ones(1, dtype=torch.float32))


@pytest.mark.cpu
@pytest.mark.worker
@pytest.mark.skipif(not FP8_AVAILABLE, reason="FP8 support not available")
@pytest.mark.parametrize("use_chunked_prefill", [True, False])
def test_set_scale_for_fp8_scale_indices_subset(monkeypatch, use_chunked_prefill):
    """Test _set_scale_for_fp8 with a subset of scale indices."""
    monkeypatch.setenv("VLLM_SPYRE_USE_CHUNKED_PREFILL", str(use_chunked_prefill))
    
    model_runner = FP8InstrumentedModelRunner.build_fp8(
        monkeypatch=monkeypatch,
        use_chunked_prefill=use_chunked_prefill,
    )
    
    # Create attention metadata with non-contiguous scale indices
    scale_indices = torch.tensor([0, 2], dtype=torch.int64)
    attn_metadata = SpyreAttentionMetadata(
        slot_mapping=torch.zeros(2, 1, dtype=torch.int64),
        current_tkv_mask=torch.ones(2, dtype=torch.bool),
        left_padded_prompt_mask=torch.zeros(2, dtype=torch.bool),
        block_table=torch.zeros(2, 4, dtype=torch.int64),
        is_prefill=False,
        scale_indices=scale_indices,
    )
    
    # Call the function
    model_runner.model.model._set_scale_for_fp8(attn_metadata)
    
    # Verify scales are set correctly
    for layer_idx, (k, v) in enumerate(model_runner.model.model.past_key_value_states):
        if use_chunked_prefill:
            # Static scaling
            assert k._scale.shape[0] == 2
            assert v._scale.shape[0] == 2
            assert torch.allclose(k._scale, torch.ones(2, dtype=torch.float32))
            assert torch.allclose(v._scale, torch.ones(2, dtype=torch.float32))
        else:
            # Scales should be extracted from current_kv_scales at indices [0, 2]
            assert k._scale.shape[0] == 2
            assert v._scale.shape[0] == 2


@pytest.mark.cpu
@pytest.mark.worker
@pytest.mark.skipif(not FP8_AVAILABLE, reason="FP8 support not available")
def test_set_scale_for_fp8_dynamic_marking(monkeypatch):
    """Test that dynamic marking is applied correctly for non-chunked prefill decode."""
    monkeypatch.setenv("VLLM_SPYRE_USE_CHUNKED_PREFILL", "False")
    
    model_runner = FP8InstrumentedModelRunner.build_fp8(
        monkeypatch=monkeypatch,
        use_chunked_prefill=False,
    )
    
    # Decode with multiple requests
    scale_indices = torch.tensor([0, 1], dtype=torch.int64)
    attn_metadata = SpyreAttentionMetadata(
        slot_mapping=torch.zeros(2, 1, dtype=torch.int64),
        current_tkv_mask=torch.ones(2, dtype=torch.bool),
        left_padded_prompt_mask=torch.zeros(2, dtype=torch.bool),
        block_table=torch.zeros(2, 4, dtype=torch.int64),
        is_prefill=False,
        scale_indices=scale_indices,
    )
    
    with patch('torch._dynamo.mark_dynamic') as mock_mark_dynamic:
        model_runner.model.model._set_scale_for_fp8(attn_metadata)
        
        # Verify mark_dynamic was called for decode (is_dynamic_flag=1)
        # Should be called for each layer's k and v scales
        expected_calls = len(model_runner.model.model.past_key_value_states) * 2
        assert mock_mark_dynamic.call_count == expected_calls
        
        # Verify the second argument (is_dynamic_flag) is 1 for decode
        for call in mock_mark_dynamic.call_args_list:
            args, kwargs = call
            assert args[1] == 1, "Expected is_dynamic_flag=1 for decode"

# Made with Bob

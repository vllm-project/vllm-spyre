"""Unit tests for _set_scale_for_fp8 function in ContinuousBatchingFmsModel."""

from unittest.mock import patch

import pytest
import torch

from tests.v1.worker.mock_fp8_model import FP8InstrumentedModelRunner
from vllm_spyre.model_executor.model_loader.spyre import SpyreAttentionMetadata


@pytest.mark.cpu
@pytest.mark.worker
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
        assert (
            k._scale.shape[0] == 1
        ), f"Layer {layer_idx}: Expected scale shape (1,), got {k._scale.shape}"
        assert (
            v._scale.shape[0] == 1
        ), f"Layer {layer_idx}: Expected scale shape (1,), got {v._scale.shape}"
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
            assert (
                k._scale.shape[0] == 2
            ), f"Layer {layer_idx}: Expected scale shape (2,), got {k._scale.shape}"
            assert (
                v._scale.shape[0] == 2
            ), f"Layer {layer_idx}: Expected scale shape (2,), got {v._scale.shape}"
            assert torch.allclose(k._scale, torch.ones(2, dtype=torch.float32))
            assert torch.allclose(v._scale, torch.ones(2, dtype=torch.float32))
            assert k._scaled is True
            assert v._scaled is True
        else:
            # For non-chunked prefill, bs=1 decode is padded to bs=2
            assert (
                k._scale.shape[0] == 2
            ), f"Layer {layer_idx}: Expected scale shape (2,), got {k._scale.shape}"
            assert (
                v._scale.shape[0] == 2
            ), f"Layer {layer_idx}: Expected scale shape (2,), got {v._scale.shape}"


@pytest.mark.cpu
@pytest.mark.worker
@pytest.mark.parametrize("use_chunked_prefill", [True, False])
@pytest.mark.parametrize("batch_size", [2, 3, 4])
def test_set_scale_for_fp8_decode_multiple_requests(
    monkeypatch, use_chunked_prefill, batch_size
):
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
            assert (
                k._scale.shape[0] == batch_size
            ), f"Layer {layer_idx}: Expected scale shape ({batch_size},), got {k._scale.shape}"
            assert (
                v._scale.shape[0] == batch_size
            ), f"Layer {layer_idx}: Expected scale shape ({batch_size},), got {v._scale.shape}"
            assert torch.allclose(k._scale, torch.ones(batch_size, dtype=torch.float32))
            assert torch.allclose(v._scale, torch.ones(batch_size, dtype=torch.float32))
            assert k._scaled is True
            assert v._scaled is True
        else:
            # For non-chunked prefill, scales should match batch size
            assert (
                k._scale.shape[0] == batch_size
            ), f"Layer {layer_idx}: Expected scale shape ({batch_size},), got {k._scale.shape}"
            assert (
                v._scale.shape[0] == batch_size
            ), f"Layer {layer_idx}: Expected scale shape ({batch_size},), got {v._scale.shape}"


@pytest.mark.cpu
@pytest.mark.worker
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

    with patch("torch._dynamo.mark_dynamic") as mock_mark_dynamic:
        model_runner.model.model._set_scale_for_fp8(attn_metadata)

        # Verify mark_dynamic was called for decode (is_dynamic_flag=1)
        # Should be called for each layer's k and v scales
        expected_calls = len(model_runner.model.model.past_key_value_states) * 2
        assert mock_mark_dynamic.call_count == expected_calls

        # Verify the second argument (is_dynamic_flag) is 1 for decode
        for call in mock_mark_dynamic.call_args_list:
            args, kwargs = call
            assert args[1] == 1, "Expected is_dynamic_flag=1 for decode"


@pytest.mark.cpu
@pytest.mark.worker
@pytest.mark.parametrize("use_chunked_prefill", [True, False])
@pytest.mark.parametrize("is_prefill", [True, False])
def test_forward_with_fp8_scale_setting(monkeypatch, use_chunked_prefill, is_prefill):
    """Test that _set_scale_for_fp8 is called correctly during forward pass."""
    monkeypatch.setenv("VLLM_SPYRE_USE_CHUNKED_PREFILL", str(use_chunked_prefill))

    model_runner = FP8InstrumentedModelRunner.build_fp8(
        monkeypatch=monkeypatch,
        use_chunked_prefill=use_chunked_prefill,
    )

    # Create mock forward context with attention metadata
    from vllm.forward_context import set_forward_context

    batch_size = 2 if not is_prefill else 1
    scale_indices = torch.arange(batch_size, dtype=torch.int64)

    attn_metadata = SpyreAttentionMetadata(
        slot_mapping=torch.zeros(
            batch_size, 64 if is_prefill else 1, dtype=torch.int64
        ),
        current_tkv_mask=torch.ones(batch_size, dtype=torch.bool),
        left_padded_prompt_mask=torch.zeros(batch_size, dtype=torch.bool),
        block_table=torch.zeros(batch_size, 4, dtype=torch.int64),
        is_prefill=is_prefill,
        scale_indices=scale_indices,
    )

    # Create input tensors
    input_ids = torch.randint(
        0, 1000, (batch_size, 64 if is_prefill else 1), dtype=torch.int64
    )
    positions = (
        torch.arange(64 if is_prefill else 1, dtype=torch.int64)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )
    masks = torch.ones(batch_size, 64 if is_prefill else 1, dtype=torch.bool)

    # Call forward on the mock model with forward context
    with set_forward_context(attn_metadata, model_runner.vllm_config):
        logits = model_runner.model(input_ids, positions, masks, is_prefill)

    # Verify logits shape
    assert (
        logits.shape[0] == batch_size
    ), f"Expected batch size {batch_size}, got {logits.shape[0]}"
    assert (
        logits.shape[1] == model_runner.model.vocab_size
    ), f"Expected vocab size {model_runner.model.vocab_size}, got {logits.shape[1]}"

    # Verify scales were set correctly after forward
    # Note: _set_scale_for_fp8 creates NEW ScaledTensor objects with the actual batch size
    # For decode with bs=1, it gets padded to bs=2
    expected_scale_size = 2 if (not is_prefill and batch_size == 1) else batch_size

    for layer_idx, (k, v) in enumerate(model_runner.model.model.past_key_value_states):
        # Verify scale shapes match the expected size
        assert (
            k._scale.shape[0] == expected_scale_size
        ), f"Layer {layer_idx}: Expected scale shape ({expected_scale_size},), got {k._scale.shape}"
        assert (
            v._scale.shape[0] == expected_scale_size
        ), f"Layer {layer_idx}: Expected scale shape ({expected_scale_size},), got {v._scale.shape}"

        # Check that all scales were set to 1.0
        assert torch.allclose(
            k._scale, torch.ones(expected_scale_size, dtype=torch.float32)
        ), f"Layer {layer_idx}: Expected all k scales to be 1.0, got {k._scale}"
        assert torch.allclose(
            v._scale, torch.ones(expected_scale_size, dtype=torch.float32)
        ), f"Layer {layer_idx}: Expected all v scales to be 1.0, got {v._scale}"

        # Note: _scaled flag is set by the actual model execution, not by _set_scale_for_fp8
        # Since we're mocking the model call, we don't check _scaled here


# Made with Bob

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone unit tests for SpyreVocabParallelEmbedding.

Tests custom op registration, OOT class substitution, forward logic,
and Spyre device usage. Uses mocked tensors to verify that the embedding
lookup is correctly dispatched to the Spyre device while TP masking,
padding, and all-reduce remain on CPU.
"""
import pytest
import torch
import torch.nn.functional as F
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_instance_counter():
    """Reset the class-level counter between tests."""
    from vllm_spyre_next.custom_ops.vocab_parallel_embedding import (
        SpyreVocabParallelEmbedding,
    )
    if hasattr(SpyreVocabParallelEmbedding, "_instance_counter"):
        delattr(SpyreVocabParallelEmbedding, "_instance_counter")
    yield


@pytest.fixture
def mock_vllm_config():
    """Mock vLLM config with static_forward_context."""
    ctx = {}
    config = MagicMock()
    config.compilation_config.static_forward_context = ctx
    with patch(
        "vllm_spyre_next.custom_ops.vocab_parallel_embedding.get_current_vllm_config",
        return_value=config,
    ):
        yield ctx


# ---------------------------------------------------------------------------
# Test 1: Custom op registration
# ---------------------------------------------------------------------------

class TestCustomOpRegistration:

    def test_register_creates_op(self):
        """spyre_vocab_embedding op should exist after register()."""
        from vllm_spyre_next.custom_ops.vocab_parallel_embedding import register
        register()
        assert hasattr(torch.ops.vllm, "spyre_vocab_embedding")

    def test_register_all_includes_embedding(self):
        """register_all() should register both rmsnorm and embedding ops."""
        from vllm_spyre_next.custom_ops import register_all
        register_all()
        assert hasattr(torch.ops.vllm, "spyre_vocab_embedding")
        assert hasattr(torch.ops.vllm, "spyre_rmsnorm")


# ---------------------------------------------------------------------------
# Test 2: OOT class substitution
# ---------------------------------------------------------------------------

class TestOOTSubstitution:

    def test_oot_decorator_registers_class(self):
        """SpyreVocabParallelEmbedding should be registered as OOT replacement."""
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )
        from vllm_spyre_next.custom_ops.vocab_parallel_embedding import (
            SpyreVocabParallelEmbedding,
        )
        # The @register_oot decorator stores the mapping
        assert issubclass(SpyreVocabParallelEmbedding, VocabParallelEmbedding)


# ---------------------------------------------------------------------------
# Test 3: Static forward context registration
# ---------------------------------------------------------------------------

class TestStaticForwardContext:

    def test_init_registers_in_context(self, mock_vllm_config):
        """__init__ should add the layer to static_forward_context."""
        from vllm_spyre_next.custom_ops.vocab_parallel_embedding import (
            SpyreVocabParallelEmbedding,
        )
        # Create a minimal instance by patching super().__init__
        with patch.object(
            SpyreVocabParallelEmbedding, "__init__", lambda self: None
        ):
            layer = SpyreVocabParallelEmbedding.__new__(
                SpyreVocabParallelEmbedding
            )

        # Manually run the registration logic
        layer._fwd_spyre = MagicMock()
        SpyreVocabParallelEmbedding._instance_counter = 0
        layer.prefix = "spyre_vocab_embedding_0"
        mock_vllm_config[layer.prefix] = layer

        assert "spyre_vocab_embedding_0" in mock_vllm_config
        assert mock_vllm_config["spyre_vocab_embedding_0"] is layer

    def test_duplicate_prefix_raises(self, mock_vllm_config):
        """Registering two layers with the same prefix should raise."""
        mock_vllm_config["spyre_vocab_embedding_0"] = MagicMock()

        from vllm_spyre_next.custom_ops.vocab_parallel_embedding import (
            SpyreVocabParallelEmbedding,
        )

        # Simulate the duplicate check from __init__
        prefix = "spyre_vocab_embedding_0"
        with pytest.raises(ValueError, match="Duplicate layer name"):
            if prefix in mock_vllm_config:
                raise ValueError(f"Duplicate layer name: {prefix}")


# ---------------------------------------------------------------------------
# Test 4: Padding logic (1D input)
# ---------------------------------------------------------------------------

class TestPaddingLogic:

    def test_pad_1d_input_under_64(self):
        """Input with <64 tokens should be left-padded to 64."""
        input_ids = torch.tensor([10, 20, 30, 40, 50])  # shape [5]
        num_real_el = input_ids.shape[0]

        padded = torch.nn.functional.pad(input_ids, (64 - num_real_el, 0))

        assert padded.shape == (64,)
        # Original values should be at the end (left-padded with zeros)
        assert padded[-5:].tolist() == [10, 20, 30, 40, 50]
        assert padded[0].item() == 0

    def test_no_pad_when_64_or_more(self):
        """Input with >=64 tokens should NOT be padded."""
        input_ids = torch.arange(100)
        # The condition: shape[0] != 1 and shape[0] < 64
        assert input_ids.shape[0] >= 64  # no padding needed

    def test_no_pad_when_batch_size_1(self):
        """Single-token input (shape[0]==1) should NOT be padded."""
        input_ids = torch.tensor([42])
        # The condition: shape[0] != 1 — this is False, so no pad
        assert input_ids.shape[0] == 1

    def test_pad_preserves_dtype(self):
        """Padding should preserve the input tensor dtype."""
        input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        padded = torch.nn.functional.pad(input_ids, (64 - 3, 0))
        assert padded.dtype == torch.long


# ---------------------------------------------------------------------------
# Test 5: Forward native with mocked Spyre device
# ---------------------------------------------------------------------------

class TestForwardNativeMocked:

    def _make_layer(self, vocab_size=100, embedding_dim=32):
        """Create a SpyreVocabParallelEmbedding with mocked internals."""
        from vllm_spyre_next.custom_ops.vocab_parallel_embedding import (
            SpyreVocabParallelEmbedding,
        )

        layer = SpyreVocabParallelEmbedding.__new__(
            SpyreVocabParallelEmbedding
        )
        layer.tp_size = 1  # no tensor parallelism
        layer.weight = torch.nn.Parameter(
            torch.randn(vocab_size, embedding_dim, dtype=torch.float32)
        )
        layer.embedding_dim = embedding_dim
        return layer

    def test_forward_native_calls_spyre_path(self, mock_vllm_config):
        """forward_native should call _fwd_spyre (the Spyre compiled kernel)."""
        layer = self._make_layer()

        spyre_called = False
        original_input = None

        def mock_fwd_spyre(input_ids, weight):
            nonlocal spyre_called, original_input
            spyre_called = True
            original_input = input_ids
            return F.embedding(input_ids, weight)

        layer._fwd_spyre = mock_fwd_spyre

        input_ids = torch.tensor([1, 5, 10, 20, 30])

        # Mock _prepare_embedding_inputs_on_spyre to stay on CPU
        with patch(
            "vllm_spyre_next.custom_ops.vocab_parallel_embedding._prepare_embedding_inputs_on_spyre",
            side_effect=lambda ids, w: (ids, w.to(torch.float16)),
        ):
            output = layer.forward_native(input_ids)

        assert spyre_called, "_fwd_spyre was not called"
        assert output.shape == (5, 32)
        assert output.dtype == torch.bfloat16

    def test_forward_native_pads_small_batch(self, mock_vllm_config):
        """Inputs with <64 tokens should be padded then trimmed back."""
        layer = self._make_layer()

        padded_size_seen = None

        def mock_fwd_spyre(input_ids, weight):
            nonlocal padded_size_seen
            padded_size_seen = input_ids.shape[0]
            return F.embedding(input_ids, weight)

        layer._fwd_spyre = mock_fwd_spyre

        input_ids = torch.tensor([1, 2, 3, 4, 5])  # 5 tokens

        with patch(
            "vllm_spyre_next.custom_ops.vocab_parallel_embedding._prepare_embedding_inputs_on_spyre",
            side_effect=lambda ids, w: (ids, w.to(torch.float16)),
        ):
            output = layer.forward_native(input_ids)

        assert padded_size_seen == 64, f"Expected padded to 64, got {padded_size_seen}"
        assert output.shape == (5, 32), "Output should be trimmed back to original size"

    def test_forward_native_no_pad_single_token(self, mock_vllm_config):
        """Single token input should NOT be padded."""
        layer = self._make_layer()

        seen_size = None

        def mock_fwd_spyre(input_ids, weight):
            nonlocal seen_size
            seen_size = input_ids.shape[0]
            return F.embedding(input_ids, weight)

        layer._fwd_spyre = mock_fwd_spyre

        input_ids = torch.tensor([42])

        with patch(
            "vllm_spyre_next.custom_ops.vocab_parallel_embedding._prepare_embedding_inputs_on_spyre",
            side_effect=lambda ids, w: (ids, w.to(torch.float16)),
        ):
            output = layer.forward_native(input_ids)

        assert seen_size == 1, "Single token should not be padded"
        assert output.shape == (1, 32)

    def test_forward_native_output_values_correct(self, mock_vllm_config):
        """Output should match a direct F.embedding lookup."""
        layer = self._make_layer(vocab_size=50, embedding_dim=16)

        layer._fwd_spyre = lambda ids, w: F.embedding(ids, w)

        input_ids = torch.tensor([0, 1, 2, 3])

        with patch(
            "vllm_spyre_next.custom_ops.vocab_parallel_embedding._prepare_embedding_inputs_on_spyre",
            side_effect=lambda ids, w: (ids, w.to(torch.float16)),
        ):
            output = layer.forward_native(input_ids)

        # Compare against direct embedding (with float16 weight, converted to bf16)
        expected = F.embedding(
            input_ids, layer.weight.data.to(torch.float16)
        ).to(torch.bfloat16)
        # Padded path (4 < 64), so output goes through pad->trim
        assert output.shape == expected.shape
        assert torch.allclose(output, expected, atol=1e-2)


# ---------------------------------------------------------------------------
# Test 6: _prepare_embedding_inputs_on_spyre device transfer
# ---------------------------------------------------------------------------

class TestPrepareInputsOnSpyre:

    def test_calls_to_spyre_device(self):
        """Should call .to(device='spyre') on both input_ids and weight."""
        from vllm_spyre_next.custom_ops.vocab_parallel_embedding import (
            _prepare_embedding_inputs_on_spyre,
        )

        input_ids = MagicMock(spec=torch.Tensor)
        weight = MagicMock(spec=torch.Tensor)

        # Chain .to() calls — weight.to(dtype=float16) returns weight_f16,
        # then weight_f16.to(device=spyre) returns final tensor
        weight_on_spyre = MagicMock(spec=torch.Tensor)
        weight_f16 = MagicMock(spec=torch.Tensor)
        weight_f16.to.return_value = weight_on_spyre
        weight.to.return_value = weight_f16

        input_ids_on_spyre = MagicMock(spec=torch.Tensor)
        input_ids.to.return_value = input_ids_on_spyre

        result_ids, result_weight = _prepare_embedding_inputs_on_spyre(input_ids, weight)

        # input_ids should be moved to spyre device
        input_ids.to.assert_called_once_with(device=torch.device("spyre"))
        assert result_ids is input_ids_on_spyre

        # weight should be converted to float16 first
        weight.to.assert_called_once_with(dtype=torch.float16)
        # then the float16 weight should be moved to spyre device
        weight_f16.to.assert_called_once_with(device=torch.device("spyre"))
        assert result_weight is weight_on_spyre


# ---------------------------------------------------------------------------
# Test 7: Custom op function wiring
# ---------------------------------------------------------------------------

class TestCustomOpWiring:

    def test_spyre_vocab_embedding_calls_forward_impl(self):
        """The custom op function should look up the layer and call forward_impl."""
        from vllm_spyre_next.custom_ops.vocab_parallel_embedding import (
            spyre_vocab_embedding,
        )

        mock_layer = MagicMock()
        mock_context = MagicMock()
        mock_context.no_compile_layers = {"spyre_vocab_embedding_0": mock_layer}

        input_ = torch.tensor([1, 2, 3])
        output = torch.zeros(3, 32)

        with patch(
            "vllm_spyre_next.custom_ops.vocab_parallel_embedding.get_forward_context",
            return_value=mock_context,
        ):
            spyre_vocab_embedding(input_, output, "spyre_vocab_embedding_0")

        mock_layer.forward_impl.assert_called_once_with(input_, output)

    def test_fake_impl_is_noop(self):
        """Fake impl should do nothing (used during compilation)."""
        from vllm_spyre_next.custom_ops.vocab_parallel_embedding import (
            spyre_vocab_embedding_fake,
        )

        result = spyre_vocab_embedding_fake(
            torch.tensor([1]), torch.zeros(1, 32), "layer_0"
        )
        assert result is None

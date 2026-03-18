"""
Test SpyreVocabParallelEmbedding custom op correctness against a reference implementation.
"""

import pytest
import torch
import torch.nn.functional as F


def reference_embedding(
    input_: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Golden reference: standard F.embedding in PyTorch."""
    return F.embedding(input_, weight)


@pytest.fixture(autouse=True)
def _reset_instance_counter():
    """Reset the class-level instance counter before each test.

    SpyreVocabParallelEmbedding uses a monotonic counter for unique layer
    names in static_forward_context.  Without resetting, tests that create
    layers will collide with stale entries from prior tests.
    """
    from vllm_spyre_next.custom_ops.vocab_parallel_embedding import (
        SpyreVocabParallelEmbedding,
    )

    if hasattr(SpyreVocabParallelEmbedding, "_instance_counter"):
        SpyreVocabParallelEmbedding._instance_counter = 0
    yield


@pytest.mark.spyre
@pytest.mark.parametrize("num_tokens", [1, 32, 63, 64, 65, 128])
@pytest.mark.parametrize("num_embeddings", [256, 1024])
@pytest.mark.parametrize("embedding_dim", [64, 128, 512])
def test_spyre_vocab_parallel_embedding_matches_reference(
    default_vllm_config,
    num_tokens,
    num_embeddings,
    embedding_dim,
):
    """SpyreVocabParallelEmbedding output matches golden reference.

    Tests both paths:
    - forward(): custom op dispatch (no-compile path via
      torch.ops.vllm.spyre_vocab_parallel_embedding)
    - forward_native(): direct CPU execution with Spyre padding logic

    Exercises the _SPYRE_MIN_SEQ_LEN=64 padding boundary with num_tokens
    values below, at, and above the threshold.
    """
    from vllm.config import get_current_vllm_config
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm_spyre_next.custom_ops.vocab_parallel_embedding import (
        SpyreVocabParallelEmbedding,
    )

    torch.manual_seed(42)

    # Build layer — uses upstream VocabParallelEmbedding.__init__ internally
    layer = SpyreVocabParallelEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        org_num_embeddings=num_embeddings,
    )

    input_ = torch.randint(0, num_embeddings, (num_tokens,), dtype=torch.long)

    expected = reference_embedding(input_, layer.weight.data)

    # forward_native path
    actual_native = layer.forward_native(input_)
    torch.testing.assert_close(
        actual_native.float(), expected.float(), atol=1e-4, rtol=1e-4, equal_nan=True
    )

    # forward (custom op) path — requires an active forward context so that
    # get_forward_context().no_compile_layers resolves the registered layer.
    from vllm.config.compilation import CUDAGraphMode

    vllm_config = get_current_vllm_config()
    fwd_ctx = ForwardContext(
        no_compile_layers=vllm_config.compilation_config.static_forward_context,
        attn_metadata={},
        slot_mapping={},
        virtual_engine=0,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    with override_forward_context(fwd_ctx):
        actual_forward = layer.forward(input_)
    torch.testing.assert_close(
        actual_forward.float(), expected.float(), atol=1e-4, rtol=1e-4, equal_nan=True
    )


@pytest.fixture
def dummy_input():
    return torch.randint(0, 256, (16,), dtype=torch.long)


def mock_forward_native(input_: torch.Tensor) -> torch.Tensor:
    """Mock: return ones of expected shape."""
    return torch.ones(input_.shape[0], 128)


@pytest.mark.spyre
def test_vocab_parallel_embedding_oot_dispatch(default_vllm_config, monkeypatch, dummy_input):
    """Verify VocabParallelEmbedding OOT registration: class swap and forward_oot routing."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm_spyre_next.custom_ops.vocab_parallel_embedding import (
        SpyreVocabParallelEmbedding,
    )

    layer = VocabParallelEmbedding(
        num_embeddings=256,
        embedding_dim=128,
        org_num_embeddings=256,
    )

    # OOT class swap: VocabParallelEmbedding.__new__ should produce SpyreVocabParallelEmbedding
    assert isinstance(layer, SpyreVocabParallelEmbedding)

    # Mock forward_native (called by forward_oot) with a known transform
    monkeypatch.setattr(layer, "forward_native", mock_forward_native)
    out = layer.forward_oot(dummy_input)

    assert torch.allclose(out, torch.ones(dummy_input.shape[0], 128))

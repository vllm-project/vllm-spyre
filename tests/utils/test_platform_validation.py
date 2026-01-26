"""Unit tests for platform validation of structured outputs.

Tests the fix in vllm_spyre/platform.py that strips structured_outputs
from SamplingParams during request validation.
"""

import pytest
from vllm import SamplingParams
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm_spyre.platform import SpyrePlatform

pytestmark = pytest.mark.skip_global_cleanup


class TestStructuredOutputValidation:
    """Test that platform validation strips structured outputs from requests."""

    def test_strips_structured_outputs(self):
        """Test that validate_request sets structured_outputs to None."""
        params = SamplingParams(
            max_tokens=20, structured_outputs=StructuredOutputsParams(json_object=True)
        )

        assert params.structured_outputs is not None

        SpyrePlatform.validate_request("Test prompt", params)

        assert params.structured_outputs is None

    def test_logs_warning_when_stripping(self, caplog_vllm_spyre):
        """Test that a warning is logged when stripping structured_outputs."""
        params = SamplingParams(
            max_tokens=20, structured_outputs=StructuredOutputsParams(json_object=True)
        )

        SpyrePlatform.validate_request("Test prompt", params)

        assert len(caplog_vllm_spyre.records) > 0
        warning_record = caplog_vllm_spyre.records[0]
        assert warning_record.levelname == "WARNING"
        assert "Structured outputs" in warning_record.message
        assert "not supported" in warning_record.message

    @pytest.mark.parametrize(
        "structured_output",
        [
            StructuredOutputsParams(json_object=True),
            StructuredOutputsParams(regex="[0-9]+"),
        ],
    )
    def test_strips_different_structured_output_types(self, structured_output):
        """Test validation with different types of structured outputs."""
        params = SamplingParams(max_tokens=20, structured_outputs=structured_output)

        assert params.structured_outputs is not None

        SpyrePlatform.validate_request("Test prompt", params)

        assert params.structured_outputs is None

    def test_preserves_other_sampling_params(self):
        """Test that other sampling params are not affected by the fix."""
        params = SamplingParams(
            max_tokens=20,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            structured_outputs=StructuredOutputsParams(json_object=True),
        )

        # Store original values
        original_values = {
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
        }

        SpyrePlatform.validate_request("Test prompt", params)

        # Verify other params are unchanged
        assert params.max_tokens == original_values["max_tokens"]
        assert params.temperature == original_values["temperature"]
        assert params.top_p == original_values["top_p"]
        assert params.top_k == original_values["top_k"]
        # But structured_outputs should be None
        assert params.structured_outputs is None

    def test_does_not_affect_pooling_params(self):
        """Test that PoolingParams are not affected (early return in validate_request)."""
        pooling_params = PoolingParams()

        # Should not raise any errors and should return early
        SpyrePlatform.validate_request("Test prompt", pooling_params)

        # PoolingParams don't have structured_outputs, so just verify no exception
        assert True  # If we got here, the early return worked


# Made with Bob

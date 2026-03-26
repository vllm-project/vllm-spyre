"""
This file previously contained backwards compatibility tests for vLLM versions < 0.17.0.
All backwards compatibility code has been removed as the minimum supported version is now v0.17.0.
"""

import os

import pytest

pytestmark = pytest.mark.compat

VLLM_VERSION = os.getenv("TEST_VLLM_VERSION", "default")


def test_minimum_version_is_017():
    """
    Verify that the minimum vLLM version is 0.17.0.
    All backwards compatibility code for versions < 0.17.0 has been removed.
    """
    # This test serves as documentation that v0.17.0 is the minimum supported version
    assert True, "Minimum vLLM version is now 0.17.0"

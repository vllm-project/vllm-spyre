"""Tests for our environment configs"""

import os

import pytest

from vllm_spyre import envs


@pytest.mark.cpu
def test_env_vars_are_cached(monkeypatch):
    monkeypatch.setenv("VLLM_SPYRE_NUM_CPUS", "42")
    assert envs.VLLM_SPYRE_NUM_CPUS == 42

    # Future reads don't query the environment every time, so this should not
    # return the updated value
    monkeypatch.setenv("VLLM_SPYRE_NUM_CPUS", "77")
    assert envs.VLLM_SPYRE_NUM_CPUS == 42


@pytest.mark.cpu
def test_env_vars_override(monkeypatch):
    monkeypatch.setenv("VLLM_SPYRE_NUM_CPUS", "42")
    assert envs.VLLM_SPYRE_NUM_CPUS == 42

    # This override both sets the environment variable and updates our cache
    envs.override("VLLM_SPYRE_NUM_CPUS", "77")
    assert envs.VLLM_SPYRE_NUM_CPUS == 77
    assert os.getenv("VLLM_SPYRE_NUM_CPUS") == "77"


@pytest.mark.cpu
def test_env_vars_override_with_bad_value(monkeypatch):
    monkeypatch.setenv("VLLM_SPYRE_NUM_CPUS", "42")
    assert envs.VLLM_SPYRE_NUM_CPUS == 42

    # The environment should not be updated if the config is invalid
    envs.override("VLLM_SPYRE_NUM_CPUS", "notanumber")
    assert envs.VLLM_SPYRE_NUM_CPUS == 42
    assert os.getenv("VLLM_SPYRE_NUM_CPUS") == "42"


@pytest.mark.cpu
def test_sendnn_decoder_backwards_compat(monkeypatch):
    # configuring the deprecated `sendnn_decoder` backend will swap to the new
    # `sendnn` backend instead
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", "sendnn_decoder")
    assert envs.VLLM_SPYRE_DYNAMO_BACKEND == "sendnn"

import sys

import pytest
from spyre_util import patch_environment
from vllm.config import VllmConfig
from vllm.entrypoints.cli.main import main as vllm_cli_main


class ProcessedVllmConfig(Exception):

    def __init__(self, vllm_config):
        super().__init__("VllmConfig")
        self.config = vllm_config


def get_final_vllm_config(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", argv)
    patch_environment(
        use_cb=True,
        warmup_shapes=None,
        backend='eager',
        monkeypatch=monkeypatch,
        use_chunked_prefill=True,
    )
    # monkeypatch to capture the processed VllmConfig
    orig_func = VllmConfig.__post_init__

    def wrapped_post_init(self) -> None:
        orig_func(self)
        raise ProcessedVllmConfig(self)

    monkeypatch.setattr(VllmConfig, "__post_init__", wrapped_post_init)

    with pytest.raises(ProcessedVllmConfig) as vllm_config_container:
        vllm_cli_main()

    return vllm_config_container.value.config


def test_default_config_and_args(monkeypatch):

    vllm_config = get_final_vllm_config(
        ["vllm", "serve", "ibm-ai-platform/micro-g3.3-8b-instruct-1b"],
        monkeypatch)

    assert vllm_config.scheduler_config.max_num_batched_tokens == 1024

from typing import Optional

import torch
from llm_cache import patch_environment
from spyre_util import ModelInfo
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor


def test_custom_logits_processor(model: ModelInfo, backend, monkeypatch,
                                 warmup_shapes, cb):
    '''
    Simple test to check if custom logits processors are being registered 
    '''

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    has_invoked_logits_processor = False

    class DummyLogitsProcessor(LogitsProcessor):

        def __init__(self, vllm_config: VllmConfig, device: torch.device,
                     is_pin_memory: bool):
            # Required to register LogitsProcessor
            pass

        def is_argmax_invariant(self) -> bool:
            return False

        def update_state(self, batch_update: Optional[BatchUpdate]):
            # Required to register LogitsProcessor
            pass

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            nonlocal has_invoked_logits_processor
            has_invoked_logits_processor = True
            return logits

    patch_environment(cb == 1, warmup_shapes if cb == 0 else None, backend,
                      monkeypatch)

    spyre_model = LLM(model=model.name,
                      revision=model.revision,
                      max_model_len=128,
                      logits_processors=[DummyLogitsProcessor])
    prompt = "Hello Logits Processors"
    params = SamplingParams(max_tokens=5, temperature=0, logprobs=0)

    spyre_model.generate(prompt, params)[0]

    assert has_invoked_logits_processor

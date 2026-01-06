import pytest
import torch
from llm_cache import patch_environment
from llm_cache_util import force_engine_shutdown
from spyre_util import ModelInfo
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor, MoveDirectionality


def test_custom_logits_processor(
    model: ModelInfo, backend, monkeypatch, max_num_seqs, max_model_len, warmup_shapes, mode: str
):
    """
    Simple test to check if custom logits processors are being registered
    """

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    has_invoked_logits_processor = False

    class DummyLogitsProcessor(LogitsProcessor):
        def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
            # Required to register LogitsProcessor
            pass

        def is_argmax_invariant(self) -> bool:
            return False

        def update_state(self, batch_update: BatchUpdate | None):
            # Required to register LogitsProcessor
            pass

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            nonlocal has_invoked_logits_processor
            has_invoked_logits_processor = True
            return logits

    patch_environment(
        use_cb=mode in ["cb", "cp", "pc"],
        warmup_shapes=warmup_shapes if mode == "sb" else None,
        backend=backend,
        use_chunked_prefill=mode in ["cp", "pc"],
        monkeypatch=monkeypatch,
    )

    spyre_model = LLM(
        model=model.name,
        revision=model.revision,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=128 if mode in ["cp", "pc"] else None,
        enable_prefix_caching=mode == "pc",
        logits_processors=[DummyLogitsProcessor],
    )
    prompt = "Hello Logits Processors"
    params = SamplingParams(max_tokens=5, temperature=0, logprobs=0)

    spyre_model.generate(prompt, params)
    force_engine_shutdown(spyre_model)

    assert has_invoked_logits_processor


@pytest.mark.cb
def test_cb_logits_processor(model: ModelInfo, backend, monkeypatch, max_model_len):
    """
    Test if the state of logits for CB are correct due to the switch of
    prefill/decode in a step engine. The LLM is initialized with bs=2,
    we send 3 requests, one of them should be waiting for the other 2
    to complete. The first request should finish and give its slot to
    the last one. The logits processors will do a greedy sampling
    decoding to emulate the 'state' of the logit processor. After
    the generation we assert that the generated output is the same
    for the spy and vllm.
    """

    # Same process to ease things
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # Hack to collect outputs from logits, the key
    # is the max_tokens to ease identify the requests
    spy_outputs: dict[int, list[int]] = {}

    class SpyLogitsProcessor(LogitsProcessor):
        """
        This logits processor collect the tokens
        """

        def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
            self.req_info: dict[int, SamplingParams] = {}

        def is_argmax_invariant(self) -> bool:
            return False

        def update_state(self, batch_update: BatchUpdate | None):
            if not batch_update:
                return

            for index, params, _, _ in batch_update.added:
                self.req_info[index] = params
                nonlocal spy_outputs
                spy_outputs[params.max_tokens] = []

            if self.req_info:
                # Process removed requests.
                for index in batch_update.removed:
                    self.req_info.pop(index, None)

                # Process moved requests, unidirectional move (a->b) and swap
                # (a<->b)
                for adx, bdx, direct in batch_update.moved:
                    a_val = self.req_info.pop(adx, None)
                    b_val = self.req_info.pop(bdx, None)
                    if a_val is not None:
                        self.req_info[bdx] = a_val
                    if direct == MoveDirectionality.SWAP and b_val is not None:
                        self.req_info[adx] = b_val

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            if not self.req_info:
                return
            batch_size = logits.shape[0]
            nonlocal spy_outputs
            for i in range(batch_size):
                params = self.req_info[i]
                token_id = logits[i].argmax(-1).reshape(-1).item()
                spy_outputs[params.max_tokens].append(token_id)
            return logits

    patch_environment(True, None, backend, monkeypatch)

    spyre_model = LLM(
        model=model.name,
        revision=model.revision,
        max_model_len=max_model_len,
        max_num_seqs=2,
        logits_processors=[SpyLogitsProcessor],
    )
    prompt = ["Hello Logits Processors"] * 3
    params0 = SamplingParams(max_tokens=5, temperature=0, logprobs=0, ignore_eos=True)
    params1 = SamplingParams(max_tokens=10, temperature=0, logprobs=0, ignore_eos=True)
    params2 = SamplingParams(max_tokens=7, temperature=0, logprobs=0, ignore_eos=True)

    # clear from the warmup
    spy_outputs = {}
    params = [params0, params1, params2]
    outputs = spyre_model.generate(prompt, params)
    force_engine_shutdown(spyre_model)

    assert spy_outputs[5] == outputs[0].outputs[0].token_ids
    assert spy_outputs[10] == outputs[1].outputs[0].token_ids
    assert spy_outputs[7] == outputs[2].outputs[0].token_ids

import math
from typing import Optional

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.v1.sample.logits_processor import (BatchUpdate, LogitsProcessor,
                                             MoveDirectionality)


class GoldenTokenInjectorState:

    def __init__(self, input_dict: dict):
        self.expected_token_ids = input_dict["expected_token_ids"]
        self.expected_logprobs = input_dict["expected_logprobs"]
        self.error_threshold = input_dict["error_threshold"]

        self.current_token_idx = 0
        self.has_error = False


class GoldenTokenInjector(LogitsProcessor):
    """Logit processor to inject expected token during generation for tests"""

    def __init__(self, vllm_config: VllmConfig, device: torch.device,
                 is_pin_memory: bool):
        self.injectors: dict[int, GoldenTokenInjectorState] = {}
        # NOTE: This logit processor hold a tokenizer for each instance.
        # for couple requests that does not
        self.tokenizer = get_tokenizer(vllm_config.model_config.tokenizer)

    def is_argmax_invariant(self) -> bool:
        """Never impacts greedy sampling"""
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        # This method keeps the indices consistent of request while the
        # persistent batch is changing.
        if not batch_update:
            return

        # Process added requests.
        for index, params, _, _ in batch_update.added:
            assert params is not None
            if params.extra_args and (
                    injector_dict :=
                    params.extra_args.get("golden_token_injector")):
                self.injectors[index] = GoldenTokenInjectorState(injector_dict)

        if self.injectors:
            # Process removed requests.
            for index in batch_update.removed:
                self.injectors.pop(index, None)

            # Process moved requests, unidirectional move (a->b) and swap
            # (a<->b)
            for adx, bdx, direct in batch_update.moved:
                a_val = self.injectors.pop(adx, None)
                b_val = self.injectors.pop(bdx, None)
                if a_val is not None:
                    self.injectors[bdx] = a_val
                if direct == MoveDirectionality.SWAP and b_val is not None:
                    self.injectors[adx] = b_val

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.injectors:
            return logits

        # Calculate logprobs for the current model execution
        logprobs = F.log_softmax(logits, dim=-1)

        for req_idx, injector in self.injectors.items():

            if injector.has_error:
                # There was an error already for inject tokens for this
                # request skip until the end.
                continue

            expected_token_id = injector.expected_token_ids[
                injector.current_token_idx]
            token_id = torch.argmax(logits[req_idx], dim=-1)

            if expected_token_id == token_id:
                # Expectation is met, nothing to do.
                injector.current_token_idx += 1
                continue

            # Get the logprob for the expected token
            lp = logprobs[req_idx][expected_token_id].reshape(-1)
            prob = torch.exp(lp)

            expected_logprob = \
                injector.expected_logprobs[injector.current_token_idx]
            expected_prob = math.exp(expected_logprob)

            if not math.isclose(expected_prob,
                                prob.item(),
                                abs_tol=injector.error_threshold):
                err = abs(expected_prob - prob.item())

                print("Token probability is out of the acceptable threshold "
                      f"{err} > {injector.error_threshold} at request "
                      f"'{req_idx}' token idx '{injector.current_token_idx}'."
                      "Token injection will be skipped")
                injector.has_error = True
                continue

            full_prob = torch.ones(1, dtype=prob.dtype)  # 100%

            # Keep the same logprob for the expected token and
            # redistribute evenly the probability among the other
            # token ids.
            # NOTE: se are setting logprobs to the logits, if we recalculate
            # the softmax again over this distribution we shall find the same
            # values, but with some minimal difference. The intention is
            # inject the golden token but preserving the original logprob.

            other_token_ids_count = logits.shape[1] - 1
            other_logprobs = torch.log(
                (full_prob - prob) / other_token_ids_count)

            if lp < other_logprobs:
                print("The logprob is lower than the redistributed "
                      "logprobs for the token ids "
                      f"({lp.item()} < {other_logprobs.item()}), this "
                      "suggests that the generation diverged too much "
                      "from the expectation.")
                injector.has_error = True
                continue

            logits[req_idx] = other_logprobs
            logits[req_idx][expected_token_id] = lp

            print(f"Golden token injection for request idx '{req_idx}'"\
                  f" at index '{injector.current_token_idx}':")

            token = self.tokenizer.decode([token_id])
            expected_token = self.tokenizer.decode([expected_token_id])
            print(f"'{token}' {prob.item() * 100} -> "
                  f"'{expected_token}' {expected_logprob * 100}%")
            injector.current_token_idx += 1

        return logits

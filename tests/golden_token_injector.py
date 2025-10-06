import math
from typing import Optional

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.v1.sample.logits_processor import (BatchUpdate, LogitsProcessor,
                                             MoveDirectionality)


class ExpectationState:
    '''
    This class controls the state of the generation.
    Args:
        expected_token_ids: Expected tokens ids
        expected_logprobs: Expected logprobs
        error_threshold: Acceptable threshold to keep the injection. If it is 
            over the threshold, we stop the injection and give feedback at the
            end of the generation that this token is diverging too much.
        label: Used to identify the request, ideally it would be the request 
            id. However we might not have that yet, therefore we have the
            opportunity to add a more human friendly label. It is used to log 
            which requests are being injected with the golden token.
    '''

    def __init__(self,
                 expected_token_ids: list[int],
                 expected_logprobs: list[float],
                 error_threshold: float,
                 label: Optional[str] = None):

        self.token_ids: list[int] = expected_token_ids
        self.logprobs: list[float] = expected_logprobs
        self.threshold: float = error_threshold
        self.label: Optional[str] = label

        self.current_token_idx = 0
        self.has_error = False


class GoldenTokenInjector(LogitsProcessor):
    """Logit processor to inject expected token during generation for tests"""

    def __init__(self, vllm_config: VllmConfig, device: torch.device,
                 is_pin_memory: bool):
        self.req_states: dict[int, ExpectationState] = {}
        # NOTE: This logit processor hold a tokenizer for each instance.
        # for couple requests that does not have too much impact.
        # But since this is used mostly for validation, it would be fine
        # to keep them.
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
                self.req_states[index] = ExpectationState(**injector_dict)

        if not self.req_states:
            return

        # Process removed requests.
        for index in batch_update.removed:
            self.req_states.pop(index, None)

        # Process moved requests, unidirectional move (a->b) and swap
        # (a<->b)
        for adx, bdx, direct in batch_update.moved:
            a_val = self.req_states.pop(adx, None)
            b_val = self.req_states.pop(bdx, None)
            if a_val is not None:
                self.req_states[bdx] = a_val
            if direct == MoveDirectionality.SWAP and b_val is not None:
                self.req_states[adx] = b_val

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_states:
            return logits

        # Calculate logprobs for the current model execution
        logprobs = F.log_softmax(logits, dim=-1)

        for req_idx, expectation in self.req_states.items():

            if expectation.has_error:
                # There was an error already for inject tokens for this
                # request, skip until the end of its generation.
                continue

            expected_token_id = expectation.token_ids[
                expectation.current_token_idx]
            token_id = torch.argmax(logits[req_idx], dim=-1)

            if expected_token_id == token_id:
                # Expectation is met, nothing to do.
                expectation.current_token_idx += 1
                continue

            # Get the logprob for the expected token
            lp = logprobs[req_idx][expected_token_id].reshape(-1)
            prob = torch.exp(lp).item()

            expected_logprob = \
                expectation.logprobs[expectation.current_token_idx]
            expected_prob = math.exp(expected_logprob)

            # Label to identify request, if the label was set in the state,
            # use it, otherwise it will be the index of the request in the
            # batch

            label = f"'{expectation.label}'" if expectation.label is not None \
                else f"idx '{req_idx}'"

            # We'll inject only if the error is below the threshold
            if not math.isclose(
                    expected_prob, prob, abs_tol=expectation.threshold):
                err = abs(expected_prob - prob)

                print("Token probability is out of the acceptable threshold "
                      f"{err:.2f} > {expectation.threshold:.2f} at request "
                      f"{label} token idx '{expectation.current_token_idx}'."
                      " Token injection will be skipped.")
                expectation.has_error = True
                continue

            full_prob = torch.ones(1, dtype=logprobs.dtype)  # 100%

            # Keep the same logprob for the expected token and
            # redistribute evenly the probability among the other
            # token ids.
            # NOTE: we are setting logprobs to the logits, if we recalculate
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
                expectation.has_error = True
                continue

            logits[req_idx] = other_logprobs
            logits[req_idx][expected_token_id] = lp

            # Decode the tokens for better human readability
            token = self.tokenizer.decode([token_id])
            expected_token = self.tokenizer.decode([expected_token_id])
            old_prob = logprobs[req_idx][token_id].exp().item()

            print(f"Golden token injection for request {label}"\
                  f" at token index '{expectation.current_token_idx}':")
            print(f"'{token}' ({old_prob * 100:.2f}%) replaced by"
                  f" '{expected_token}' ({prob * 100:.2f}%);"
                  f" baseline: ({expected_prob * 100:.2f}%)")
            expectation.current_token_idx += 1

        return logits

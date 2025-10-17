import json
import math
from typing import TYPE_CHECKING, Optional, cast

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.v1.sample.logits_processor import (BatchUpdate, LogitsProcessor,
                                             process_dict_updates)

from vllm_spyre.v1.sample.spyre_logits_processor import SpyreLogitsProcessor

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm import SamplingParams
    from vllm.config import VllmConfig

else:
    VllmConfig = None
    SamplingParams = None


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
                 expected_logprobs: Optional[list[float]],
                 error_threshold: Optional[float],
                 label: Optional[str] = None):

        self.token_ids: list[int] = expected_token_ids
        self.logprobs: Optional[list[float]] = expected_logprobs
        self.threshold: Optional[float] = error_threshold
        self.label: Optional[str] = label

        self.current_token_idx = 0
        self.has_error = False


class GoldenTokenInjector(SpyreLogitsProcessor, LogitsProcessor):
    """Logit processor to inject expected token during generation for tests"""

    def __init__(self, vllm_config: VllmConfig, device: torch.device,
                 is_pin_memory: bool):
        self.req_states: dict[int, ExpectationState] = {}
        self.tokenizer = get_tokenizer(vllm_config.model_config.tokenizer)

        self.prefill_index: Optional[int] = None

    def is_argmax_invariant(self) -> bool:
        """Never impacts greedy sampling"""
        return False

    def set_prefill(self, idx: int) -> None:
        self.prefill_index = idx

    @staticmethod
    def add_req_states(
            params: SamplingParams, prompt_tok_ids: list[int] | None,
            output_tok_ids: list[int]) -> Optional[ExpectationState]:

        if params.extra_args and (
                injector_dict :=
                params.extra_args.get("golden_token_injector")):

            # OpenAI API can pass this parameter as string, so
            # we will just parse as the expected dict
            if isinstance(injector_dict, str):
                injector_dict = json.loads(injector_dict)
            elif not isinstance(injector_dict, dict):
                raise ValueError(
                    "Golden token injector accepts only str or dict.")

            return ExpectationState(**injector_dict)

        return None

    def update_state(self, batch_update: Optional[BatchUpdate]):
        process_dict_updates(self.req_states, batch_update,
                             self.add_req_states)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_states:
            return logits

        # Calculate logprobs for the current model execution
        logprobs = F.log_softmax(logits, dim=-1)

        if self.prefill_index:
            expectation = self.req_states[self.prefill_index]
            # zero because for prefill there's only a request in the batch
            self.inject_token(logits, logprobs, 0, expectation)
            self.prefill_index = None
        else:
            for req_idx, expectation in self.req_states.items():
                self.inject_token(logits, logprobs, req_idx, expectation)

        return logits

    def inject_token(self, logits: torch.Tensor, logprobs: torch.Tensor,
                     req_idx: int, expectation: ExpectationState):
        if expectation.has_error:
            # There was an error already for inject tokens for this
            # request, skip until the end of its generation.
            return

        expected_token_id = expectation.token_ids[
            expectation.current_token_idx]
        token_id = torch.argmax(logits[req_idx], dim=-1)

        if expected_token_id == token_id:
            # Expectation is met, nothing to do.
            expectation.current_token_idx += 1
            return

        # Label to identify request, if the label was set in the state,
        # use it, otherwise it will be the index of the request in the
        # batch

        label = f"'{expectation.label}'" if expectation.label is not None \
            else f"idx '{req_idx}'"

        # Decode the tokens for better human readability
        token = self.tokenizer.decode([token_id])
        expected_token = self.tokenizer.decode([expected_token_id])

        if expectation.logprobs is None or \
            expectation.threshold is None:

            # Always inject the token
            logits[req_idx] = -math.inf
            logits[req_idx][expected_token_id] = 0.0

            logger.info("Golden token injection for request %s"\
                    " at token index '%d': "
                    "'%s'  replaced by '%s'",
                    label,
                    expectation.current_token_idx,
                    token,
                    expected_token)

            return

        # Check if the token is injectable based on a threshold
        token_lp = logprobs[req_idx][expected_token_id].reshape(-1)
        prob = torch.exp(token_lp).item()

        expected_logprob = \
            cast(torch.Tensor, expectation.logprobs)[
                expectation.current_token_idx
            ]
        expected_prob = math.exp(expected_logprob)

        # We'll inject only if the error is below the threshold
        if not math.isclose(expected_prob,
                            prob,
                            abs_tol=cast(float, expectation.threshold)):
            err = abs(expected_prob - prob)

            logger.err(
                "Token probability is out of the acceptable threshold "
                "%.2f > %.2f at request "
                "%s token idx '%s'."
                " Token injection will be skipped.", err,
                expectation.threshold, label, expectation.current_token_idx)
            expectation.has_error = True
            return

        full_prob = torch.ones(1, dtype=logprobs.dtype)  # 100%

        # Keep the same logprob for the expected token and
        # redistribute evenly the probability among the other
        # token ids.
        # NOTE: we are setting logprobs to the logits, if we recalculate
        # the softmax again over this distribution we shall find the same
        # values, but with some minimal difference. The intention is
        # inject the golden token but preserving the original logprob.

        other_token_ids_count = logits.shape[1] - 1
        other_logprobs = torch.log((full_prob - prob) / other_token_ids_count)

        if token_lp < other_logprobs:
            logger.warning(
                "The logprob is lower than the redistributed "
                "logprobs for the token ids "
                "(%.4f < %.4f), this "
                "suggests that the generation diverged too much "
                "from the expectation.", token_lp.item(),
                other_logprobs.item())
            expectation.has_error = True
            return

        logits[req_idx] = other_logprobs
        logits[req_idx][expected_token_id] = token_lp

        old_prob = logprobs[req_idx][token_id].exp().item()

        logger.info("Golden token injection for request %s"\
                " at token index '%d': "
                "'%s' (%.2f%%) replaced by "
                "'%s' (%.2f%%);"
                " baseline: (%.2f%%)",
                label,
                expectation.current_token_idx,
                token,
                old_prob * 100,
                expected_token,
                prob * 100,
                expected_prob * 100)
        expectation.current_token_idx += 1

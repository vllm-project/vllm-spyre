import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from transformers import AutoModel
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import Pooler, PoolingType

from vllm_spyre.platform import SpyrePlatform
from vllm_spyre.v1.worker.spyre_model_runner import (BaseSpyreModelRunner,
                                                     ModelForwardInputs,
                                                     WarmupShapesMixin)
# yapf conflicts with ruff for this block
# yapf: disable
from vllm_spyre.v1.worker.spyre_pooling_input_batch import (
    PoolingInputBatch, PoolingRequestState)

# yapf: enable

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
else:
    SchedulerOutput = None
    NewRequestData = None

from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

import vllm_spyre.envs as envs_spyre

logger = init_logger(__name__)

BACKEND_LIST = ['sendnn', 'inductor']


@dataclass(frozen=True)
class PoolingForwardInputs(ModelForwardInputs):
    """ Used by the SpyrePoolingModelRunner. """
    token_type_ids: Optional[torch.Tensor] = None


class SpyrePoolingModelRunner(WarmupShapesMixin,
                              BaseSpyreModelRunner[PoolingInputBatch,
                                                   PoolingRequestState,
                                                   PoolingForwardInputs]):

    def __init__(
        self,
        vllm_config: VllmConfig,
        is_driver_worker: bool,
    ):
        super().__init__(vllm_config=vllm_config,
                         is_driver_worker=is_driver_worker)

        # position_ids of all the sequences in current batch
        self._position_ids: torch.Tensor = None

        pooler_config = vllm_config.model_config.pooler_config
        if hasattr(Pooler, "from_config_with_defaults"):
            # TODO: remove this when we no longer support
            # vllm version v0.9.2
            self.pooler = Pooler.from_config_with_defaults(
                pooler_config,
                pooling_type=PoolingType.CLS,
                normalize=True,
                softmax=False)
        else:
            self.pooler = Pooler.for_embed(
                pooler_config=vllm_config.model_config.pooler_config,
                default_pooling_type=PoolingType.CLS,
                default_normalize=True,
                default_softmax=False)

    def build_input_batch(self) -> PoolingInputBatch:
        return PoolingInputBatch(
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.model_config.max_model_len,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
        )

    def load_model(self, prompt_lens: Iterable[int],
                   num_decode_tokens: Iterable[int]) -> None:
        self.model = AutoModel.from_pretrained(self.model_config.model)
        self.model.eval()
        torch.set_grad_enabled(False)
        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND in BACKEND_LIST:
            # Lazy import to avoid load torch_sendnn runtime before it is really
            # necessary. This solve issues of running forked tests that share
            # some resources from parent to children which can have problems
            # of caching even though the test run in isolated subprocesses.
            try:
                from torch_sendnn import torch_sendnn  # noqa: F401
            except ImportError:
                print("WARNING: Disabled: torch_sendnn")

            self.model = torch.compile(
                self.model,
                mode="default",
                dynamic=False,
                backend=envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND)

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    def pad_input_ids(
        self,
        input_ids_list: list[torch.Tensor],
        token_type_list: list[torch.Tensor],
        min_pad_length: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        padded_input_ids_list, mask_list, \
            position_ids_list, padded_token_type_list = (
            self._prepare_pad_input_ids(input_ids_list, min_pad_length,
                                        token_type_list))

        input_ids = torch.stack(padded_input_ids_list)
        if padded_token_type_list:
            token_type_ids = torch.stack(padded_token_type_list)
        else:
            token_type_ids = None
        mask = torch.stack(mask_list)
        position_ids = torch.stack(position_ids_list)

        return input_ids, token_type_ids, position_ids, mask

    def update_states(self, scheduler_output: SchedulerOutput):
        assert len(scheduler_output.scheduled_cached_reqs.req_ids) == 0

        if scheduler_output.finished_req_ids:
            for req_id in scheduler_output.finished_req_ids:
                self.input_batch.remove_request(req_id)
                self.requests.pop(req_id, None)

    def _prepare_prompt(
        self,
        new_requests: list[NewRequestData],
    ) -> PoolingForwardInputs:
        assert len(new_requests) > 0
        input_token_list: list[torch.Tensor] = []
        token_type_list: list[torch.Tensor] = []
        padded_batch_size, min_pad_length_batch = self._get_padded_batch_size(
            new_requests)

        # Internal state is reset here.
        # We don't support continuous batching, so we know all previous requests
        # have finished decoding.
        self.input_batch.clear_requests()
        self.requests = {}

        # Build batch and prepare input_token1
        for request_data in new_requests:
            # retrieve initial (unpadded) tokens
            prompt_tokens = request_data.prompt_token_ids

            input_token_list.append(
                torch.tensor(prompt_tokens,
                             dtype=torch.long,
                             device=torch.device("cpu")))

            #if request_data.token_type_ids is not None:
            #    token_type_list.append(
            #        torch.tensor(request_data.token_type_ids,
            #                     dtype=torch.long,
            #                     device=torch.device("cpu")))

            # Add new requests to the cached states.
            req_id = request_data.req_id
            pooling_params = request_data.pooling_params
            assert pooling_params is not None

            req_state = PoolingRequestState(
                req_id=req_id,
                prompt_token_ids=request_data.prompt_token_ids,
                #token_type_ids=request_data.token_type_ids,
                pooling_params=pooling_params,
            )
            self.requests[req_id] = req_state
            self.input_batch.add_request(req_state)

        self.input_batch.padded_batch_size = padded_batch_size

        if token_type_list:
            assert len(input_token_list) == len(token_type_list)

        # padding to compiled batch size
        while len(input_token_list) < padded_batch_size:
            input_token_list.append(
                torch.zeros(min_pad_length_batch,
                            dtype=torch.long,
                            device=torch.device("cpu")))
            if token_type_list:
                token_type_list.append(
                    torch.zeros(min_pad_length_batch,
                                dtype=torch.long,
                                device=torch.device("cpu")))

        # get position ids and attention mask
        input_tokens, token_type_ids, position_ids, mask = self.pad_input_ids(
            input_token_list,
            token_type_list,
            min_pad_length=min_pad_length_batch)

        model_input = PoolingForwardInputs(
            input_tokens=input_tokens,
            token_type_ids=token_type_ids,
            input_positions=position_ids,
            input_masks=mask,
            is_prompt=True,
        )

        self._mark_input_tensors(model_input)

        return model_input

    def prepare_model_input(
            self, scheduler_output: SchedulerOutput) -> PoolingForwardInputs:

        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        # Also assuming that new sequences are prefills
        is_prompt = len(scheduler_output.scheduled_new_reqs) > 0

        # Prepare input tensors.
        assert is_prompt
        # Assert no running requests
        assert len(scheduler_output.scheduled_cached_reqs.req_ids) == 0
        return self._prepare_prompt(scheduler_output.scheduled_new_reqs)

    def _mark_input_tensors(self, model_input: PoolingForwardInputs) -> None:

        super()._mark_input_tensors(model_input=model_input)
        if not self.warmup_mode:
            # Only mark tensors when we're warming up and compiling the graphs
            return

        if model_input.token_type_ids is not None:
            torch._dynamo.mark_static(model_input.token_type_ids, 0)
            torch._dynamo.mark_static(model_input.token_type_ids, 1)

    @SpyrePlatform.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        **kwargs,
    ) -> ModelRunnerOutput:

        t0 = time.time()

        self.update_states(scheduler_output)

        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOuptut if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT

        model_input = self.prepare_model_input(scheduler_output)

        model_kwargs: dict[str, Any] = {}
        if model_input.token_type_ids:
            model_kwargs["token_type_ids"] =\
                  model_input.token_type_ids

        # Execute the model
        attn_metadata = self.build_attn_metadata(model_input)
        with set_forward_context(attn_metadata, self.vllm_config):
            outputs = self.model(
                input_ids=model_input.input_tokens,
                # TODO: verify this position id thing
                #position_ids=model_input.input_positions,
                attention_mask=model_input.input_masks)
            hidden_states = outputs["last_hidden_state"]

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return EMPTY_MODEL_RUNNER_OUTPUT

        t1 = time.time() - t0
        logger.debug("t_batch: %.2fms", (t1 * 1000))

        pooling_metadata = self.input_batch.make_pooling_metadata()

        # prepare unpadded output for the pooler
        hidden_state_list: list[torch.Tensor] = []
        for hidden_state, prompt_len in zip(hidden_states,
                                            pooling_metadata.prompt_lens):
            # we're left padding
            hidden_state_list.append(hidden_state[-prompt_len:])

        raw_pooler_output = self.pooler(hidden_states=hidden_state_list,
                                        pooling_metadata=pooling_metadata)

        pooler_output: list[Optional[torch.Tensor]] = []

        for raw_output in raw_pooler_output:
            pooler_output.append(raw_output.data.to("cpu"))

        model_output = ModelRunnerOutput(
            req_ids=self.input_batch.requests_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
        )
        return model_output

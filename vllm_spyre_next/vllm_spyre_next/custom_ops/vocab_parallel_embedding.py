# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    get_masked_input_and_mask,
)
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def _prepare_embedding_inputs_on_spyre(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare inputs for Spyre device.

    - input_ids: kept as integer dtype, moved to spyre device
    - weight: converted to float16, moved to spyre device
    """
    spyre_input_ids = input_ids.to(device=torch.device("spyre"))
    spyre_weight = weight.to(dtype=torch.float16).to(
        device=torch.device("spyre")
    )
    return spyre_input_ids, spyre_weight


@VocabParallelEmbedding.register_oot(name="VocabParallelEmbedding")
class SpyreVocabParallelEmbedding(VocabParallelEmbedding):
    """OOT version of VocabParallelEmbedding for IBM's Spyre device.

    This implementation uses a custom op registration to avoid being compiled
    by torch.compile, similar to how SpyreRMSNorm handles its operations.
    The layer is registered in static_forward_context and accessed via
    no_compile_layers during forward pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        logger.debug("Building custom VocabParallelEmbedding")

        # Compile the Spyre-specific forward implementation
        # This compilation is separate from the main model compilation
        self._fwd_spyre = torch.compile(
            self._forward_static_spyre, dynamic=False
        )

        # Register this layer in the static forward context
        # This allows it to be accessed during the custom op execution
        compilation_config = get_current_vllm_config().compilation_config
        if not hasattr(SpyreVocabParallelEmbedding, "_instance_counter"):
            SpyreVocabParallelEmbedding._instance_counter = 0
        self.prefix = (
            f"spyre_vocab_embedding_"
            f"{SpyreVocabParallelEmbedding._instance_counter}"
        )
        SpyreVocabParallelEmbedding._instance_counter += 1

        if self.prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.prefix}")
        compilation_config.static_forward_context[self.prefix] = self

    def forward(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor:
        """Forward method that uses a custom op to avoid torch.compile.

        This delegates to the custom op which will call forward_impl
        outside of the compilation graph.
        """
        # Create output tensor with embedding dimensions
        output = torch.empty(
            *input_.shape,
            self.embedding_dim,
            dtype=self.weight.dtype,
            device=input_.device,
        )

        # Call the custom op - this will NOT be compiled
        torch.ops.vllm.spyre_vocab_embedding(
            input_,
            output,
            self.prefix,
        )

        return output

    def forward_impl(
        self,
        input_: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Implementation called by the custom op.

        This executes outside of torch.compile's graph.
        """
        result = self.forward_native(input_)
        output.copy_(result)

    @staticmethod
    def _forward_static_spyre(
        input_ids: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorch-native implementation for Spyre device.

        This method is compiled separately via self._fwd_spyre.
        Simple embedding table lookup.
        """
        return F.embedding(input_ids, weight)

    def forward_native(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward().

        This method handles the Spyre-specific device operations including:
        - TP masking
        - Padding for minimum batch size requirements
        - Data transfer to/from Spyre device
        - Calling the compiled Spyre kernel
        - TP all-reduce
        """
        # TP masking (on CPU)
        if self.tp_size > 1:
            masked_input, input_mask = get_masked_input_and_mask(
                input_,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
        else:
            masked_input = input_

        # Store original batch size for later trimming
        num_real_el = masked_input.shape[0]

        # Pad to minimum batch size of 64 if needed
        if masked_input.shape[0] != 1 and masked_input.shape[0] < 64:
            masked_input = torch.nn.functional.pad(
                masked_input, (0, 0, 64 - num_real_el, 0)
            )

        # Transfer to Spyre device
        spyre_input, spyre_weight = _prepare_embedding_inputs_on_spyre(
            masked_input.long(), self.weight.data
        )

        # Execute the Spyre-compiled kernel
        out = self._fwd_spyre(spyre_input, spyre_weight)

        # Transfer result back to CPU
        output_parallel = out.cpu()

        # Remove padding to restore original batch size
        output_parallel = output_parallel[:num_real_el, :]

        # Convert to expected output dtype
        output_parallel = output_parallel.to(torch.bfloat16)

        # TP mask output and all-reduce (on CPU)
        if self.tp_size > 1:
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)

        output = tensor_model_parallel_all_reduce(output_parallel)
        return output

    def forward_oot(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(input_)


# Custom op implementation
def spyre_vocab_embedding(
    input_: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom op that calls the SpyreVocabParallelEmbedding layer
    outside of compilation."""
    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    layer.forward_impl(input_, output)


def spyre_vocab_embedding_fake(
    input_: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Fake implementation for shape inference during compilation."""
    return


def register():
    # Register the custom op
    direct_register_custom_op(
        op_name="spyre_vocab_embedding",
        op_func=spyre_vocab_embedding,
        mutates_args=["output"],
        fake_impl=spyre_vocab_embedding_fake,
    )
    logger.info("Registered custom op: SpyreVocabParallelEmbedding")
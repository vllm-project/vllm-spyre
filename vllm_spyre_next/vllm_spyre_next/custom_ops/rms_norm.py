# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import torch.utils._pytree as pytree

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm

logger = init_logger(__name__)


def _prepare_inputs_on_spyre(*args):
    def _convert_to_spyre(arg):
        return (
            arg.to(dtype=torch.float16).to(device=torch.device("spyre"))
            if isinstance(arg, torch.Tensor)
            else arg
        )

    return pytree.tree_map(_convert_to_spyre, args)[0]


@RMSNorm.register_oot(name="RMSNorm")
class SpyreRMSNorm(RMSNorm):
    """OOT version of RMSNorm for IBM's Spyre device

    This implementation uses a custom op registration to avoid being compiled
    by torch.compile, similar to how MambaMixer2 handles its operations.
    The layer is registered in static_forward_context and accessed via
    no_compile_layers during forward pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        logger.debug("Building custom RMS norm")

        # Compile the Spyre-specific forward implementation
        # This compilation is separate from the main model compilation
        self._fwd_spyre = torch.compile(self._forward_static_spyre, dynamic=False)

        # Register this layer in the static forward context
        # This allows it to be accessed during the custom op execution
        compilation_config = get_current_vllm_config().compilation_config
        # Use a unique prefix for this layer - you may want to pass this as a parameter
        # For now, we'll use a counter-based approach
        if not hasattr(SpyreRMSNorm, "_instance_counter"):
            SpyreRMSNorm._instance_counter = 0
        self.prefix = f"spyre_rmsnorm_{SpyreRMSNorm._instance_counter}"
        SpyreRMSNorm._instance_counter += 1

        if self.prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.prefix}")
        compilation_config.static_forward_context[self.prefix] = self

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method that uses a custom op to avoid torch.compile.

        This delegates to the custom op which will call forward_impl
        outside of the compilation graph.
        """
        # Create output tensor
        output = torch.empty_like(x)

        # Call the custom op - this will NOT be compiled
        torch.ops.vllm.spyre_rmsnorm(
            x,
            output,
            self.prefix,
            residual,
        )

        if residual is not None:
            # The custom op will have updated residual in-place if needed
            return output, residual
        return output

    def forward_impl(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> None:
        """
        Implementation called by the custom op.
        This executes outside of torch.compile's graph.
        """
        result = self.forward_native(x, residual)

        if residual is not None:
            # Unpack tuple result
            output_data, residual_data = result
            output.copy_(output_data)
            residual.copy_(residual_data)
        else:
            output.copy_(result)

    @staticmethod
    def _forward_static_spyre(
        x: torch.Tensor,
        variance_epsilon: float,
        hidden_size: int,
        orig_dtype: torch.dtype,
        weight: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        variance_size_override: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation for Spyre device.

        This method is compiled separately via self._fwd_spyre.
        """
        x = x.transpose(1, 0).contiguous()

        if residual is not None:
            # residual promoted f16->f32 automatically,
            # otherwise Inductor eliminates the casts to and from f16,
            # increasing memory usage (and complicating pattern matching)
            x = x + residual
            residual = x.to(orig_dtype)

        if variance_size_override is None:
            x_var = x
        else:
            if hidden_size < variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{variance_size_override}, but found: {hidden_size}"
                )

            x_var = x[:, :, :variance_size_override]

        variance = x_var * x_var
        variance = variance.mean(dim=0)
        x = x * torch.rsqrt(variance + variance_epsilon)[None, :]

        x = x.transpose(1, 0).contiguous()

        if weight is not None:
            x = x * weight

        if residual is None:
            return x
        else:
            return x, residual

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward().

        This method handles the Spyre-specific device operations including:
        - Padding for minimum batch size requirements
        - Data transfer to/from Spyre device
        - Calling the compiled Spyre kernel
        """
        if residual is not None:
            raise NotImplementedError("TODO: Residual support not yet implemented")

        # Store original batch size for later trimming
        num_real_el = x.shape[0]

        # Pad to minimum batch size of 64 if needed
        if x.shape[0] != 1 and x.shape[0] < 64:
            x = torch.nn.functional.pad(x, (0, 0, 64 - num_real_el, 0))

        # Execute the Spyre-compiled kernel
        # _prepare_inputs_on_spyre handles device transfer and dtype conversion
        out = self._fwd_spyre(
            _prepare_inputs_on_spyre([x])[0],
            _prepare_inputs_on_spyre([torch.ones(x.shape[0]) * self.variance_epsilon])[0],
            self.hidden_size,
            torch.float16,
            _prepare_inputs_on_spyre([self.weight.data])[0] if self.has_weight else None,
            residual,
            self.variance_size_override,
        )

        # Transfer result back to CPU
        spyre_out = out.cpu()

        # Remove padding to restore original batch size
        spyre_out = spyre_out[:num_real_el, :]

        # Convert to expected output dtype
        return spyre_out.to(torch.bfloat16)

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(x, residual)


# Custom op implementation
def spyre_rmsnorm(
    x: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    residual: torch.Tensor | None = None,
) -> None:
    """Custom op that calls the SpyreRMSNorm layer outside of compilation."""

    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    layer.forward_impl(x, output, residual)


def spyre_rmsnorm_fake(
    x: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    residual: torch.Tensor | None = None,
) -> None:
    """Fake implementation for shape inference during compilation."""
    return


def register():
    # Register the custom op
    direct_register_custom_op(
        op_name="spyre_rmsnorm",
        op_func=spyre_rmsnorm,
        mutates_args=["output"],
        fake_impl=spyre_rmsnorm_fake,
    )
    logger.info("Registered custom op: SpyreRMSNorm")

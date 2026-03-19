# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre-specific RMSNorm implementation using out-of-tree (OOT) registration.

This module provides a custom RMSNorm layer optimized for IBM's Spyre device,
replacing the upstream vLLM implementation (vllm/model_executor/layers/layernorm.py)
when instantiated.

Architecture Overview:
    1. OOT Registration: @RMSNorm.register_oot() replaces upstream class at instantiation
    2. Custom Op Pattern: Uses torch.ops.vllm.spyre_rmsnorm to bypass torch.compile
    3. Static Forward Context: Registers in compilation_config.static_forward_context
    4. No-Compile Execution: Retrieved via forward_context.no_compile_layers during forward

Key Components:
    - SpyreRMSNorm: Main layer class with Spyre-specific optimizations
    - spyre_rmsnorm: Custom op implementation (executes outside torch.compile)
    - spyre_rmsnorm_fake: Fake implementation for shape inference
    - register(): Registers the custom op with vLLM

Spyre Device Constraints:
    - Minimum batch size: 64 (due to spyre constraint, automatically padded)
    - Device dtype: float16 (via prepare_inputs_on_spyre)
    - Output dtype: bfloat16 (converted on CPU)
    - Algorithm: Transpose-based computation with torch.ops.spyre.full()

Limitations:
    Currently the implementation in `_forward_vLLM_native` is similar to the
    upstream implementation in `forward_static` from llm/model_executor/layers/layernorm.py,
    but it DOES NOT use the promotion of the data types, as this is not
    yet supported in torch-spyre.

References:
    - Upstream RMSNorm: vllm/model_executor/layers/layernorm.py
"""

import torch
import torch.utils._pytree as pytree

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from functools import lru_cache

from .utils import convert_for_spyre, convert_from_spyre

logger = init_logger(__name__)


@RMSNorm.register_oot(name="RMSNorm")
class SpyreRMSNorm(RMSNorm):
    """Out-of-tree (OOT) RMSNorm implementation for IBM's Spyre device.

    This replaces the upstream vLLM RMSNorm (vllm/model_executor/layers/layernorm.py)
    when instantiated, providing Spyre-specific optimizations and device handling.
    """

    def __init__(self, *args, **kwargs):
        """Initialize SpyreRMSNorm layer.

        Compiles the Spyre kernel based on VLLM_SPYRE_NEXT_RMSNORM_KERNEL
        environment variable and registers this instance in static_forward_context.
        """
        super().__init__(*args, **kwargs)

        logger.debug("Building custom RMS norm")

        self._fwd_spyre = torch.compile(self.forward_spyre, dynamic=False)

        logger.warning(
            "SpyreRMSNorm: no dtype promotion is performed, \
            expect numerical differences to upstream vLLM."
        )

        # Register in static_forward_context for custom op access
        # Pattern: Each instance gets unique name via counter to avoid collisions
        compilation_config = get_current_vllm_config().compilation_config
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
        """Forward pass using custom op to bypass torch.compile.

        Delegates to torch.ops.vllm.spyre_rmsnorm which retrieves this layer
        from forward_context.no_compile_layers and calls forward_impl outside
        the compilation graph. This prevents torch.compile from inlining the
        Spyre-specific operations.

        Args:
            x: Input tensor [batch_size, hidden_size]
            residual: Optional residual tensor

        Returns:
            Normalized output, or (output, residual) tuple if residual provided
        """
        output = torch.empty_like(x)

        # Custom op call - executes outside torch.compile graph
        torch.ops.vllm.spyre_rmsnorm(x, output, self.prefix, residual)

        if residual is not None:
            return output, residual
        return output

    def forward_impl(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> None:
        """Implementation called by custom op, executes outside torch.compile.

        Called by spyre_rmsnorm custom op via forward_context.no_compile_layers.
        Delegates to forward_native for actual computation, then copies results
        to pre-allocated output tensors.

        Args:
            x: Input tensor
            output: Pre-allocated output tensor (modified in-place)
            residual: Optional residual tensor (modified in-place if provided)
        """
        result = self.forward_native(x, residual)

        if residual is not None:
            output_data, residual_data = result
            output.copy_(output_data)
            residual.copy_(residual_data)
        else:
            output.copy_(result)

    @staticmethod
    def forward_spyre(
        x: torch.Tensor,
        variance_epsilon: float,
        hidden_size: int,
        weight: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        variance_size_override: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Spyre-optimized RMS norm using transpose-based computation (active implementation).

        Based on upstream vLLM's forward_static (vllm/model_executor/layers/layernorm.py)
        but adapted for Spyre device with transpose operations and torch.ops.spyre.full().
        Compiled separately via torch.compile in __init__.

        Key differences from upstream:
            - Uses transpose(-1, -2) for computation efficiency on Spyre
            - Creates epsilon tensor via torch.ops.spyre.full() instead of scalar
            - No dtype promotion support (torch-spyre limitation)
        """
        if residual is not None:
            x = x + residual
            residual = x

        if x.shape[-1] != hidden_size:
            raise ValueError(f"Expected hidden_size to be {hidden_size}, but found: {x.shape[-1]}")

        x = x.transpose(-1, -2).contiguous()

        variance_epsilon = torch.ops.spyre.full(
            x.shape, variance_epsilon, dtype=torch.float16, device="spyre"
        )

        if variance_size_override is None:
            x_var = x
        else:
            if hidden_size < variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{variance_size_override}, but found: {hidden_size}"
                )

            x_var = x[:, :, :variance_size_override]

        # After transpose, hidden dim is now dim=0
        variance = x_var.pow(2).mean(dim=0, keepdim=True)

        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x.transpose(-1, -2).contiguous()

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
        """Spyre device execution with padding, device transfer, and dtype conversion.

        Handles Spyre-specific constraints:
            1. Minimum batch size: Pads to 64 if needed
            2. Device transfer: CPU -> Spyre (float16) via prepare_inputs_on_spyre
            3. Kernel execution: Calls compiled _fwd_spyre
            4. Result transfer: Spyre -> CPU, trim padding, convert to bfloat16

        Limitations:
            - variance_size_override not implemented (raises NotImplementedError)

        Args:
            x: Input tensor [batch_size, hidden_size] on CPU
            residual: Optional residual

        Returns:
            Normalized output [batch_size, hidden_size] in bfloat16
        """
        x_dtype = x.dtype
        x_device = x.device

        if self.variance_size_override is not None:
            raise NotImplementedError("TODO: variance_size_override not yet implemented")

        orig_batch_size = x.shape[0]

        # Pad to minimum batch size of 64 (Spyre constraint)
        # Pad at END so original data stays at indices [0:orig_batch_size]
        if x.shape[0] < 64:
            pad_amount = 64 - x.shape[0]
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_amount))
            if residual is not None:
                residual = torch.nn.functional.pad(residual, (0, 0, 0, pad_amount))

        # Execute compiled kernel on Spyre device
        # convert_for_spyre: CPU tensor -> Spyre device (float16)
        outs = self._fwd_spyre(
            convert_for_spyre(x, dtype=torch.float16),
            self.variance_epsilon,
            self.hidden_size,
            convert_for_spyre(self.weight.data, dtype=torch.float16) if self.has_weight else None,
            convert_for_spyre(residual, dtype=torch.float16),
            self.variance_size_override,
        )

        # Transfer back to CPU and restore original shape
        return pytree.tree_map(
            lambda el: el[:orig_batch_size, :],
            convert_from_spyre(outs, dtype=x_dtype, device=x_device),
        )

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """OOT forward method - delegates to forward_native."""
        return self.forward_native(x, residual)


# Custom op implementation (executed outside torch.compile)
def spyre_rmsnorm(
    x: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    residual: torch.Tensor | None = None,
) -> None:
    """Custom op implementation - retrieves layer and executes outside compilation.

    Called by SpyreRMSNorm.forward() via torch.ops.vllm.spyre_rmsnorm.
    Retrieves the layer instance from forward_context.no_compile_layers using
    layer_name, then calls forward_impl to execute the actual computation.

    This pattern prevents torch.compile from inlining Spyre-specific operations.
    Similar to mamba_mixer2 (vllm/model_executor/layers/mamba/mamba_mixer2.py).

    Args:
        x: Input tensor
        output: Pre-allocated output tensor (modified in-place)
        layer_name: Unique layer identifier in static_forward_context
        residual: Optional residual tensor
    """
    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    layer.forward_impl(x, output, residual)


def spyre_rmsnorm_fake(
    x: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    residual: torch.Tensor | None = None,
) -> None:
    """Fake implementation for shape/dtype inference during torch.compile.

    Provides metadata to torch.compile without executing actual computation.
    """
    return


@lru_cache(maxsize=1)
def register():
    """Register the spyre_rmsnorm custom op with vLLM.

    Registers torch.ops.vllm.spyre_rmsnorm with:
        - op_func: Actual implementation (spyre_rmsnorm)
        - fake_impl: Shape inference implementation (spyre_rmsnorm_fake)
        - mutates_args: Indicates 'output' is modified in-place
    """
    direct_register_custom_op(
        op_name="spyre_rmsnorm",
        op_func=spyre_rmsnorm,
        mutates_args=["output"],
        fake_impl=spyre_rmsnorm_fake,
    )
    logger.info("Registered custom op: SpyreRMSNorm")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre-specific SiluAndMul implementation using out-of-tree (OOT) registration.

This module provides a custom SiluAndMul (SwiGLU) activation layer optimized for
IBM's Spyre device, replacing the upstream vLLM implementation from
vllm/model_executor/layers/activation.py when instantiated.

Architecture Overview:
    1. OOT Registration: @SiluAndMul.register_oot() replaces upstream class at instantiation
    2. Custom Op Pattern: Uses torch.ops.vllm.spyre_siluandmul to bypass torch.compile
    3. Static Forward Context: Registers in compilation_config.static_forward_context
    4. No-Compile Execution: Retrieved via forward_context.no_compile_layers during forward

Key Components:
    - SpyreSiluAndMul: Main layer class with Spyre-specific optimizations
    - spyre_siluandmul: Custom op implementation (executes outside torch.compile)
    - spyre_siluandmul_fake: Fake implementation for shape inference
    - register(): Registers the custom op with vLLM

Spyre Device Constraints:
    - Device dtype: float16 (via convert_for_spyre)
    - Output dtype: matches input dtype (converted on CPU)

Output Shape Note:
    Unlike RMSNorm (same input/output shape), SiluAndMul halves the last dimension:
    input shape: [..., 2*d] -> output shape: [..., d]

References:
    - Upstream SiluAndMul: vllm/model_executor/layers/activation.py
"""

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from functools import lru_cache

from .utils import convert_for_spyre, convert_from_spyre

logger = init_logger(__name__)


@SiluAndMul.register_oot(name="SiluAndMul")
class SpyreSiluAndMul(SiluAndMul):
    """Out-of-tree (OOT) SiluAndMul implementation for IBM's Spyre device.

    This replaces the upstream vLLM SiluAndMul (vllm/model_executor/layers/activation.py)
    when instantiated, providing Spyre-specific optimizations and device handling.

    Computes: x -> silu(x[..., :d]) * x[..., d:] where d = x.shape[-1] // 2
    """

    def __init__(self, *args, **kwargs):
        """Initialize SpyreSiluAndMul layer.

        Compiles the Spyre kernel and registers this instance in static_forward_context.
        """
        super().__init__(*args, **kwargs)

        logger.debug("Building custom SiluAndMul")

        self._fwd_spyre = torch.compile(self.forward_static, dynamic=False)

        # Register in static_forward_context for custom op access
        # Pattern: Each instance gets unique name via counter to avoid collisions
        compilation_config = get_current_vllm_config().compilation_config
        if not hasattr(SpyreSiluAndMul, "_instance_counter"):
            SpyreSiluAndMul._instance_counter = 0
        self.prefix = f"spyre_siluandmul_{SpyreSiluAndMul._instance_counter}"
        SpyreSiluAndMul._instance_counter += 1

        if self.prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.prefix}")
        compilation_config.static_forward_context[self.prefix] = self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using custom op to bypass torch.compile.

        Delegates to torch.ops.vllm.spyre_siluandmul which retrieves this layer
        from forward_context.no_compile_layers and calls forward_impl outside
        the compilation graph.

        Args:
            x: Input tensor [..., 2*d]

        Returns:
            Activated output tensor [..., d]
        """
        d = x.shape[-1] // 2
        output = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)

        # Custom op call - executes outside torch.compile graph
        torch.ops.vllm.spyre_siluandmul(x, output, self.prefix)

        return output

    def forward_impl(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Implementation called by custom op, executes outside torch.compile.

        Called by spyre_siluandmul custom op via forward_context.no_compile_layers.
        Delegates to forward_native for actual computation, then copies result
        to pre-allocated output tensor.

        Args:
            x: Input tensor [..., 2*d]
            output: Pre-allocated output tensor [..., d] (modified in-place)
        """
        result = self.forward_native(x)
        output.copy_(result)

    @staticmethod
    def forward_static(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Spyre-optimized silu+multiply kernel compiled via torch.compile.

        Computes silu(x1) * x2 on the Spyre device, relying on torch-spyre's
        registered aten::silu.out kernel. The two halves are passed in as
        separate tensors because the Spyre device does not yet support tensor
        slicing (strided views); the split is therefore performed on CPU before
        this method is called (see forward_native).

        Args:
            x1: First half of the gated input, shape [..., d], on Spyre device
                (float16).  silu is applied to this half.
            x2: Second half of the gated input, shape [..., d], on Spyre device
                (float16).  Acts as the multiplicative gate.

        Returns:
            Output tensor of shape [..., d] on the Spyre device (float16).
        """
        return F.silu(x1) * x2

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """Spyre device execution: CPU slicing workaround, device transfer, kernel call.

        The Spyre device does not currently support strided tensor views (slicing),
        so the input is split into its two halves on the CPU before being
        transferred to the device.  Once tensor slicing is supported this method
        should revert to the simpler single-tensor path (see commented-out block).

        Execution steps:
            1. Slice on CPU: split x into x1 = x[..., :d] and x2 = x[..., d:]
            2. Device transfer: convert x1 and x2 independently to Spyre (float16)
               via convert_for_spyre
            3. Kernel execution: call compiled _fwd_spyre(x1_spyre, x2_spyre)
            4. Result transfer: Spyre -> original device, restore original dtype

        Args:
            x: Input tensor of shape [..., 2*d] on CPU with arbitrary float dtype.

        Returns:
            Activated output tensor of shape [..., d] on the original device with
            the original dtype.
        """
        x_dtype = x.dtype
        x_device = x.device

        # Note: Workaround with tensor slicing on CPU
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        out = self._fwd_spyre(
            convert_for_spyre(x1, dtype=torch.float16),
            convert_for_spyre(x2, dtype=torch.float16),
        )

        # Transfer back to CPU and restore original shape
        return convert_from_spyre(out, dtype=x_dtype, device=x_device)

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        """OOT forward method - delegates to forward_native."""
        return self.forward_native(x)


# Custom op implementation (executed outside torch.compile)
def spyre_siluandmul(
    x: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom op implementation - retrieves layer and executes outside compilation.

    Called by SpyreSiluAndMul.forward() via torch.ops.vllm.spyre_siluandmul.
    Retrieves the layer instance from forward_context.no_compile_layers using
    layer_name, then calls forward_impl to execute the actual computation.

    This pattern prevents torch.compile from inlining Spyre-specific operations.
    Similar to mamba_mixer2 (vllm/model_executor/layers/mamba/mamba_mixer2.py).

    Args:
        x: Input tensor [..., 2*d]
        output: Pre-allocated output tensor [..., d] (modified in-place)
        layer_name: Unique layer identifier in static_forward_context
    """
    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    layer.forward_impl(x, output)


def spyre_siluandmul_fake(
    x: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Fake implementation for shape/dtype inference during torch.compile.

    Provides metadata to torch.compile without executing actual computation.
    """
    return


@lru_cache(maxsize=1)
def register():
    """Register the spyre_siluandmul custom op with vLLM.

    Registers torch.ops.vllm.spyre_siluandmul with:
        - op_func: Actual implementation (spyre_siluandmul)
        - fake_impl: Shape inference implementation (spyre_siluandmul_fake)
        - mutates_args: Indicates 'output' is modified in-place
    """
    direct_register_custom_op(
        op_name="spyre_siluandmul",
        op_func=spyre_siluandmul,
        mutates_args=["output"],
        fake_impl=spyre_siluandmul_fake,
    )
    logger.info("Registered custom op: SpyreSiluAndMul")

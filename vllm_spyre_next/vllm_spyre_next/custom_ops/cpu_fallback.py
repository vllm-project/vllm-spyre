# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic CPU fallback infrastructure for Spyre custom ops.

Provides a single custom op (torch.ops.vllm.spyre_cpu_fallback) that can
execute any layer on CPU while the rest of the model runs on Spyre.
This is needed for example when a layer is not yet properly wrapped
for execution via torch-spyre. For example RoPE, Embedding or MLP.
Compatible with fullgraph=True compilation — no dynamo graph breaks.

Usage: For each layer that needs CPU fallback, create a thin OOT wrapper:
    1. Inherit from SpyreCpuFallbackMixin + the target layer class
    2. Call self._init_cpu_fallback("name") in __init__
    3. Override forward_oot to pre-allocate output and call the custom op
    4. Override cpu_forward to call the parent's forward_native

This is temporary — once all layers compile natively on Spyre, these
fallbacks can be removed.
"""

import torch

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import convert, register_spyre_dispatch, _fake_impl

logger = init_logger(__name__)


class SpyreCpuFallbackMixin:
    """Mixin for OOT layers that fall back to CPU execution.

    Handles registration in static_forward_context with unique prefixes.
    Intercepts .to() to keep parameters on CPU when the model is moved
    to Spyre. Subclasses must implement cpu_forward().
    """

    _instance_counter: int = 0

    def _init_cpu_fallback(
        self,
        prefix_name: str,
        output_device: torch.device | None = None,
    ) -> None:
        """Register this layer instance in static_forward_context.

        Args:
            prefix_name: Short identifier for the layer type (e.g. "embedding").
            output_device: Device for pre-allocated output tensors in forward().
                Defaults to torch.device("spyre"). Set to torch.device("cpu")
                for layers whose output must stay on CPU (e.g. QKV for split).
        """
        self._output_device = output_device or torch.device("spyre")

        cls = type(self)
        cls._instance_counter += 1
        self.prefix = f"spyre_cpu_{prefix_name}_{cls._instance_counter}"

        compilation_config = get_current_vllm_config().compilation_config
        if self.prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.prefix}")
        compilation_config.static_forward_context[self.prefix] = self

        logger.info(
            "CPU fallback: %s (%s, output_device=%s)",
            self.prefix,
            cls.__name__,
            self._output_device,
        )

    def _apply(self, fn, recurse=True):
        """No-op: keep all parameters on CPU."""
        return self

    def cpu_forward(self, *args, **kwargs):
        """Execute the layer's original forward on CPU."""
        raise NotImplementedError


def spyre_cpu_fallback(
    input: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom op implementation

    1. Converts input tensor to cpu
    2. Executes layer on cpu
    3. Copies result to pre-allocated output buffer (spyre)
    """
    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    cpu_input = convert(input, device="cpu")
    result = layer.cpu_forward(cpu_input)
    output.copy_(result)


def register():
    """Register the generic spyre_cpu_fallback custom op."""
    direct_register_custom_op(
        op_name="spyre_cpu_fallback",
        op_func=spyre_cpu_fallback,
        mutates_args=["output"],
        fake_impl=_fake_impl,
    )
    register_spyre_dispatch("spyre_cpu_fallback", spyre_cpu_fallback)
    logger.info("Registered custom op: spyre_cpu_fallback")

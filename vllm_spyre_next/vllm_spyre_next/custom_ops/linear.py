# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre CPU fallback for linear layers (MLP and attention projections).

Remove this file once Spyre supports linear ops natively.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

from .cpu_fallback import SpyreCpuFallbackMixin

logger = init_logger(__name__)


# --- QKVParallelLinear (inherits ColumnParallelLinear.forward) ---


@QKVParallelLinear.register_oot(name="QKVParallelLinear")
class SpyreQKVParallelLinear(SpyreCpuFallbackMixin, QKVParallelLinear):
    """OOT QKVParallelLinear that falls back to CPU execution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # QKV output must stay on CPU: granite.py does qkv.split() which
        # creates strided views not supported on Spyre.
        self._init_cpu_fallback("qkv_linear", output_device=torch.device("cpu"))

    def forward(self, input_):
        output = torch.empty(
            *input_.shape[:-1],
            self.output_size_per_partition,
            dtype=input_.dtype,
            device=self._output_device,
        )
        torch.ops.vllm.spyre_cpu_fallback(input_, output, self.prefix)
        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def cpu_forward(self, cpu_input):
        result = ColumnParallelLinear.forward(self, cpu_input)
        if isinstance(result, tuple):
            return result[0]
        return result


# --- MergedColumnParallelLinear (inherits ColumnParallelLinear.forward) ---


@MergedColumnParallelLinear.register_oot(name="MergedColumnParallelLinear")
class SpyreMergedColumnParallelLinear(SpyreCpuFallbackMixin, MergedColumnParallelLinear):
    """OOT MergedColumnParallelLinear that falls back to CPU execution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_cpu_fallback("merged_col_linear")  # default: Spyre output

    def forward(self, input_):
        output = torch.empty(
            *input_.shape[:-1],
            self.output_size_per_partition,
            dtype=input_.dtype,
            device=self._output_device,
        )
        torch.ops.vllm.spyre_cpu_fallback(input_, output, self.prefix)
        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def cpu_forward(self, cpu_input):
        result = ColumnParallelLinear.forward(self, cpu_input)
        if isinstance(result, tuple):
            return result[0]
        return result


# --- RowParallelLinear ---


@RowParallelLinear.register_oot(name="RowParallelLinear")
class SpyreRowParallelLinear(SpyreCpuFallbackMixin, RowParallelLinear):
    """OOT RowParallelLinear that falls back to CPU execution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_cpu_fallback("row_linear")  # default: Spyre output

    def forward(self, input_):
        output = torch.empty(
            *input_.shape[:-1],
            self.output_size,
            dtype=input_.dtype,
            device=self._output_device,
        )
        torch.ops.vllm.spyre_cpu_fallback(input_, output, self.prefix)
        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def cpu_forward(self, cpu_input):
        result = RowParallelLinear.forward(self, cpu_input)
        if isinstance(result, tuple):
            return result[0]
        return result


def register():
    # No per-layer op registration needed — uses generic spyre_cpu_fallback.
    # OOT class registration happens at import time via decorators.
    pass

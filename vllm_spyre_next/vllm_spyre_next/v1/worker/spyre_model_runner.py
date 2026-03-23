"""Spyre-specific model runner for vLLM v1.

Inherits from GPUModelRunner (not CPUModelRunner) to preserve the CpuGpuBuffer
dual-buffer pattern where .cpu = CPU staging and .gpu = Spyre device tensors.

CPUModelRunner has two blockers:
  1. `assert device == torch.device("cpu")` — prevents using torch.device("spyre")
  2. `_postprocess_tensors()` — collapses .gpu → .cpu, destroying device placement

Instead, we copy ~15 lines of trivial stubs from CPUModelRunner (CUDA mocking,
disabled flags, no-op stubs) and get ~96% code reuse from GPUModelRunner.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import CompilationMode, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.cpu_model_runner import _torch_cuda_wrapper
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

# Dtypes natively supported on Spyre device.
# int32 is NOT supported — promoted to int64.
_SPYRE_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32,
                           torch.int64, torch.bool}


class SpyreCpuGpuBuffer(CpuGpuBuffer):
    """CpuGpuBuffer adapted for Spyre device constraints.

    - .cpu: always on CPU with original dtype (for numpy staging)
    - .gpu: on Spyre with dtype promotion (int32 → int64)
    - copy_to_gpu: handles dtype conversion and synchronous copies
      (Spyre does not support non_blocking)

    Unsupported dtypes keep .gpu on CPU as a fallback.
    """

    def __init__(
        self,
        *size: int | torch.SymInt,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
        with_numpy: bool = True,
    ) -> None:
        # CPU side: original dtype, always on CPU
        self.cpu = torch.zeros(*size, dtype=dtype, device="cpu",
                               pin_memory=False)

        # GPU (Spyre) side: promote int32 → int64
        spyre_dtype = dtype
        if dtype == torch.int32:
            spyre_dtype = torch.int64

        if spyre_dtype in _SPYRE_SUPPORTED_DTYPES:
            self.gpu = torch.zeros(*size, dtype=spyre_dtype, device=device)
        else:
            # Unsupported dtype (e.g. bool): keep .gpu on CPU
            self.gpu = torch.zeros(*size, dtype=dtype, device="cpu")

        self.np: np.ndarray
        if with_numpy:
            if dtype == torch.bfloat16:
                raise ValueError(
                    "Bfloat16 torch tensors cannot be directly cast to a "
                    "numpy array, so call with with_numpy=False"
                )
            self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        # Synchronous copy with optional dtype promotion (int32 → int64).
        # Spyre does not support non_blocking=True.
        src = self.cpu if n is None else self.cpu[:n]
        dst = self.gpu if n is None else self.gpu[:n]
        if src.dtype != dst.dtype:
            return dst.copy_(src.to(dst.dtype))
        return dst.copy_(src)

    def copy_to_cpu(self, n: int | None = None) -> torch.Tensor:
        src = self.gpu if n is None else self.gpu[:n]
        dst = self.cpu if n is None else self.cpu[:n]
        if src.dtype != dst.dtype:
            return dst.copy_(src.to(dst.dtype))
        return dst.copy_(src)


class TorchSpyreModelRunner(GPUModelRunner):
    """Model runner for IBM's Spyre device.

    Treats Spyre as the 'GPU' device in vLLM's CpuGpuBuffer pattern:
    - .cpu tensors on CPU (numpy staging for scheduler)
    - .gpu tensors on Spyre (fed to the compiled model)

    Inherits from GPUModelRunner (not CPUModelRunner) to preserve
    the dual-buffer device placement pattern.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Run GPUModelRunner.__init__ with device="cpu" to avoid crashes
        # during initialization. Many components (logits processors, input
        # batch buffers) create tensors on `self.device`, and Spyre does not
        # support all dtypes (int32, int64, bool) or non_blocking copies.
        # By initializing with CPU, all buffers and logitsprocs tensors are
        # safely created on CPU. We restore self.device to Spyre afterward
        # for model loading and forward execution.
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, torch.device("cpu"))

        # Check if the model dtype is different from float16,
        # which is only currently in torch-spyre
        if vllm_config.model_config.dtype != torch.float16:
            raise ValueError(f'The model dtype needs to be torch.float16 for spyre, but was specified to be {vllm_config.model_config.dtype}')

        # Restore the real Spyre device for model loading and execution
        self.device = device

        # Disable GPU-specific features (same as CPUModelRunner)
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        # NOTE: Because we initialized with device="cpu", persistent buffers
        # (input_ids, positions, seq_lens, etc.) were created with .gpu on
        # CPU. We now recreate them with SpyreCpuGpuBuffer so .gpu is on
        # Spyre (with int32→int64 promotion). This is done lazily via the
        # _make_buffer override — the buffers are recreated by calling
        # the parent's buffer setup again, but that's not practical for
        # all buffers. Instead, we patch existing buffers in-place.
        self._promote_buffers_to_spyre()

    def load_model(self, load_dummy_weights: bool = False) -> None:
        """Load model, move to Spyre, and compile."""
        logger.info("Loading model %s...", self.model_config.model)

        # Step 1: Load model on CPU (safe — all ops work on CPU).
        # OOT-registered layers (VocabParallelEmbedding, RMSNorm, SiluAndMul)
        # are automatically replaced at instantiation via register_oot().
        self.model = get_model(vllm_config=self.vllm_config)
        self.model_memory_usage = 0  # No GPU memory profiling for Spyre

        if self.lora_config:
            self.model = self.load_lora_model(
                self.model, self.vllm_config, self.device
            )

        # Step 2: Move model to Spyre device.
        self.model = self.model.to(device=self.device)

        # Step 3: Compile for Spyre
        self._compile_for_spyre()

        logger.info("Model loaded and compiled for Spyre.")

    def _compile_for_spyre(self) -> None:
        """Apply torch.compile with appropriate backend."""
        mode = self.compilation_config.mode

        if mode == CompilationMode.NONE:
            logger.info("Compilation disabled (eager mode)")
            return

        if mode == CompilationMode.STOCK_TORCH_COMPILE:
            backend = self.compilation_config.init_backend(self.vllm_config)
            # Custom ops (spyre_rmsnorm, spyre_cpu_fallback, etc.) are opaque
            # to dynamo but don't cause graph breaks — fullgraph=True is safe.
            self.model = torch.compile(
                self.model,
                backend=backend,
                fullgraph=True,
            )
            return

        if mode == CompilationMode.VLLM_COMPILE:
            # vLLM's custom compilation uses the @support_torch_compile
            # decorator on model classes. The decorator handles compilation
            # during model instantiation. Nothing extra needed here.
            return

        logger.warning("Unsupported compilation mode %s for Spyre", mode)

    def warming_up_model(self) -> None:
        """Trigger Spyre compilation via dummy forward pass."""
        logger.info("Warming up model for Spyre compilation...")
        with _set_spyre_compilation_settings(self.vllm_config):
            self._dummy_run(
                min(
                    max(16, self.max_num_reqs),
                    self.scheduler_config.max_num_batched_tokens,
                )
            )
        logger.info("Spyre warmup done.")

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors=None,
    ) -> Any:
        """Run upstream execute_model, then transfer logits to CPU for sampling.

        Sampling operations (torch.multinomial, torch.topk) may not be
        supported on Spyre, so logits must be on CPU before sample_tokens().
        """
        result = super().execute_model(scheduler_output, intermediate_tensors)

        # Transfer logits and hidden states to CPU for sampling
        if self.execute_model_state is not None:
            state = self.execute_model_state
            # logits
            if state.logits is not None and state.logits.device.type == "spyre":
                # Synchronous copy — non_blocking not yet supported on Spyre
                new_logits = state.logits.to("cpu")
                self.execute_model_state = state._replace(logits=new_logits)
                state = self.execute_model_state
            # hidden_states
            if isinstance(state.hidden_states, torch.Tensor):
                if state.hidden_states.device.type == "spyre":
                    new_hs = state.hidden_states.to("cpu")
                    self.execute_model_state = state._replace(
                        hidden_states=new_hs
                    )
                    state = self.execute_model_state
            # sample_hidden_states
            if isinstance(state.sample_hidden_states, torch.Tensor):
                if state.sample_hidden_states.device.type == "spyre":
                    new_shs = state.sample_hidden_states.to("cpu")
                    self.execute_model_state = state._replace(
                        sample_hidden_states=new_shs
                    )

        return result

    # --- KV cache allocation ---

    def _allocate_kv_cache_tensors(
        self, kv_cache_config,
    ) -> dict[str, torch.Tensor]:
        """Allocate KV cache tensors on CPU instead of Spyre.

        Spyre device memory is limited and may not support int8 tensors
        (used as raw storage). KV cache stays on CPU; the attention backend
        (TorchSDPA) operates on CPU tensors. The model receives KV cache
        references and handles device placement internally.

        TODO: Move KV cache to Spyre once device memory is sufficient and
        int8/view operations are supported.
        """
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            tensor = torch.zeros(
                kv_cache_tensor.size, dtype=torch.int8, device="cpu"
            )
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys()), (
            "Some layers are not correctly initialized"
        )
        return kv_cache_raw_tensors

    # --- Stubs copied from CPUModelRunner ---
    # These are trivial overrides that GPUModelRunner expects.

    def _init_device_properties(self) -> None:
        # No CUDA/GPU device properties to query for Spyre
        pass

    def _sync_device(self) -> None:
        # TODO: Replace with torch.spyre.synchronize() when available.
        # For now, all copies are synchronous (no non_blocking), so
        # explicit sync is not needed.
        pass

    def get_dp_padding(
        self, num_tokens: int
    ) -> tuple[int, torch.Tensor | None]:
        return 0, None

    def get_model(self) -> nn.Module:
        return self.model

    # --- Buffer management ---

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True
    ) -> SpyreCpuGpuBuffer:
        """Create a SpyreCpuGpuBuffer with .gpu on Spyre."""
        return SpyreCpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            pin_memory=False,
            with_numpy=numpy,
        )

    def _promote_buffers_to_spyre(self) -> None:
        """Move CpuGpuBuffer .gpu tensors to Spyre device.

        After super().__init__ with device="cpu", all buffers have .gpu on
        CPU. This method moves them to Spyre with dtype promotion
        (int32 → int64) where needed.
        """
        for attr_name in dir(self):
            obj = getattr(self, attr_name, None)
            if not isinstance(obj, CpuGpuBuffer):
                continue
            if obj.gpu.device == self.device:
                continue
            # Promote int32 → int64 (Spyre doesn't support int32)
            spyre_dtype = obj.gpu.dtype
            if spyre_dtype == torch.int32:
                spyre_dtype = torch.int64
            if spyre_dtype in _SPYRE_SUPPORTED_DTYPES:
                obj.gpu = obj.gpu.to(dtype=spyre_dtype, device=self.device)
            logger.debug(
                "Promoted buffer %s to Spyre (%s → %s, device=%s)",
                attr_name,
                obj.cpu.dtype,
                obj.gpu.dtype,
                obj.gpu.device,
            )


@contextmanager
def _set_spyre_compilation_settings(config: VllmConfig):
    """Context manager for Spyre-specific compilation settings during warmup.

    Similar to _set_global_compilation_settings in cpu_model_runner.py but
    adapted for Spyre's compilation requirements.
    """
    import torch._inductor.config as torch_inductor_config

    inductor_config = config.compilation_config.inductor_compile_config
    freezing_value = torch_inductor_config.freezing
    try:
        if inductor_config.get("max_autotune", False):
            torch_inductor_config.freezing = True
        yield
    finally:
        torch_inductor_config.freezing = freezing_value

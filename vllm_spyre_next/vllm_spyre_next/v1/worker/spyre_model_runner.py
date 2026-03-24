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

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.cpu_model_runner import _torch_cuda_wrapper
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _parse_target_device(args, kwargs):
    """Extract target device from .to() arguments."""
    if args:
        if isinstance(args[0], (str, torch.device)):
            return torch.device(args[0])
        if isinstance(args[0], torch.Tensor):
            return args[0].device
    if "device" in kwargs:
        return torch.device(kwargs["device"])
    return None


class SpyreSafeView(torch.Tensor):
    """Tensor subclass that handles Spyre's sliced-view copy bug.

    torch-spyre crashes in copy_device_to_host when copying a sliced
    view (tensor[:n]) from Spyre to CPU. This affects ALL dtypes.
    Full (non-sliced) tensor copies work fine.

    This subclass wraps Spyre tensors and:
    - __getitem__: returns a SpyreSafeView of the real Spyre view
      (writes via .copy_() go to actual Spyre memory)
    - .to("cpu") / .cpu(): if this tensor is a view, goes through
      the parent: parent.to("cpu")[slice_idx] (safe path)

    Remove this class once torch-spyre fixes sliced view copies.
    See dtype_repro.py in torch-spyre for the reproduction.
    """

    @staticmethod
    def __new__(cls, data):
        return torch.Tensor._make_subclass(cls, data)

    def __getitem__(self, idx):
        real_slice = torch.Tensor.__getitem__(self, idx)
        if isinstance(real_slice, torch.Tensor) \
                and real_slice.device.type == "spyre":
            safe = torch.Tensor._make_subclass(SpyreSafeView, real_slice)
            safe._parent_ref = self
            safe._slice_idx = idx
            return safe
        return real_slice

    def to(self, *args, **kwargs):
        target = _parse_target_device(args, kwargs)
        if target is not None and target.type == "cpu" \
                and self.device.type == "spyre":
            parent = getattr(self, "_parent_ref", None)
            idx = getattr(self, "_slice_idx", None)
            if parent is not None and idx is not None:
                # View: move parent to CPU (recursively safe), then slice
                return parent.to("cpu")[idx]
            # Full tensor: direct .to("cpu") is safe
        return torch.Tensor.to(self, *args, **kwargs)

    def cpu(self):
        return self.to("cpu")


def spyre_safe_view(tensor: torch.Tensor) -> torch.Tensor:
    """Wrap a Spyre tensor in SpyreSafeView for safe slicing.

    Returns the tensor unchanged if not on Spyre.
    Remove calls to this function once torch-spyre fixes sliced copies.
    """
    if tensor.device.type == "spyre":
        return torch.Tensor._make_subclass(SpyreSafeView, tensor)
    return tensor


class SpyreCpuGpuBuffer(CpuGpuBuffer):
    """CpuGpuBuffer with Spyre-safe copies and split dtypes.

    Supports different dtypes on CPU vs Spyre (e.g. int32 CPU, int64 Spyre).
    All copies are synchronous (Spyre doesn't support non_blocking).

    .gpu is wrapped in SpyreSafeView to handle Spyre's sliced-view copy
    bug. Slicing .gpu returns real Spyre views (writes work), but
    .to("cpu") on a view goes through the parent tensor safely.
    """

    def __init__(
        self,
        *size: int | torch.SymInt,
        cpu_dtype: torch.dtype,
        gpu_dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
        with_numpy: bool = True,
    ) -> None:
        self.cpu = torch.zeros(*size, dtype=cpu_dtype, device="cpu",
                               pin_memory=pin_memory)
        gpu_raw = torch.zeros(*size, dtype=gpu_dtype, device=device)
        self.gpu = spyre_safe_view(gpu_raw)
        self.np: "np.ndarray"
        if with_numpy:
            if cpu_dtype == torch.bfloat16:
                raise ValueError(
                    "Bfloat16 torch tensors cannot be directly cast to a "
                    "numpy array, so call SpyreCpuGpuBuffer with "
                    "with_numpy=False"
                )
            self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        src = self.cpu if n is None else self.cpu[:n]
        # Write to Spyre slice (host-to-device writes are safe)
        dst = self.gpu if n is None else torch.Tensor.__getitem__(self.gpu, slice(None, n))
        if src.dtype != dst.dtype:
            dst.copy_(src.to(dst.dtype))
        else:
            dst.copy_(src)
        return dst

    def copy_to_cpu(self, n: int | None = None) -> torch.Tensor:
        # Read from Spyre via safe view (device-to-host on views crashes)
        src = self.gpu if n is None else self.gpu[:n]
        dst = self.cpu if n is None else self.cpu[:n]
        # src is a SpyreSafeView — .to("cpu") is safe
        cpu_src = src.to("cpu") if src.device.type == "spyre" else src
        if cpu_src.dtype != dst.dtype:
            return dst.copy_(cpu_src.to(dst.dtype))
        return dst.copy_(cpu_src)


class TorchSpyreModelRunner(GPUModelRunner):
    """Model runner for IBM's Spyre device.

    Treats Spyre as the 'GPU' device in vLLM's CpuGpuBuffer pattern:
    - .cpu tensors on CPU (numpy staging for scheduler)
    - .gpu tensors on Spyre (fed to the compiled model)

    Inherits from GPUModelRunner (not CPUModelRunner) to preserve
    the dual-buffer device placement pattern.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Store the real Spyre device before super().__init__ so that
        # _make_buffer can place .gpu tensors on Spyre directly.
        self._spyre_device = device

        # Run GPUModelRunner.__init__ with device="cpu" to avoid crashes
        # during initialization. Many components (logits processors, input
        # batch buffers) create tensors on `self.device`, and Spyre does not
        # support all dtypes (int32, int64, bool) or non_blocking copies.
        # By initializing with CPU, all buffers and logitsprocs tensors are
        # safely created on CPU. _make_buffer overrides the .gpu placement
        # to use the real Spyre device with dtype promotion.
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, torch.device("cpu"))

        # Check if the model dtype is different from float16,
        # which is only currently in torch-spyre
        if vllm_config.model_config.dtype != torch.float16:
            raise ValueError(f'The model dtype needs to be torch.float16 for spyre, but was specified to be {vllm_config.model_config.dtype}')

        # Keep self.device = CPU (set by super().__init__).
        # For enforce_eager: all OOT layers have CPU weights (via _apply
        # no-op in SpyreCpuFallbackMixin), so the model runs entirely on
        # CPU. SiluAndMul uses Spyre briefly via convert_for_spyre().
        # For compiled mode: _model_forward will move inputs to Spyre.
        # self._spyre_device is available for explicit Spyre placement.

        # Disable GPU-specific features (same as CPUModelRunner)
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        # Replace Triton kernels with CPU-compatible implementations
        # (same as CPUModelRunner._postprocess_triton)
        self._postprocess_triton()

    def _postprocess_triton(self) -> None:
        """Replace Triton kernels with CPU-compatible implementations.

        Triton is not available on Spyre, so we use the same CPU fallbacks
        that CPUModelRunner uses.
        """
        import vllm.utils.cpu_triton_utils as cpu_tl
        import vllm.v1.worker.block_table

        vllm.v1.worker.block_table._compute_slot_mapping_kernel = (
            cpu_tl.compute_slot_mapping_kernel
        )

    def load_model(self, load_dummy_weights: bool = False) -> None:
        """Load model and optionally compile for Spyre."""
        logger.info("Loading model %s...", self.model_config.model)

        # Load model on CPU. OOT-registered layers (VocabParallelEmbedding,
        # RMSNorm, SiluAndMul, Linear, etc.) are automatically replaced at
        # instantiation via register_oot(). Their _apply() no-op keeps
        # weights on CPU regardless of .to() calls.
        self.model = get_model(vllm_config=self.vllm_config)
        self.model_memory_usage = 0  # No GPU memory profiling for Spyre

        if self.lora_config:
            self.model = self.load_lora_model(
                self.model, self.vllm_config, self.device
            )

        # Compile for Spyre (no-op if enforce_eager=True)
        self._compile_for_spyre()

        logger.info("Model loaded and compiled for Spyre.")

    def _compile_for_spyre(self) -> None:
        """Apply torch.compile for Spyre with static shapes.

        Spyre compilation is handled here (not by vLLM's @support_torch_compile)
        because Spyre requires static shapes — dynamic shapes (SymInt) are not
        supported by the Spyre Inductor backend.

        Supported modes:
        - enforce_eager=True: no compilation (eager execution)
        - CompilationMode.NONE: Spyre-managed compilation with torch.compile
        Other vLLM compilation modes (VLLM_COMPILE, STOCK_TORCH_COMPILE) are
        not supported — the platform forces CompilationMode.NONE in
        apply_config_platform_defaults().
        """
        from vllm.config import CompilationMode

        mode = self.compilation_config.mode
        if mode != CompilationMode.NONE:
            raise ValueError(
                f"Unsupported compilation mode {mode} for Spyre. "
                f"Only CompilationMode.NONE is supported. Spyre handles "
                f"compilation internally via _compile_for_spyre(). "
                f"Use enforce_eager=True to disable compilation entirely."
            )

        if self.vllm_config.model_config.enforce_eager:
            logger.info("Compilation disabled (enforce_eager=True)")
            return

        # Custom ops (spyre_rmsnorm, spyre_cpu_fallback, etc.) are opaque
        # to dynamo but don't cause graph breaks — fullgraph=True is safe.
        # dynamic=False ensures static shapes (Spyre can't handle SymInt).
        self.model = torch.compile(
            self.model,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        logger.info("Model compiled for Spyre (backend=inductor)")

    # No _dummy_run override needed — self.device is CPU, so
    # logit_indices and all tensors are created on CPU naturally.

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

    # No execute_model override needed — with self.device=CPU, all model
    # outputs (logits, hidden_states) are already on CPU. No Spyre→CPU
    # transfer required for sampling.

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
        """Create a SpyreCpuGpuBuffer with Spyre-aware placement.

        - .cpu stays on CPU with original dtype (for numpy/scheduler).
        - float16 .gpu on Spyre, wrapped in SpyreSafeView (sliced-view
          copy bug — see dtype_repro.py).
        - int/bool .gpu stays on CPU. These are infrastructure tensors
          (slot mapping, block table, query_start_loc) used by CPU-based
          attention/KV cache. The C++ compute_slot_mapping_kernel expects
          int32 on CPU — Spyre's int64 promotion would break it.
        """
        if dtype == torch.float16:
            # float16 goes on Spyre, wrapped in SpyreSafeView.
            return SpyreCpuGpuBuffer(
                *size,
                cpu_dtype=dtype,
                gpu_dtype=dtype,
                device=self._spyre_device,
                pin_memory=False,
                with_numpy=numpy,
            )

        # Integer and bool buffers stay on CPU (infrastructure tensors
        # for slot mapping, block table, query_start_loc, etc.).
        return SpyreCpuGpuBuffer(
            *size,
            cpu_dtype=dtype,
            gpu_dtype=dtype,
            device="cpu",
            pin_memory=False,
            with_numpy=numpy,
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

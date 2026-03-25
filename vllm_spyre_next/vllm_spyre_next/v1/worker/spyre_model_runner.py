"""Spyre-specific model runner for vLLM v1.

Inherits from GPUModelRunner (not CPUModelRunner) to preserve the CpuGpuBuffer
dual-buffer pattern where .cpu = CPU staging and .gpu = Spyre device tensors.

All tensors live on Spyre (inputs, layer parameters for Spyre-native layers)
except KV cache (stays on CPU). CPU fallback layers keep weights on CPU via
_apply no-op in SpyreCpuFallbackMixin.

Dtype casting: float→float16, int→int64, bool→int64 (Spyre-supported dtypes).
Slicing: not supported on Spyre — SpyreSafeView moves to CPU, slices, moves back.
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

from vllm_spyre_next.custom_ops.utils import spyre_to_cpu

logger = init_logger(__name__)


def _spyre_dtype(dtype: torch.dtype) -> torch.dtype | None:
    """Map a dtype to a Spyre-supported dtype, or None if not suitable.

    Spyre supports float16 for floats and int64 for integers.
    torch-spyre stores int64 as int32 internally (storage = N*4 not N*8),
    which breaks tensor slicing for views >50% of storage. The 2x buffer
    workaround in SpyreCpuGpuBuffer handles this by allocating double-size
    backing tensors so any valid slice stays within bounds.

    Bool tensors stay on CPU (only used in speculative decode / multimodal).

    Returns:
        torch.float16 for float dtypes, torch.int64 for int dtypes,
        None for bool/unsupported dtypes.
    """
    if dtype.is_floating_point:
        return torch.float16
    if dtype in (torch.int32, torch.int64):
        return torch.int64
    return None  # bool tensors stay on CPU


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
                # View: move parent to CPU, then slice
                return parent.to("cpu")[idx]
            # Use spyre_to_cpu which handles subclasses, views, etc.
            return spyre_to_cpu(self)
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

    For int64 tensors on Spyre, uses a 2x backing buffer workaround:
    torch-spyre stores int64 as int32 internally (storage = N*4 not N*8),
    so any slice >50% of storage fails. By allocating 2x the size, the
    actual .gpu view covers exactly 50% and all sub-slices stay in bounds.
    This only works for 1D tensors; 2D int tensors fall back to CPU.

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

        # Determine if this is a 1D int tensor on Spyre needing 2x backing.
        # Normalize shape: *size may be individual ints or a single tuple.
        shape = (size[0] if len(size) == 1 and isinstance(size[0], tuple)
                 else size)
        ndim = len(shape) if isinstance(shape, tuple) else 1
        is_1d_int_on_spyre = (
            device.type == "spyre"
            and not gpu_dtype.is_floating_point
            and ndim == 1
        )

        if is_1d_int_on_spyre:
            # 2x backing: allocate double the size so any valid slice of
            # the original size stays ≤50% of storage (within bounds).
            # Storage math: backing 2N has storage=2N*4=8N bytes;
            # view of N elements needs N*8=8N bytes — exactly fits.
            #
            # CRITICAL: Use set_() instead of slicing to create gpu_raw.
            # Slicing (_int_backing[:n]) creates a VIEW, which causes two
            # problems:
            # 1. spyre_to_cpu follows the view chain to the root (2N
            #    elements) and tries to D2H copy it — reads 16N bytes
            #    from 8N bytes of storage → hangs/crashes.
            # 2. Direct D2H of views crashes in torch-spyre DMA code.
            #
            # set_() creates a tensor that SHARES the backing storage but
            # is NOT a view (_is_view()=False). spyre_to_cpu treats it as
            # a full tensor: D2H copies N*8=8N bytes from 8N byte storage.
            n = shape[0] if isinstance(shape, tuple) else shape
            self._int_backing = torch.zeros(
                2 * n, dtype=gpu_dtype, device=device)
            gpu_raw = torch.empty(0, dtype=gpu_dtype, device=device)
            gpu_raw.set_(
                self._int_backing.untyped_storage(),
                0,          # byte offset
                (n,),       # size
                (1,),       # stride
            )
        elif (device.type == "spyre" and not gpu_dtype.is_floating_point
              and ndim > 1):
            # 2D+ int tensors: 2x backing doesn't work due to stride math.
            # Fall back to CPU placement.
            logger.warning_once(
                "2D int tensor (shape %s) cannot use 2x backing on Spyre, "
                "keeping on CPU.", shape)
            gpu_raw = torch.zeros(*size, dtype=gpu_dtype, device="cpu")
        else:
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

        # Phase 1: Init with device="cpu" to avoid dtype/device errors.
        # Many components create tensors on self.device during init, and
        # Spyre doesn't support all dtypes (int32, bool) natively.
        # _make_buffer (overridden below) already places .gpu on Spyre
        # via self._spyre_device regardless of self.device.
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, torch.device("cpu"))

        # Check if the model dtype is different from float16,
        # which is only currently supported in torch-spyre
        if vllm_config.model_config.dtype != torch.float16:
            raise ValueError(
                f'The model dtype needs to be torch.float16 for spyre, '
                f'but was specified to be {vllm_config.model_config.dtype}'
            )

        # Phase 2: Restore self.device to Spyre. Future tensor creations
        # (in execute_model, etc.) will use the Spyre device.
        self.device = self._spyre_device

        # Move plain tensors created during Phase 1 (on CPU) to Spyre.
        self._move_init_tensors_to_spyre()

        # Disable GPU-specific features (same as CPUModelRunner)
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        # Replace Triton kernels with Spyre-aware implementations
        self._postprocess_triton()

    def _move_init_tensors_to_spyre(self) -> None:
        """Adjust plain tensors created during __init__.

        CpuGpuBuffer attributes are already handled by _make_buffer.
        Integer tensors (positions, seq_lens, etc.) stay on CPU because
        Spyre's int64→int32 storage mismatch breaks tensor slicing.
        Only float tensors would be moved to Spyre (currently none).
        """
        # All plain tensor attributes (positions, seq_lens,
        # num_computed_tokens) are integer types — keep on CPU.
        pass

    def _postprocess_triton(self) -> None:
        """Replace Triton kernels with CPU-compatible implementations.

        Triton is not available on Spyre. The C++ compute_slot_mapping_kernel
        expects specific dtypes (int32 for query_start_loc/block_table,
        int64 for positions/slot_mapping). This wrapper handles dtype casts.
        """
        import vllm.utils.cpu_triton_utils as cpu_tl
        import vllm.v1.worker.block_table

        # The C++ kernel (cpu_tl) expects int32 for query_start_loc/block_table
        # and int64 for positions/slot_mapping.  Wrap with dtype casts.
        original_impl = cpu_tl._compute_slot_mapping_kernel_impl

        def spyre_slot_mapping_impl(
            num_tokens, max_num_tokens, query_start_loc, positions,
            block_table, block_table_stride, block_size, slot_mapping,
            **kwargs,
        ):
            cpu_qsl = query_start_loc.to(dtype=torch.int32, device="cpu")
            cpu_pos = positions.to(dtype=torch.int64, device="cpu")
            cpu_bt = block_table.to(dtype=torch.int32, device="cpu")
            cpu_sm = slot_mapping.to(dtype=torch.int64, device="cpu")
            original_impl(
                num_tokens, max_num_tokens, cpu_qsl, cpu_pos,
                cpu_bt, block_table_stride, block_size, cpu_sm, **kwargs,
            )
            # Copy mutated slot_mapping back
            slot_mapping.copy_(cpu_sm.to(
                dtype=slot_mapping.dtype, device=slot_mapping.device))

        # _FuncWrapper provides __getitem__ for Triton-style grid[(...)] syntax
        vllm.v1.worker.block_table._compute_slot_mapping_kernel = (
            cpu_tl._FuncWrapper(spyre_slot_mapping_impl)
        )

    def load_model(self, load_dummy_weights: bool = False) -> None:
        """Load model and compile for Spyre."""
        logger.info("Loading model %s...", self.model_config.model)

        # Load model on CPU. OOT-registered layers (VocabParallelEmbedding,
        # RMSNorm, SiluAndMul, Linear, etc.) are automatically replaced at
        # instantiation via register_oot().
        self.model = get_model(vllm_config=self.vllm_config)
        self.model_memory_usage = 0  # No GPU memory profiling for Spyre

        if self.lora_config:
            self.model = self.load_lora_model(
                self.model, self.vllm_config, self.device
            )

        # Move Spyre-native layer weights to Spyre device.
        # SpyreCpuFallbackMixin._apply() no-op keeps fallback layer weights
        # on CPU (linear, embedding, rotary, lm_head). Only Spyre-native
        # layers (e.g. RMSNorm with compiled Spyre kernel) get moved.
        self.model.to(device=self._spyre_device)
        logger.info("Spyre-native layer weights moved to %s", self._spyre_device)

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

    # --- Model forward override ---

    def _model_forward(self, *args, **kwargs):
        """Run model forward and move outputs from Spyre to CPU.

        After the model forward, hidden_states are on Spyre. Downstream
        operations (logits slicing, compute_logits/lm_head, sampling) all
        run on CPU. Moving here avoids slicing Spyre tensors and ensures
        lm_head (weights on CPU) receives CPU inputs.
        """
        result = super()._model_forward(*args, **kwargs)

        if isinstance(result, torch.Tensor):
            if result.device.type == "spyre":
                return spyre_to_cpu(result)
        elif isinstance(result, tuple):
            return tuple(
                spyre_to_cpu(t) if isinstance(t, torch.Tensor)
                and t.device.type == "spyre" else t
                for t in result
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
        """Create a SpyreCpuGpuBuffer with tensors on Spyre where possible.

        - .cpu stays on CPU with original dtype (for numpy/scheduler).
        - .gpu: float→float16 on Spyre, int→int64 on Spyre (with 2x backing
          for 1D to work around torch-spyre storage bug), bool→CPU.
        - SpyreSafeView wraps .gpu for safe Spyre→CPU copies.
        - 2D int tensors (mrope/xdrope) fall back to CPU in the buffer init.
        """
        gpu_dtype = _spyre_dtype(dtype)
        if gpu_dtype is not None:
            # Float or int tensor → target Spyre device.
            # SpyreCpuGpuBuffer handles 2x backing for 1D int and
            # falls back to CPU for 2D int internally.
            return SpyreCpuGpuBuffer(
                *size,
                cpu_dtype=dtype,
                gpu_dtype=gpu_dtype,
                device=self._spyre_device,
                pin_memory=False,
                with_numpy=numpy,
            )
        # Bool / unsupported → keep on CPU (both .cpu and .gpu)
        return SpyreCpuGpuBuffer(
            *size,
            cpu_dtype=dtype,
            gpu_dtype=dtype,
            device=torch.device("cpu"),
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

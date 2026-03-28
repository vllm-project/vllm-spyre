"""Spyre-specific model runner for vLLM v1.

Inherits from GPUModelRunner (not CPUModelRunner) to preserve the CpuGpuBuffer
dual-buffer pattern where .cpu = CPU staging and .gpu = Spyre device tensors.

Hidden states flow on Spyre between decoder layers. Only float tensors go to
Spyre device (.gpu = float16 on Spyre). Int and bool tensors stay on CPU
(.gpu aliased to .cpu, same as CPUModelRunner).

The model is wrapped with _SpyreOutputWrapper so that the final hidden_states
are automatically converted from Spyre to CPU for downstream operations
(logits indexing, lm_head, sampling).
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

from vllm_spyre_next.custom_ops.utils import convert

logger = init_logger(__name__)


class SpyreCpuGpuBuffer(CpuGpuBuffer):
    """CpuGpuBuffer with Spyre-safe copies and split dtypes.

    For float dtypes: .cpu on CPU, .gpu on Spyre (float16).
    For int/bool dtypes: .gpu aliased to .cpu (CPUModelRunner pattern).
    All copies are synchronous (Spyre doesn't support non_blocking).
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
        if device.type == "spyre":
            self.gpu = torch.zeros(*size, dtype=gpu_dtype, device=device)
        else:
            # int/bool: alias gpu = cpu (CPUModelRunner pattern)
            self.gpu = self.cpu
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
        if self.gpu is self.cpu:
            # Aliased (int/bool) — no copy needed
            return self.gpu if n is None else self.gpu[:n]
        src = self.cpu if n is None else self.cpu[:n]
        dst = self.gpu if n is None else self.gpu[:n]
        dst.copy_(src)
        return dst

    def copy_to_cpu(self, n: int | None = None) -> torch.Tensor:
        if self.gpu is self.cpu:
            # Aliased (int/bool) — no copy needed
            return self.cpu if n is None else self.cpu[:n]
        src = self.gpu if n is None else self.gpu[:n]
        dst = self.cpu if n is None else self.cpu[:n]
        cpu_src = convert(src, device="cpu")
        dst.copy_(cpu_src)
        return dst


class _SpyreOutputWrapper:
    """Transparent wrapper that converts model forward() outputs to CPU.

    The model's final hidden_states come out on Spyre. Downstream operations
    (indexing via logits_indices, compute_logits/lm_head, sampling) all run
    on CPU. Wrapping at the model level ensures ALL call sites get CPU
    outputs — both execute_model (via _model_forward) and _dummy_run
    (which calls self.model(...) directly).

    Attribute access delegates to the wrapped model so that
    self.model.compute_logits, self.model.config, etc. work unchanged.
    """

    def __init__(self, model: nn.Module):
        # Use object.__setattr__ to avoid triggering __setattr__ override
        object.__setattr__(self, "_model", model)

    def __call__(self, *args, **kwargs):
        result = self._model(*args, **kwargs)
        if isinstance(result, torch.Tensor):
            if result.device.type == "spyre":
                return convert(result, device="cpu")
            return result
        if isinstance(result, tuple):
            return tuple(
                convert(t, device="cpu") if isinstance(t, torch.Tensor)
                and t.device.type == "spyre" else t
                for t in result
            )
        return result

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __setattr__(self, name, value):
        setattr(self._model, name, value)


class TorchSpyreModelRunner(GPUModelRunner):
    """Model runner for IBM's Spyre device.

    Treats Spyre as the 'GPU' device in vLLM's CpuGpuBuffer pattern:
    - .cpu tensors on CPU (numpy staging for scheduler)
    - .gpu tensors on Spyre for floats, aliased to CPU for int/bool

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

        # Keep self.device as CPU. Upstream code uses self.device for tensor
        # creation (.to(self.device)), and most of these are int tensors
        # (input_ids, positions, logit_indices) that must stay on CPU.
        # Float tensors reach Spyre through _make_buffer (uses _spyre_device)
        # and the model's OOT layers handle Spyre placement internally.

        # Disable GPU-specific features (same as CPUModelRunner)
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        # Replace Triton kernels with Spyre-aware implementations
        self._postprocess_triton()

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

        # Move layer weights to Spyre device.
        # SpyreCpuFallbackMixin._apply() no-op keeps CPU fallback layer
        # weights on CPU (linear, embedding, rotary, lm_head).
        # Spyre-native layers (RMSNorm, SiluAndMul) get their weights moved.
        self.model.to(device=self._spyre_device)
        logger.info("Spyre-native layer weights moved to %s", self._spyre_device)

        # Compile for Spyre (no-op if enforce_eager=True)
        self._compile_for_spyre()

        # Wrap model so ALL forward() calls (execute_model, _dummy_run, etc.)
        # automatically convert Spyre outputs to CPU. This ensures downstream
        # indexing (logits_indices), lm_head (CPU weights), and sampling all
        # receive CPU tensors without needing per-call-site overrides.
        self.model = _SpyreOutputWrapper(self.model)

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
        """Create a SpyreCpuGpuBuffer with float tensors on Spyre.

        - Float dtypes: .cpu on CPU, .gpu on Spyre as float16
        - Int/bool dtypes: .gpu aliased to .cpu (stays on CPU)
        """
        if dtype.is_floating_point:
            return SpyreCpuGpuBuffer(
                *size,
                cpu_dtype=dtype,
                gpu_dtype=torch.float16,
                device=self._spyre_device,
                pin_memory=False,
                with_numpy=numpy,
            )
        # Int/bool → CPU-only (aliased)
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

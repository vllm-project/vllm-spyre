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
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn as nn
import torch.utils._pytree as pytree

from vllm.config import CompilationMode, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.cpu_model_runner import _torch_cuda_wrapper
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class SpyreCpuFallback(nn.Module):
    """Wraps a module to execute on CPU while the rest of the model runs on Spyre.

    - Intercepts .to() to keep the inner module on CPU
    - Uses @torch._dynamo.disable to force a graph break
    - Transfers tensors: Spyre -> CPU -> execute -> CPU -> Spyre
    """

    def __init__(self, module: nn.Module, spyre_device: torch.device):
        super().__init__()
        self._cpu_module = module  # stays on CPU
        self._spyre_device = spyre_device

    def to(self, *args, **kwargs):
        # Do NOT move the inner module — it must stay on CPU.
        return self

    @torch._dynamo.disable
    def forward(self, *args, **kwargs):
        # Transfer inputs: Spyre -> CPU
        cpu_args = pytree.tree_map(
            lambda t: t.to("cpu") if isinstance(t, torch.Tensor) else t,
            args,
        )
        cpu_kwargs = pytree.tree_map(
            lambda t: t.to("cpu") if isinstance(t, torch.Tensor) else t,
            kwargs,
        )

        # Execute on CPU
        outputs = self._cpu_module(*cpu_args, **cpu_kwargs)

        # Transfer outputs: CPU -> Spyre
        return pytree.tree_map(
            lambda t: t.to(self._spyre_device)
            if isinstance(t, torch.Tensor)
            else t,
            outputs,
        )


class TorchSpyreModelRunner(GPUModelRunner):
    """Model runner for IBM's Spyre device.

    Treats Spyre as the 'GPU' device in vLLM's CpuGpuBuffer pattern:
    - .cpu tensors on CPU (numpy staging for scheduler)
    - .gpu tensors on Spyre (fed to the compiled model)

    Inherits from GPUModelRunner (not CPUModelRunner) to preserve
    the dual-buffer device placement pattern.
    """

    # Layer types that must fall back to CPU execution.
    # Override or configure per-model as needed.
    SPYRE_CPU_FALLBACK_LAYERS: ClassVar[set[str]] = set()

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Mock CUDA APIs — same mechanism as CPUModelRunner.
        # GPUModelRunner.__init__ creates torch.Event and torch.cuda.Stream
        # objects which don't exist without CUDA.
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        # Disable GPU-specific features (same as CPUModelRunner)
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        # NOTE: We intentionally do NOT call _postprocess_tensors().
        # CpuGpuBuffer.gpu tensors remain on the Spyre device.
        # This preserves the CPU -> Spyre transfer via copy_to_gpu().

        # CAVEAT: CpuGpuBuffer creates device tensors via
        # torch.zeros_like(cpu_tensor, device="spyre"). This may fail
        # for dtypes not yet supported on Spyre (e.g., int32, int64).
        # If initialization fails here, override _make_buffer() with
        # dtype-aware routing — see the comprehensive plan Section 8.

    def load_model(self, load_dummy_weights: bool = False) -> None:
        """Load model, apply fallbacks, move to Spyre, and compile."""
        logger.info("Loading model %s...", self.model_config.model)

        # Step 1: Load model on CPU (safe — all ops work on CPU)
        self.model = get_model(vllm_config=self.vllm_config)
        self.model_memory_usage = 0  # No GPU memory profiling for Spyre

        if self.lora_config:
            self.model = self.load_lora_model(
                self.model, self.vllm_config, self.device
            )

        # Step 2: Wrap unsupported layers for CPU fallback.
        # Must happen BEFORE .to("spyre") so wrapped layers stay on CPU.
        if self.SPYRE_CPU_FALLBACK_LAYERS:
            self._apply_cpu_fallbacks()

        # Step 3: Move model to Spyre device.
        # SpyreCpuFallback.to() is a no-op, keeping wrapped layers on CPU.
        self.model = self.model.to(device=self.device)

        # Step 4: Compile for Spyre
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
            # fullgraph=False when we have CPU fallback layers or custom op
            # escapes, since those cause graph breaks.
            has_graph_breaks = bool(self.SPYRE_CPU_FALLBACK_LAYERS)
            self.model = torch.compile(
                self.model,
                backend=backend,
                fullgraph=not has_graph_breaks,
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

    # --- CPU Fallback Infrastructure ---

    def _apply_cpu_fallbacks(self) -> None:
        """Wrap layers listed in SPYRE_CPU_FALLBACK_LAYERS with SpyreCpuFallback."""
        modules_dict = dict(self.model.named_modules())
        for name, module in list(modules_dict.items()):
            if type(module).__name__ in self.SPYRE_CPU_FALLBACK_LAYERS:
                parent_name, _, child_name = name.rpartition(".")
                parent = (
                    self.model if not parent_name else modules_dict[parent_name]
                )
                setattr(
                    parent,
                    child_name,
                    SpyreCpuFallback(module, self.device),
                )
                logger.info(
                    "CPU fallback: %s (%s)", name, type(module).__name__
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

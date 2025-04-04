# This import wraps the importing of some vLLM classes based on the version

try:
    # vllm v0.8.2+
    from vllm.v1.core.sched.output import CachedRequestData  # noqa: F401
    from vllm.v1.core.sched.output import NewRequestData  # noqa: F401
    from vllm.v1.core.sched.output import SchedulerOutput  # noqa: F401
except ImportError:
    from vllm.v1.core.scheduler import CachedRequestData  # noqa: F401
    from vllm.v1.core.scheduler import NewRequestData  # noqa: F401
    from vllm.v1.core.scheduler import SchedulerOutput  # noqa: F401

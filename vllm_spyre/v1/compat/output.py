# This import wraps the importing of some vLLM classes based on the version
import inspect
from dataclasses import dataclass
from typing import ClassVar

# yapf conflicts with ruff for this block
# yapf: disable
try:
    # vllm v0.8.2+
    from vllm.v1.core.sched.output import (
        NewRequestData as UpstreamNewRequestData)
except ImportError:
    from vllm.v1.core.scheduler import NewRequestData as UpstreamNewRequestData
# yapf: enable


@dataclass
class NewRequestData(UpstreamNewRequestData):
    _legacy_signature: ClassVar[bool] = False

    def __init__(self, *args, **kwargs):
        # The prompt field was removed in https://github.com/vllm-project/vllm/pull/17214/files#diff-cafd89ce8a698a56acb24ada62831cbc7a980782f78a52d1742ba238031f296cL25
        # NB: this assumes kwargs are used at the call site
        if self._legacy_signature:
            kwargs['prompt'] = None

        super().__init__(*args, **kwargs)


if 'prompt' in inspect.signature(UpstreamNewRequestData.__init__).parameters:
    NewRequestData._legacy_signature = True

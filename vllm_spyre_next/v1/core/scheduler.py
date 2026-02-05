# SPDX-License-Identifier: Apache-2.0

from vllm.v1.core.sched.scheduler import Scheduler


class TorchSpyreScheduler(Scheduler):
    """Torch Spyre specific scheduler class inheriting from the V1 scheduler."""

    raise NotImplementedError

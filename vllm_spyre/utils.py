import contextlib
import math

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)


@contextlib.contextmanager
def stagger_region(limit: int, world_size: int, rank: int):
    """
    Limit the number of concurrent processes into this region of code.
    Processes yield from this function when they are allowed to enter the region of code.
    Processes return from this function when all of the processes have completed the region of code.
    :param limit: Number of concurrent processes allowed in the code region if > 0.
    """
    if limit > 0 and limit != world_size:
        for _set in range(math.ceil(world_size / float(limit))):
            if rank < (_set + 1) * limit:
                break
            torch.distributed.barrier()
        logger.info("Stagger Region Enter (Set: %d) of %d", _set + 1,
                    math.ceil(world_size / float(limit)))
    yield {}
    if limit > 0 and limit != world_size:
        for _set in range(math.ceil(world_size / float(limit))):
            if rank >= (_set + 1) * limit:
                continue
            torch.distributed.barrier()
        logger.inbfo("Stagger Region: All Complete")

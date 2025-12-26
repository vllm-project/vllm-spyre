import contextlib
import importlib.abc
import importlib.machinery
import math
import sys

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


@contextlib.contextmanager
def stagger_region(limit: int, world_size: int, rank: int):
    """
    Limit the number of concurrent processes into this region of code.
    Processes yield from this function when they are allowed to enter the
    region of code. Processes return from this function when all of the
    processes have completed the region of code.

    :param limit: Number of concurrent processes allowed if > 0.
    :param world_size: Total world size, usually TP * PP.
    :param rank: Rank of calling worker process.
    """
    if limit > 0 and limit < world_size:
        for _set in range(math.ceil(world_size / float(limit))):
            if rank < (_set + 1) * limit:
                break
            torch.distributed.barrier()
        logger.info("Stagger Region Enter (Set: %d) of %d", _set + 1,
                    math.ceil(world_size / float(limit)))
    yield {}

    # TODO: make sure this isn't called excessively

    if limit > 0 and limit < world_size:
        logger.info("Rank %d Done With Stagger Region", rank)
        for _set in range(math.ceil(world_size / float(limit))):
            if rank >= (_set + 1) * limit:
                continue
            torch.distributed.barrier()
        logger.info("Stagger Region: All Complete")


def exact_div(a: int, b: int) -> int:
    q, r = divmod(a, b)
    if r != 0:
        raise ValueError(f"{a} is not exactly divisible by {b}")
    return q


# Utilities to monkeypatch modules in upstream vLLM

# Supports changing the default values for configurations. Complexity comes from
# circular imports, import order/initialization, and various defaulting
# approaches:
# - Field default values in the config dataclasses
# - CLI argparse defaults (mostly pull from the dataclasses)
# - __post_init__ functions on config dataclasses that modify values
# - EngineArgs's _set_default_args setting defaults based on runtime
#   configuration
#
# TODO: Consider adding a plugin extension point upstream to support this
# use-case


## Import hook patching
#
# Register hooks to modify a module that can't be imported yet (e.g. circular
# import)
class PatchLoader(importlib.abc.Loader):
    """Wrapped loader to patch a module after it is executed."""

    def __init__(self, real_loader, patch_func):
        self.real_loader = real_loader
        self.patch_func = patch_func

    def create_module(self, spec):
        # Delegate module creation
        if hasattr(self.real_loader, "create_module"):
            return self.real_loader.create_module(spec)
        # Return None to let default import machinery create it
        return None

    def exec_module(self, module):
        self.real_loader.exec_module(module)
        self.patch_func(module)


class PatchingFinder(importlib.abc.MetaPathFinder):
    """Intercepts the import of a module to install a PatchLoader."""

    def __init__(self, target_fullname, patch_func):
        self.target_fullname = target_fullname
        self.patch_func = patch_func

    def find_spec(self, fullname, path, target=None):
        if fullname != self.target_fullname:
            return None

        # Use default PathFinder machinery to avoid recursion with
        # importlib.util.find_spec
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None

        spec.loader = PatchLoader(spec.loader, self.patch_func)
        return spec


def install_patch(target_fullname, patch_func):
    """Installs patch_func to modify the target_fullname module.

    Patch application may happen right away if target_fullname is fully
    initialized or via an import hook after target_fullname is imported.

    Raises a ValueError if the same target is patched multiple times.
    """
    # If the module is already in sys.modules, we can patch directly
    if (mod := sys.modules.get(target_fullname)) is not None:
        logger.info("Installing %s patch on %s", patch_func.__name__,
                    target_fullname)
        patch_func(mod)
        return
    # else we register a hook to patch at a subsequent import
    logger.info("Installing %s as import hook patch on %s",
                patch_func.__name__, target_fullname)

    # Install PatchingFinder at the front of sys.meta_path so it runs first.
    # Avoid installing multiple times.
    for f in sys.meta_path:
        if isinstance(f,
                      PatchingFinder) and f.target_fullname == target_fullname:
            raise ValueError(
                f"Cannot install second patch for {target_fullname}")
    sys.meta_path.insert(0, PatchingFinder(target_fullname, patch_func))

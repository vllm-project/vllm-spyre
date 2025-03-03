import json
import os
import sys
import tempfile

import torch

# ==============================================================
# Common utilities
# ==============================================================
#-------------
# Discover the world size and my rank (envars set by torchrun)
# https://pytorch.org/docs/stable/elastic/run.html#environment-variables
#-------------
local_rank = int(os.getenv("LOCAL_RANK", 0))
rank = int(os.getenv("RANK", 0))
world_rank = rank
world_size = int(os.getenv("WORLD_SIZE", 1))

def dprint(text):
    print(f"[{rank:2d}/{world_size:2d}]: {text}")

# ==============================================================
# Common setup
# ==============================================================
def spyre_setup(rank=0, world_size=1, local_rank=0, local_size=1, verbose=False):
    # -------------
    # Envar setup for backend
    # -------------
    # Environment variable created by the runtime to identify the specific Spyre card that is assigned to this rank
    spyre_config_file_envar = "AIU_CONFIG_FILE_" + str(rank)

    # Default to senulator backend unless user specified otherwise
    os.environ.setdefault("FLEX_COMPUTE", "SENULATOR")
    os.environ.setdefault("FLEX_DEVICE", "MOCK")

    # Each rank needs a unique space to write its binaries
    # For both 'export' and '__pycache'
    # https://docs.python.org/3/library/sys.html#sys.pycache_prefix
    with tempfile.TemporaryDirectory() as exportdir:
        os.environ.setdefault("DEEPRT_EXPORT_DIR", exportdir)
        os.environ.setdefault("DTCOMPILER_EXPORT_DIR", exportdir)
    if world_size > 1:
        os.environ["DEEPRT_EXPORT_DIR"] += f"/{rank}"
        os.environ["DTCOMPILER_EXPORT_DIR"] += f"/{rank}"
        sys.pycache_prefix=os.getenv("DEEPRT_EXPORT_DIR")+"/py-" + str(rank)
    os.environ.setdefault("DTCOMPILER_KEEP_EXPORT", "0")

    # Inform Flex of the size of this job
    os.environ.setdefault("FLEX_RDMA_WORLD_SIZE", str(world_size))
    os.environ.setdefault("FLEX_RDMA_WORLD_RANK", str(rank))
    os.environ.setdefault("FLEX_RDMA_LOCAL_SIZE", str(world_size))
    os.environ.setdefault("FLEX_RDMA_LOCAL_RANK", str(rank))
    for peer_rank in range(world_size):
        pcie_env_str="AIU_WORLD_RANK_"+str(peer_rank)
        flex_env_str="FLEX_RDMA_PCI_BUS_ADDR_"+str(peer_rank)
        if os.getenv(pcie_env_str) is None:
            raise RuntimeError(f"Error: The environment variable {pcie_env_str} is not defined")
        if os.getenv(flex_env_str) is None:
            raise RuntimeError(f"Error: The environment variable {flex_env_str} is not defined")
    if os.getenv("DUMP_MEMMAP") is not None:
        if os.getenv("SDSC_REF_DIR") is None:
            os.environ["SDSC_REF_DIR"] = os.environ["DEEPRT_EXPORT_DIR"]
        else:
            os.environ["SDSC_REF_DIR"] += f"/{rank}"
        assert (
            os.getenv("DUMP_MEMMAP_DIR") is not None
        ), "DUMP_MEMMAP_DIR not set while DUMP_MEMMAP set"
        os.environ["DUMP_MEMMAP_DIR"] += f"/{rank}"
        os.makedirs(
            os.environ["DUMP_MEMMAP_DIR"], exist_ok=True
        )  # directory needs to exist

    for peer_rank in range(world_size):
        pcie_env_str = "AIU_WORLD_RANK_" + str(peer_rank)
        flex_env_str = "FLEX_RDMA_PCI_BUS_ADDR_" + str(peer_rank)
        if os.getenv("FLEX_COMPUTE") == "SENULATOR":
            if os.getenv(pcie_env_str) is not None:
                os.environ[flex_env_str] = os.getenv(pcie_env_str)
            else:
                os.environ[pcie_env_str] = f"0000:{rank:02x}:01.0"
                os.environ[flex_env_str] = f"0000:{rank:02x}:01.0"
        else:
            if os.getenv(flex_env_str) is None:
                if os.getenv("PCIDEVICE_IBM_COM_SENTIENT_PF") is not None:
                    os.environ[pcie_env_str] = os.getenv(
                        "PCIDEVICE_IBM_COM_SENTIENT_PF"
                    )

                if os.getenv(pcie_env_str) is not None:
                    os.environ[flex_env_str] = os.getenv(pcie_env_str)
                else:
                    raise RuntimeError(
                        f"[{rank}/{world_size}]: ERROR: {flex_env_str} and {pcie_env_str} were not set for peer {peer_rank}."
                    )
        if rank == 0 and verbose:
            dprint(f"PCI Addr Rank {peer_rank} {pcie_env_str}={os.environ[pcie_env_str]}")
            dprint(f"PCI Addr Rank {peer_rank} {flex_env_str}={os.environ[flex_env_str]}")

    if rank == 0 and verbose:
        dprint(f"FLEX_COMPUTE=" + os.getenv("FLEX_COMPUTE"))
        dprint(f"FLEX_DEVICE=" + os.getenv("FLEX_DEVICE"))
        dprint(f"DEEPRT_EXPORT_DIR=" + os.getenv("DEEPRT_EXPORT_DIR"))
        dprint(f"DTCOMPILER_EXPORT_DIR=" + os.getenv("DTCOMPILER_EXPORT_DIR"))
        if os.getenv(spyre_config_file_envar) is not None:
            dprint(f"{spyre_config_file_envar}=" + os.environ[spyre_config_file_envar])
        if os.getenv("SENLIB_DEVEL_CONFIG_FILE") is not None:
            dprint(f"SENLIB_DEVEL_CONFIG_FILE=" + os.environ["SENLIB_DEVEL_CONFIG_FILE"])
        if os.getenv(flex_env_str) is not None:
            dprint(f"{flex_env_str}=" + os.environ[flex_env_str])
        dprint(f"FLEX_RDMA_LOCAL_RANK=" + os.getenv("FLEX_RDMA_LOCAL_RANK"))
        dprint(f"FLEX_RDMA_LOCAL_SIZE=" + os.getenv("FLEX_RDMA_LOCAL_SIZE"))
        dprint(f"FLEX_RDMA_WORLD_RANK=" + os.getenv("FLEX_RDMA_WORLD_RANK"))
        dprint(f"FLEX_RDMA_WORLD_SIZE=" + os.getenv("FLEX_RDMA_WORLD_SIZE"))

    if os.getenv("FLEX_COMPUTE") == "SENTIENT":
        pcie_env_str = "AIU_WORLD_RANK_" + str(rank)
        if os.getenv(pcie_env_str) is not None:
            device_id = os.getenv(pcie_env_str)
        else:
            with open(os.getenv(spyre_config_file_envar)) as fd:
                data = json.load(fd)
                device_id = data["GENERAL"]["sen_bus_id"]
        dprint(f"Spyre: Enabled ({device_id})")
    else:
        dprint(f"Spyre: Disabled (Senulator)")


# ==============================================================
# Distributed setup
# ==============================================================
def spyre_dist_setup(rank, world_size, local_rank=-0, local_size=-1, verbose=False):
    if local_rank < 0:
        local_rank = rank
    if local_size < 0:
        local_size = world_size

    if os.getenv("TORCHELASTIC_RUN_ID") is None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
    elif rank == 0 or verbose:
        dprint(f"Detected running via torchrun")

    if rank == 0 or verbose:
        dprint(f"Parallel Backend: {torch.distributed.get_backend()}")

    spyre_setup(rank, world_size)
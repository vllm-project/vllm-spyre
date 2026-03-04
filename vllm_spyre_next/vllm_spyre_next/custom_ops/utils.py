"""This module contains all custom ops for spyre"""

import torch
import torch.utils._pytree as pytree
from vllm.logger import init_logger

logger = init_logger(__name__)


def prepare_inputs_on_spyre(*args):
    def _convert_to_spyre(arg):
        return (
            arg.to(dtype=torch.float16).to(device=torch.device("spyre"))
            if isinstance(arg, torch.Tensor)
            else arg
        )

    return pytree.tree_map(_convert_to_spyre, args)[0]

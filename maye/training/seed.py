import os
import random

import numpy as np
import torch

from maye import training
from maye.utils import logger


def set_seed(seed: int, debug_mode: str | int | None = None) -> int:
    world_size, rank = training.get_world_size_and_rank()
    max_val = np.iinfo(np.uint32).max - world_size + 1
    min_val = np.iinfo(np.uint32).min
    if seed < min_val or seed > max_val:
        raise ValueError(
            f"Invalid seed value provided: {seed}. Value must be in the range [{min_val}, {max_val}]"
        )
    local_seed = seed + rank
    if rank == 0:
        logger.debug(
            f"Setting manual seed to local seed {local_seed}. Local seed is seed + rank = {seed} + {rank}"
        )

    torch.manual_seed(local_seed)
    np.random.seed(local_seed)
    random.seed(local_seed)

    if debug_mode is not None:
        logger.debug(f"Setting deterministic debug mode to {debug_mode}")
        torch.set_deterministic_debug_mode(debug_mode)
        deterministic_debug_mode = torch.get_deterministic_debug_mode()
        if deterministic_debug_mode == 0:
            logger.debug("Disabling cuDNN deterministic mode")
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            logger.debug("Enabling cuDNN deterministic mode")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # reference: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return seed

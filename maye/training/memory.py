import gc
from typing import Callable

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

ACWrapPolicyType = set[type[nn.Module]] | Callable[[nn.Module, bool, int], bool]


def set_activation_checkpointing(
    model: nn.Module, auto_wrap_policy: ACWrapPolicyType, **kwargs
) -> None:
    if isinstance(auto_wrap_policy, set):
        auto_wrap_policy = ModuleWrapPolicy(auto_wrap_policy)
    apply_activation_checkpointing(model, auto_wrap_policy=auto_wrap_policy, **kwargs)


def cleanup_before_training() -> None:
    """
    Call gc collect, empty CUDA cache, and reset peak memory stats.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

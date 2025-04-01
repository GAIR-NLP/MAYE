import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        # linear warmup phase
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)

        # cosine
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )

        cosine_lr_multiple = 0.5 * (
            1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)
        )
        return max(0.0, cosine_lr_multiple)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    param_groups = optimizer.param_groups
    if len(param_groups) < 1:
        raise RuntimeError(
            f"Invalid optimizer param groups with len of: {len(param_groups)}"
        )

    # LR Schedulers are the same across all param groups for full_finetune right now
    lr = param_groups[0]["lr"]
    for group in param_groups:
        if group["lr"] != lr:
            raise RuntimeError("LR Schedulers are different across all param groups ")
    return lr

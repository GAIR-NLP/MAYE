import os

import torch
from torch import nn

from maye.utils import log_rank_zero


def compile_model(model: nn.Module, verbose: bool = False):
    backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
    log_rank_zero("Compiling model layers with torch.compile...")
    for n, m in reversed(list(model.named_modules())):
        if verbose:
            log_rank_zero(n)
        m.compile(backend=backend)


def compile_loss(loss: nn.Module) -> nn.Module:
    backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
    log_rank_zero("Compiling loss with torch.compile...")
    loss = torch.compile(loss, backend=backend)  # type: ignore
    return loss

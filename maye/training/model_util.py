import torch

from maye import utils


def disable_dropout(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0:
            utils.logger.warning(
                f"Found Dropout with value {module.p} in module {module}. Setting to zero."
            )
            module.p = 0

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils.checkpoint import checkpoint


# Uses PTD FSDP AC wrapper
# currently selective per layer checkpointing are supported
def checkpoint_wrapper(
    module: nn.Module,
    ac_mode: str,
    ac_style: int | str | None = None,
):
    if ac_mode == "full":
        return ptd_checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint,
            use_reentrant=False,
            preserve_rng_state=False,
        )

    # selective layer checkpointing...some checks in case we receive '2' or 2...
    elif ac_mode == "selective":
        """enables selective checkpointing of candidate layers.
        Usage:
        'selective_ac_option' with a positive 'int' value in config controls which layers to checkpoint.
        1 == checkpointing every one (all).
        2 == checkpoint every 2nd one
        """
        if ac_style is None:
            ac_style = 1
        every_x_layer = int(ac_style)

        if not (every_x_layer >= 0):
            raise ValueError(
                f"Selective layer AC policy (every_x_layer) expects a positive integer, received {every_x_layer}"
            )

        checkpoint_wrapper.__dict__.setdefault("_count", 0)

        checkpoint_wrapper._count += 1
        if not every_x_layer or checkpoint_wrapper._count % every_x_layer == 0:
            return ptd_checkpoint_wrapper(
                module,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                checkpoint_fn=checkpoint,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        # skip activation checkpointing and store activations for this layer
        else:
            return module

    else:
        raise NotImplementedError(
            "Unknown AC type or AC config. Only selective op and selective layer ac implemented currently."
        )


def apply_selective_activation_checkpointing(
    model: nn.Module,
    ac_mode: str,
    ac_option: int | str | None = None,
) -> None:
    for layer_id, transformer_block in enumerate(model.layers):
        if ac_mode in ("full", "selective"):
            transformer_block = checkpoint_wrapper(
                transformer_block,
                ac_mode,
                ac_option,
            )
        model.layers[layer_id] = transformer_block

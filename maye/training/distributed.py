import os
from itertools import chain
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._composable.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed._tensor import distribute_tensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh

from maye import utils

FSDPPolicyType = Callable[[nn.Module, bool, int], bool]


_DISTRIBUTED_STATE_DICT_API_IS_AVAILABLE = False


def is_distributed() -> bool:
    port = os.environ.get("MASTER_PORT", "")
    addr = os.environ.get("MASTER_ADDR", "")
    size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", -1))
    avlb = dist.is_available()
    return bool(port and addr and size >= 1 and rank >= 0 and avlb)


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        device = tensor.device
        if dist.get_backend() == "nccl":
            tensor = tensor.to(utils.get_device("cuda"))
        dist.broadcast(tensor, src=src, group=None)
        return tensor.to(device)
    else:
        return tensor


def get_distributed_backend(device_type: str, offload_ops_to_cpu: bool = False) -> str:
    default_device_backend_map = dist.Backend.default_device_backend_map
    backend = "nccl"
    if device_type in default_device_backend_map:
        backend = default_device_backend_map[device_type]
    if offload_ops_to_cpu:
        backend = f"{device_type}:{backend},cpu:gloo"
    return backend


def set_torch_num_threads() -> None:
    num_threads = (os.cpu_count() or 1) // (
        dist.get_world_size() if dist.is_initialized() else 1
    )
    torch.set_num_threads(num_threads)
    utils.logger.info(f"Set intra op parallelism no. of threads to {num_threads}")


def get_world_size_and_rank() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    else:
        return 1, 0


def validate_no_params_on_meta_device(model: nn.Module) -> None:
    for n, p in chain(model.named_parameters(), model.named_buffers()):
        if p.is_meta:
            raise RuntimeError(f"Unexpected param or buffer {n} on meta device.")


def load_from_full_model_state_dict(
    model: nn.Module,
    full_sd: dict[str, Any],
    device: torch.device,
    strict: bool = False,
    cpu_offload: bool = False,
):
    meta_sharded_sd = model.state_dict()

    if _DISTRIBUTED_STATE_DICT_API_IS_AVAILABLE:
        for param_name in full_sd.keys():
            sharded_meta_param = meta_sharded_sd.get(param_name)
            full_sd[param_name] = full_sd[param_name].to(sharded_meta_param.dtype)
        options = StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            strict=strict,
            cpu_offload=cpu_offload,
        )
        return set_model_state_dict(
            model=model, model_state_dict=full_sd, options=options
        )
    else:
        sharded_sd = {}
        for param_name, full_tensor in full_sd.items():
            sharded_meta_param = meta_sharded_sd.get(param_name)
            full_tensor = full_tensor.to(sharded_meta_param.dtype).to(device)
            if not hasattr(sharded_meta_param, "device_mesh"):
                # In cases where parts of the model aren't sharded, some parameters will be plain tensors
                sharded_tensor = full_tensor
            else:
                sharded_tensor = distribute_tensor(
                    full_tensor,
                    sharded_meta_param.device_mesh,
                    sharded_meta_param.placements,
                )
            if cpu_offload:
                sharded_tensor = sharded_tensor.cpu()
            sharded_sd[param_name] = nn.Parameter(sharded_tensor)

        # choose `assign=True` since we cannot call `copy_` on meta tensor
        return model.load_state_dict(sharded_sd, strict=strict, assign=True)


def gather_cpu_state_dict(
    model: nn.Module,
    is_rank_zero: bool,
    device: torch.device,
) -> dict[str, Any]:
    if _DISTRIBUTED_STATE_DICT_API_IS_AVAILABLE:
        cpu_state_dict = {}
        options = StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            cpu_offload=True,
        )
        cpu_state_dict = get_model_state_dict(model=model, options=options)
        if is_rank_zero:
            return cpu_state_dict
        else:
            return {}
    else:
        cpu_state_dict = {}
        sharded_sd = model.state_dict()
        for param_name, param in sharded_sd.items():
            if param.is_cpu:
                # Move back to device if offloaded to CPU
                param = param.to(device)
            if hasattr(param, "_local_tensor"):
                # Gather DTensor
                param = param.full_tensor()
            if is_rank_zero:
                cpu_state_dict[param_name] = param.cpu()
            dist.barrier()
        return cpu_state_dict


def get_shard_conditions(
    name: str,
    names_to_match: list[str] | None = None,
) -> bool:
    if names_to_match and name in names_to_match:
        return True

    name_list = name.split(".")
    if len(name_list) >= 2:
        return name_list[-2] in ["layers", "blocks"] and str.isdigit(name_list[-1])

    return False


def shard_model(
    model: nn.Module,
    shard_conditions: list[Callable],
    *,
    cpu_offload: bool,
    reshard_after_forward: bool = True,
    dp_mesh: DeviceMesh | None = None,
) -> None:
    fsdp_kwargs = {"reshard_after_forward": reshard_after_forward, "mesh": dp_mesh}
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    # Shard the model with FSDP, iterating in reverse to start with
    # lowest-level modules first
    num_layers_sharded = 0
    for n, m in reversed(list(model.named_modules())):
        if any([shard_condition(n) for shard_condition in shard_conditions]):
            fully_shard(m, **fsdp_kwargs)
            num_layers_sharded += 1

    if num_layers_sharded == 0:
        raise ValueError(
            "No layer modules were sharded. Please check if shard conditions are working as expected."
        )

    # Finally shard the entire model to account for any stragglers
    fully_shard(model, **fsdp_kwargs)

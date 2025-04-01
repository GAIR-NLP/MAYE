import contextlib
from typing import Generator, Iterable

import torch
import torch.distributed as dist
from torch.cuda import nccl

PRECISION_STR_TO_DTYPE: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def verify_bf16_support() -> bool:
    cuda_support = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )
    mps_support = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    return cuda_support or mps_support


def get_dtype(
    dtype: str | None = None, device: torch.device | None = None
) -> torch.dtype:
    # None defaults to float32
    if dtype is None:
        return torch.float32

    # Convert to torch.dtype
    torch_dtype = PRECISION_STR_TO_DTYPE[dtype]

    # dtype must be one of the supported precisions
    if torch_dtype not in PRECISION_STR_TO_DTYPE.values():
        raise ValueError(
            f"Dtype {torch_dtype} must be one of {', '.join(list(PRECISION_STR_TO_DTYPE.keys()))} for finetuning."
        )

    # TODO (rohan-varma): prefer to use get_default_device() here to figure out whether user is training on
    # CPU or GPU, but it is not supported in versions of torch we test.
    if (
        torch_dtype == torch.bfloat16
        and device != torch.device("cpu")
        and not verify_bf16_support()
    ):
        raise RuntimeError(
            "bf16 precision was requested but not available on this hardware. Please use fp32 precision instead."
        )

    return torch_dtype


@contextlib.contextmanager
def set_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def validate_expected_param_dtype(
    named_params: Iterable[tuple[str, torch.nn.Parameter]], dtype: torch.dtype
) -> None:
    for name, param in named_params:
        if param.dtype != dtype:
            raise ValueError(
                f"Parameter {name} has dtype {param.dtype}, but expected {dtype}"
            )

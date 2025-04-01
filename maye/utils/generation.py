import torch


def get_causal_mask_from_padding_mask(
    padding_mask: torch.Tensor, target_seq_len: int | None = None
) -> torch.Tensor:
    bsz, seq_len = padding_mask.shape
    target_seq_len = seq_len if target_seq_len is None else target_seq_len

    if target_seq_len < seq_len:
        raise AssertionError(
            "target_seq_len cannot be shorter than the sequence length of the padding mask."
        )

    mask = torch.tril(
        torch.ones(
            seq_len, target_seq_len, device=padding_mask.device, dtype=torch.bool
        ),
        diagonal=0,
    ).repeat(bsz, 1, 1)
    mask.narrow(2, 0, seq_len).mul_(padding_mask[:, None, :].expand(-1, seq_len, -1))
    mask.diagonal(dim1=1, dim2=2).copy_(torch.Tensor([True]))
    return mask


def get_position_ids_from_padding_mask(padding_mask: torch.Tensor):
    return ((padding_mask.cumsum(-1) - 1) * padding_mask).to(torch.int)

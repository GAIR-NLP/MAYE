import torch
import torch.nn.functional as F

from maye.utils.collate import CROSS_ENTROPY_IGNORE_INDEX


def truncate_sequence_at_first_stop_token(
    sequences: torch.Tensor, stop_tokens: torch.Tensor, fill_value: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    eos_mask = torch.isin(sequences, stop_tokens)
    seq_lens = torch.cumsum(eos_mask, dim=1)
    padding_mask = (seq_lens > 1) | ((seq_lens == 1) & ~eos_mask)
    sequences[padding_mask] = fill_value
    return padding_mask, sequences


def masked_mean(
    x: torch.Tensor, mask: torch.Tensor, dim: int | None = None
) -> torch.Tensor:
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim)


def logits_to_logprobs(
    logits: torch.Tensor, sequences: torch.Tensor, temperature: float = 1.0
):
    return torch.gather(
        input=F.log_softmax(logits / temperature, dim=-1),
        dim=2,
        index=sequences.unsqueeze(-1),
    ).squeeze(-1)


def compute_entropy_from_log_probs(
    log_probs: torch.Tensor, padding_masks: torch.Tensor
):
    probs = log_probs.exp()
    entropy = -probs * log_probs
    entropy_loss = masked_mean(entropy, mask=padding_masks)
    return entropy_loss


def get_batch_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_pad_token_id: int = CROSS_ENTROPY_IGNORE_INDEX,
    return_average_logprobs: bool = False,
):
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            "Logits (batch and sequence length dim) and labels must have the same shape."
        )
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id

    labels[labels == label_pad_token_id] = 0
    # take log-likelihood of the labels given our model
    per_token_log_probs = logits_to_logprobs(logits, labels, temperature=1.0)

    if return_average_logprobs:
        return masked_mean(per_token_log_probs, loss_mask, dim=-1)
    else:
        return (per_token_log_probs * loss_mask).sum(-1)


def truncate_sequence_for_logprobs(
    query_response_logits: torch.Tensor, context_length: int
) -> torch.Tensor:
    return query_response_logits[:, context_length - 1 : -1]


def get_unmasked_sequence_lengths(mask: torch.Tensor) -> torch.Tensor:
    # calculate per-batch-element sequence lengths by finding last valid tokens
    sequence_lengths = (~mask).cumsum(dim=-1).argmax(dim=-1).to(dtype=torch.long)

    return sequence_lengths.clip(0, mask.shape[1] - 1)

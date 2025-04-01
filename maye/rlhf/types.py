from typing import NamedTuple

import torch


class Trajectory(NamedTuple):
    query_responses: torch.Tensor
    logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    masks: torch.Tensor
    position_ids: torch.Tensor
    response_padding_masks: torch.Tensor
    scores: torch.Tensor
    acc_rewards: torch.Tensor
    format_rewards: torch.Tensor
    language_rewards: torch.Tensor
    seq_lens: torch.Tensor


class PPOStats(NamedTuple):
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    entropy: torch.Tensor
    ratios: torch.Tensor
    clipfrac: torch.Tensor
    approx_policy_kls: torch.Tensor

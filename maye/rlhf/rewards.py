import re
from typing import Callable

import torch


def get_reward_penalty_mask(
    sequences: torch.Tensor,
    seq_lens: torch.Tensor,
    stop_tokens: torch.Tensor,
    penalise_no_eos: bool = True,
    min_response_length: int | None = None,
) -> torch.Tensor:
    reward_penalty_mask = torch.zeros_like(seq_lens).to(torch.bool)

    # since sequences will have been truncated at EOS, we can mask based on the presence of any padding tokens
    if penalise_no_eos:
        eos_mask = torch.isin(sequences, stop_tokens)
        reward_penalty_mask = ~eos_mask.any(-1)

    if min_response_length is not None:
        reward_penalty_mask |= ~(seq_lens >= min_response_length)
    return reward_penalty_mask


def get_rewards_from_ref(
    scores: torch.Tensor,
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    kl_coeff: float,
    valid_score_idxs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_ratio = ref_logprobs - logprobs
    ratio = torch.exp(log_ratio)

    kl = (ratio - 1) - log_ratio
    kl = torch.clamp(kl, min=-10, max=10)
    kl_reward = -kl_coeff * kl

    total_reward = kl_reward.clone().float()

    # adding reward to kl at final valid position
    # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L153

    if valid_score_idxs is not None:
        total_reward.scatter_add_(
            1, valid_score_idxs.unsqueeze(-1), scores.unsqueeze(-1)
        )
    else:
        total_reward[:, -1] += scores

    return total_reward, kl, kl_reward


def masked_mean(
    x: torch.Tensor, mask: torch.Tensor, dim: int | None = None
) -> torch.Tensor:
    return (x * mask).sum(dim=dim) / (mask.sum(dim=dim) + 1e-8)


def masked_var(
    centered_values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True
) -> torch.Tensor:
    var = masked_mean(centered_values.pow(2), mask)
    if unbiased:
        mask_sum = mask.sum() + 1e-8
        bessel_correction = mask_sum / (mask_sum - 1)
        var = var * bessel_correction
    return var


def whiten(
    x: torch.Tensor, mask: torch.Tensor | None = None, shift_mean: bool = True
) -> torch.Tensor:
    if mask is not None:
        mean = masked_mean(x, mask)
        var = masked_var(x - mean, mask)
    else:
        mean, var = x.mean(), x.var()
    whitened = (x - mean) * torch.rsqrt(var + 1e-8)
    if shift_mean:
        whitened += mean
    return whitened


def estimate_advantages(
    rewards: torch.Tensor,
    gamma: float,
    masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, response_length = rewards.shape
    returns = torch.zeros_like(rewards)

    running_return = torch.zeros(batch_size, dtype=rewards.dtype, device=rewards.device)

    # estimate advantage for every predicted token position
    for t in reversed(range(response_length)):
        running_return = rewards[:, t] + gamma * running_return
        returns[:, t] = running_return

    advantages = returns

    # normalize advantages across the batch of trajectories to reduce variance
    advantages = whiten(advantages, mask=masks)
    if masks is not None:
        advantages[~masks] = 0.0

    return advantages, returns


def accuracy_reward_fn(
    predictions: list[str],
    solutions: list[str],
    judge_fn: Callable[[str, str], bool],
) -> torch.Tensor:
    rewards = [0.0] * len(predictions)
    for idx, (prediction, solution) in enumerate(zip(predictions, solutions)):
        answer_correct_flag = judge_fn(prediction, solution)
        if answer_correct_flag:
            rewards[idx] += 1.0
    return torch.tensor(rewards)


def format_reward_fn(texts: list[str]):
    rewards = [0.0] * len(texts)
    for idx, text in enumerate(texts):
        matches = re.findall(r"<think>.*?</think>", text, re.DOTALL)
        if len(matches) == 1:
            rewards[idx] += 1.0
    return torch.tensor(rewards)


def language_reward_fn(texts: list[str]):
    rewards = [0.0] * len(texts)
    for idx, text in enumerate(texts):
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        if len(text) > 0:
            proportion = len(chinese_chars) / len(text)
        else:
            proportion = 0
        rewards[idx] = -0.1 * proportion
    return torch.tensor(rewards)

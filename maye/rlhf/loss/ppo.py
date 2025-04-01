import torch
import torch.nn as nn

from maye import rlhf


class PPOLoss(nn.Module):
    def __init__(
        self,
        epsilon_high: float = 0.2,
        epsilon_low: float = 0.2,
        kl_loss_coeff: float = 0.01,
    ):
        super().__init__()
        self.epsilon_high = epsilon_high
        self.epsilon_low = epsilon_low
        self.kl_loss_coeff = kl_loss_coeff

    def forward(
        self,
        pi_old_logprobs: torch.Tensor,
        pi_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        padding_masks: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        ratios = torch.exp(pi_logprobs - pi_old_logprobs)
        clipped_ratios = torch.clamp(
            ratios, 1.0 - self.epsilon_low, 1.0 + self.epsilon_high
        )

        policy_losses_clipped = -advantages * clipped_ratios
        policy_losses_unclipped = -advantages * ratios

        clipfrac = (policy_losses_clipped > policy_losses_unclipped).to(
            pi_logprobs.dtype
        )
        clipfrac = (
            clipfrac.mean()
            if padding_masks is None
            else rlhf.masked_mean(clipfrac, padding_masks)
        )

        policy_loss = torch.maximum(policy_losses_clipped, policy_losses_unclipped)
        policy_loss = (
            policy_loss.mean()
            if padding_masks is None
            else rlhf.masked_mean(policy_loss, padding_masks)
        )

        kl_loss = (
            torch.exp(ref_logprobs - pi_logprobs) - (ref_logprobs - pi_logprobs) - 1
        )
        kl_loss = rlhf.masked_mean(kl_loss, padding_masks)
        kl_loss = torch.clamp(kl_loss, min=-10, max=10)

        loss = policy_loss + self.kl_loss_coeff * kl_loss
        entropy = rlhf.compute_entropy_from_log_probs(pi_logprobs, padding_masks)

        return (
            loss,
            policy_loss.detach(),
            kl_loss.detach(),
            entropy.detach(),
            ratios.mean().detach(),
            clipfrac.detach(),
        )

"""RLCD: reinforcement learning for calibrated decisions.

TypeSafe names this method for Jev but does not publish it. This module
follows the recipe Laya describes and ships in its fine-tuning notebook:

1. Treat the logits as the mean of a Gaussian policy. Sample ``group_size``
   noisy logit vectors per question (zero-mean noise, projected so it sums to
   zero over the real options, which leaves the softmax "level" unchanged).
2. Softmax each sample into a reported distribution ``q``.
3. Reward ``q`` with a strictly proper scoring rule against the observed
   outcome (one-hot or a soft annotator distribution): log score plus
   spherical score, minus the ranked probability score for ordinal questions.
4. Advantage = reward minus the group mean (GRPO-style baseline), normalised.
5. REINFORCE: maximise ``advantage * log N(z | logits, sigma^2)``.

A soft cross-entropy term is added as supervised guidance, as Laya does; on
its own, cross-entropy against soft targets is itself the log scoring rule.
Because every term is a proper scoring rule, the expected reward is
maximised by reporting the true outcome distribution, which is what pushes
the model towards calibrated probabilities rather than just a correct argmax.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import MASKED_LOGIT
from .serialization import QUESTION_TYPES

LOG_FLOOR = -9.21  # log(1e-4): bounds the log score so one miss cannot dominate a batch


def proper_scoring_reward(
    probabilities: Tensor,
    target: Tensor,
    question_types: Tensor,
    option_mask: Tensor,
    spherical_weight: float = 0.5,
    rps_weight: float = 1.0,
) -> Tensor:
    """Reward for reported ``probabilities`` ``[..., batch, options]`` given ``target``.

    Higher is better. Log and spherical scores apply to every question; the
    ranked probability score (squared CDF distance, a proper rule that also
    respects ordering) is subtracted for score questions.
    """
    mask = option_mask.to(probabilities.dtype)
    probabilities = probabilities * mask
    target = target * mask
    log_score = (target * probabilities.clamp_min(1e-12).log().clamp_min(LOG_FLOOR)).sum(-1)
    spherical = (target * probabilities).sum(-1) / probabilities.norm(dim=-1).clamp_min(1e-9)
    reward = log_score + spherical_weight * spherical

    is_score = (question_types == QUESTION_TYPES["score"]).to(probabilities.dtype)
    if rps_weight and is_score.any():
        levels = option_mask.sum(-1).clamp_min(2).to(probabilities.dtype)
        cdf_gap = probabilities.cumsum(-1) - target.cumsum(-1)
        rps = (cdf_gap.square() * mask).sum(-1) / (levels - 1)
        reward = reward - rps_weight * rps * is_score
    return reward


def soft_cross_entropy(logits: Tensor, target: Tensor, option_mask: Tensor) -> Tensor:
    """Mean cross-entropy against a (possibly soft) target distribution."""
    log_probs = F.log_softmax(logits.masked_fill(~option_mask, MASKED_LOGIT), dim=-1)
    return -(target * log_probs * option_mask).sum(-1).mean()


@dataclass
class RLCDConfig:
    group_size: int = 4
    sigma: float = 0.3
    ce_weight: float = 1.0
    spherical_weight: float = 0.75
    rps_weight: float = 1.0


def rlcd_loss(
    logits: Tensor,
    target: Tensor,
    question_types: Tensor,
    option_mask: Tensor,
    config: RLCDConfig = RLCDConfig(),
    sigma: float | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Policy-gradient loss with a group-mean baseline, plus soft cross-entropy.

    ``sigma`` overrides ``config.sigma`` so callers can anneal exploration.
    Returns the loss and scalar diagnostics for logging.
    """
    sigma = config.sigma if sigma is None else sigma
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    mask = option_mask.to(logits.dtype)
    option_count = mask.sum(-1, keepdim=True)

    noise = torch.randn(
        (config.group_size, *logits.shape),
        device=logits.device,
        dtype=logits.dtype,
        generator=generator,
    ) * sigma * mask
    noise = (noise - noise.sum(-1, keepdim=True) / option_count) * mask
    samples = logits.detach().unsqueeze(0) + noise
    reported = F.softmax(samples.masked_fill(~option_mask, MASKED_LOGIT), dim=-1)

    with torch.no_grad():
        reward = proper_scoring_reward(
            reported,
            target.unsqueeze(0),
            question_types,
            option_mask,
            config.spherical_weight,
            config.rps_weight,
        )
        advantage = reward - reward.mean(0, keepdim=True)
        advantage = advantage / (advantage.std() + 1e-6)

    log_policy = -((samples - logits.unsqueeze(0)).square() * mask).sum(-1) / (2 * sigma**2)
    policy_loss = -(advantage * log_policy).mean()
    ce_loss = soft_cross_entropy(logits, target, option_mask)
    loss = policy_loss + config.ce_weight * ce_loss

    return loss, {
        "loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "cross_entropy": float(ce_loss.detach()),
        "mean_reward": float(reward.mean()),
    }

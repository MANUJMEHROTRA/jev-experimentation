"""Temperature scaling and decision metrics.

Calibration is fitted on a held-out slice that never entered training: on
training items the model is near-certain and near-correct, so a temperature
fitted there measures fit, not calibration. Laya fits one temperature per
question type (and optionally per option count); this module does per type.
"""

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import MASKED_LOGIT
from .serialization import QUESTION_TYPES

TEMPERATURE_MIN = 0.5
TEMPERATURE_MAX = 5.0


def fit_temperature(logits: Tensor, target: Tensor, option_mask: Tensor) -> float:
    """Single temperature minimising cross-entropy of ``softmax(logits / T)`` to ``target``."""
    logits = logits.detach().float().cpu().masked_fill(~option_mask.cpu(), MASKED_LOGIT)
    target = target.detach().float().cpu()
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        log_probs = F.log_softmax(logits / log_temperature.exp(), dim=-1)
        loss = -(target * log_probs).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(TEMPERATURE_MIN, TEMPERATURE_MAX))


def fit_temperatures_by_type(
    logits: Tensor,
    target: Tensor,
    option_mask: Tensor,
    question_types: Tensor,
    min_examples: int = 10,
) -> Tensor:
    """One temperature per question type; types with too few examples keep 1.0."""
    temperatures = torch.ones(len(QUESTION_TYPES))
    for type_index in range(len(QUESTION_TYPES)):
        selected = question_types.cpu() == type_index
        if int(selected.sum()) >= min_examples:
            temperatures[type_index] = fit_temperature(
                logits[selected], target[selected], option_mask[selected]
            )
    return temperatures


def expected_calibration_error(confidence: Tensor, correct: Tensor, bins: int = 15) -> float:
    """Weighted mean |accuracy - confidence| over equal-width confidence bins."""
    if confidence.numel() == 0:
        return float("nan")
    confidence = confidence.float().cpu()
    correct = correct.float().cpu()
    edges = torch.linspace(0, 1, bins + 1)
    error = 0.0
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        lower_ok = confidence >= low if index == 0 else confidence > low
        selected = lower_ok & (confidence <= high)
        if selected.any():
            gap = (confidence[selected].mean() - correct[selected].mean()).abs()
            error += float(selected.float().mean() * gap)
    return error


def reliability_bins(confidence: Tensor, correct: Tensor, bins: int = 10) -> list[dict]:
    """Per-bin mean confidence, accuracy, and count, for reliability diagrams."""
    confidence = confidence.float().cpu()
    correct = correct.float().cpu()
    edges = torch.linspace(0, 1, bins + 1)
    rows = []
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        lower_ok = confidence >= low if index == 0 else confidence > low
        selected = lower_ok & (confidence <= high)
        count = int(selected.sum())
        rows.append({
            "low": float(low),
            "high": float(high),
            "count": count,
            "confidence": float(confidence[selected].mean()) if count else None,
            "accuracy": float(correct[selected].mean()) if count else None,
        })
    return rows


def decision_metrics(
    probabilities: Tensor,
    target: Tensor,
    option_mask: Tensor,
    question_types: Tensor,
    gold_index: Tensor,
) -> dict[str, object]:
    """Accuracy, soft accuracy, Brier, and ECE overall and per question type.

    ``gold_index`` is the gold label's option index. ``target`` is the gold
    distribution (soft when annotators disagree). Soft accuracy is the
    probability that a draw from the model agrees with a draw from the
    annotators; Brier is squared distance to the gold distribution.
    """
    mask = option_mask.float()
    probabilities = probabilities.float() * mask
    target = target.float() * mask
    confidence, predicted = probabilities.max(-1)
    correct = (predicted == gold_index).float()

    def summarise(selected: Tensor) -> dict[str, float]:
        if not selected.any():
            return {"n": 0}
        return {
            "n": int(selected.sum()),
            "accuracy": float(correct[selected].mean()),
            "soft_accuracy": float((probabilities * target).sum(-1)[selected].mean()),
            "brier": float((probabilities - target).square().sum(-1)[selected].mean()),
            "mean_confidence": float(confidence[selected].mean()),
            "ece": expected_calibration_error(confidence[selected], correct[selected]),
        }

    everything = torch.ones_like(question_types, dtype=torch.bool)
    return {
        "overall": summarise(everything),
        **{
            name: summarise(question_types == index)
            for name, index in QUESTION_TYPES.items()
        },
        "reliability": reliability_bins(confidence, correct),
    }

"""Batching and inference helpers shared by the notebook and evaluation code."""

import random
from typing import Iterator

import torch
from torch import Tensor

from .data import DecisionItem, collate, model_inputs
from .model import MASKED_LOGIT, MarkerDecisionModel


def length_bucketed_batches(
    items: list[DecisionItem], batch_size: int, shuffle: bool = True, seed: int = 0
) -> Iterator[list[DecisionItem]]:
    """Group similar-length items so padding stays small, then shuffle the batches."""
    order = sorted(range(len(items)), key=lambda index: len(items[index].input_ids))
    batches = [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
    if shuffle:
        random.Random(seed).shuffle(batches)
    for batch in batches:
        yield [items[index] for index in batch]


@torch.no_grad()
def collect_logits(
    model: MarkerDecisionModel,
    items: list[DecisionItem],
    pad_token_id: int,
    device: torch.device | str,
    batch_size: int = 32,
    pad_to_multiple_of: int = 1,
) -> dict[str, Tensor]:
    """Raw logits for every item, padded to the split's largest option count.

    Returned tensors are on CPU and in ``items`` order, alongside the targets,
    masks, types, and gold indices needed for calibration and metrics.
    """
    if not items:
        raise ValueError("items must not be empty")
    model.eval()
    option_count = max(len(item.markers) for item in items)
    logits = torch.full((len(items), option_count), MASKED_LOGIT)
    target = torch.zeros((len(items), option_count))
    option_mask = torch.zeros((len(items), option_count), dtype=torch.bool)
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        batch = collate(chunk, pad_token_id, pad_to_multiple_of)
        output = model(**model_inputs(batch, device))
        logits[start:start + len(chunk), : output.logits.shape[1]] = output.logits.float().cpu()
    for row, item in enumerate(items):
        target[row, : len(item.target)] = torch.tensor(item.target)
        option_mask[row, : len(item.markers)] = True

    return {
        "logits": logits,
        "target": target,
        "option_mask": option_mask,
        "question_types": torch.tensor([item.question_type for item in items]),
        "gold_index": torch.tensor([item.gold_index for item in items]),
    }


def apply_temperatures(logits: Tensor, option_mask: Tensor, question_types: Tensor,
                       temperatures: Tensor) -> Tensor:
    """Softmax ``logits`` with each row divided by its question type's temperature."""
    scaled = logits / temperatures[question_types].unsqueeze(1)
    return scaled.masked_fill(~option_mask, MASKED_LOGIT).softmax(-1)

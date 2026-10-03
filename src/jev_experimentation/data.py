"""Load typed-decision data and batch it for :class:`MarkerDecisionModel`.

The default dataset is ``LocalLLaMA/typed-decisions`` on Hugging Face, the
benchmark Laya fine-tunes and reports against Jev on. Each row is one case
(a JSON state from one of four workflows) with five typed questions and a
gold answer per question that includes a soft annotator distribution.
"""

import json
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch

from .serialization import encode_question, option_keys, serialize_state

DATASET_ID = "LocalLLaMA/typed-decisions"


@dataclass
class DecisionItem:
    """One (case, question) pair ready for the model."""

    case_id: str
    workflow: str
    question_id: str
    question_type: int
    keys: list[str]
    input_ids: list[int]
    markers: list[int]
    target: list[float]
    gold_index: int


def gold_distribution(
    question: Mapping[str, Any], gold: Mapping[str, Any]
) -> tuple[list[str], list[float], int]:
    """Option keys, normalised gold distribution, and gold label index.

    Falls back to a one-hot on the label when the gold has no probabilities,
    and to the distribution's argmax when the label is missing.
    """
    keys = option_keys(question)
    probabilities = gold.get("probabilities") or {}
    target = [float(probabilities.get(key, 0.0)) for key in keys]
    label = gold.get("label")
    if label is not None:
        label = str(label).lower() if question["type"] == "noul" else str(label)
    if sum(target) <= 0:
        if label not in keys:
            raise ValueError("gold has neither probabilities nor a known label")
        target = [float(key == label) for key in keys]
    total = sum(target)
    target = [value / total for value in target]
    gold_index = keys.index(label) if label in keys else max(range(len(keys)), key=target.__getitem__)
    return keys, target, gold_index


def _parse(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def build_items(
    rows: Iterable[Mapping[str, Any]],
    tokenizer,
    max_len: int = 512,
    head_max_len: int = 192,
) -> list[DecisionItem]:
    """Expand dataset rows (one per case) into one item per answered question."""
    items = []
    for row in rows:
        state = _parse(row["state"])
        questions = _parse(row["questions"])
        gold = _parse(row["gold"])
        state_ids = tokenizer(serialize_state(state), add_special_tokens=False)["input_ids"]
        for question_id, question in questions.items():
            if question_id not in gold:
                continue
            keys, target, gold_index = gold_distribution(question, gold[question_id])
            encoded = encode_question(
                tokenizer, state, question, max_len, head_max_len, state_ids=list(state_ids)
            )
            items.append(DecisionItem(
                case_id=str(row["id"]),
                workflow=str(row.get("workflow", "")),
                question_id=question_id,
                question_type=encoded.question_type,
                keys=keys,
                input_ids=encoded.input_ids,
                markers=encoded.markers,
                target=target,
                gold_index=gold_index,
            ))
    return items


def load_typed_decisions(
    split: str, config: str = "all", limit: int | None = None, seed: int = 0
):
    """Load a split of the Hugging Face dataset as a list of row dicts.

    Rows are grouped by workflow, so ``limit`` takes a seeded random sample
    rather than the first rows, which would all come from one workflow.
    """
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, config, split=split)
    if limit is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(limit, len(dataset))))
    return list(dataset)


def split_by_case(
    items: list[DecisionItem], holdout_fraction: float, seed: int = 0
) -> tuple[list[DecisionItem], list[DecisionItem]]:
    """Split items so all questions about one case land on the same side.

    Splitting individual questions would leak a case's state into both sides.
    """
    case_ids = sorted({item.case_id for item in items})
    random.Random(seed).shuffle(case_ids)
    holdout = set(case_ids[: round(len(case_ids) * holdout_fraction)])
    kept = [item for item in items if item.case_id not in holdout]
    held_out = [item for item in items if item.case_id in holdout]
    return kept, held_out


def collate(
    items: list[DecisionItem], pad_token_id: int, pad_to_multiple_of: int = 1
) -> dict[str, torch.Tensor]:
    """Right-pad sequences and option lists into model-ready tensors.

    ``pad_to_multiple_of`` rounds the sequence length up so batches fall into a
    few shapes, which keeps per-shape kernel and allocator caches (notably on
    Apple MPS) from growing with every distinct length.
    """
    batch_size = len(items)
    sequence_length = max(len(item.input_ids) for item in items)
    sequence_length = -(-sequence_length // pad_to_multiple_of) * pad_to_multiple_of
    option_count = max(len(item.markers) for item in items)

    input_ids = torch.full((batch_size, sequence_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, sequence_length), dtype=torch.long)
    marker_positions = torch.zeros((batch_size, option_count), dtype=torch.long)
    marker_mask = torch.zeros((batch_size, option_count), dtype=torch.bool)
    target = torch.zeros((batch_size, option_count))
    for row, item in enumerate(items):
        length, options = len(item.input_ids), len(item.markers)
        input_ids[row, :length] = torch.tensor(item.input_ids)
        attention_mask[row, :length] = 1
        marker_positions[row, :options] = torch.tensor(item.markers)
        marker_mask[row, :options] = True
        target[row, :options] = torch.tensor(item.target)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_positions": marker_positions,
        "marker_mask": marker_mask,
        "question_types": torch.tensor([item.question_type for item in items]),
        "target": target,
        "gold_index": torch.tensor([item.gold_index for item in items]),
    }


MODEL_INPUTS = ("input_ids", "attention_mask", "marker_positions", "marker_mask", "question_types")


def model_inputs(batch: Mapping[str, torch.Tensor], device: torch.device | str) -> dict:
    """The subset of a collated batch that :class:`MarkerDecisionModel` takes, on ``device``."""
    return {name: batch[name].to(device) for name in MODEL_INPUTS}

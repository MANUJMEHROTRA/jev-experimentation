"""Typed decision models: a cross-encoder baseline and a Laya-style marker model."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel

from .serialization import QUESTION_TYPES

MASKED_LOGIT = -1e4


QuestionType = Literal["choice", "score", "noul"]


@dataclass
class DecisionOutput:
    """Batched scores over each question's padded candidate options."""

    logits: Tensor
    probabilities: Tensor
    option_mask: Tensor


class TypedDecisionModel(nn.Module):
    """Cross-encoder baseline: one encoder pass per (state, question, option).

    Inputs have shape ``[batch, options, sequence]``. Each candidate sequence
    should contain the state, question instructions, and that option's text.
    ``option_mask`` marks real options and excludes batch padding.
    """

    def __init__(self, encoder: nn.Module, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")

        self.encoder = encoder
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    @classmethod
    def from_pretrained(cls, model_name: str, dropout: float = 0.1):
        """Load a Hugging Face encoder and attach the decision scorer."""
        encoder = AutoModel.from_pretrained(model_name)
        return cls(encoder, encoder.config.hidden_size, dropout=dropout)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        option_mask: Tensor,
    ) -> DecisionOutput:
        if input_ids.ndim != 3:
            raise ValueError("input_ids must have shape [batch, options, sequence]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        if option_mask.shape != input_ids.shape[:2]:
            raise ValueError("option_mask must have shape [batch, options]")
        if not option_mask.any(dim=1).all():
            raise ValueError("each example must contain at least one real option")

        batch_size, option_count, sequence_length = input_ids.shape
        flat_input_ids = input_ids.reshape(batch_size * option_count, sequence_length)
        flat_attention_mask = attention_mask.reshape(
            batch_size * option_count, sequence_length
        )
        encoded = self.encoder(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
        )
        token_states = encoded.last_hidden_state
        candidate_states = token_states[:, 0]
        logits = self.scorer(candidate_states).reshape(batch_size, option_count)
        logits = logits.masked_fill(~option_mask.bool(), torch.finfo(logits.dtype).min)
        probabilities = F.softmax(logits, dim=-1)

        return DecisionOutput(logits, probabilities, option_mask.bool())


def _attention_heads(hidden_size: int) -> int:
    heads = max(1, hidden_size // 64)
    while hidden_size % heads:
        heads -= 1
    return heads


class MarkerDecisionModel(nn.Module):
    """Laya-style System 1 model: all options of a question scored in one pass.

    Each sequence holds a question head with one ``[MASK]`` marker per option
    followed by the state (see :mod:`jev_experimentation.serialization`).

    1. A pretrained bidirectional encoder contextualises every token, so each
       marker sees the instructions, its own option text, the other options,
       and the state.
    2. A learned question-type embedding is added to every token.
    3. A small transformer head (``head_layers`` pre-norm layers) refines the
       states for the decision task.
    4. The hidden state at each marker is scored by an MLP to one logit.
    5. Logits are softmaxed over that question's real options, after dividing
       by a per-type temperature fitted on held-out data.

    Inputs: ``input_ids``/``attention_mask`` ``[batch, seq]``,
    ``marker_positions``/``marker_mask`` ``[batch, options]`` and
    ``question_types`` ``[batch]`` (0 choice, 1 score, 2 noul).
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        head_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")

        self.encoder = encoder
        self.head_layers = head_layers
        self.dropout = dropout
        self.type_embedding = nn.Embedding(len(QUESTION_TYPES), hidden_size)
        if head_layers > 0:
            layer = nn.TransformerEncoderLayer(
                hidden_size,
                _attention_heads(hidden_size),
                4 * hidden_size,
                dropout,
                batch_first=True,
                norm_first=True,
            )
            self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
        else:
            self.head = None
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.register_buffer("temperature", torch.ones(len(QUESTION_TYPES)))

    @classmethod
    def from_pretrained(cls, model_name: str, head_layers: int = 2, dropout: float = 0.1):
        """Load a Hugging Face encoder and attach a fresh decision head."""
        encoder = AutoModel.from_pretrained(model_name)
        return cls(encoder, encoder.config.hidden_size, head_layers, dropout)

    def save_pretrained(self, directory: str | Path) -> None:
        """Save the encoder in Hugging Face format and the head as a state dict."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path / "encoder")
        head_state = {
            name: value for name, value in self.state_dict().items()
            if not name.startswith("encoder.")
        }
        torch.save(head_state, path / "decision_head.pt")
        config = {"head_layers": self.head_layers, "dropout": self.dropout}
        (path / "decision_config.json").write_text(json.dumps(config, indent=2))

    @classmethod
    def load_pretrained(cls, directory: str | Path, map_location: str = "cpu"):
        """Inverse of :meth:`save_pretrained`, including fitted temperatures."""
        path = Path(directory)
        config = json.loads((path / "decision_config.json").read_text())
        encoder = AutoModel.from_pretrained(path / "encoder")
        model = cls(encoder, encoder.config.hidden_size, **config)
        head_state = torch.load(path / "decision_head.pt", map_location=map_location)
        result = model.load_state_dict(head_state, strict=False)
        missing = [key for key in result.missing_keys if not key.startswith("encoder.")]
        if missing or result.unexpected_keys:
            raise ValueError(
                f"decision head mismatch: missing {missing}, unexpected {result.unexpected_keys}"
            )
        return model

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        marker_positions: Tensor,
        marker_mask: Tensor,
        question_types: Tensor,
    ) -> DecisionOutput:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [batch, sequence]")
        if marker_positions.shape != marker_mask.shape or marker_mask.ndim != 2:
            raise ValueError("marker_positions and marker_mask must have shape [batch, options]")
        if question_types.shape != input_ids.shape[:1]:
            raise ValueError("question_types must have shape [batch]")
        if not marker_mask.any(dim=1).all():
            raise ValueError("each example must contain at least one real option")

        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden + self.type_embedding(question_types)[:, None, :]
        if self.head is not None:
            hidden = self.head(hidden, src_key_padding_mask=~attention_mask.bool())

        index = marker_positions.clamp_min(0)[:, :, None].expand(-1, -1, hidden.size(-1))
        marker_states = torch.gather(hidden, 1, index)
        logits = self.scorer(marker_states).squeeze(-1).float()
        option_mask = marker_mask.bool()
        logits = logits.masked_fill(~option_mask, MASKED_LOGIT)

        temperature = self.temperature[question_types].unsqueeze(1)
        probabilities = F.softmax(logits / temperature, dim=-1)
        return DecisionOutput(logits, probabilities, option_mask)


def decision_loss(
    output: DecisionOutput,
    target_indices: Tensor,
    example_mask: Tensor | None = None,
) -> Tensor:
    """Cross-entropy baseline for one gold candidate per example."""
    if target_indices.shape != output.logits.shape[:1]:
        raise ValueError("target_indices must have shape [batch]")
    if not output.option_mask.gather(1, target_indices.long().unsqueeze(1)).all():
        raise ValueError("each target must refer to a real option")

    losses = F.cross_entropy(output.logits, target_indices.long(), reduction="none")
    if example_mask is None:
        return losses.mean()
    if example_mask.shape != target_indices.shape:
        raise ValueError("example_mask must have shape [batch]")
    weights = example_mask.to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1)


def decode_decisions(
    output: DecisionOutput,
    question_types: Sequence[QuestionType],
    option_labels: Sequence[Sequence[str]],
    ordinal_values: Sequence[Sequence[float]] | None = None,
) -> list[dict[str, object]]:
    """Convert candidate distributions into the three typed answer forms."""
    batch_size = output.probabilities.shape[0]
    if len(question_types) != batch_size or len(option_labels) != batch_size:
        raise ValueError("question_types and option_labels must match batch size")
    if ordinal_values is not None and len(ordinal_values) != batch_size:
        raise ValueError("ordinal_values must match batch size")

    results = []
    for row, question_type in enumerate(question_types):
        option_count = int(output.option_mask[row].sum().item())
        labels = list(option_labels[row])
        if len(labels) != option_count:
            raise ValueError("each label list must match its number of real options")
        probabilities = output.probabilities[row, :option_count]

        if question_type == "choice":
            best_index = int(probabilities.argmax().item())
            results.append({
                "type": "choice",
                "choice": labels[best_index],
                "probabilities": dict(zip(labels, probabilities.tolist())),
                "confidence": float(probabilities.max().item()),
            })
        elif question_type == "score":
            if ordinal_values is None or len(ordinal_values[row]) != option_count:
                raise ValueError("score questions require one ordinal value per option")
            values = torch.tensor(
                ordinal_values[row],
                device=probabilities.device,
                dtype=probabilities.dtype,
            )
            results.append({
                "type": "score",
                "score": float((probabilities * values).sum().item()),
                "probabilities": dict(zip(labels, probabilities.tolist())),
            })
        elif question_type == "noul":
            if option_count != 2:
                raise ValueError("noul questions require [false, true] options")
            results.append({"type": "noul", "noul": float(probabilities[1].item())})
        else:
            raise ValueError(f"unsupported question type: {question_type}")

    return results
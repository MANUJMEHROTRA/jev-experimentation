"""A Jev/Laya-shaped ``predict(state, questions)`` interface over the marker model."""

import time
from typing import Any, Mapping

import torch

from .data import DecisionItem, collate, model_inputs
from .model import MarkerDecisionModel, decode_decisions
from .serialization import QUESTION_TYPE_NAMES, encode_question, serialize_state


class DecisionAgent:
    """Answer every typed question about one state in a single batched forward pass.

    ``questions`` uses the schema shared by Jev and Laya::

        {"department": {"type": "choice", "instructions": "...",
                        "criteria": {"billing": "...", "technical": "..."}},
         "urgency": {"type": "score", "instructions": "...",
                     "criteria": ["not urgent", "soon", "blocking"]},
         "churn_risk": {"type": "noul", "instructions": "..."}}
    """

    def __init__(
        self,
        model: MarkerDecisionModel,
        tokenizer,
        max_len: int = 512,
        head_max_len: int = 192,
        device: torch.device | str | None = None,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.head_max_len = head_max_len
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)

    @torch.no_grad()
    def predict(
        self, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Any]:
        if not questions:
            raise ValueError("questions must not be empty")
        started = time.perf_counter()
        state_ids = self.tokenizer(serialize_state(state), add_special_tokens=False)["input_ids"]
        items, dropped = [], 0
        for question_id, question in questions.items():
            encoded = encode_question(
                self.tokenizer, state, question, self.max_len, self.head_max_len,
                state_ids=list(state_ids),
            )
            dropped = max(dropped, encoded.state_tokens_dropped)
            items.append(DecisionItem(
                case_id="request",
                workflow="",
                question_id=question_id,
                question_type=encoded.question_type,
                keys=encoded.keys,
                input_ids=encoded.input_ids,
                markers=encoded.markers,
                target=[0.0] * len(encoded.keys),
                gold_index=0,
            ))

        batch = collate(items, self.tokenizer.pad_token_id)
        output = self.model(**model_inputs(batch, self.device))
        output.probabilities = output.probabilities.cpu()
        output.option_mask = output.option_mask.cpu()
        decoded = decode_decisions(
            output,
            [QUESTION_TYPE_NAMES[item.question_type] for item in items],
            [item.keys for item in items],
            ordinal_values=[list(range(len(item.keys))) for item in items],
        )
        answers = {}
        for item, answer in zip(items, decoded):
            if answer["type"] == "noul":
                answer["confidence"] = max(answer["noul"], 1 - answer["noul"])
            elif answer["type"] == "score":
                answer["confidence"] = max(answer["probabilities"].values())
            answers[item.question_id] = answer

        return {
            "answers": answers,
            "usage": {
                "input_tokens": int(batch["attention_mask"].sum()),
                "output_tokens": 0,
                "state_tokens_dropped": dropped,
            },
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

"""Turn a state plus one typed question into a single encoder sequence.

The layout follows the one in Laya's released ``laya/common.py``
(Apache-2.0, https://github.com/NandhaKishorM/laya)::

    [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]

Every option starts with its own ``[MASK]`` token. The model reads the
contextualised hidden state at each marker and scores it, so the answer space
is whatever options the caller sends, not a fixed classifier vocabulary. The
question "head" goes first so the state, which can be arbitrarily long, is the
part that gets truncated.

This is an independent reimplementation for study; it is token-layout
compatible in spirit but not guaranteed byte-identical to Laya's tokenisation.
"""

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

QUESTION_TYPES = {"choice": 0, "score": 1, "noul": 2}
QUESTION_TYPE_NAMES = {index: name for name, index in QUESTION_TYPES.items()}

OPTION_TOKEN_CAP = 48
MIN_INSTRUCTION_TOKENS = 8
MIN_OPTION_TOKENS = 4


@dataclass
class EncodedQuestion:
    """One question rendered as token ids, with the position of each option marker."""

    input_ids: list[int]
    markers: list[int]
    question_type: int
    keys: list[str]
    state_tokens_dropped: int = 0


def serialize_state(state: str | Mapping[str, Any] | Sequence[Any]) -> str:
    """Strings pass through; structured state becomes compact JSON."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def option_keys(question: Mapping[str, Any]) -> list[str]:
    """Answer keys in marker order: choice keys, score levels "0".."n-1", or noul false/true."""
    question_type = question["type"]
    criteria = question.get("criteria")
    if question_type == "choice":
        if not criteria:
            raise ValueError("choice questions need a non-empty criteria mapping")
        return [str(key) for key in criteria]
    if question_type == "score":
        if not criteria:
            raise ValueError("score questions need a non-empty criteria list")
        return [str(level) for level in range(len(criteria))]
    if question_type == "noul":
        return ["false", "true"]
    raise ValueError(f"unsupported question type: {question_type}")


def render_options(question: Mapping[str, Any]) -> list[str]:
    """Option text in marker order. Noul is always ordered [false, true]."""
    question_type = question["type"]
    criteria = question.get("criteria")
    if question_type == "choice":
        return [
            str(key) if description in (None, "") else f"{key}: {_render(description)}"
            for key, description in criteria.items()
        ]
    if question_type == "score":
        return [f"level {level}: {_render(text)}" for level, text in enumerate(criteria)]
    if question_type == "noul":
        criteria = criteria or {}
        false_text = criteria.get("false") or "no, the statement does not hold"
        true_text = criteria.get("true") or "yes, the statement holds"
        return [f"false: {_render(false_text)}", f"true: {_render(true_text)}"]
    raise ValueError(f"unsupported question type: {question_type}")


def _tokens(tokenizer, text: str, max_length: int | None = None) -> list[int]:
    text = text.replace(tokenizer.mask_token, " ")
    if max_length is None:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])
    return list(
        tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)[
            "input_ids"
        ]
    )


def encode_question(
    tokenizer,
    state: str | Mapping[str, Any] | Sequence[Any],
    question: Mapping[str, Any],
    max_len: int = 512,
    head_max_len: int = 192,
    state_ids: list[int] | None = None,
) -> EncodedQuestion:
    """Build ``[CLS] head [SEP] [MASK] opt ... [SEP] state [SEP]`` for one question.

    The head (instructions plus options) is capped at ``head_max_len`` tokens.
    When the options alone would not fit, every option is cut to an equal share
    of the budget, which is where large option sets start to lose information.
    The state fills whatever room is left inside ``max_len``. Pass ``state_ids``
    to tokenise a shared state once across several questions.
    """
    if head_max_len >= max_len:
        raise ValueError("head_max_len must leave room for the state inside max_len")

    question_type = question["type"]
    keys = option_keys(question)
    options = render_options(question)
    instruction_ids = _tokens(tokenizer, f"{question_type} question: {question['instructions']}")
    option_ids = [
        [tokenizer.mask_token_id] + _tokens(tokenizer, " " + text, OPTION_TOKEN_CAP)
        for text in options
    ]

    option_budget = head_max_len - sum(len(ids) for ids in option_ids)
    if option_budget < 16:
        per_option = max(MIN_OPTION_TOKENS, (head_max_len - 16) // len(option_ids))
        option_ids = [ids[:per_option] for ids in option_ids]
        option_budget = head_max_len - sum(len(ids) for ids in option_ids)
    instruction_ids = instruction_ids[: max(MIN_INSTRUCTION_TOKENS, option_budget)]

    input_ids = [tokenizer.cls_token_id] + instruction_ids + [tokenizer.sep_token_id]
    markers = []
    for ids in option_ids:
        markers.append(len(input_ids))
        input_ids.extend(ids)
    input_ids.append(tokenizer.sep_token_id)

    if state_ids is None:
        state_ids = _tokens(tokenizer, serialize_state(state))
    room = max(0, max_len - len(input_ids) - 1)
    kept_state = state_ids[:room]
    input_ids = input_ids + kept_state + [tokenizer.sep_token_id]

    if len(input_ids) > max_len or any(marker >= max_len for marker in markers):
        raise ValueError("question head does not fit inside max_len; raise max_len")

    return EncodedQuestion(
        input_ids=input_ids,
        markers=markers,
        question_type=QUESTION_TYPES[question_type],
        keys=keys,
        state_tokens_dropped=len(state_ids) - len(kept_state),
    )

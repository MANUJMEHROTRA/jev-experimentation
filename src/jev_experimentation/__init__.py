"""Experiments with typed (System 1) decision models inspired by Laya and Jev."""

from .agent import DecisionAgent
from .calibration import (
    decision_metrics,
    expected_calibration_error,
    fit_temperature,
    fit_temperatures_by_type,
)
from .data import (
    DATASET_ID,
    DecisionItem,
    build_items,
    collate,
    gold_distribution,
    load_typed_decisions,
    model_inputs,
    split_by_case,
)
from .model import (
    DecisionOutput,
    MarkerDecisionModel,
    TypedDecisionModel,
    decode_decisions,
    decision_loss,
)
from .rlcd import RLCDConfig, proper_scoring_reward, rlcd_loss, soft_cross_entropy
from .serialization import (
    QUESTION_TYPES,
    EncodedQuestion,
    encode_question,
    option_keys,
    render_options,
)
from .training import apply_temperatures, collect_logits, length_bucketed_batches

__all__ = [
    "apply_temperatures",
    "collect_logits",
    "length_bucketed_batches",
    "DATASET_ID",
    "QUESTION_TYPES",
    "DecisionAgent",
    "DecisionItem",
    "DecisionOutput",
    "EncodedQuestion",
    "MarkerDecisionModel",
    "RLCDConfig",
    "TypedDecisionModel",
    "build_items",
    "collate",
    "decision_loss",
    "decision_metrics",
    "decode_decisions",
    "encode_question",
    "expected_calibration_error",
    "fit_temperature",
    "fit_temperatures_by_type",
    "gold_distribution",
    "load_typed_decisions",
    "model_inputs",
    "option_keys",
    "proper_scoring_reward",
    "render_options",
    "rlcd_loss",
    "soft_cross_entropy",
    "split_by_case",
]

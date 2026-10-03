import tempfile
import unittest

import torch

from fakes import TinyEncoder
from jev_experimentation import (
    DecisionOutput,
    MarkerDecisionModel,
    TypedDecisionModel,
    decode_decisions,
    decision_loss,
)


class MarkerDecisionModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = MarkerDecisionModel(TinyEncoder(hidden_size=8), hidden_size=8, dropout=0)
        self.inputs = {
            "input_ids": torch.tensor([[1, 9, 2, 3, 10, 3, 11, 2, 12, 2], [1, 9, 2, 3, 10, 3, 11, 2, 0, 0]]),
            "attention_mask": torch.tensor([[1] * 10, [1] * 8 + [0, 0]]),
            "marker_positions": torch.tensor([[4, 6], [4, 0]]),
            "marker_mask": torch.tensor([[True, True], [True, False]]),
            "question_types": torch.tensor([2, 0]),
        }

    def test_scores_markers_and_masks_padded_options(self):
        output = self.model(**self.inputs)

        self.assertEqual(output.logits.shape, (2, 2))
        self.assertEqual(output.probabilities[1, 1].item(), 0)
        torch.testing.assert_close(output.probabilities.sum(-1), torch.ones(2))

    def test_temperature_softens_only_its_question_type(self):
        before = self.model(**self.inputs).probabilities
        self.model.temperature[2] = 4.0
        after = self.model(**self.inputs).probabilities

        self.assertLess(after[0].max(), before[0].max())
        torch.testing.assert_close(after[1], before[1])

    def test_option_logit_depends_on_marker_position_not_option_index(self):
        swapped = dict(self.inputs, marker_positions=torch.tensor([[6, 4], [4, 0]]))

        original = self.model(**self.inputs).logits
        reordered = self.model(**swapped).logits

        torch.testing.assert_close(reordered[0], original[0].flip(0))

    def test_gradients_reach_encoder_and_head(self):
        output = self.model(**self.inputs)
        decision_loss(output, torch.tensor([1, 0])).backward()

        self.assertIsNotNone(self.model.encoder.embedding.weight.grad)
        self.assertIsNotNone(self.model.type_embedding.weight.grad)
        self.assertIsNotNone(self.model.scorer[1].weight.grad)

    def test_save_and_load_round_trip_keeps_outputs_and_temperatures(self):
        from transformers import BertConfig, BertModel

        config = BertConfig(vocab_size=64, hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=32)
        model = MarkerDecisionModel(BertModel(config), hidden_size=16, dropout=0).eval()
        model.temperature[1] = 2.5
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = MarkerDecisionModel.load_pretrained(directory).eval()

        torch.testing.assert_close(loaded(**self.inputs).probabilities,
                                   model(**self.inputs).probabilities)
        self.assertEqual(loaded.temperature[1].item(), 2.5)

    def test_rejects_examples_without_options(self):
        inputs = dict(self.inputs, marker_mask=torch.tensor([[True, True], [False, False]]))

        with self.assertRaisesRegex(ValueError, "at least one real option"):
            self.model(**inputs)


class TypedDecisionModelTests(unittest.TestCase):
    def setUp(self):
        self.model = TypedDecisionModel(TinyEncoder(), hidden_size=4, dropout=0)
        self.input_ids = torch.tensor([
            [[1, 2, 0], [3, 4, 0], [5, 0, 0]],
            [[6, 7, 0], [8, 0, 0], [0, 0, 0]],
        ])
        self.attention_mask = self.input_ids.ne(0).long()
        self.option_mask = torch.tensor([
            [True, True, True],
            [True, True, False],
        ])

    def test_masks_padded_options_and_normalizes_each_distribution(self):
        output = self.model(self.input_ids, self.attention_mask, self.option_mask)

        self.assertEqual(output.probabilities.shape, (2, 3))
        self.assertEqual(output.probabilities[1, 2].item(), 0)
        torch.testing.assert_close(
            output.probabilities.sum(dim=1), torch.ones(2)
        )

    def test_loss_backpropagates(self):
        output = self.model(self.input_ids, self.attention_mask, self.option_mask)
        loss = decision_loss(output, torch.tensor([1, 0]))

        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(self.model.scorer[0].weight.grad)

    def test_decodes_choice_score_and_noul(self):
        probabilities = torch.tensor([[0.2, 0.8], [0.25, 0.75], [0.4, 0.6]])
        mask = torch.ones_like(probabilities, dtype=torch.bool)
        output = DecisionOutput(probabilities.log(), probabilities, mask)

        results = decode_decisions(
            output,
            ["choice", "score", "noul"],
            [["billing", "technical"], ["low", "high"], ["false", "true"]],
            ordinal_values=[[0, 1], [0, 2], [0, 1]],
        )

        self.assertEqual(results[0]["choice"], "technical")
        self.assertAlmostEqual(results[1]["score"], 1.5)
        self.assertAlmostEqual(results[2]["noul"], 0.6)

    def test_noul_requires_two_options(self):
        probabilities = torch.tensor([[1.0]])
        output = DecisionOutput(
            probabilities.log(), probabilities, torch.ones_like(probabilities, dtype=torch.bool)
        )

        with self.assertRaisesRegex(ValueError, r"\[false, true\]"):
            decode_decisions(output, ["noul"], [["true"]])


if __name__ == "__main__":
    unittest.main()
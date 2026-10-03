import unittest

import torch

from fakes import TinyEncoder, WordTokenizer
from jev_experimentation import (
    DecisionAgent,
    MarkerDecisionModel,
    RLCDConfig,
    apply_temperatures,
    build_items,
    collate,
    collect_logits,
    decision_metrics,
    expected_calibration_error,
    fit_temperature,
    gold_distribution,
    length_bucketed_batches,
    model_inputs,
    proper_scoring_reward,
    rlcd_loss,
    split_by_case,
)

QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "money", "technical": "bugs", "other": "rest"}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]},
    "refund": {"type": "noul", "instructions": "Refund requested?"},
}
GOLD = {
    "team": {"label": "billing", "probabilities": {"billing": 0.8, "technical": 0.2, "other": 0.0}},
    "urgency": {"label": "2", "probabilities": {"0": 0.0, "1": 0.25, "2": 0.75}},
    "refund": {"label": "True", "probabilities": {"false": 0.1, "true": 0.9}},
}


def rows(count):
    return [
        {"id": f"case{i}", "workflow": "support", "state": {"body": f"billed twice {i}"},
         "questions": QUESTIONS, "gold": GOLD}
        for i in range(count)
    ]


class ScoringRuleTests(unittest.TestCase):
    def test_reward_is_maximised_by_reporting_the_target(self):
        target = torch.tensor([[0.7, 0.2, 0.1]])
        mask = torch.ones_like(target, dtype=torch.bool)
        reports = torch.tensor([[[0.7, 0.2, 0.1]], [[0.9, 0.05, 0.05]], [[0.4, 0.4, 0.2]]])

        for qtype in (0, 1):
            reward = proper_scoring_reward(reports, target, torch.tensor([qtype]), mask)
            self.assertEqual(int(reward.argmax()), 0)

    def test_ranked_probability_penalises_distant_levels_more(self):
        target = torch.tensor([[1.0, 0.0, 0.0]])
        mask = torch.ones_like(target, dtype=torch.bool)
        near = torch.tensor([[0.5, 0.5, 0.0]])
        far = torch.tensor([[0.5, 0.0, 0.5]])
        score_type = torch.tensor([1])

        self.assertGreater(
            proper_scoring_reward(near, target, score_type, mask).item(),
            proper_scoring_reward(far, target, score_type, mask).item(),
        )

    def test_padded_options_do_not_change_reward(self):
        target = torch.tensor([[0.6, 0.4, 0.0]])
        probabilities = torch.tensor([[0.5, 0.5, 0.0]])
        mask = torch.tensor([[True, True, False]])

        padded = proper_scoring_reward(probabilities, target, torch.tensor([0]), mask)
        plain = proper_scoring_reward(probabilities[:, :2], target[:, :2], torch.tensor([0]),
                                      mask[:, :2])

        torch.testing.assert_close(padded, plain)


class RLCDTests(unittest.TestCase):
    def test_loss_is_finite_differentiable_and_seeded(self):
        logits = torch.zeros(2, 3, requires_grad=True)
        target = torch.tensor([[1.0, 0.0, 0.0], [0.3, 0.7, 0.0]])
        mask = torch.tensor([[True, True, True], [True, True, False]])
        types = torch.tensor([0, 2])

        first, stats = rlcd_loss(logits, target, types, mask,
                                 generator=torch.Generator().manual_seed(1))
        second, _ = rlcd_loss(logits, target, types, mask,
                              generator=torch.Generator().manual_seed(1))
        first.backward()

        self.assertTrue(torch.isfinite(first))
        self.assertEqual(first.item(), second.item())
        self.assertEqual(set(stats), {"loss", "policy_loss", "cross_entropy", "mean_reward"})
        self.assertEqual(logits.grad[1, 2].item(), 0)

    def test_policy_gradient_alone_moves_probability_toward_target(self):
        torch.manual_seed(0)
        logits = torch.zeros(1, 3, requires_grad=True)
        target = torch.tensor([[0.0, 1.0, 0.0]])
        mask = torch.ones_like(target, dtype=torch.bool)
        optimizer = torch.optim.SGD([logits], lr=0.05)
        config = RLCDConfig(group_size=16, sigma=0.5, ce_weight=0.0)

        for _ in range(200):
            optimizer.zero_grad()
            loss, _ = rlcd_loss(logits, target, torch.tensor([0]), mask, config)
            loss.backward()
            optimizer.step()

        self.assertEqual(int(logits.argmax()), 1)


class CalibrationTests(unittest.TestCase):
    def test_fit_temperature_recovers_overconfidence(self):
        generator = torch.Generator().manual_seed(0)
        true_logits = torch.randn(4000, 3, generator=generator)
        labels = torch.multinomial(true_logits.softmax(-1), 1, generator=generator).squeeze(1)
        target = torch.nn.functional.one_hot(labels, 3).float()
        mask = torch.ones_like(target, dtype=torch.bool)

        temperature = fit_temperature(true_logits * 2.5, target, mask)

        self.assertAlmostEqual(temperature, 2.5, delta=0.25)

    def test_ece_is_zero_when_confidence_matches_accuracy(self):
        confidence = torch.full((10,), 0.8)
        correct = torch.tensor([1.0] * 8 + [0.0] * 2)

        self.assertAlmostEqual(expected_calibration_error(confidence, correct), 0.0)
        self.assertAlmostEqual(expected_calibration_error(confidence, torch.ones(10)), 0.2, places=5)


class DataAndAgentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tokenizer = WordTokenizer()
        self.model = MarkerDecisionModel(TinyEncoder(hidden_size=8), hidden_size=8, dropout=0)

    def test_gold_distribution_normalises_and_finds_label(self):
        keys, target, index = gold_distribution(QUESTIONS["refund"], GOLD["refund"])
        self.assertEqual((keys, index), (["false", "true"], 1))
        self.assertAlmostEqual(sum(target), 1.0)

        _, one_hot, index = gold_distribution(QUESTIONS["team"], {"label": "other"})
        self.assertEqual((one_hot, index), ([0.0, 0.0, 1.0], 2))

    def test_items_collate_and_train_end_to_end(self):
        items = build_items(rows(4), self.tokenizer, max_len=64, head_max_len=40)
        self.assertEqual(len(items), 12)

        batch = collate(items, self.tokenizer.pad_token_id)
        output = self.model(**model_inputs(batch, "cpu"))
        loss, _ = rlcd_loss(output.logits, batch["target"], batch["question_types"],
                            batch["marker_mask"])
        loss.backward()
        metrics = decision_metrics(output.probabilities.detach(), batch["target"],
                                   batch["marker_mask"], batch["question_types"],
                                   batch["gold_index"])

        self.assertEqual(batch["marker_positions"].shape, (12, 3))
        padded = collate(items, self.tokenizer.pad_token_id, pad_to_multiple_of=32)
        self.assertEqual(padded["input_ids"].shape[1] % 32, 0)
        self.assertEqual(padded["attention_mask"].sum(), batch["attention_mask"].sum())
        self.assertEqual(metrics["overall"]["n"], 12)
        self.assertEqual(metrics["noul"]["n"], 4)

    def test_split_by_case_keeps_cases_together(self):
        items = build_items(rows(10), self.tokenizer, max_len=64, head_max_len=40)

        kept, held_out = split_by_case(items, 0.3, seed=0)

        self.assertEqual(len(held_out), 9)
        self.assertFalse({i.case_id for i in kept} & {i.case_id for i in held_out})

    def test_collect_logits_matches_per_batch_forward_and_covers_every_item(self):
        items = build_items(rows(5), self.tokenizer, max_len=64, head_max_len=40)
        batches = list(length_bucketed_batches(items, batch_size=4, seed=3))

        collected = collect_logits(self.model, items, self.tokenizer.pad_token_id, "cpu",
                                   batch_size=4)
        direct = self.model(**model_inputs(collate(items[:3], 0), "cpu")).logits
        probabilities = apply_temperatures(collected["logits"], collected["option_mask"],
                                           collected["question_types"], torch.ones(3))

        self.assertEqual(sorted(len(b) for b in batches), [3, 4, 4, 4])
        torch.testing.assert_close(collected["logits"][:3, :3], direct, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(15))
        self.assertEqual(collected["option_mask"].sum().item(), 5 * (3 + 3 + 2))

    def test_agent_returns_typed_answers_in_one_call(self):
        agent = DecisionAgent(self.model, self.tokenizer, max_len=64, head_max_len=40)

        result = agent.predict({"body": "billed twice"}, QUESTIONS)
        answers = result["answers"]

        self.assertIn(answers["team"]["choice"], QUESTIONS["team"]["criteria"])
        self.assertAlmostEqual(sum(answers["team"]["probabilities"].values()), 1.0, places=5)
        self.assertTrue(0 <= answers["urgency"]["score"] <= 2)
        self.assertTrue(0 <= answers["refund"]["noul"] <= 1)
        self.assertEqual(result["usage"]["output_tokens"], 0)


if __name__ == "__main__":
    unittest.main()

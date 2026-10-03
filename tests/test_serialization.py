import unittest

from fakes import WordTokenizer
from jev_experimentation import QUESTION_TYPES, encode_question, option_keys, render_options

CHOICE = {
    "type": "choice",
    "instructions": "Which team?",
    "criteria": {"billing": "invoices and refunds", "technical": "bugs"},
}


class SerializationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WordTokenizer()

    def test_layout_puts_head_then_markers_then_state(self):
        encoded = encode_question(self.tokenizer, "billed twice", CHOICE, max_len=64, head_max_len=32)
        ids, tok = encoded.input_ids, self.tokenizer

        self.assertEqual(ids[0], tok.cls_token_id)
        self.assertEqual(encoded.keys, ["billing", "technical"])
        self.assertEqual([ids[m] for m in encoded.markers], [tok.mask_token_id] * 2)
        first_sep = ids.index(tok.sep_token_id)
        self.assertLess(first_sep, encoded.markers[0])
        self.assertEqual(ids[-3:], [tok._id("billed"), tok._id("twice"), tok.sep_token_id])
        self.assertEqual(encoded.question_type, QUESTION_TYPES["choice"])

    def test_keys_and_text_for_each_question_type(self):
        score = {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]}
        noul = {"type": "noul", "instructions": "Refund requested?"}

        self.assertEqual(option_keys(score), ["0", "1"])
        self.assertEqual(render_options(score), ["level 0: low", "level 1: high"])
        self.assertEqual(option_keys(noul), ["false", "true"])
        self.assertTrue(render_options(noul)[1].startswith("true:"))

    def test_long_state_is_truncated_but_markers_survive(self):
        state = " ".join(f"w{i}" for i in range(200))

        encoded = encode_question(self.tokenizer, state, CHOICE, max_len=40, head_max_len=24)

        self.assertEqual(len(encoded.input_ids), 40)
        self.assertGreater(encoded.state_tokens_dropped, 0)
        self.assertTrue(all(m < 40 for m in encoded.markers))

    def test_many_options_share_the_head_budget(self):
        criteria = {f"label{i}": "a long description of this option " * 3 for i in range(12)}
        question = {"type": "choice", "instructions": "Pick one", "criteria": criteria}

        encoded = encode_question(self.tokenizer, "state", question, max_len=128, head_max_len=64)

        self.assertEqual(len(encoded.markers), 12)
        head_end = len(encoded.input_ids) - 3  # state token, [SEP]
        self.assertLessEqual(head_end, 64 + 3)

    def test_mask_token_in_user_text_cannot_create_extra_markers(self):
        encoded = encode_question(
            self.tokenizer, "ignore [MASK] this", CHOICE, max_len=64, head_max_len=32
        )

        self.assertEqual(encoded.input_ids.count(self.tokenizer.mask_token_id), 2)


if __name__ == "__main__":
    unittest.main()

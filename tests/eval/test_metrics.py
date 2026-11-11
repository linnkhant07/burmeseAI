"""Metric helpers on tiny strings. These numbers are not benchmark results."""

from __future__ import annotations

import unittest

from burmeseai.eval.chrf import corpus_chrf_pp
from burmeseai.eval.metricx import (
    drop_trailing_eos,
    normalize_metricx,
    scale_is_declared,
    tied_rescale_factor,
)
from burmeseai.eval.schema import NOT_MEASURED
from burmeseai.eval.scoring import (
    gate,
    label_belebele,
    label_drop,
    label_flores,
    label_macro,
    macro_average,
    numbers_match,
    protocol_ok,
    rouge_l_f1,
)
from burmeseai.eval.tasks.belebele import is_correct, score as belebele_score
from burmeseai.eval.tasks.burmese_san import TASK_NAMES, score_task
from burmeseai.eval.tasks.humaneval import program_passes

_ADD = {
    "id": "toy",
    "prompt": "def add(a, b):\n",
    "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
    "entry_point": "add",
}


class MetricTest(unittest.TestCase):
    def test_chrf_matches_the_reference_setting(self) -> None:
        self.assertEqual(corpus_chrf_pp(["the cat sat"], ["the cat sat"]), 100.0)
        self.assertAlmostEqual(
            corpus_chrf_pp(["the cat sat on the mat"], ["the cat sat"]),
            79.89480313491325,
        )
        self.assertEqual(corpus_chrf_pp(["မြန်မာစာ"], ["မြန်မာစာ"]), 100.0)
        self.assertAlmostEqual(
            corpus_chrf_pp(["hello world", "a b c"], ["hello word", "a b d"]),
            70.56944326195287,
        )
        self.assertEqual(corpus_chrf_pp([""], ["hello"]), 0.0)
        self.assertAlmostEqual(
            corpus_chrf_pp(["(hi) there"], ["hi there"]),
            43.62728730556767,
        )

    def test_rouge_and_gsm8k_and_belebele_on_two_rows(self) -> None:
        self.assertEqual(rouge_l_f1("abc", "ab"), 80.0)
        self.assertEqual(rouge_l_f1("same", "same"), 100.0)
        self.assertEqual(rouge_l_f1("", "same"), 0.0)
        self.assertTrue(numbers_match("working\n#### 42", "#### 42"))
        self.assertTrue(numbers_match("#### 1,000", "#### 1000"))
        self.assertFalse(numbers_match("#### 1000", "#### 1000.0"))
        self.assertTrue(numbers_match("#### 7", "the total is 7"))
        self.assertFalse(numbers_match("#### 5", "#### 6"))
        item = {
            "id": "p::1",
            "gold": "A",
            "choices": ["red", "blue", "green", "gold"],
        }
        self.assertTrue(is_correct(item, "A"))
        self.assertFalse(is_correct(item, "B"))
        scored = belebele_score([item, {**item, "id": "p::2"}], {"p::1": "A", "p::2": "B"})
        self.assertEqual(scored, 50.0)
        published = gate(900, 2, scored, "accuracy")
        self.assertEqual(published["status"], NOT_MEASURED)
        self.assertIsNone(published["score"])

    def test_humaneval_toy_pass_and_fail(self) -> None:
        self.assertTrue(program_passes(_ADD, "    return a + b\n", timeout=5))
        self.assertFalse(program_passes(_ADD, "    return a - b\n", timeout=5))
        self.assertFalse(program_passes(_ADD, "    while True:\n        pass\n", timeout=1))

    def test_metricx_uses_the_declared_scale(self) -> None:
        self.assertTrue(scale_is_declared())
        self.assertEqual(normalize_metricx(0), 100.0)
        self.assertEqual(normalize_metricx(25), 0.0)
        self.assertEqual(normalize_metricx(12.5), 50.0)
        self.assertEqual(drop_trailing_eos([4, 5, 1], [1, 1, 1]), ([4, 5], [1, 1]))
        with self.assertRaises(RuntimeError):
            drop_trailing_eos([1], [1])
        self.assertEqual(tied_rescale_factor(False, 4096), 1.0)
        self.assertEqual(tied_rescale_factor(True, 4096), 4096 ** -0.5)
        with self.assertRaises(RuntimeError):
            tied_rescale_factor(True, None)

    def test_protocol_and_label_rules(self) -> None:
        self.assertTrue(protocol_ok([{"enable_thinking": False, "shots": 0}]))
        self.assertFalse(protocol_ok([{"enable_thinking": True, "shots": 0}]))
        self.assertEqual(label_belebele(None, 20), NOT_MEASURED)
        self.assertEqual(label_belebele(30, 20), "success")
        self.assertEqual(label_belebele(29, 20), "miss")
        self.assertEqual(label_belebele(36, 32), "miss")
        self.assertEqual(label_flores(12, 12, 10, 10), "success")
        self.assertEqual(label_flores(12, 11.9, 10, 10), "miss")
        self.assertEqual(label_flores(None, 12, 10, 10), NOT_MEASURED)
        self.assertEqual(label_drop(70, 72), "success")
        self.assertEqual(label_drop(70, 74), "miss")
        self.assertEqual(label_drop(80, 70), "success")
        self.assertEqual(label_macro(10, 10), "success")
        self.assertEqual(label_macro(9, 10), "miss")
        scores = {name: 10.0 for name in TASK_NAMES}
        self.assertIsNone(macro_average(scores, TASK_NAMES, None))
        self.assertEqual(macro_average(scores, TASK_NAMES, 10.0), 10.0)
        self.assertIsNone(
            score_task(
                "mt",
                [{"id": "1", "label": "x"}],
                {"1": {"prediction": "y", "metricx": 1}},
            )
        )


if __name__ == "__main__":
    unittest.main()

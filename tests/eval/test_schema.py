"""The empty report names every model and task and contains no measured score."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from burmeseai.eval.driver import build_report
from burmeseai.eval.frozen_humaneval import FROZEN_HUMANEVAL_TASK_IDS
from burmeseai.eval.schema import (
    ARC_EMPTY,
    BELEBELE_ITEMS,
    BURMESE_SAN_ITEMS,
    BURMESE_SAN_TASKS,
    FLORES_PAIRS,
    GSM8K_ITEMS,
    NOT_MEASURED,
    empty_report,
)

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = (
    "77.3",
    "60.4",
    "59.9",
    "72.35",
    "66.46",
    "51.61",
    "54.68",
    "23.07",
    "11.62",
    "11.70",
    "8.93",
    "2.41",
    "3.16",
)


class SchemaTest(unittest.TestCase):
    def test_empty_report_names_the_harness(self) -> None:
        report = empty_report(FROZEN_HUMANEVAL_TASK_IDS)
        names = [row["name"] for row in report["models"]]
        self.assertEqual(
            names,
            [
                "Qwen3-0.6B",
                "Gemma 4 E2B",
                "Gemma 4 E4B",
                "SEA-LION v4.5 E2B-IT",
                "Padauk Q8",
                "Arc 1.0",
            ],
        )
        sources = {row["id"]: row["source"] for row in report["models"]}
        self.assertEqual(sources["qwen3-0.6b"], "Qwen/Qwen3-0.6B")
        self.assertEqual(sources["sea-lion-v4.5-e2b-it"], "aisingapore/Gemma-SEA-LION-v4.5-E2B-IT")
        padauk = report["models"][4]
        self.assertEqual(padauk["source"], "WYNN747/ai4burmese-padauk-gguf")
        self.assertEqual(padauk["filename"], "padauk-gemma-q8_0.gguf")
        self.assertEqual(report["models"][5]["weights"], ARC_EMPTY)
        self.assertFalse(report["enable_thinking"])
        self.assertEqual(report["shots"], 0)

        belebele = report["tasks"]["belebele"]
        self.assertEqual(belebele["expected_items"], BELEBELE_ITEMS)
        self.assertEqual(belebele["language"], "mya_Mymr")
        self.assertEqual(belebele["choices"], 4)
        self.assertEqual(belebele["shots"], 0)
        self.assertFalse(belebele["enable_thinking"])
        self.assertEqual(belebele["chance"], 25)

        flores = report["tasks"]["flores"]
        self.assertEqual(flores["corpus"], "facebook/flores")
        self.assertEqual(flores["split"], "devtest")
        self.assertEqual(flores["expected_pairs"], FLORES_PAIRS)
        self.assertEqual(
            flores["directions"],
            ["eng_Latn_to_mya_Mymr", "mya_Mymr_to_eng_Latn"],
        )

        san = report["tasks"]["burmese_san"]
        self.assertEqual(san["expected_items"], BURMESE_SAN_ITEMS)
        self.assertEqual(san["task_count"], 7)
        self.assertEqual(san["release_license"], "CC-BY-SA 4.0")
        self.assertEqual(sum(row["items"] for row in san["licenses"]), BURMESE_SAN_ITEMS)
        licenses = {row["task"]: row["license"] for row in san["licenses"]}
        self.assertEqual(licenses["sa"], "Unknown")
        self.assertEqual(licenses["qa"], "CC BY-NC 4.0")
        self.assertEqual(licenses["td"], "CC BY-NC-SA 4.0")
        self.assertEqual(licenses["cr"], "CC BY 4.0")
        self.assertEqual(licenses["nli"], "CC BY-NC 4.0")
        self.assertEqual(licenses["as"], "CC BY-NC-SA 4.0")
        self.assertEqual(licenses["mt"], "CC BY-SA 4.0")
        self.assertEqual(
            [row["task"] for row in BURMESE_SAN_TASKS],
            ["qa", "sa", "td", "cr", "nli", "as", "mt"],
        )

        self.assertEqual(report["tasks"]["gsm8k"]["expected_items"], GSM8K_ITEMS)
        self.assertEqual(report["tasks"]["gsm8k"]["metric"], "exact_match")
        frozen = report["tasks"]["humaneval"]["frozen_task_ids"]
        self.assertEqual(frozen, list(FROZEN_HUMANEVAL_TASK_IDS))
        self.assertEqual(len(frozen), 32)
        self.assertEqual(frozen[0], "HumanEval/0")
        self.assertEqual(frozen[-1], "HumanEval/155")
        self.assertNotIn("HumanEval/1", frozen)

        measurements = {row["id"]: row for row in report["measurements"]}
        for key in (
            "resident_memory",
            "tokens_per_second",
            "apple_silicon",
            "mlx",
            "android_8gb",
            "colloquial",
        ):
            self.assertEqual(measurements[key]["status"], NOT_MEASURED)
            self.assertIsNone(measurements[key]["value"])
        self.assertEqual(measurements["resident_memory"]["unit"], "bytes")
        self.assertEqual(measurements["tokens_per_second"]["unit"], "tokens/sec")
        self.assertIn("20 human-written", measurements["colloquial"]["note"])

        for task_id, block in report["labels"].items():
            self.assertEqual(block["status"], NOT_MEASURED, task_id)
        self._assert_no_measured_score(report)

    def test_missing_dirs_keep_the_same_empty_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_report(root / "data", root / "weights", root / "runs")
        self._assert_no_measured_score(report)
        self.assertEqual(report["models"][5]["weights"], ARC_EMPTY)
        for task in report["tasks"].values():
            for cell in task["models"].values():
                self.assertEqual(cell["status"], NOT_MEASURED)
                self.assertIn("data absent", cell["reason"])
        blob = json.dumps(report)
        for token in FORBIDDEN:
            self.assertNotIn(token, blob)

    def test_sources_do_not_carry_external_scores(self) -> None:
        text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "burmeseai" / "eval").rglob("*.py")
        )
        for token in FORBIDDEN:
            self.assertNotIn(token, text)

    def _assert_no_measured_score(self, report: dict) -> None:
        for task in report["tasks"].values():
            for cell in task["models"].values():
                self.assertEqual(cell["status"], NOT_MEASURED)
                self.assertIsNone(cell["score"])
                for direction in ("eng_Latn_to_mya_Mymr", "mya_Mymr_to_eng_Latn"):
                    if direction in cell:
                        self.assertIsNone(cell[direction])
                for nested in cell.get("tasks", {}).values():
                    self.assertEqual(nested["status"], NOT_MEASURED)
                    self.assertIsNone(nested["score"])


if __name__ == "__main__":
    unittest.main()

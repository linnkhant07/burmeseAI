"""The four review fixes: MetricX, thinking, report status, and FLORES digests."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from burmeseai.eval.driver import (
    build_report,
    predictions_path,
    report_stdout_status,
    run_task,
)
from burmeseai.eval.generate import call_chat_template, gguf_template_kwargs
from burmeseai.eval.schema import NOT_MEASURED
from burmeseai.eval.tasks.burmese_san import EXPECTED, EXPECTED_LABELS, MT_DIRECTIONS, TASK_NAMES
from burmeseai.eval.tasks.flores import PINNED_CORPUS, can_publish

PLANT = 17.125


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prompts() -> dict[str, str]:
    return {name: "prompt {input} {choices} {direction}" for name in TASK_NAMES}


def _full_items() -> list[dict]:
    items = []

    def add(task: str, label: str, count: int, direction: str | None = None) -> None:
        choices = list(EXPECTED_LABELS.get(task, {}))
        for index in range(count):
            items.append({
                "id": f"{task}-{label}-{direction}-{index}",
                "task": task,
                "input": f"{task} source {index}",
                "label": label,
                "choices": choices,
                "direction": direction,
            })

    for label, count in EXPECTED_LABELS["sa"].items():
        add("sa", label, count)
    for label, count in EXPECTED_LABELS["td"].items():
        add("td", label, count)
    for label, count in EXPECTED_LABELS["cr"].items():
        add("cr", label, count)
    for label, count in EXPECTED_LABELS["nli"].items():
        add("nli", label, count)
    add("qa", "span", EXPECTED["qa"])
    add("as", "summary", EXPECTED["as"])
    for direction, count in MT_DIRECTIONS.items():
        add("mt", "translation", count, direction)
    return items


def _prediction(item: dict) -> dict:
    row = {
        "id": item["id"],
        "prediction": item["label"],
        "enable_thinking": False,
        "shots": 0,
    }
    if item["task"] == "mt":
        row["metricx"] = PLANT
    return row


class ReviewFixTest(unittest.TestCase):
    def test_template_rejection_is_not_retried(self) -> None:
        class Tokenizer:
            def __init__(self) -> None:
                self.calls = 0

            def apply_chat_template(self, messages, **kwargs):
                self.calls += 1
                if "enable_thinking" in kwargs:
                    raise TypeError("rejected")
                return "omitted"

        tokenizer = Tokenizer()
        with self.assertRaises(TypeError):
            call_chat_template(tokenizer, [])
        self.assertEqual(tokenizer.calls, 1)

    def test_gguf_call_must_accept_the_flag(self) -> None:
        def bare(messages, temperature, max_tokens):
            return messages

        with self.assertRaises(RuntimeError):
            gguf_template_kwargs(inspect.signature(bare).parameters)

        def variable(messages, **kwargs):
            return kwargs

        kwargs = gguf_template_kwargs(inspect.signature(variable).parameters)
        self.assertEqual(kwargs["chat_template_kwargs"], {"enable_thinking": False})

    def test_pinned_flores_requires_the_digest(self) -> None:
        pairs = [{"id": str(index), "eng_Latn": "a", "mya_Mymr": "b"} for index in range(1012)]
        self.assertTrue(can_publish({
            "corpus": PINNED_CORPUS,
            "pairs": pairs,
            "pin_match": True,
        }))
        self.assertFalse(can_publish({
            "corpus": PINNED_CORPUS,
            "pairs": pairs,
            "pin_match": False,
        }))
        self.assertFalse(can_publish({
            "corpus": None,
            "pairs": pairs,
            "pin_match": False,
        }))

    def test_unpinned_flores_is_not_labeled_and_not_averaged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            devtest = root / "data" / "flores" / "devtest"
            devtest.mkdir(parents=True)
            english = "\n".join(f"english {index}" for index in range(1012)) + "\n"
            burmese = "\n".join(f"burmese {index}" for index in range(1012)) + "\n"
            (devtest / "eng_Latn.devtest").write_text(english, encoding="utf-8")
            (devtest / "mya_Mymr.devtest").write_text(burmese, encoding="utf-8")
            report = build_report(root / "data", root / "weights", root / "runs")
            self.assertIsNone(report["tasks"]["flores"]["corpus"])
            cell = report["tasks"]["flores"]["models"]["qwen3-0.6b"]
            self.assertEqual(cell["status"], NOT_MEASURED)
            self.assertIsNone(cell["score"])
            self.assertIsNone(cell["eng_Latn_to_mya_Mymr"])
            self.assertIsNone(cell["mya_Mymr_to_eng_Latn"])

            (root / "data" / "flores" / "manifest.json").write_text(
                json.dumps({"corpus": "openlanguagedata/flores_plus"}),
                encoding="utf-8",
            )
            rows = []
            for index in range(1012):
                for direction in ("eng_Latn_to_mya_Mymr", "mya_Mymr_to_eng_Latn"):
                    rows.append({
                        "id": str(index),
                        "direction": direction,
                        "prediction": (
                            f"burmese {index}"
                            if direction.endswith("mya_Mymr")
                            else "xxxx"
                        ),
                        "enable_thinking": False,
                        "shots": 0,
                    })
            _write_jsonl(predictions_path(root / "runs", "qwen3-0.6b", "flores"), rows)
            report = build_report(root / "data", root / "weights", root / "runs")
            self.assertEqual(
                report["tasks"]["flores"]["corpus"],
                "openlanguagedata/flores_plus",
            )
            self.assertNotEqual(report["tasks"]["flores"]["corpus"], PINNED_CORPUS)
            cell = report["tasks"]["flores"]["models"]["qwen3-0.6b"]
            self.assertEqual(cell["status"], "measured")
            self.assertIsNone(cell["score"])
            self.assertIsInstance(cell["eng_Latn_to_mya_Mymr"], float)
            self.assertIsInstance(cell["mya_Mymr_to_eng_Latn"], float)
            self.assertNotEqual(
                cell["eng_Latn_to_mya_Mymr"],
                cell["mya_Mymr_to_eng_Latn"],
            )

    def test_burmese_san_checkpoint_and_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            items = _full_items()
            _write_jsonl(data / "burmese-san" / "items.jsonl", items)
            _write_jsonl(
                predictions_path(root / "runs", "qwen3-0.6b", "burmese_san"),
                [_prediction(item) for item in items],
            )
            report = build_report(data, root / "weights", root / "runs")
            cell = report["tasks"]["burmese_san"]["models"]["qwen3-0.6b"]
            self.assertEqual(cell["status"], NOT_MEASURED)
            self.assertIsNone(cell["score"])
            self.assertEqual(
                cell["tasks"]["mt"]["reason"],
                "Burmese prompt templates are absent",
            )
            self.assertIsNone(cell["tasks"]["mt"]["score"])
            self.assertNotIn(str(PLANT), json.dumps(cell))

            (data / "burmese-san" / "prompts.json").write_text(
                json.dumps(_prompts()),
                encoding="utf-8",
            )
            report = build_report(data, root / "weights", root / "runs")
            cell = report["tasks"]["burmese_san"]["models"]["qwen3-0.6b"]
            self.assertEqual(cell["tasks"]["qa"]["status"], "measured")
            self.assertEqual(cell["tasks"]["qa"]["score"], 100.0)
            self.assertEqual(cell["tasks"]["mt"]["status"], NOT_MEASURED)
            self.assertIn("checkpoint is absent", cell["tasks"]["mt"]["reason"])
            self.assertEqual(cell["tasks"]["as"]["status"], NOT_MEASURED)
            self.assertIn("XL-Sum", cell["tasks"]["as"]["reason"])
            self.assertIsNone(cell["score"])
            self.assertNotIn(str(PLANT), json.dumps(cell))
            self.assertEqual(cell["tasks"]["mt"]["metric"], "metricx_24")

            def fake_metricx(examples, checkpoint):
                self.assertTrue(all("metricx" not in example for example in examples))
                self.assertEqual(len(examples), EXPECTED["mt"])
                return 40.0

            with (
                patch("burmeseai.eval.driver.metricx.find_checkpoint", return_value=root / "checkpoint"),
                patch("burmeseai.eval.driver.metricx.normalized_mean", side_effect=fake_metricx),
                patch("burmeseai.eval.tasks.burmese_san.mean_rouge_l", return_value=50.0),
            ):
                report = build_report(data, root / "weights", root / "runs")
            cell = report["tasks"]["burmese_san"]["models"]["qwen3-0.6b"]
            self.assertEqual(cell["tasks"]["mt"]["score"], 40.0)
            self.assertEqual(cell["tasks"]["as"]["score"], 50.0)
            self.assertEqual(cell["status"], "measured")
            self.assertAlmostEqual(cell["score"], (100.0 * 5 + 50.0 + 40.0) / 7)
            printed = {"status": report_stdout_status(report)}
            self.assertEqual(printed["status"], "measured")
            self.assertNotIn("score", printed)

    def test_missing_prompts_do_not_write_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_jsonl(
                root / "data" / "burmese-san" / "items.jsonl",
                [{
                    "id": "qa-0",
                    "task": "qa",
                    "input": "passage",
                    "label": "span",
                }],
            )
            cell = run_task(
                "qwen3-0.6b",
                "burmese_san",
                root / "data",
                root / "weights",
                root / "runs",
                generate=lambda prompt, enable_thinking: "span",
            )
            self.assertEqual(cell["reason"], "Burmese prompt templates are absent")
            self.assertIsNone(cell["score"])
            self.assertFalse((root / "runs").exists())

    def test_empty_report_stdout_stays_unmeasured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_report(root / "data", root / "weights", root / "runs")
        self.assertEqual(report_stdout_status(report), NOT_MEASURED)


if __name__ == "__main__":
    unittest.main()

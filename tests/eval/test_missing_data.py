"""Absent files stay unmeasured, and only a run writes predictions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from burmeseai.eval.driver import build_report, predictions_path, run_task
from burmeseai.eval.schema import ARC_EMPTY, NOT_MEASURED, PADAUK_FILENAME

ROOT = Path(__file__).resolve().parents[2]


def _belebele_row(number: int = 1) -> dict:
    return {
        "link": "http://example.test/passage",
        "question_number": number,
        "flores_passage": "A short passage about a market.",
        "question": "What is named?",
        "mc_answer1": "red",
        "mc_answer2": "blue",
        "mc_answer3": "green",
        "mc_answer4": "gold",
        "correct_answer_num": "1",
        "dialect": "mya_Mymr",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


class MissingDataTest(unittest.TestCase):
    def test_report_does_not_create_prediction_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            report = build_report(root / "data", root / "weights", runs)
            self.assertFalse(runs.exists())
            self.assertIsNone(report["tasks"]["belebele"]["models"]["qwen3-0.6b"]["score"])
            self.assertEqual(
                report["measurements"][0]["id"],
                "resident_memory",
            )
            self.assertIsNone(report["measurements"][0]["value"])
            self.assertIsNone(report["measurements"][1]["value"])

    def test_run_without_files_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            for model, task in (
                ("qwen3-0.6b", "belebele"),
                ("qwen3-0.6b", "flores"),
                ("qwen3-0.6b", "burmese_san"),
                ("qwen3-0.6b", "gsm8k"),
                ("qwen3-0.6b", "humaneval"),
                ("arc-1.0", "belebele"),
            ):
                cell = run_task(model, task, root / "data", root / "weights", runs)
                self.assertEqual(cell["status"], NOT_MEASURED)
                self.assertIsNone(cell["score"])
                self.assertIn("data absent", cell["reason"])
            self.assertFalse(runs.exists())

    def test_short_belebele_file_is_not_a_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            runs = root / "runs"
            _write_jsonl(data / "belebele" / "mya_Mymr.jsonl", [_belebele_row()])
            cell = run_task(
                "qwen3-0.6b",
                "belebele",
                data,
                root / "weights",
                runs,
                generate=lambda prompt, enable_thinking: "A",
            )
            self.assertEqual(cell["status"], NOT_MEASURED)
            self.assertIsNone(cell["score"])
            self.assertIn("found 1", cell["reason"])
            path = predictions_path(runs, "qwen3-0.6b", "belebele")
            self.assertTrue(path.is_file())
            saved = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIs(saved["enable_thinking"], False)
            self.assertEqual(saved["shots"], 0)
            before = path.read_bytes()
            report = build_report(data, root / "weights", runs)
            self.assertEqual(path.read_bytes(), before)
            published = report["tasks"]["belebele"]["models"]["qwen3-0.6b"]
            self.assertEqual(published["status"], NOT_MEASURED)
            self.assertIsNone(published["score"])

    def test_weights_absent_does_not_write_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            runs = root / "runs"
            _write_jsonl(data / "gsm8k" / "test.jsonl", [
                {"question": "What is 1+1?", "answer": "1+1=2\n#### 2"},
            ])
            cell = run_task("gemma4-e2b", "gsm8k", data, root / "weights", runs)
            self.assertEqual(cell["reason"], "weights absent")
            self.assertIsNone(cell["score"])
            self.assertFalse(runs.exists())

    def test_padauk_ignores_a_different_filename_and_an_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "weights" / "padauk-q8"
            weights.mkdir(parents=True)
            (weights / "other.gguf").write_bytes(b"not-the-q8-file")
            (weights / PADAUK_FILENAME).write_bytes(b"")
            report = build_report(root / "data", root / "weights", root / "runs")
            padauk = next(row for row in report["models"] if row["id"] == "padauk-q8")
            self.assertEqual(padauk["weights"], "absent")
            (weights / PADAUK_FILENAME).write_bytes(b"x")
            report = build_report(root / "data", root / "weights", root / "runs")
            padauk = next(row for row in report["models"] if row["id"] == "padauk-q8")
            self.assertEqual(padauk["weights"], "present")
            self.assertIsNone(report["tasks"]["flores"]["models"]["padauk-q8"]["score"])

    def test_arc_stays_empty_until_a_weight_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arc = root / "weights" / "arc-1.0"
            arc.mkdir(parents=True)
            report = build_report(root / "data", root / "weights", root / "runs")
            row = next(item for item in report["models"] if item["id"] == "arc-1.0")
            self.assertEqual(row["weights"], ARC_EMPTY)
            cell = run_task("arc-1.0", "gsm8k", root / "data", root / "weights", root / "runs")
            self.assertIsNone(cell["score"])
            self.assertFalse((root / "runs").exists())
            (arc / "model.gguf").write_bytes(b"x")
            report = build_report(root / "data", root / "weights", root / "runs")
            row = next(item for item in report["models"] if item["id"] == "arc-1.0")
            self.assertEqual(row["weights"], "present")
            self.assertIsNone(report["tasks"]["gsm8k"]["models"]["arc-1.0"]["score"])

    def test_one_flores_side_and_a_named_corpus_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            devtest = data / "flores" / "devtest"
            devtest.mkdir(parents=True)
            (devtest / "eng_Latn").write_text("hello\n", encoding="utf-8")
            report = build_report(data, root / "weights", root / "runs")
            cell = report["tasks"]["flores"]["models"]["qwen3-0.6b"]
            self.assertEqual(cell["reason"], "data absent")
            self.assertIsNone(cell["eng_Latn_to_mya_Mymr"])
            self.assertIsNone(cell["mya_Mymr_to_eng_Latn"])
            (devtest / "mya_Mymr").write_text("မင်္ဂလာပါ\n", encoding="utf-8")
            (data / "flores" / "manifest.json").write_text(
                json.dumps({"corpus": "openlanguagedata/flores_plus"}),
                encoding="utf-8",
            )
            report = build_report(data, root / "weights", root / "runs")
            self.assertEqual(report["tasks"]["flores"]["corpus"], "openlanguagedata/flores_plus")
            cell = report["tasks"]["flores"]["models"]["gemma4-e4b"]
            self.assertEqual(cell["status"], NOT_MEASURED)
            self.assertIsNone(cell["score"])
            self.assertIsNone(cell["eng_Latn_to_mya_Mymr"])

    def test_bad_protocol_is_not_scored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            runs = root / "runs"
            _write_jsonl(data / "belebele" / "mya_Mymr.jsonl", [_belebele_row()])
            _write_jsonl(
                predictions_path(runs, "qwen3-0.6b", "belebele"),
                [{
                    "id": "http://example.test/passage::1",
                    "prediction": "A",
                    "enable_thinking": True,
                    "shots": 0,
                }],
            )
            report = build_report(data, root / "weights", runs)
            cell = report["tasks"]["belebele"]["models"]["qwen3-0.6b"]
            self.assertEqual(cell["status"], NOT_MEASURED)
            self.assertIsNone(cell["score"])
            self.assertIn("enable_thinking", cell["reason"])

    def test_cli_report_and_run_on_empty_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT)
            out = root / "report.json"
            report_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "burmeseai.eval",
                    "report",
                    "--data-dir",
                    str(root / "data"),
                    "--weights-dir",
                    str(root / "weights"),
                    "--predictions-dir",
                    str(root / "runs"),
                    "--out",
                    str(out),
                ],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(report_proc.returncode, 0, report_proc.stderr)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertIsNone(payload["tasks"]["gsm8k"]["models"]["qwen3-0.6b"]["score"])
            self.assertIsNone(payload["measurements"][0]["value"])
            self.assertIsNone(payload["measurements"][1]["value"])
            self.assertFalse((root / "runs").exists())
            run_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "burmeseai.eval",
                    "run",
                    "--model",
                    "arc-1.0",
                    "--task",
                    "humaneval",
                    "--data-dir",
                    str(root / "data"),
                    "--weights-dir",
                    str(root / "weights"),
                    "--predictions-dir",
                    str(root / "runs"),
                ],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(run_proc.returncode, 0, run_proc.stderr)
            body = json.loads(run_proc.stdout)
            self.assertEqual(body[0]["status"], NOT_MEASURED)
            self.assertIsNone(body[0]["score"])
            self.assertFalse((root / "runs").exists())


if __name__ == "__main__":
    unittest.main()

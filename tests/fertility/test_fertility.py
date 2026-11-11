"""The missing FLORES path must not invent a fertility premium."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import burmeseai.fertility as fertility


def banned_premiums() -> list[str]:
    pairs = [("11", "62"), ("8", "93"), ("2", "41"), ("3", "16")]
    return [left + "." + right for left, right in pairs]


def assert_no_banned_text(testcase: unittest.TestCase, text: str) -> None:
    for banned in banned_premiums():
        testcase.assertNotIn(banned, text)


class FertilityTests(unittest.TestCase):
    def test_missing_corpus_does_not_invent_a_premium(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "report.json"
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = fertility.main(
                    [
                        "--english",
                        str(root / "missing-eng.devtest"),
                        "--burmese",
                        str(root / "missing-my.devtest"),
                        "--qwen-tokenizer",
                        str(root / "missing-tokenizer.json"),
                        "--extended-tokenizer",
                        str(root / "missing-extended.json"),
                        "--gemma-tokenizer",
                        str(root / "missing-gemma.json"),
                        "--out",
                        str(out),
                    ]
                )
            text = out.read_text(encoding="utf-8")
            self.assertEqual(stdout.getvalue(), text)
        self.assertEqual(code, 0)
        report = json.loads(text)
        self.assertEqual(report["status"], "baseline not measured")
        self.assertIsNone(report["gate"])
        self.assertFalse(report["corpus"]["pin_match"])
        self.assertIsNone(report["corpus"]["pairs"])
        self.assertEqual(
            [row["tokenizer"] for row in report["rows"]],
            [
                "cl100k_base",
                "o200k_base",
                "Unmodified Qwen3-0.6B",
                "Extended Qwen3-0.6B",
                "Gemma 4 E2B",
            ],
        )
        citations = [row["external"] for row in report["rows"]]
        self.assertEqual(sum("11.70" in (item or "") for item in citations), 1)
        self.assertIn("11.70", citations[0])
        for row in report["rows"]:
            self.assertIsNone(row["measured_premium"])
            self.assertIsNone(row["burmese_tokens"])
            self.assertIsNone(row["english_tokens"])
            self.assertEqual(row["detail"], "not counted")
            measured = {
                "measured_premium": row["measured_premium"],
                "burmese_tokens": row["burmese_tokens"],
                "english_tokens": row["english_tokens"],
            }
            self.assertNotIn("11.70", json.dumps(measured))
        assert_no_banned_text(self, text)

    def test_right_length_unpinned_corpus_does_not_invent_a_premium(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            english = root / "eng_Latn.devtest"
            burmese = root / "mya_Mymr.devtest"
            english.write_text("not the flores devtest\n" * fertility.PINNED_PAIRS, encoding="utf-8")
            burmese.write_text("က\n" * fertility.PINNED_PAIRS, encoding="utf-8")
            out = root / "report.json"
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = fertility.main(
                    [
                        "--english",
                        str(english),
                        "--burmese",
                        str(burmese),
                        "--qwen-tokenizer",
                        str(root / "tokenizer.json"),
                        "--out",
                        str(out),
                    ]
                )
            self.assertEqual(code, 0)
            report = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "baseline not measured")
        self.assertFalse(report["corpus"]["pin_match"])
        self.assertIsNone(report["corpus"]["pairs"])
        self.assertIsNotNone(report["corpus"]["observed_english_sha256"])
        self.assertNotEqual(
            report["corpus"]["observed_english_sha256"],
            fertility.PINNED_ENGLISH_SHA256,
        )
        for row in report["rows"]:
            self.assertIsNone(row["measured_premium"])
            self.assertIsNone(row["burmese_tokens"])
            self.assertIsNone(row["english_tokens"])
        assert_no_banned_text(self, json.dumps(report))

    def test_unpinned_report_does_not_call_the_encoder(self) -> None:
        def encode(_text: str) -> list[int]:
            raise AssertionError("encoder ran without the pinned corpus")

        report = fertility.build_report(
            pin_match=False,
            english=["hello"],
            burmese=["က"],
            encoders=[
                fertility.Encoder("cl100k_base", fertility.EXTERNAL_CL100K, encode, "ready", None),
                fertility.Encoder("o200k_base", None, encode, "ready", None),
                fertility.Encoder(fertility.BASELINE_TOKENIZER, None, encode, "ready", None),
                fertility.Encoder("Extended Qwen3-0.6B", None, encode, "ready", None),
                fertility.Encoder("Gemma 4 E2B", None, encode, "ready", None),
            ],
            corpus={"pin_match": False, "pairs": None},
            qwen_tokenizer={"pin_match": False},
        )
        self.assertEqual(report["status"], "baseline not measured")
        self.assertIsNone(report["gate"])
        for row in report["rows"]:
            self.assertIsNone(row["measured_premium"])

    def test_premium_is_the_token_sum_ratio(self) -> None:
        def encode(text: str) -> list[int]:
            return [1] * len(text)

        encoders = [
            fertility.Encoder("cl100k_base", fertility.EXTERNAL_CL100K, None, "tokenizer was not available", None),
            fertility.Encoder("o200k_base", None, None, "tokenizer was not available", None),
            fertility.Encoder(fertility.BASELINE_TOKENIZER, None, encode, "fixture encoder", "abc"),
            fertility.Encoder("Extended Qwen3-0.6B", None, None, "not built", None),
            fertility.Encoder("Gemma 4 E2B", None, None, "tokenizer file was not local", None),
        ]
        report = fertility.build_report(
            pin_match=True,
            english=["ab", "c"],
            burmese=["abcd", "ef"],
            encoders=encoders,
            corpus={"pin_match": True, "pairs": 2},
            qwen_tokenizer={"pin_match": True},
        )
        baseline = report["rows"][2]
        self.assertEqual(baseline["burmese_tokens"], 6)
        self.assertEqual(baseline["english_tokens"], 3)
        self.assertEqual(baseline["measured_premium"], "6/3")
        self.assertEqual(
            baseline["measured_premium"],
            f"{baseline['burmese_tokens']}/{baseline['english_tokens']}",
        )
        self.assertEqual(report["status"], "measured")
        self.assertEqual(report["gate"]["decision"], "skip")
        self.assertIsNone(report["gate"]["extended_premium"])
        self.assertIsNone(report["rows"][0]["measured_premium"])
        self.assertIsNone(report["rows"][3]["measured_premium"])
        self.assertIsNone(report["rows"][4]["measured_premium"])
        self.assertNotEqual(baseline["measured_premium"], "11.70")
        assert_no_banned_text(self, json.dumps(report))

    def test_premium_under_threshold_skips_extension(self) -> None:
        def encode(text: str) -> list[int]:
            return [1] * len(text)

        report = fertility.build_report(
            pin_match=True,
            english=["abcd"],
            burmese=["xy"],
            encoders=[
                fertility.Encoder(fertility.BASELINE_TOKENIZER, None, encode, "fixture encoder", None)
            ],
            corpus={"pin_match": True},
            qwen_tokenizer={"pin_match": True},
        )
        self.assertEqual(report["rows"][0]["measured_premium"], "2/4")
        self.assertEqual(report["gate"]["decision"], "skip")
        self.assertEqual(report["status"], "measured")

    def test_threshold_boundary_extends(self) -> None:
        def encode(text: str) -> list[int]:
            return [1] * len(text)

        report = fertility.build_report(
            pin_match=True,
            english=["ab"],
            burmese=["abcdef"],
            encoders=[
                fertility.Encoder(fertility.BASELINE_TOKENIZER, None, encode, "fixture encoder", None)
            ],
            corpus={"pin_match": True},
            qwen_tokenizer={"pin_match": True},
        )
        self.assertEqual(report["rows"][0]["measured_premium"], "6/2")
        self.assertEqual(report["gate"]["decision"], "extend")

    def test_zero_english_tokens_leave_the_cell_empty(self) -> None:
        report = fertility.build_report(
            pin_match=True,
            english=["ab"],
            burmese=["cd"],
            encoders=[
                fertility.Encoder(fertility.BASELINE_TOKENIZER, None, lambda _text: [], "fixture encoder", None)
            ],
            corpus={"pin_match": True},
            qwen_tokenizer={"pin_match": True},
        )
        self.assertEqual(report["status"], "baseline not measured")
        self.assertIsNone(report["rows"][0]["measured_premium"])
        self.assertEqual(report["rows"][0]["english_tokens"], 0)

    def test_other_tokenizer_does_not_clear_the_baseline(self) -> None:
        def encode(text: str) -> list[int]:
            return [1] * len(text)

        report = fertility.build_report(
            pin_match=True,
            english=["ab"],
            burmese=["abcd"],
            encoders=[
                fertility.Encoder("cl100k_base", fertility.EXTERNAL_CL100K, encode, "fixture encoder", None),
                fertility.Encoder(fertility.BASELINE_TOKENIZER, None, None, "tokenizer file was not local", None),
            ],
            corpus={"pin_match": True},
            qwen_tokenizer={"pin_match": False},
        )
        self.assertEqual(report["status"], "baseline not measured")
        self.assertEqual(report["rows"][0]["measured_premium"], "4/2")
        self.assertIsNone(report["rows"][1]["measured_premium"])
        self.assertIsNone(report["gate"])

    def test_source_keeps_petrov_out_of_measured_cells(self) -> None:
        source = Path(fertility.__file__).read_text(encoding="utf-8")
        assert_no_banned_text(self, source)
        self.assertIn("11.70", source)
        self.assertNotIn('"measured_premium": "11.70"', source)
        self.assertEqual(source.count("11.70"), 1)

    def test_module_entrypoint_exits_zero_when_corpus_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            proc = subprocess.run(
                [sys.executable, "-m", "burmeseai.fertility", "--out", str(out)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout, out.read_text(encoding="utf-8"))
        report = json.loads(proc.stdout)
        self.assertEqual(report["status"], "baseline not measured")
        for row in report["rows"]:
            self.assertIsNone(row["measured_premium"])
        assert_no_banned_text(self, proc.stdout)


if __name__ == "__main__":
    unittest.main()

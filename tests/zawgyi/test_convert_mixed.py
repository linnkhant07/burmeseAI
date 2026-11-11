"""Hand-built mixed lines for convert_mixed.

Expected Unicode was stored after a local run of convertLine from
github.com/arckara/zawgyi at e66860c30b616227b070ab12cb9e995dd99917fb.
The span path has to match those strings. A whole-line probability above
0.95 is computed only here, as the comparison those mixed rows are meant
to beat.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from burmeseai import zawgyi as zawgyi_mod
from burmeseai.zawgyi import PINNED_COMMIT, convert_mixed

HERE = Path(__file__).resolve().parent
FIXTURE = json.loads((HERE / "fixtures.json").read_text(encoding="utf-8"))
WHOLE_LINE_THRESHOLD = 0.95

# sha256 of the vendored files at the pinned commit.
PINNED_SHA256 = {
    "NOTICE": "bcd7693e6e869c26779b3a2508dbb7db8b3da13136a16912df4e499f48408eb3",
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "src/convert.mjs": "db766f099203c47e40cacc1890db460b525b1a91643900d30d861abc73e8c5fa",
    "src/detector.mjs": "58ec460e2624b1eccdf9e0672c60af4b9d5cace315019875004b40737318f6e1",
    "src/signatures.mjs": "9e4c039fbac5975bcb31d5fbc1bcf56bcac57e99d53c6cc1c65e89638eb6eb13",
    "src/transliterate.mjs": "17d9a6a13eec9a5c2002eb8192ee74c40c8e3803128e701ec54a73c6ccc0994f",
    "third_party/myanmar-tools/LICENSE.md": "3bda6435719550f9eba4837f8ef67883d21228e97f25feca566a5a8301abd9d4",
    "third_party/myanmar-tools/U2Z.js": "0e094382ab70b15e2a36d417c833a1b0b85122117c52d0bcb99e556b6f6b7686",
    "third_party/myanmar-tools/Z2U.js": "4ee41f93899d1fbb07795c3ddadfaad039ff840a1cbb46caee6862d57e3ca205",
    "third_party/myanmar-tools/zawgyiUnicodeModel.dat": "7df7bdf7c7bf2dbc83ebbbb6f75ce7b6a1886cc64147b778176a03a803241b8a",
}


def _node_other_than_20() -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        for name in ("node", "nodejs"):
            candidate = Path(directory) / name
            if not candidate.is_file() or not os.access(candidate, os.X_OK):
                continue
            version = subprocess.check_output([str(candidate), "--version"], text=True).strip()
            if not version.startswith("v20."):
                return str(candidate)
    return None


def _row(name: str) -> dict:
    for row in FIXTURE["rows"]:
        if row["name"] == name:
            return row
    raise KeyError(name)


def _engine_report(lines: list[str]) -> list[dict]:
    proc = subprocess.run(
        [
            zawgyi_mod._node20_executable(),
            str(HERE / "whole_line_compare.mjs"),
            str(zawgyi_mod._ENGINE),
        ],
        input=json.dumps(
            {"threshold": WHOLE_LINE_THRESHOLD, "lines": lines},
            ensure_ascii=False,
        ).encode("utf-8"),
        capture_output=True,
        check=False,
        timeout=60,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(err or f"compare helper exited {proc.returncode}")
    return json.loads(proc.stdout.decode("utf-8"))


class ConvertMixedTests(unittest.TestCase):
    def test_pinned_commit_and_notice(self) -> None:
        self.assertEqual(PINNED_COMMIT, "e66860c30b616227b070ab12cb9e995dd99917fb")
        self.assertEqual(FIXTURE["commit"], PINNED_COMMIT)
        notice = (zawgyi_mod._ENGINE / "NOTICE").read_text(encoding="utf-8")
        self.assertTrue(notice.startswith("zawgyi\nCopyright 2026 Arckara\n"))
        self.assertIn("Apache License", notice)
        self.assertIn("Google LLC", notice)
        self.assertIn("myanmar-tools", notice)
        for rel, digest in PINNED_SHA256.items():
            got = hashlib.sha256((zawgyi_mod._ENGINE / rel).read_bytes()).hexdigest()
            self.assertEqual(got, digest, rel)

    def test_node_20(self) -> None:
        version = subprocess.check_output(
            [zawgyi_mod._node20_executable(), "--version"],
            text=True,
        ).strip()
        self.assertRegex(version, r"^v20\.")

    def test_other_node_major_is_refused(self) -> None:
        other = _node_other_than_20()
        if other is None:
            self.skipTest("no non-20 node on PATH")
        previous = os.environ.get("BURMESEAI_NODE")
        os.environ["BURMESEAI_NODE"] = other
        try:
            with self.assertRaises(RuntimeError):
                zawgyi_mod._node20_executable()
        finally:
            if previous is None:
                os.environ.pop("BURMESEAI_NODE", None)
            else:
                os.environ["BURMESEAI_NODE"] = previous

    def test_product_source_has_no_whole_line_threshold(self) -> None:
        source = Path(zawgyi_mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("0.95", source)
        self.assertIn("convertLine", source)

    def test_span_path_matches_stored_unicode(self) -> None:
        reports = _engine_report([row["input"] for row in FIXTURE["rows"]])
        self.assertEqual(len(reports), len(FIXTURE["rows"]))
        for row, report in zip(FIXTURE["rows"], reports):
            with self.subTest(row=row["name"]):
                self.assertEqual(report["text"], row["expected"])
                self.assertEqual(report["label"], row["label"])
                self.assertEqual(report["words"], row["words"])
                self.assertEqual(convert_mixed(row["input"]), row["expected"])
                if row["beats_whole_line"]:
                    self.assertNotEqual(report["whole"], row["expected"])
                    self.assertNotEqual(convert_mixed(row["input"]), report["whole"])
                else:
                    self.assertEqual(report["whole"], row["expected"])

    def test_spaceless_burmese_line_is_one_span(self) -> None:
        greeting = _row("spaceless-greeting")
        thanks = _row("spaceless-zawgyi")
        jammed = _row("spaceless-jammed")
        self.assertEqual(jammed["input"], greeting["input"] + thanks["input"])
        self.assertIsNone(re.search(r"\s", jammed["input"]))
        self.assertEqual(jammed["words"], 1)
        one_span = convert_mixed(jammed["input"])
        split_spans = convert_mixed(greeting["input"]) + convert_mixed(thanks["input"])
        self.assertEqual(one_span, jammed["expected"])
        self.assertNotEqual(one_span, split_spans)
        for row in FIXTURE["rows"]:
            if not row["one_span"]:
                continue
            with self.subTest(row=row["name"]):
                self.assertIsNone(re.search(r"\s", row["input"]))
                self.assertEqual(row["words"], 1)
                self.assertEqual(convert_mixed(row["input"]), row["expected"])

    def test_line_breaks_stay_outside_the_span(self) -> None:
        first = _row("zawgyi-word-inside-unicode")
        second = _row("spaceless-greeting")
        text = first["input"] + "\r\n" + second["input"] + "\n"
        expected = first["expected"] + "\r\n" + second["expected"] + "\n"
        self.assertEqual(convert_mixed(text), expected)
        self.assertEqual(convert_mixed("\n"), "\n")
        self.assertEqual(convert_mixed(""), "")
        self.assertEqual(convert_mixed("   "), "   ")

    def test_text_must_be_a_string(self) -> None:
        with self.assertRaises(TypeError):
            convert_mixed(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

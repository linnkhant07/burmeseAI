"""Mixed-line Zawgyi to Unicode.

``convert_mixed`` calls ``convertLine`` from github.com/arckara/zawgyi at
commit e66860c30b616227b070ab12cb9e995dd99917fb. The process is Node.js 20.
The files under ``vendor/arckara-zawgyi`` are that commit, including its
Apache-2.0 NOTICE.

Each line is passed to ``convertLine`` whole. A line with no spaces is one
span. The wrapper does not decide a line by a single probability cutoff.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

PINNED_COMMIT = "e66860c30b616227b070ab12cb9e995dd99917fb"

_ENGINE = Path(__file__).resolve().parent / "vendor" / "arckara-zawgyi"
_LINE_BREAK = re.compile(r"(\r\n|\n|\r)")
_CACHED_NODE20: str | None = None

_RUNNER = r"""
import { convertLine } from {source};
import fs from "node:fs";

const lines = JSON.parse(fs.readFileSync(0, "utf8"));
if (!Array.isArray(lines)) {
  process.stderr.write("expected a JSON array of lines\n");
  process.exit(2);
}
const out = [];
for (const line of lines) {
  if (typeof line !== "string") {
    process.stderr.write("expected a JSON array of strings\n");
    process.exit(2);
  }
  out.push(convertLine(line).text);
}
process.stdout.write(JSON.stringify(out));
"""


def convert_mixed(text: str) -> str:
    """Return Unicode text, converting Zawgyi spans inside each line."""
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    if text == "":
        return ""
    parts = _LINE_BREAK.split(text)
    lines = parts[0::2]
    breaks = parts[1::2]
    converted = _convert_lines(lines)
    out: list[str] = []
    for i, line in enumerate(converted):
        out.append(line)
        if i < len(breaks):
            out.append(breaks[i])
    return "".join(out)


def _convert_lines(lines: list[str]) -> list[str]:
    entry = _ENGINE / "src" / "convert.mjs"
    notice = _ENGINE / "NOTICE"
    if not entry.is_file() or not notice.is_file():
        raise RuntimeError(
            "pinned zawgyi convertLine is missing from vendor/arckara-zawgyi"
        )
    script = _RUNNER.replace("{source}", json.dumps(entry.as_uri()))
    proc = subprocess.run(
        [_node20_executable(), "--input-type=module", "-e", script],
        input=json.dumps(lines, ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        check=False,
        timeout=60,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(err or f"convertLine exited {proc.returncode}")
    result = json.loads(proc.stdout.decode("utf-8"))
    if (
        not isinstance(result, list)
        or len(result) != len(lines)
        or not all(isinstance(item, str) for item in result)
    ):
        raise RuntimeError("convertLine returned an unexpected payload")
    return result


def _node20_executable() -> str:
    override = os.environ.get("BURMESEAI_NODE")
    if override:
        major = _node_major(override)
        if major != 20:
            raise RuntimeError(f"convertLine requires Node.js 20, not {override}")
        return override
    global _CACHED_NODE20
    if _CACHED_NODE20 is not None:
        return _CACHED_NODE20
    for candidate in _node_candidates():
        if _node_major(candidate) == 20:
            _CACHED_NODE20 = candidate
            return candidate
    raise RuntimeError("convertLine requires Node.js 20")


def _node_candidates() -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        if not path.is_file() or not os.access(path, os.X_OK):
            return
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        found.append(key)

    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        for name in ("node", "nodejs"):
            add(Path(directory) / name)
    nvm = Path.home() / ".nvm" / "versions" / "node"
    if nvm.is_dir():
        for exe in sorted(nvm.glob("v20.*/bin/node")):
            add(exe)
    return found


def _node_major(executable: str) -> int | None:
    try:
        proc = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    match = re.match(r"v(\d+)\.", proc.stdout.strip())
    if match is None:
        return None
    return int(match.group(1))

"""Frozen HumanEval subset, pass@1, one completion per problem.

Completions run in a subprocess. A timeout or a bad exit is a failed problem,
which is a measured miss for that item. The subset itself is fixed.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from burmeseai.eval.frozen_humaneval import FROZEN_HUMANEVAL_TASK_IDS
from burmeseai.eval.io_util import read_records

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def load(data_dir: Path) -> list[dict] | None:
    path = data_dir / "humaneval" / "HumanEval.jsonl"
    rows = read_records(path)
    if rows is None:
        rows = read_records(data_dir / "humaneval" / "HumanEval.json")
    if rows is None:
        return None
    by_id = {}
    for index, row in enumerate(rows):
        task_id = str(row.get("task_id", ""))
        if task_id not in FROZEN_HUMANEVAL_TASK_IDS:
            continue
        for field in ("prompt", "test", "entry_point"):
            if field not in row:
                raise ValueError(f"HumanEval row {index} is missing {field}")
        if not _IDENTIFIER.fullmatch(str(row["entry_point"])):
            raise ValueError(f"HumanEval row {index} has a bad entry point")
        by_id[task_id] = {
            "id": task_id,
            "prompt": str(row["prompt"]),
            "test": str(row["test"]),
            "entry_point": str(row["entry_point"]),
        }
    if not by_id:
        return None
    return [by_id[task_id] for task_id in FROZEN_HUMANEVAL_TASK_IDS if task_id in by_id]


def subset_complete(items: list[dict]) -> bool:
    found = {item["id"] for item in items}
    return found == set(FROZEN_HUMANEVAL_TASK_IDS)


def prompt(item: dict) -> str:
    return (
        "Complete the Python function. Reply with the function only.\n\n"
        f"{item['prompt']}"
    )


def strip_completion(problem_prompt: str, prediction: str) -> str:
    match = _FENCE.search(prediction)
    text = match.group(1) if match else prediction
    if text.startswith(problem_prompt):
        text = text[len(problem_prompt) :]
    return text


def program_passes(item: dict, prediction: str, timeout: float = 5.0) -> bool:
    if len(prediction) > 100_000:
        return False
    completion = strip_completion(item["prompt"], prediction)
    program = (
        f"{item['prompt']}{completion}\n"
        f"{item['test']}\n"
        f"check({item['entry_point']})\n"
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.py"
            path.write_text(program, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(path)],
                cwd=tmp,
                capture_output=True,
                timeout=timeout,
                env={
                    "PATH": "",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                check=False,
            )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


def score(items: list[dict], predictions: dict[str, str], timeout: float = 5.0) -> float | None:
    if not subset_complete(items):
        return None
    if any(item["id"] not in predictions for item in items):
        return None
    ordered = [item for item in items if item["id"] in FROZEN_HUMANEVAL_TASK_IDS]
    hits = sum(
        program_passes(item, predictions[item["id"]], timeout=timeout) for item in ordered
    )
    return 100.0 * hits / len(ordered)


if __name__ == "__main__":
    from burmeseai.eval.driver import main

    sys.exit(main(["run", "--task", "humaneval", *sys.argv[1:]]))

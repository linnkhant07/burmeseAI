"""GSM8K test exact match, 0-shot, thinking off.

The gold number is the last #### number in the reference, with commas removed.
The prediction uses the same marker when it is present, otherwise its last number.
The strings must be equal. 1,000 matches 1000. 1000.0 does not match 1000.
"""

from __future__ import annotations

import sys
from pathlib import Path

from burmeseai.eval.io_util import read_records
from burmeseai.eval.scoring import numbers_match

def load(data_dir: Path) -> list[dict] | None:
    rows = read_records(data_dir / "gsm8k" / "test.jsonl")
    if rows is None:
        rows = read_records(data_dir / "gsm8k" / "test.json")
    if rows is None:
        return None
    items = []
    for index, row in enumerate(rows):
        if "question" not in row or "answer" not in row:
            raise ValueError(f"GSM8K row {index} is missing question or answer")
        items.append(
            {
                "id": str(row.get("id", index)),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
            }
        )
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate GSM8K ids")
    return items


def prompt(item: dict) -> str:
    return (
        f"{item['question']}\n\n"
        "Solve the problem. End with a line of the form #### <number>."
    )


def score(items: list[dict], predictions: dict[str, str]) -> float | None:
    if not items or any(item["id"] not in predictions for item in items):
        return None
    hits = sum(
        numbers_match(item["answer"], predictions[item["id"]]) for item in items
    )
    return 100.0 * hits / len(items)


if __name__ == "__main__":
    from burmeseai.eval.driver import main

    sys.exit(main(["run", "--task", "gsm8k", *sys.argv[1:]]))

"""Belebele mya_Mymr: 900 four-choice questions, 0-shot, thinking off."""

from __future__ import annotations

import sys
from pathlib import Path

from burmeseai.eval.io_util import read_records
from burmeseai.eval.scoring import parse_choice

LETTERS = "ABCD"
FIELDS = (
    "link",
    "question_number",
    "flores_passage",
    "question",
    "mc_answer1",
    "mc_answer2",
    "mc_answer3",
    "mc_answer4",
    "correct_answer_num",
)


def data_path(data_dir: Path) -> Path:
    jsonl = data_dir / "belebele" / "mya_Mymr.jsonl"
    if jsonl.is_file():
        return jsonl
    return data_dir / "belebele" / "mya_Mymr.json"


def load(data_dir: Path) -> list[dict] | None:
    rows = read_records(data_path(data_dir))
    if rows is None:
        return None
    items = [_normalize(row, index) for index, row in enumerate(rows)]
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate Belebele ids")
    return items


def _normalize(row: dict, index: int) -> dict:
    missing = [field for field in FIELDS if field not in row]
    if missing:
        raise ValueError(f"Belebele row {index} missing {missing[0]}")
    dialect = row.get("dialect")
    if dialect is not None and str(dialect) != "mya_Mymr":
        raise ValueError(f"Belebele row {index} is not mya_Mymr")
    number = int(str(row["correct_answer_num"]))
    if number not in (1, 2, 3, 4):
        raise ValueError(f"Belebele row {index} has a bad answer number")
    item_id = f"{row['link']}::{row['question_number']}"
    return {
        "id": item_id,
        "passage": str(row["flores_passage"]),
        "question": str(row["question"]),
        "choices": [
            str(row["mc_answer1"]),
            str(row["mc_answer2"]),
            str(row["mc_answer3"]),
            str(row["mc_answer4"]),
        ],
        "gold": LETTERS[number - 1],
    }


def prompt(item: dict) -> str:
    letters = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(LETTERS, item["choices"])
    )
    return (
        f"{item['passage']}\n\n"
        f"{item['question']}\n"
        f"{letters}\n\n"
        "Reply with the letter of the correct answer."
    )


def is_correct(item: dict, prediction: str) -> bool:
    letter = parse_choice(prediction)
    if letter is not None:
        return letter == item["gold"]
    hits = [
        index
        for index, choice in enumerate(item["choices"])
        if choice and choice in prediction
    ]
    if len(hits) == 1:
        return LETTERS[hits[0]] == item["gold"]
    return False


def score(items: list[dict], predictions: dict[str, str]) -> float | None:
    if not items or any(item["id"] not in predictions for item in items):
        return None
    correct = sum(is_correct(item, predictions[item["id"]]) for item in items)
    return 100.0 * correct / len(items)


if __name__ == "__main__":
    from burmeseai.eval.driver import main

    sys.exit(main(["run", "--task", "belebele", *sys.argv[1:]]))

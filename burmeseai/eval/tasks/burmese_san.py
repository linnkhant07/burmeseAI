"""BURMESE-SAN: 3,920 items, seven tasks.

Accuracy covers QA, sentiment, toxicity, causal reasoning, and NLI.
Abstractive summarization uses character ROUGE-L F1.
Machine translation stays unmeasured: this process does not compute MetricX-24
and does not put another metric in that cell.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from burmeseai.eval.io_util import read_records
from burmeseai.eval.scoring import label_match, mean_rouge, span_match

TASK_NAMES = ("qa", "sa", "td", "cr", "nli", "as", "mt")
EXPECTED = {
    "qa": 120,
    "sa": 600,
    "td": 400,
    "cr": 400,
    "nli": 600,
    "as": 100,
    "mt": 1700,
}
EXPECTED_LABELS = {
    "sa": {"positive": 200, "negative": 200, "neutral": 200},
    "td": {"clean": 200, "hate": 200},
    "cr": {"cause": 200, "effect": 200},
    "nli": {"contradiction": 200, "entailment": 200, "neutral": 200},
}
MT_DIRECTIONS = {"eng_to_mya": 850, "mya_to_eng": 850}

_PROMPTS = {
    "qa": "Answer the question with the span from the passage and nothing else.",
    "sa": "Reply with one label: positive, negative, or neutral.",
    "td": "Reply with one label: clean or hate.",
    "cr": "Reply with one label: cause or effect.",
    "nli": "Reply with one label: contradiction, entailment, or neutral.",
    "as": "Write a one-sentence summary of the passage.",
    "mt": "Translate the sentence. Reply with the translation only.",
}


def load(data_dir: Path) -> list[dict] | None:
    directory = data_dir / "burmese-san"
    rows = read_records(directory / "items.jsonl")
    if rows is None:
        rows = read_records(directory / "items.json")
    if rows is None:
        return None
    items = [_normalize(row, index) for index, row in enumerate(rows)]
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate BURMESE-SAN ids")
    return items


def _normalize(row: dict, index: int) -> dict:
    task = str(row.get("task", "")).strip().lower()
    if task not in EXPECTED:
        raise ValueError(f"BURMESE-SAN row {index} has an unknown task")
    if "input" not in row or "label" not in row:
        raise ValueError(f"BURMESE-SAN row {index} is missing input or label")
    item = {
        "id": str(row.get("id", index)),
        "task": task,
        "input": str(row["input"]),
        "label": str(row["label"]),
        "choices": [str(choice) for choice in row.get("choices") or []],
        "direction": row.get("direction"),
    }
    if task == "mt":
        direction = str(item["direction"] or "")
        if direction not in MT_DIRECTIONS:
            raise ValueError(f"BURMESE-SAN row {index} has a bad MT direction")
        item["direction"] = direction
    return item


def prompt(item: dict) -> str:
    instruction = _PROMPTS[item["task"]]
    choices = ""
    if item["choices"]:
        rendered = ", ".join(item["choices"])
        choices = f"\nLabels: {rendered}"
    direction = ""
    if item["task"] == "mt":
        direction = f"\nDirection: {item['direction']}"
    return f"{instruction}{choices}{direction}\n\n{item['input']}"


def grouped(items: list[dict]) -> dict[str, list[dict]]:
    groups = {name: [] for name in TASK_NAMES}
    for item in items:
        groups[item["task"]].append(item)
    return groups


def task_ready(task: str, items: list[dict]) -> bool:
    if len(items) != EXPECTED[task]:
        return False
    labels = EXPECTED_LABELS.get(task)
    if labels is not None:
        found = Counter(_label_key(item["label"]) for item in items)
        if any(found.get(label, 0) != count for label, count in labels.items()):
            return False
    if task == "mt":
        directions = Counter(item["direction"] for item in items)
        if any(directions.get(name, 0) != count for name, count in MT_DIRECTIONS.items()):
            return False
    return True


def _label_key(label: str) -> str:
    return " ".join(label.split()).casefold()


def score_task(task: str, items: list[dict], predictions: dict[str, dict]) -> float | None:
    if task == "mt" or not items:
        return None
    if any(item["id"] not in predictions for item in items):
        return None
    if task == "as":
        pairs = [
            (str(predictions[item["id"]].get("prediction", "")), item["label"])
            for item in items
        ]
        return mean_rouge(pairs)
    if task == "qa":
        hits = sum(
            span_match(str(predictions[item["id"]].get("prediction", "")), item["label"])
            for item in items
        )
        return 100.0 * hits / len(items)
    hits = 0
    for item in items:
        text = str(predictions[item["id"]].get("prediction", ""))
        if label_match(text, item["label"], item["choices"] or None):
            hits += 1
    return 100.0 * hits / len(items)


if __name__ == "__main__":
    from burmeseai.eval.driver import main

    sys.exit(main(["run", "--task", "burmese_san", *sys.argv[1:]]))

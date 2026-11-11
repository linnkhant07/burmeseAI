"""JSONL readers. Prediction files are written by the driver, not here."""

from __future__ import annotations

import json
from pathlib import Path


def read_records(path: Path) -> list[dict] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list) or not payload:
            return None
        if not all(isinstance(row, dict) for row in payload):
            raise ValueError(f"{path} is not a list of objects")
        return payload
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        if not isinstance(row, dict):
            raise ValueError(f"{path} line {line_number} is not an object")
        rows.append(row)
    return rows or None

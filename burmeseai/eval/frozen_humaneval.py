"""Frozen HumanEval subset.

The ids are fixed: HumanEval/0, then every fifth problem, through HumanEval/155.
A later run scores this list and does not draw a new sample.
"""

from __future__ import annotations

FROZEN_HUMANEVAL_TASK_IDS: tuple[str, ...] = tuple(
    f"HumanEval/{index}" for index in range(0, 156, 5)
)

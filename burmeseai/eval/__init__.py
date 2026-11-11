"""Eval harness for Belebele, FLORES, BURMESE-SAN, GSM8K, and HumanEval.

A score is recorded only after a run writes predictions for a complete set.
Missing data, missing weights, and a short file stay "not measured".
"""

from burmeseai.eval.schema import NOT_MEASURED

__all__ = ["NOT_MEASURED"]

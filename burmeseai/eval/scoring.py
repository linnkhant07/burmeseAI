"""Score helpers and the label rules from the eval plan.

Label functions return a word only when both numbers were passed in.
They are not called while a report cell is still empty.
"""

from __future__ import annotations

import re

from burmeseai.eval.schema import NOT_MEASURED

_HASH_NUMBER = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
_ANY_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_LETTER = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
_LETTER_LINE = re.compile(r"[\(\[]?([ABCD])[\)\].:]?", re.IGNORECASE)


def gate(expected: int, found: int, score: float | None, metric: str) -> dict:
    """Publish a score only when the file has the full expected count."""
    if found != expected or score is None:
        cell = {"status": NOT_MEASURED, "score": None, "metric": metric}
        if found != expected:
            cell["reason"] = f"expected {expected} items, found {found}"
        return cell
    return {
        "status": "measured",
        "score": score,
        "metric": metric,
        "items": found,
    }


def protocol_ok(rows: list[dict]) -> bool:
    return all(
        row.get("enable_thinking") is False and row.get("shots") == 0 for row in rows
    )


def parse_choice(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        match = _LETTER_LINE.fullmatch(stripped)
        if match:
            return match.group(1).upper()
    match = _LETTER.search(text)
    if match:
        return match.group(1).upper()
    return None


def canonical_number(token: str) -> str:
    number = token.replace(",", "").strip()
    if number.startswith("+"):
        number = number[1:]
    return number


def gold_number(answer: str) -> str | None:
    matches = _HASH_NUMBER.findall(answer)
    if not matches:
        return None
    return canonical_number(matches[-1])


def predicted_number(text: str) -> str | None:
    matches = _HASH_NUMBER.findall(text)
    if matches:
        return canonical_number(matches[-1])
    numbers = _ANY_NUMBER.findall(text)
    if not numbers:
        return None
    return canonical_number(numbers[-1])


def numbers_match(gold: str, prediction: str) -> bool:
    expected = gold_number(gold)
    found = predicted_number(prediction)
    return expected is not None and found is not None and expected == found


def normalized_text(text: str) -> str:
    return " ".join(text.split())


def span_match(prediction: str, gold: str) -> bool:
    expected = normalized_text(str(gold))
    if not expected:
        return False
    if normalized_text(prediction) == expected:
        return True
    lines = [normalized_text(line) for line in prediction.splitlines()]
    return expected in lines


def label_match(prediction: str, gold: str, choices: list[str] | None = None) -> bool:
    expected = normalized_text(str(gold))
    found = normalized_text(prediction)
    if found == expected or found.casefold() == expected.casefold():
        return True
    first = prediction.splitlines()[0] if prediction.splitlines() else ""
    if normalized_text(first).casefold() == expected.casefold():
        return True
    if not choices:
        return False
    hits = []
    haystack = prediction.casefold()
    for choice in choices:
        needle = normalized_text(str(choice))
        if needle and needle.casefold() in haystack:
            hits.append(needle)
    if len(hits) != 1:
        return False
    return hits[0].casefold() == expected.casefold()


def rouge_l_f1(hypothesis: str, reference: str) -> float:
    """Character ROUGE-L F1 on a 0-100 scale, with whitespace removed."""
    hyp = list("".join(hypothesis.split()))
    ref = list("".join(reference.split()))
    if not hyp or not ref:
        return 0.0
    length = _lcs_length(hyp, ref)
    precision = length / len(hyp)
    recall = length / len(ref)
    denom = precision + recall
    if denom == 0.0:
        return 0.0
    return 100.0 * 2.0 * precision * recall / denom


def mean_rouge(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        raise ValueError("no pairs")
    return sum(rouge_l_f1(hyp, ref) for hyp, ref in pairs) / len(pairs)


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for char in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            if char == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def label_belebele(candidate: float | None, baseline: float | None) -> str:
    if candidate is None or baseline is None:
        return NOT_MEASURED
    if candidate >= baseline + 5.0 and candidate >= 30.0:
        return "success"
    return "miss"


def label_flores(
    candidate_en_my: float | None,
    candidate_my_en: float | None,
    baseline_en_my: float | None,
    baseline_my_en: float | None,
) -> str:
    values = (candidate_en_my, candidate_my_en, baseline_en_my, baseline_my_en)
    if any(value is None for value in values):
        return NOT_MEASURED
    assert candidate_en_my is not None and candidate_my_en is not None
    assert baseline_en_my is not None and baseline_my_en is not None
    if (
        candidate_en_my >= baseline_en_my + 2.0
        and candidate_my_en >= baseline_my_en + 2.0
    ):
        return "success"
    return "miss"


def label_macro(candidate: float | None, baseline: float | None) -> str:
    if candidate is None or baseline is None:
        return NOT_MEASURED
    if candidate >= baseline:
        return "success"
    return "miss"


def label_drop(
    candidate: float | None, baseline: float | None, max_drop: float = 3.0
) -> str:
    """A drop is baseline minus candidate. An improvement is within the cap."""
    if candidate is None or baseline is None:
        return NOT_MEASURED
    if baseline - candidate <= max_drop:
        return "success"
    return "miss"


def macro_average(
    task_scores: dict[str, float | None],
    task_names: tuple[str, ...],
    mt_normalized: float | None,
) -> float | None:
    """Mean of seven 0-100 scores. Raw MetricX-24 is not one of those scores."""
    values: list[float] = []
    for name in task_names:
        if name == "mt":
            if mt_normalized is None:
                return None
            values.append(mt_normalized)
            continue
        score = task_scores.get(name)
        if score is None:
            return None
        values.append(score)
    if len(values) != len(task_names) or not values:
        return None
    return sum(values) / len(values)

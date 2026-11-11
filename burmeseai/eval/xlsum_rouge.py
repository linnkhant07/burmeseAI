"""XL-Sum multilingual ROUGE-L for Burmese summaries.

The scorer has to be the XL-Sum RougeScorer with lang \"burmese\". If that
scorer is not installed, this module returns None and the task stays
unmeasured. The 0-100 value is SEA-HELM's normalize_score(fmeasure, 0, 1) * 100,
the generative baseline on a fraction whose native maximum is 1.
"""

from __future__ import annotations


def mean_rouge_l(pairs: list[tuple[str, str]]) -> float | None:
    scorer = _burmese_scorer()
    if scorer is None or not pairs:
        return None
    measures = []
    for hypothesis, reference in pairs:
        scored = scorer.score(reference, hypothesis)
        measures.append(float(scored["rougeL"].fmeasure))
    mean = sum(measures) / len(measures)
    return max((mean - 0.0) / 1.0, 0.0) * 100.0


def _burmese_scorer():
    for loader in (_from_multilingual_package, _from_rouge_score):
        scorer = loader()
        if scorer is not None and _uses_burmese(scorer):
            return scorer
    return None


def _from_multilingual_package():
    try:
        from multilingual_rouge_scoring.rouge_scorer import RougeScorer
    except ImportError:
        return None
    try:
        return RougeScorer(["rougeL"], use_stemmer=False, lang="burmese")
    except (TypeError, ImportError, LookupError):
        return None


def _from_rouge_score():
    try:
        from rouge_score.rouge_scorer import RougeScorer
    except ImportError:
        return None
    try:
        return RougeScorer(["rougeL"], use_stemmer=False, lang="burmese")
    except TypeError:
        return None


def _uses_burmese(scorer) -> bool:
    tokenizer = getattr(scorer, "_tokenizer", None)
    return getattr(tokenizer, "lang", None) == "burmese"

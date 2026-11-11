"""Corpus chrF++ on the sacrebleu chrF2++ setting.

Character n-grams of order 1 through 6, word n-grams of order 1 and 2,
beta 2, whitespace removed before the character n-grams. Counts are summed
across the corpus before the single F score. The result is on a 0-100 scale.
"""

from __future__ import annotations

from collections import Counter

CHAR_ORDER = 6
WORD_ORDER = 2
BETA = 2
_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def corpus_chrf_pp(hypotheses: list[str], references: list[str]) -> float:
    if len(hypotheses) != len(references):
        raise ValueError("hypothesis and reference lengths differ")
    if not hypotheses:
        raise ValueError("no sentences")
    width = (CHAR_ORDER + WORD_ORDER) * 3
    totals = [0] * width
    for hypothesis, reference in zip(hypotheses, references):
        stats = _segment_stats(hypothesis, reference)
        totals = [left + right for left, right in zip(totals, stats)]
    return _f_score(totals)


def _segment_stats(hypothesis: str, reference: str) -> list[int]:
    hyp_ngrams = _char_ngrams(hypothesis) + _word_ngrams(hypothesis)
    ref_ngrams = _char_ngrams(reference) + _word_ngrams(reference)
    stats: list[int] = []
    for hyp_counts, ref_counts in zip(hyp_ngrams, ref_ngrams):
        stats.extend(_match(hyp_counts, ref_counts))
    return stats


def _char_ngrams(text: str) -> list[Counter[str]]:
    compact = "".join(text.split())
    orders = []
    for n in range(1, CHAR_ORDER + 1):
        grams = [compact[i : i + n] for i in range(max(0, len(compact) - n + 1))]
        orders.append(Counter(grams))
    return orders


def _word_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for word in text.split():
        if len(word) == 1:
            tokens.append(word)
        elif word[-1] in _PUNCT:
            tokens.append(word[:-1])
            tokens.append(word[-1])
        elif word[0] in _PUNCT:
            tokens.append(word[0])
            tokens.append(word[1:])
        else:
            tokens.append(word)
    return tokens


def _word_ngrams(text: str) -> list[Counter[str]]:
    tokens = _word_tokens(text)
    orders = []
    for n in range(1, WORD_ORDER + 1):
        grams = [" ".join(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]
        orders.append(Counter(grams))
    return orders


def _match(hyp: Counter[str], ref: Counter[str]) -> list[int]:
    matched = 0
    hyp_total = 0
    for gram, count in hyp.items():
        hyp_total += count
        if gram in ref:
            matched += min(count, ref[gram])
    if not ref:
        return [0, 0, 0]
    return [hyp_total, sum(ref.values()), matched]


def _f_score(stats: list[int]) -> float:
    factor = BETA ** 2
    precision_sum = 0.0
    recall_sum = 0.0
    effective = 0
    order = CHAR_ORDER + WORD_ORDER
    for index in range(order):
        n_hyp, n_ref, n_match = stats[3 * index : 3 * index + 3]
        if n_hyp > 0 and n_ref > 0:
            precision_sum += n_match / n_hyp
            recall_sum += n_match / n_ref
            effective += 1
    if effective == 0:
        return 0.0
    precision = precision_sum / effective
    recall = recall_sum / effective
    denom = (factor * precision) + recall
    if denom == 0.0:
        return 0.0
    return 100.0 * (1.0 + factor) * precision * recall / denom

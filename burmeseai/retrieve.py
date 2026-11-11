"""BM25 over overlapping Unicode grapheme n-grams.

Widths are 3, 4, and 5. A grapheme here is an extended grapheme cluster:
base characters keep their combining marks, and a Myanmar virama keeps the
consonants it stacks. An empty index refuses. A top score below the cutoff
refuses. This module does not ship safety instructions, and it does not
fetch text.
"""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass

NGRAM_WIDTHS = (3, 4, 5)
K1 = 1.5
B = 0.75
# A top score below this floor is a miss. Equal to the floor is a hit.
# The floor is a product cutoff, not a measured recall figure. It refuses a
# shared trigram such as the overlap of "health" and "heat", and it keeps a
# full hit such as the Myanmar earthquake label.
SCORE_CUTOFF = 1.1

# Grapheme_Cluster_Break = SpacingMark in the Myanmar block. Other Mc marks
# in the block, including U+102B, U+102C, and U+1038, are Other.
_MYANMAR_SPACING_MARK = frozenset({0x1031, 0x103B, 0x103C, 0x1056, 0x1057, 0x1084})
# Indic_Conjunct_Break for the Myanmar block. Linker is U+1039 only.
_INCB_CONSONANT = (
    (0x1000, 0x102A),
    (0x103F, 0x103F),
    (0x1050, 0x1055),
    (0x105A, 0x105D),
    (0x1061, 0x1061),
    (0x1065, 0x1066),
    (0x106E, 0x1070),
    (0x1075, 0x1081),
    (0x108E, 0x108E),
)
_INCB_EXTEND = (
    (0x102D, 0x1030),
    (0x1032, 0x1037),
    (0x103A, 0x103A),
    (0x103D, 0x103E),
    (0x1058, 0x1059),
    (0x105E, 0x1060),
    (0x1071, 0x1074),
    (0x1082, 0x1082),
    (0x1085, 0x1086),
    (0x108D, 0x108D),
    (0x109D, 0x109D),
)
_MYANMAR_VIRAMA = "\u1039"


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    license: str
    topic: str = ""
    source: str = "project-written"

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("chunk id is required")
        if not isinstance(self.text, str):
            raise ValueError("chunk text must be a string")
        if not isinstance(self.license, str) or not self.license.strip():
            raise ValueError("chunk license is required")
        if not isinstance(self.topic, str) or not isinstance(self.source, str):
            raise ValueError("chunk topic and source must be strings")


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    text: str
    license: str
    score: float
    topic: str
    source: str


@dataclass(frozen=True)
class SearchResult:
    refused: bool
    reason: str | None
    hits: list[Hit]
    top_score: float | None


def grapheme_clusters(text: str) -> list[str]:
    """Split text into extended grapheme clusters.

    The rules used are the Unicode extended grapheme cluster boundaries that
    matter for Burmese and for ordinary letters: do not break before
    combining marks or spacing marks, do not break a carriage return from its
    line feed, keep Hangul jamo syllables together, pair regional indicators,
    and do not break a Myanmar virama from the consonants it joins.
    """
    text = unicodedata.normalize("NFC", text)
    if not text:
        return []
    chars = list(text)
    clusters: list[str] = []
    start = 0
    state = _incb_state_after("none", chars[0])
    ri_count = 1 if _is_regional_indicator(chars[0]) else 0
    for index in range(len(chars) - 1):
        current = chars[index]
        nxt = chars[index + 1]
        if _should_break(current, nxt, incb_state=state, ri_count=ri_count):
            clusters.append("".join(chars[start : index + 1]))
            start = index + 1
            state = _incb_state_after("none", nxt)
            ri_count = 1 if _is_regional_indicator(nxt) else 0
        else:
            state = _incb_state_after(state, nxt)
            ri_count = ri_count + 1 if _is_regional_indicator(nxt) else 0
    clusters.append("".join(chars[start:]))
    return clusters


def content_runs(text: str) -> list[list[str]]:
    """Grapheme runs used for n-grams, split on spaces and punctuation."""
    folded = unicodedata.normalize("NFC", text).casefold()
    folded = unicodedata.normalize("NFC", folded)
    runs: list[list[str]] = []
    current: list[str] = []
    for cluster in grapheme_clusters(folded):
        if _is_content(cluster):
            current.append(cluster)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def ngrams(text: str, widths: tuple[int, ...] = NGRAM_WIDTHS) -> list[str]:
    """Overlapping grapheme n-grams. Shorter than the smallest width yields none."""
    tokens: list[str] = []
    for run in content_runs(text):
        for width in widths:
            if width < 1 or len(run) < width:
                continue
            for index in range(len(run) - width + 1):
                tokens.append("".join(run[index : index + width]))
    return tokens


class Index:
    """In-memory BM25 index. Chunks are supplied by the caller."""

    def __init__(self, chunks: list[Chunk] | tuple[Chunk, ...] | None = None) -> None:
        self.chunks: list[Chunk] = []
        self._postings: dict[str, dict[int, int]] = {}
        self._doc_len: list[int] = []
        self._avgdl = 0.0
        if chunks:
            for chunk in chunks:
                self.add(chunk)

    def add(self, chunk: Chunk) -> None:
        if any(existing.id == chunk.id for existing in self.chunks):
            raise ValueError(f"duplicate chunk id {chunk.id}")
        tokens = ngrams(chunk.text)
        doc_index = len(self.chunks)
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            self._postings.setdefault(token, {})[doc_index] = count
        self._doc_len.append(len(tokens))
        self.chunks.append(chunk)
        self._avgdl = sum(self._doc_len) / len(self._doc_len)

    def search(self, query: str, top_k: int = 5) -> SearchResult:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self.chunks:
            return SearchResult(refused=True, reason="empty_index", hits=[], top_score=None)
        if self._avgdl == 0:
            return SearchResult(refused=True, reason="low_score", hits=[], top_score=0.0)
        query_tokens = ngrams(query)
        scored = [(self._score(query_tokens, index), index) for index in range(len(self.chunks))]
        scored.sort(key=lambda item: (-item[0], self.chunks[item[1]].id))
        top_score = scored[0][0]
        if top_score < SCORE_CUTOFF:
            return SearchResult(refused=True, reason="low_score", hits=[], top_score=top_score)
        hits: list[Hit] = []
        for score, index in scored:
            if score < SCORE_CUTOFF or len(hits) >= top_k:
                break
            chunk = self.chunks[index]
            hits.append(
                Hit(
                    chunk_id=chunk.id,
                    text=chunk.text,
                    license=chunk.license,
                    score=score,
                    topic=chunk.topic,
                    source=chunk.source,
                )
            )
        return SearchResult(refused=False, reason=None, hits=hits, top_score=top_score)

    def _score(self, query_tokens: list[str], doc_index: int) -> float:
        if not query_tokens:
            return 0.0
        doc_len = self._doc_len[doc_index]
        norm = K1 * (1.0 - B + B * doc_len / self._avgdl)
        score = 0.0
        for token in query_tokens:
            postings = self._postings.get(token)
            if not postings:
                continue
            tf = postings.get(doc_index, 0)
            if tf == 0:
                continue
            df = len(postings)
            idf = math.log(1.0 + (len(self.chunks) - df + 0.5) / (df + 0.5))
            score += idf * (tf * (K1 + 1.0)) / (tf + norm)
        return score


def load_chunks(path: str) -> list[Chunk]:
    """Read a local JSON list of chunks. This does not open a network path."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("index file must be a list of chunks")
    chunks: list[Chunk] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each chunk must be an object")
        try:
            chunks.append(
                Chunk(
                    id=item["id"],
                    text=item["text"],
                    license=item["license"],
                    topic=item.get("topic", ""),
                    source=item.get("source", "project-written"),
                )
            )
        except KeyError as exc:
            raise ValueError(f"chunk is missing {exc.args[0]}") from exc
        except TypeError as exc:
            raise ValueError("chunk fields must be strings") from exc
    return chunks


def _is_content(cluster: str) -> bool:
    for char in cluster:
        category = unicodedata.category(char)
        if category[0] in "LN" or category in ("Mn", "Mc", "Me"):
            return True
    return False


def _is_regional_indicator(char: str) -> bool:
    return 0x1F1E6 <= ord(char) <= 0x1F1FF


def _is_prepend(char: str) -> bool:
    code = ord(char)
    if 0x0600 <= code <= 0x0605 or 0x111C2 <= code <= 0x111C3 or 0x11A84 <= code <= 0x11A89:
        return True
    return code in {
        0x06DD,
        0x070F,
        0x08E2,
        0x0D4E,
        0x110BD,
        0x110CD,
        0x1193F,
        0x11941,
        0x11A3A,
        0x11D46,
    }


def _is_control(char: str) -> bool:
    if char in ("\r", "\n"):
        return True
    if char == "\u200d" or _is_prepend(char):
        return False
    return unicodedata.category(char) in ("Cc", "Cf", "Cs")


def _is_extend_or_zwj(char: str) -> bool:
    if char == "\u200d":
        return True
    if unicodedata.category(char) in ("Mn", "Me"):
        return True
    code = ord(char)
    return 0xFE00 <= code <= 0xFE0F or 0xE0100 <= code <= 0xE01EF


def _is_spacing_mark(char: str) -> bool:
    """True for Grapheme_Cluster_Break SpacingMark, not for every Mc."""
    return ord(char) in _MYANMAR_SPACING_MARK


def _in_ranges(code: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= code <= end for start, end in ranges)


def _hangul(char: str) -> str | None:
    code = ord(char)
    if 0x1100 <= code <= 0x115F or 0xA960 <= code <= 0xA97C:
        return "L"
    if 0x1160 <= code <= 0x11A7 or 0xD7B0 <= code <= 0xD7C6:
        return "V"
    if 0x11A8 <= code <= 0x11FF or 0xD7CB <= code <= 0xD7FB:
        return "T"
    if 0xAC00 <= code <= 0xD7A3:
        return "LV" if (code - 0xAC00) % 28 == 0 else "LVT"
    return None


def _incb(char: str) -> str:
    if char == _MYANMAR_VIRAMA:
        return "Linker"
    code = ord(char)
    if _in_ranges(code, _INCB_CONSONANT):
        return "Consonant"
    if _in_ranges(code, _INCB_EXTEND):
        return "Extend"
    return "None"


def _incb_state_after(state: str, char: str) -> str:
    kind = _incb(char)
    if kind == "Consonant":
        return "consonant"
    if kind == "Linker" and state in ("consonant", "link"):
        return "link"
    if kind == "Extend" and state in ("consonant", "link"):
        return state
    return "none"


def _should_break(current: str, nxt: str, *, incb_state: str, ri_count: int) -> bool:
    if current == "\r" and nxt == "\n":
        return False
    if _is_control(current) or _is_control(nxt):
        return True
    current_hangul = _hangul(current)
    next_hangul = _hangul(nxt)
    if current_hangul and next_hangul:
        if current_hangul == "L" and next_hangul in ("L", "V", "LV", "LVT"):
            return False
        if current_hangul in ("LV", "V") and next_hangul in ("V", "T"):
            return False
        if current_hangul in ("LVT", "T") and next_hangul == "T":
            return False
    if _is_extend_or_zwj(nxt):
        return False
    if _is_spacing_mark(nxt):
        return False
    if _is_prepend(current):
        return False
    if _is_regional_indicator(current) and _is_regional_indicator(nxt) and ri_count % 2 == 1:
        return False
    if _incb(nxt) == "Consonant" and incb_state == "link":
        return False
    return True

"""Offline CLI for a Burmese and English assistant.

Pipeline: mixed-line Zawgyi conversion, then a keyword and script router,
then either safety retrieval, translation, or chat. Safety retrieval quotes
a retrieved chunk or refuses. An empty index and a score below the cutoff
both refuse and show the disclaimer. This pack is not emergency care.

The product model call passes enable_thinking false. If the weights are
not on disk, the process says the model is not loaded and still refuses a
safety miss. Resident memory and tokens per second stay unmeasured until a
local generation writes them. This module does not invent those figures,
and it does not invent medical or emergency instructions.

burmeseai.zawgyi.convert_mixed is called when that module imports. The
wrapper is owned separately. When the import fails, the original string is
kept and the router still runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

from burmeseai.retrieve import Hit, Index, SearchResult, load_chunks

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_NOT_LOADED = "The model is not loaded."
DISCLAIMER = "This pack is not emergency care."
NOT_MEASURED = "not measured"

_LATIN_SAFETY = (
    "earthquakes",
    "earthquake",
    "aftershocks",
    "aftershock",
    "quakes",
    "quake",
    "tremors",
    "tremor",
    "seismic",
    "heatwaves",
    "heatwave",
    "healthcare",
    "unhealthy",
    "healthy",
    "medicines",
    "medicine",
    "medical",
    "health",
    "heat",
)
_MYANMAR_SAFETY = (
    "ငလျင်",
    "အပူလှိုင်း",
    "အပူ",
    "ကျန်းမာရေး",
    "ကျန်းမာ",
)
_MYANMAR_TRANSLATION = (
    "ဘာသာပြန်",
    "အင်္ဂလိပ်လို",
    "မြန်မာလို",
)


@dataclass(frozen=True)
class Conversion:
    original: str
    unicode: str
    converted: bool


@dataclass(frozen=True)
class Route:
    lane: str
    direction: str | None = None


@dataclass(frozen=True)
class Generation:
    text: str
    resident_memory: str | None = None
    tokens_per_second: str | None = None


@dataclass(frozen=True)
class Reply:
    original: str
    text: str
    converted: bool
    lane: str
    direction: str | None
    refused: bool
    reason: str | None
    disclaimer: str | None
    hits: list[Hit]
    reply: str
    model: str
    model_loaded: bool
    model_called: bool
    model_note: str | None
    enable_thinking: bool
    resident_memory: str
    tokens_per_second: str
    top_score: float | None

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "text": self.text,
            "converted": self.converted,
            "lane": self.lane,
            "direction": self.direction,
            "refused": self.refused,
            "reason": self.reason,
            "disclaimer": self.disclaimer,
            "hits": [
                {
                    "id": hit.chunk_id,
                    "text": hit.text,
                    "license": hit.license,
                    "score": hit.score,
                    "topic": hit.topic,
                    "source": hit.source,
                }
                for hit in self.hits
            ],
            "reply": self.reply,
            "model": self.model,
            "model_loaded": self.model_loaded,
            "model_called": self.model_called,
            "model_note": self.model_note,
            "enable_thinking": self.enable_thinking,
            "resident_memory": self.resident_memory,
            "tokens_per_second": self.tokens_per_second,
            "top_score": self.top_score,
        }


GenerateFn = Callable[..., Generation | None]


def load_convert_mixed():
    """Import burmeseai.zawgyi.convert_mixed.

    The mixed-line converter is burmeseai.zawgyi.convert_mixed. That module
    is optional at runtime. ImportError means the wrapper is not installed,
    and the caller keeps the original string. A runtime failure, including a
    missing Node.js 20, does the same so routing and refusal still run.
    """
    try:
        from burmeseai.zawgyi import convert_mixed
    except ImportError:
        return None
    return convert_mixed


def convert_text(text: str, converter: Callable[[str], str] | None = None) -> Conversion:
    """Convert Zawgyi when a converter is available. Keep the original beside it."""
    if converter is not None:
        return Conversion(original=text, unicode=converter(text), converted=True)
    convert_mixed = load_convert_mixed()
    if convert_mixed is None:
        return Conversion(original=text, unicode=text, converted=False)
    try:
        unicode_text = convert_mixed(text)
    except (RuntimeError, OSError):
        return Conversion(original=text, unicode=text, converted=False)
    return Conversion(original=text, unicode=unicode_text, converted=True)


def route(text: str) -> Route:
    """Pick safety, translation, or chat.

    Safety keywords win, in Latin or Myanmar script. Otherwise a translation
    keyword selects translation, and the script, or an explicit target
    phrase, selects the direction. Anything else is chat.
    """
    if _is_safety(text):
        return Route("safety", None)
    explicit = _explicit_direction(text)
    if _is_translation(text) or explicit is not None:
        if explicit is not None:
            direction = explicit
        elif _dominant_script(text) == "myanmar":
            direction = "my-en"
        else:
            direction = "en-my"
        return Route("translation", direction)
    return Route("chat", None)


def compose_reply(
    *,
    lane: str,
    refused: bool,
    model_loaded: bool,
    hits: list[Hit],
    generated: str | None,
) -> str:
    """Build the user-visible text from fixed lines and retrieved chunks."""
    parts: list[str] = []
    if not model_loaded:
        parts.append(MODEL_NOT_LOADED)
    if lane == "safety":
        if not refused:
            for hit in hits:
                parts.append(f"[{hit.chunk_id} | {hit.license}] {hit.text}")
        parts.append(DISCLAIMER)
    elif generated:
        parts.append(generated)
    return "\n".join(parts)


def respond(
    text: str,
    *,
    index: Index | None = None,
    weights: str | None = None,
    converter: Callable[[str], str] | None = None,
    generate: GenerateFn | None = None,
) -> Reply:
    """Run conversion, routing, and either retrieval or a model call."""
    conversion = convert_text(text, converter)
    decision = route(conversion.unicode)
    search = SearchResult(refused=False, reason=None, hits=[], top_score=None)
    if decision.lane == "safety":
        search = (index if index is not None else Index()).search(conversion.unicode)
    generated: Generation | None = None
    model_called = False
    if decision.lane != "safety":
        messages = _messages(decision, conversion.unicode)
        if generate is None:
            generated = default_generate(messages, weights, enable_thinking=False)
        else:
            generated = generate(messages, enable_thinking=False)
        model_called = generated is not None
    model_loaded = generated is not None
    disclaimer = DISCLAIMER if decision.lane == "safety" else None
    reply_text = compose_reply(
        lane=decision.lane,
        refused=search.refused,
        model_loaded=model_loaded,
        hits=search.hits,
        generated=None if generated is None else generated.text,
    )
    return Reply(
        original=conversion.original,
        text=conversion.unicode,
        converted=conversion.converted,
        lane=decision.lane,
        direction=decision.direction,
        refused=bool(search.refused) if decision.lane == "safety" else False,
        reason=search.reason if decision.lane == "safety" else None,
        disclaimer=disclaimer,
        hits=list(search.hits) if decision.lane == "safety" and not search.refused else [],
        reply=reply_text,
        model=MODEL_ID,
        model_loaded=model_loaded,
        model_called=model_called,
        model_note=None if model_loaded else MODEL_NOT_LOADED,
        enable_thinking=False,
        resident_memory=_measured(None if generated is None else generated.resident_memory),
        tokens_per_second=_measured(None if generated is None else generated.tokens_per_second),
        top_score=search.top_score if decision.lane == "safety" else None,
    )


def default_generate(
    messages: list[dict[str, str]],
    weights: str | None,
    *,
    enable_thinking: bool = False,
) -> Generation | None:
    """Load a local checkpoint and generate. Never downloads.

    enable_thinking is passed false on the product call. A missing directory,
    a missing library, or a failed local load returns None so the caller can
    say the model is not loaded. Memory and tokens per second are recorded
    only from this process after generate returns.
    """
    if enable_thinking:
        raise ValueError("product calls pass enable_thinking false")
    if not weights or not os.path.isdir(weights):
        return None
    try:
        import torch
        import transformers
    except ImportError:
        return None
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(weights, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(weights, local_files_only=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        templated = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        started = time.perf_counter()
        with torch.no_grad():
            output = model.generate(templated, max_new_tokens=128, do_sample=False)
        elapsed = time.perf_counter() - started
        new_tokens = int(output.shape[-1] - templated.shape[-1])
        text = tokenizer.decode(output[0][templated.shape[-1] :], skip_special_tokens=True).strip()
    except (OSError, RuntimeError, TypeError, ValueError, AttributeError):
        return None
    resident = _resident_memory_bytes()
    tokens_per_second = None
    if elapsed > 0 and new_tokens >= 0:
        tokens_per_second = f"{new_tokens / elapsed:.6f}"
    return Generation(
        text=text,
        resident_memory=None if resident is None else str(resident),
        tokens_per_second=tokens_per_second,
    )


def status_payload(weights: str | None = None) -> dict:
    """Measurement rows this process can honestly report.

    Status does not load weights and does not write a speed or memory figure.
    """
    return {
        "model": MODEL_ID,
        "weights_present": bool(weights and os.path.isdir(weights)),
        "model_loaded": False,
        "model_note": MODEL_NOT_LOADED,
        "enable_thinking": False,
        "resident_memory": NOT_MEASURED,
        "tokens_per_second": NOT_MEASURED,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline Burmese and English assistant.")
    parser.add_argument("text", nargs="?", help="User text. Read stdin when omitted.")
    parser.add_argument("--index", help="Local JSON file of retrieval chunks.")
    parser.add_argument("--weights", help="Local weights directory. Nothing is downloaded.")
    parser.add_argument("--json", action="store_true", help="Print a JSON object.")
    parser.add_argument("--status", action="store_true", help="Print measurement status and exit.")
    args = parser.parse_args(argv)
    if args.status:
        payload = status_payload(args.weights)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for key, value in payload.items():
                print(f"{key}: {str(value).lower() if isinstance(value, bool) else value}")
        return 0
    if args.text is None:
        text = sys.stdin.read()
    else:
        text = args.text
    if text.endswith("\n"):
        text = text[:-1]
    index = Index()
    if args.index:
        try:
            index = Index(load_chunks(args.index))
        except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            print(f"Could not read index: {exc}", file=sys.stderr)
            return 2
    reply = respond(text, index=index, weights=args.weights)
    if args.json:
        print(json.dumps(reply.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(reply.reply)
    return 0


def _messages(decision: Route, text: str) -> list[dict[str, str]]:
    if decision.lane == "translation":
        target = "English" if decision.direction == "my-en" else "Burmese"
        system = f"Translate the user message into {target}. Return only the translation."
    else:
        system = "You are a Burmese and English assistant. Do not give medical or emergency instructions."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


def _measured(value: str | None) -> str:
    return NOT_MEASURED if value is None else value


def _resident_memory_bytes() -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        return None
    return None


def _is_safety(text: str) -> bool:
    folded = text.casefold()
    for word in _LATIN_SAFETY:
        if _has_latin_word(folded, word):
            return True
    return any(word in text for word in _MYANMAR_SAFETY)


def _is_translation(text: str) -> bool:
    folded = text.casefold()
    if any(_has_latin_word(folded, word) for word in ("translate", "translation", "translated")):
        return True
    return any(word in text for word in _MYANMAR_TRANSLATION)


def _explicit_direction(text: str) -> str | None:
    folded = text.casefold()
    found: list[tuple[int, str]] = []
    for phrase, direction in (
        ("to english", "my-en"),
        ("into english", "my-en"),
        ("in english", "my-en"),
        ("to burmese", "en-my"),
        ("into burmese", "en-my"),
        ("in burmese", "en-my"),
    ):
        start = 0
        while True:
            index = folded.find(phrase, start)
            if index < 0:
                break
            if _latin_phrase_at(folded, index, phrase):
                found.append((index, direction))
            start = index + len(phrase)
    for phrase, direction in (("အင်္ဂလိပ်လို", "my-en"), ("မြန်မာလို", "en-my")):
        index = text.find(phrase)
        if index >= 0:
            found.append((index, direction))
    if not found:
        return None
    found.sort()
    return found[-1][1]


def _dominant_script(text: str) -> str:
    latin = 0
    myanmar = 0
    for char in text:
        folded = char.casefold()
        if "a" <= folded <= "z":
            latin += 1
        elif 0x1000 <= ord(char) <= 0x109F:
            myanmar += 1
    if myanmar > latin:
        return "myanmar"
    return "latin"


def _has_latin_word(folded: str, word: str) -> bool:
    start = 0
    while True:
        index = folded.find(word, start)
        if index < 0:
            return False
        if _latin_phrase_at(folded, index, word):
            return True
        start = index + len(word)


def _latin_phrase_at(folded: str, index: int, phrase: str) -> bool:
    before = folded[index - 1] if index > 0 else ""
    after_index = index + len(phrase)
    after = folded[after_index] if after_index < len(folded) else ""
    return not _is_latin_word_char(before) and not _is_latin_word_char(after)


def _is_latin_word_char(char: str) -> bool:
    return bool(char) and ("a" <= char <= "z" or "0" <= char <= "9")


if __name__ == "__main__":
    sys.exit(main())

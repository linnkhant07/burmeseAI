"""Burmese fertility against English on the pinned FLORES-200 devtest.

Run ``python -m burmeseai.fertility``. The premium is the sum of Burmese
tokens divided by the sum of English tokens on aligned sentences. A measured
cell is that ratio, written as the two totals. If the pinned devtest files or
the pinned Qwen tokenizer are absent, the status is ``baseline not measured``
and the process exits 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

PINNED_PAIRS = 1012
PINNED_ENGLISH_SHA256 = (
    "612e9fbe87997617c0fa8fa8929654a4f49b728d96738112c2b86ef6a1d78d88"
)
PINNED_BURMESE_SHA256 = (
    "bbccb36a909003dfbaf68cf59b63729769cab723a0f90a85f68efd9ded7b1512"
)
PINNED_QWEN_SHA256 = (
    "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
)
QWEN_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
ARCHIVE_SHA256 = (
    "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
)
ARCHIVE_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
EXTENSION_THRESHOLD = 3.0

EXTERNAL_CL100K = (
    "Petrov, La Malfa, Torr, and Bibi, NeurIPS 2023, Table 1, "
    "ChatGPT/GPT-4 column, Burmese 11.70, about 2,000 FLORES-200 sentences. "
    "External citation only."
)

DEFAULT_ENGLISH = Path("data/flores/devtest/eng_Latn.devtest")
DEFAULT_BURMESE = Path("data/flores/devtest/mya_Mymr.devtest")
DEFAULT_QWEN = Path("weights/Qwen3-0.6B/tokenizer.json")
DEFAULT_EXTENDED = Path("weights/Qwen3-0.6B-extended/tokenizer.json")
DEFAULT_GEMMA = Path("weights/gemma-4-E2B/tokenizer.json")

BASELINE_TOKENIZER = "Unmodified Qwen3-0.6B"
STATUS_NOT_MEASURED = "baseline not measured"
STATUS_MEASURED = "measured"

PREMIUM_DEFINITION = (
    "premium = burmese_tokens / english_tokens, the sum of tokens on the "
    "Burmese sentences divided by the sum of tokens on the aligned English "
    "sentences. measured_premium is that exact ratio written as "
    "burmese_tokens/english_tokens."
)


class Encoder(NamedTuple):
    name: str
    external: str | None
    encode: Callable[[str], list] | None
    detail: str
    tokenizer_sha256: str | None


def format_premium(burmese_tokens: int, english_tokens: int) -> str:
    return f"{burmese_tokens}/{english_tokens}"


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row(
    spec: Encoder,
    *,
    measured_premium: str | None,
    burmese_tokens: int | None,
    english_tokens: int | None,
    detail: str,
) -> dict:
    return {
        "tokenizer": spec.name,
        "external": spec.external,
        "measured_premium": measured_premium,
        "burmese_tokens": burmese_tokens,
        "english_tokens": english_tokens,
        "tokenizer_sha256": spec.tokenizer_sha256,
        "detail": detail,
    }


def build_report(
    *,
    pin_match: bool,
    english: list[str] | None,
    burmese: list[str] | None,
    encoders: list[Encoder],
    corpus: dict,
    qwen_tokenizer: dict,
) -> dict:
    rows = []
    baseline_tokens: tuple[int, int] | None = None
    for spec in encoders:
        if not pin_match or english is None or burmese is None:
            rows.append(
                _row(
                    spec,
                    measured_premium=None,
                    burmese_tokens=None,
                    english_tokens=None,
                    detail="not counted",
                )
            )
            continue
        if spec.encode is None:
            rows.append(
                _row(
                    spec,
                    measured_premium=None,
                    burmese_tokens=None,
                    english_tokens=None,
                    detail=spec.detail,
                )
            )
            continue
        english_tokens = sum(len(spec.encode(sentence)) for sentence in english)
        burmese_tokens = sum(len(spec.encode(sentence)) for sentence in burmese)
        if english_tokens <= 0:
            rows.append(
                _row(
                    spec,
                    measured_premium=None,
                    burmese_tokens=burmese_tokens,
                    english_tokens=english_tokens,
                    detail="english token total was zero",
                )
            )
            continue
        rows.append(
            _row(
                spec,
                measured_premium=format_premium(burmese_tokens, english_tokens),
                burmese_tokens=burmese_tokens,
                english_tokens=english_tokens,
                detail=spec.detail,
            )
        )
        if spec.name == BASELINE_TOKENIZER:
            baseline_tokens = (burmese_tokens, english_tokens)

    gate = None
    status = STATUS_NOT_MEASURED
    if baseline_tokens is not None:
        burmese_tokens, english_tokens = baseline_tokens
        ratio = burmese_tokens / english_tokens
        if ratio < EXTENSION_THRESHOLD:
            decision = "skip"
            reason = (
                f"measured premium {format_premium(burmese_tokens, english_tokens)} "
                "is under 3.0, so vocabulary extension is skipped"
            )
        else:
            decision = "extend"
            reason = (
                f"measured premium {format_premium(burmese_tokens, english_tokens)} "
                "is 3.0 or higher, so vocabulary extension applies"
            )
        status = STATUS_MEASURED
        gate = {
            "baseline_tokenizer": BASELINE_TOKENIZER,
            "threshold": "3.0",
            "decision": decision,
            "reason": reason,
            "measured_premium": format_premium(burmese_tokens, english_tokens),
            "extended_premium": None,
        }

    return {
        "status": status,
        "baseline_tokenizer": BASELINE_TOKENIZER,
        "premium_definition": PREMIUM_DEFINITION,
        "corpus": corpus,
        "qwen_tokenizer": qwen_tokenizer,
        "rows": rows,
        "gate": gate,
    }


def _read_pinned_sentences(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _tiktoken_encoder(name: str) -> Encoder:
    external = EXTERNAL_CL100K if name == "cl100k_base" else None
    try:
        import tiktoken
    except ImportError:
        return Encoder(name, external, None, "tokenizer was not available", None)
    try:
        encoding = tiktoken.get_encoding(name)
    except Exception:
        return Encoder(name, external, None, "tokenizer was not available", None)
    version = getattr(tiktoken, "__version__", "unknown")

    def encode(text: str) -> list[int]:
        return encoding.encode(text, disallowed_special=())

    detail = f"tiktoken {version} encoding {name}; disallowed_special empty"
    return Encoder(name, external, encode, detail, None)


def _hf_encoder(name: str, path: Path, *, missing_detail: str, pinned_sha256: str | None) -> Encoder:
    digest = file_sha256(path)
    if digest is None:
        return Encoder(name, None, None, missing_detail, None)
    if pinned_sha256 is not None and digest != pinned_sha256:
        return Encoder(
            name,
            None,
            None,
            "tokenizer file does not match the pinned Qwen3-0.6B tokenizer.json",
            digest,
        )
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return Encoder(name, None, None, "tokenizer library was not available", digest)
    try:
        tokenizer = Tokenizer.from_file(str(path))
    except Exception:
        return Encoder(name, None, None, "tokenizer file could not be loaded", digest)
    vocab_size = tokenizer.get_vocab_size()

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False).ids

    detail = (
        f"tokenizer.json sha256 {digest}; vocab_size {vocab_size}; "
        "add_special_tokens false"
    )
    if pinned_sha256 is not None:
        detail += f"; repo Qwen/Qwen3-0.6B; revision {QWEN_REVISION}"
    return Encoder(name, None, encode, detail, digest)


def _corpus_record(english_path: Path, burmese_path: Path, english_sha: str | None, burmese_sha: str | None, pin_match: bool) -> dict:
    return {
        "dataset": "FLORES-200",
        "split": "devtest",
        "configs": ["eng_Latn", "mya_Mymr"],
        "expected_pairs": PINNED_PAIRS,
        "source": (
            "Original FLORES-200 files flores200_dataset/devtest/eng_Latn.devtest "
            "and mya_Mymr.devtest from flores200_dataset.tar.gz ("
            + ARCHIVE_URL
            + "). That tarball is the original release linked from the "
            "facebook/flores card. This is not openlanguagedata/flores_plus."
        ),
        "pinned_archive_sha256": ARCHIVE_SHA256,
        "pinned_english_sha256": PINNED_ENGLISH_SHA256,
        "pinned_burmese_sha256": PINNED_BURMESE_SHA256,
        "english_path": str(english_path),
        "burmese_path": str(burmese_path),
        "observed_english_sha256": english_sha,
        "observed_burmese_sha256": burmese_sha,
        "pin_match": pin_match,
        "pairs": PINNED_PAIRS if pin_match else None,
    }


def _qwen_record(observed_sha: str | None, pin_match: bool) -> dict:
    return {
        "repo": "Qwen/Qwen3-0.6B",
        "revision": QWEN_REVISION,
        "file": "tokenizer.json",
        "pinned_sha256": PINNED_QWEN_SHA256,
        "observed_sha256": observed_sha,
        "pin_match": pin_match,
    }


def _encoders_for_run(
    *,
    pin_match: bool,
    qwen_path: Path,
    extended_path: Path,
    gemma_path: Path,
) -> tuple[list[Encoder], str | None, bool]:
    if not pin_match:
        encoders = [
            Encoder("cl100k_base", EXTERNAL_CL100K, None, "not counted", None),
            Encoder("o200k_base", None, None, "not counted", None),
            Encoder(BASELINE_TOKENIZER, None, None, "not counted", None),
            Encoder("Extended Qwen3-0.6B", None, None, "not counted", None),
            Encoder("Gemma 4 E2B", None, None, "not counted", None),
        ]
        return encoders, None, False

    qwen = _hf_encoder(
        BASELINE_TOKENIZER,
        qwen_path,
        missing_detail="tokenizer file was not local",
        pinned_sha256=PINNED_QWEN_SHA256,
    )
    encoders = [
        _tiktoken_encoder("cl100k_base"),
        _tiktoken_encoder("o200k_base"),
        qwen,
        _hf_encoder(
            "Extended Qwen3-0.6B",
            extended_path,
            missing_detail="not built",
            pinned_sha256=None,
        ),
        _hf_encoder(
            "Gemma 4 E2B",
            gemma_path,
            missing_detail="tokenizer file was not local",
            pinned_sha256=None,
        ),
    ]
    qwen_match = qwen.encode is not None and qwen.tokenizer_sha256 == PINNED_QWEN_SHA256
    return encoders, qwen.tokenizer_sha256, qwen_match


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english", type=Path, default=DEFAULT_ENGLISH)
    parser.add_argument("--burmese", type=Path, default=DEFAULT_BURMESE)
    parser.add_argument("--qwen-tokenizer", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--extended-tokenizer", type=Path, default=DEFAULT_EXTENDED)
    parser.add_argument("--gemma-tokenizer", type=Path, default=DEFAULT_GEMMA)
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    english_sha = file_sha256(args.english)
    burmese_sha = file_sha256(args.burmese)
    pin_match = (
        english_sha == PINNED_ENGLISH_SHA256 and burmese_sha == PINNED_BURMESE_SHA256
    )
    english: list[str] | None = None
    burmese: list[str] | None = None
    if pin_match:
        english = _read_pinned_sentences(args.english)
        burmese = _read_pinned_sentences(args.burmese)
        if len(english) != PINNED_PAIRS or len(burmese) != PINNED_PAIRS:
            pin_match = False
            english = None
            burmese = None

    encoders, qwen_sha, qwen_match = _encoders_for_run(
        pin_match=pin_match,
        qwen_path=args.qwen_tokenizer,
        extended_path=args.extended_tokenizer,
        gemma_path=args.gemma_tokenizer,
    )
    report = build_report(
        pin_match=pin_match,
        english=english,
        burmese=burmese,
        encoders=encoders,
        corpus=_corpus_record(
            args.english, args.burmese, english_sha, burmese_sha, pin_match
        ),
        qwen_tokenizer=_qwen_record(qwen_sha, qwen_match),
    )
    text = json.dumps(report, indent=2) + "\n"
    sys.stdout.write(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

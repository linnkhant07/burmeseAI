"""FLORES devtest chrF++, English and Burmese, both directions.

facebook/flores is published only when both files match the fertility
digests. Any other corpus has to be named in manifest.json. The two
direction scores stay separate.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from burmeseai.eval.chrf import corpus_chrf_pp

ENG_TO_MY = "eng_Latn_to_mya_Mymr"
MY_TO_ENG = "mya_Mymr_to_eng_Latn"
PINNED_CORPUS = "facebook/flores"
PINNED_PAIRS = 1012
PINNED_ENGLISH_SHA256 = (
    "612e9fbe87997617c0fa8fa8929654a4f49b728d96738112c2b86ef6a1d78d88"
)
PINNED_BURMESE_SHA256 = (
    "bbccb36a909003dfbaf68cf59b63729769cab723a0f90a85f68efd9ded7b1512"
)


def load(data_dir: Path) -> dict | None:
    root = data_dir / "flores" / "devtest"
    english_path = _language_file(root, "eng_Latn")
    burmese_path = _language_file(root, "mya_Mymr")
    if english_path is None or burmese_path is None:
        return None
    english = english_path.read_text(encoding="utf-8").splitlines()
    burmese = burmese_path.read_text(encoding="utf-8").splitlines()
    if len(english) != len(burmese):
        raise ValueError("eng_Latn and mya_Mymr line counts differ")
    english_sha = _sha256(english_path)
    burmese_sha = _sha256(burmese_path)
    pin_match = (
        english_sha == PINNED_ENGLISH_SHA256 and burmese_sha == PINNED_BURMESE_SHA256
    )
    named = _named_corpus(data_dir)
    if pin_match:
        corpus = PINNED_CORPUS
    elif named and named != PINNED_CORPUS:
        corpus = named
    else:
        corpus = None
    pairs = [
        {"id": str(index), "eng_Latn": eng, "mya_Mymr": mya}
        for index, (eng, mya) in enumerate(zip(english, burmese))
    ]
    return {
        "corpus": corpus,
        "pairs": pairs,
        "pin_match": pin_match,
        "english_sha256": english_sha,
        "burmese_sha256": burmese_sha,
    }


def can_publish(loaded: dict) -> bool:
    """Direction scores are published only for the pin or a named other corpus."""
    if len(loaded["pairs"]) != PINNED_PAIRS:
        return False
    if loaded["pin_match"] and loaded["corpus"] == PINNED_CORPUS:
        return True
    corpus = loaded["corpus"]
    return bool(corpus) and corpus != PINNED_CORPUS


def _named_corpus(data_dir: Path) -> str | None:
    manifest = data_dir / "flores" / "manifest.json"
    if not manifest.is_file():
        return None
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    name = str(payload.get("corpus", "")).strip()
    if not name:
        raise ValueError("flores manifest has no corpus name")
    return name


def _language_file(root: Path, stem: str) -> Path | None:
    for name in (f"{stem}.devtest", stem):
        path = root / name
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt(source: str, target_name: str) -> str:
    return f"Translate the following sentence into {target_name}.\n{source}"


def prompts_for_pair(pair: dict) -> list[tuple[str, str, str]]:
    return [
        (
            pair["id"],
            ENG_TO_MY,
            prompt(pair["eng_Latn"], "Burmese"),
        ),
        (
            pair["id"],
            MY_TO_ENG,
            prompt(pair["mya_Mymr"], "English"),
        ),
    ]


def score(pairs: list[dict], predictions: dict[tuple[str, str], str]) -> dict | None:
    if not pairs:
        return None
    eng_my = []
    my_eng = []
    for pair in pairs:
        to_my = predictions.get((pair["id"], ENG_TO_MY))
        to_en = predictions.get((pair["id"], MY_TO_ENG))
        if to_my is None or to_en is None:
            return None
        eng_my.append(to_my)
        my_eng.append(to_en)
    return {
        ENG_TO_MY: corpus_chrf_pp(eng_my, [pair["mya_Mymr"] for pair in pairs]),
        MY_TO_ENG: corpus_chrf_pp(my_eng, [pair["eng_Latn"] for pair in pairs]),
    }


if __name__ == "__main__":
    from burmeseai.eval.driver import main

    sys.exit(main(["run", "--task", "flores", *sys.argv[1:]]))

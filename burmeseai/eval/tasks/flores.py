"""FLORES devtest chrF++, English and Burmese, both directions.

The pinned corpus is facebook/flores, configs eng_Latn and mya_Mymr, 1,012
devtest pairs. A manifest may name a different corpus. That name is copied
into the report. The pair count stays 1,012.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from burmeseai.eval.chrf import corpus_chrf_pp

ENG_TO_MY = "eng_Latn_to_mya_Mymr"
MY_TO_ENG = "mya_Mymr_to_eng_Latn"
PINNED_CORPUS = "facebook/flores"


def load(data_dir: Path) -> dict | None:
    root = data_dir / "flores" / "devtest"
    english = _read_lines(root / "eng_Latn")
    burmese = _read_lines(root / "mya_Mymr")
    if english is None or burmese is None:
        return None
    if len(english) != len(burmese):
        raise ValueError("eng_Latn and mya_Mymr line counts differ")
    pairs = [
        {"id": str(index), "eng_Latn": eng, "mya_Mymr": mya}
        for index, (eng, mya) in enumerate(zip(english, burmese))
    ]
    return {"corpus": _corpus_name(data_dir), "pairs": pairs}


def _corpus_name(data_dir: Path) -> str:
    manifest = data_dir / "flores" / "manifest.json"
    if not manifest.is_file():
        return PINNED_CORPUS
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    name = str(payload.get("corpus", "")).strip()
    if not name:
        raise ValueError("flores manifest has no corpus name")
    return name


def _read_lines(path: Path) -> list[str] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    return path.read_text(encoding="utf-8").splitlines()


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

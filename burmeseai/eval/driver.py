"""One driver for the six local models and five eval tasks.

`report` reads files and never writes a prediction.
`run` writes predictions only after weights and data are both present, then
scores them. A short file, a missing file, and the empty Arc path stay
"not measured". Resident memory and tokens per second stay empty.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from burmeseai.eval.frozen_humaneval import FROZEN_HUMANEVAL_TASK_IDS
from burmeseai.eval.generate import find_weights, make_generator
from burmeseai.eval import metricx
from burmeseai.eval.io_util import read_records
from burmeseai.eval.schema import (
    ARC_EMPTY,
    BELEBELE_ITEMS,
    BURMESE_SAN_ITEMS,
    FLORES_PAIRS,
    GSM8K_ITEMS,
    MODEL_ORDER,
    MODELS,
    NOT_MEASURED,
    TASK_ORDER,
    burmese_san_model_cell,
    empty_cell,
    empty_report,
    flores_cell,
)
from burmeseai.eval.scoring import gate, macro_average, protocol_ok
from burmeseai.eval.tasks import belebele, burmese_san, flores, gsm8k, humaneval

MAX_NEW_TOKENS = 512


def predictions_path(predictions_dir: Path, model_id: str, task_id: str) -> Path:
    return predictions_dir / model_id / task_id / "predictions.jsonl"


def run_meta_path(predictions_dir: Path, model_id: str, task_id: str) -> Path:
    return predictions_dir / model_id / task_id / "run.json"


def weight_label(model_id: str, weights_dir: Path) -> str:
    found = find_weights(model_id, weights_dir)
    if found is None:
        if model_id == "arc-1.0":
            return ARC_EMPTY
        return "absent"
    return "present"


def build_report(
    data_dir: Path,
    weights_dir: Path,
    predictions_dir: Path,
) -> dict:
    report = empty_report(FROZEN_HUMANEVAL_TASK_IDS)
    for index, model_id in enumerate(MODEL_ORDER):
        report["models"][index]["weights"] = weight_label(model_id, weights_dir)
    try:
        flores_payload = _load_task("flores", data_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        flores_payload = None
    if isinstance(flores_payload, dict):
        report["tasks"]["flores"]["corpus"] = flores_payload["corpus"]
    for model_id in MODEL_ORDER:
        for task_id in TASK_ORDER:
            report["tasks"][task_id]["models"][model_id] = _score_saved(
                task_id, model_id, data_dir, predictions_dir, weights_dir
            )
    return report


def run_task(
    model_id: str,
    task_id: str,
    data_dir: Path,
    weights_dir: Path,
    predictions_dir: Path,
    generate=None,
) -> dict:
    if model_id not in MODELS:
        raise ValueError(f"unknown model {model_id}")
    if task_id not in TASK_ORDER:
        raise ValueError(f"unknown task {task_id}")
    try:
        loaded = _load_task(task_id, data_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        return _blank(task_id, "data unreadable")
    if loaded is None:
        return _blank(task_id, "data absent")
    if task_id == "burmese_san" and not loaded.get("prompts"):
        return _blank(task_id, "Burmese prompt templates are absent")
    weights = find_weights(model_id, weights_dir)
    if generate is None and weights is None:
        return _blank(task_id, "weights absent")
    if generate is None:
        try:
            generate = make_generator(model_id, weights, MAX_NEW_TOKENS)
        except (OSError, RuntimeError, ValueError) as exc:
            return _blank(task_id, f"runtime unavailable: {exc.__class__.__name__}")
    rows = _generate_rows(task_id, loaded, generate)
    if rows is None:
        return _blank(task_id, "generation failed")
    _write_run(predictions_dir, model_id, task_id, rows)
    return _score_loaded(task_id, loaded, rows, weights_dir)


def _generate_rows(task_id: str, loaded, generate) -> list[dict] | None:
    rows: list[dict] = []
    try:
        if task_id == "flores":
            for pair in loaded["pairs"]:
                for item_id, direction, text in flores.prompts_for_pair(pair):
                    prediction = generate(text, enable_thinking=False)
                    rows.append(_row(item_id, prediction, direction=direction))
            return rows
        if task_id == "burmese_san":
            items = loaded["items"]
            prompts = loaded["prompts"]
            for item in items:
                prediction = generate(
                    burmese_san.prompt(item, prompts),
                    enable_thinking=False,
                )
                rows.append(_row(item["id"], prediction))
            return rows
        items = loaded
        prompt_for = {
            "belebele": belebele.prompt,
            "gsm8k": gsm8k.prompt,
            "humaneval": humaneval.prompt,
        }[task_id]
        for item in items:
            prediction = generate(prompt_for(item), enable_thinking=False)
            rows.append(_row(item["id"], prediction))
    except (OSError, RuntimeError, ValueError, TypeError, KeyError):
        return None
    return rows


def _row(item_id: str, prediction: str, direction: str | None = None) -> dict:
    row = {
        "id": item_id,
        "prediction": prediction,
        "enable_thinking": False,
        "shots": 0,
    }
    if direction is not None:
        row["direction"] = direction
    return row


def _write_run(predictions_dir: Path, model_id: str, task_id: str, rows: list[dict]) -> None:
    path = predictions_path(predictions_dir, model_id, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(path, rows)
    meta = {
        "model": model_id,
        "task": task_id,
        "enable_thinking": False,
        "shots": 0,
        "decoding": "greedy",
        "samples": 1,
        "device": "cpu",
        "max_new_tokens": MAX_NEW_TOKENS,
        "items": len(rows),
    }
    meta_path = run_meta_path(predictions_dir, model_id, task_id)
    _atomic_text(meta_path, json.dumps(meta, indent=2) + "\n")


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _atomic_text(path, payload)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _score_saved(
    task_id: str,
    model_id: str,
    data_dir: Path,
    predictions_dir: Path,
    weights_dir: Path,
) -> dict:
    try:
        loaded = _load_task(task_id, data_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        return _blank(task_id, "data unreadable")
    if loaded is None:
        return _blank(task_id, "data absent")
    path = predictions_path(predictions_dir, model_id, task_id)
    try:
        rows = read_records(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return _blank(task_id, "predictions unreadable")
    if not rows:
        return _blank(task_id, "predictions absent")
    if not protocol_ok(rows):
        return _blank(task_id, "enable_thinking must be false and shots must be 0")
    return _score_loaded(task_id, loaded, rows, weights_dir)


def _score_loaded(task_id: str, loaded, rows: list[dict], weights_dir: Path) -> dict:
    if task_id == "belebele":
        by_id = {row["id"]: str(row.get("prediction", "")) for row in rows}
        raw = belebele.score(loaded, by_id)
        return gate(BELEBELE_ITEMS, len(loaded), raw, "accuracy")
    if task_id == "flores":
        return _score_flores(loaded, rows)
    if task_id == "burmese_san":
        return _score_burmese_san(loaded, rows, weights_dir)
    if task_id == "gsm8k":
        by_id = {row["id"]: str(row.get("prediction", "")) for row in rows}
        raw = gsm8k.score(loaded, by_id)
        return gate(GSM8K_ITEMS, len(loaded), raw, "exact_match")
    if task_id == "humaneval":
        by_id = {row["id"]: str(row.get("prediction", "")) for row in rows}
        if not humaneval.subset_complete(loaded):
            return gate(len(FROZEN_HUMANEVAL_TASK_IDS), len(loaded), None, "pass@1")
        raw = humaneval.score(loaded, by_id)
        return gate(len(FROZEN_HUMANEVAL_TASK_IDS), len(FROZEN_HUMANEVAL_TASK_IDS), raw, "pass@1")
    raise ValueError(f"unknown task {task_id}")


def _score_flores(loaded: dict, rows: list[dict]) -> dict:
    pairs = loaded["pairs"]
    predictions = {
        (str(row["id"]), str(row.get("direction"))): str(row.get("prediction", ""))
        for row in rows
        if "id" in row
    }
    raw = flores.score(pairs, predictions)
    cell = {
        "status": NOT_MEASURED,
        "score": None,
        "metric": "chrf++",
        "eng_Latn_to_mya_Mymr": None,
        "mya_Mymr_to_eng_Latn": None,
    }
    if not flores.can_publish(loaded):
        if loaded.get("corpus") is None:
            cell["reason"] = "files do not match the facebook/flores devtest digests"
        else:
            cell["reason"] = f"expected {FLORES_PAIRS} pairs, found {len(pairs)}"
        return cell
    if raw is None:
        cell["reason"] = "predictions do not cover both directions"
        return cell
    cell["status"] = "measured"
    cell["score"] = None
    cell["eng_Latn_to_mya_Mymr"] = raw["eng_Latn_to_mya_Mymr"]
    cell["mya_Mymr_to_eng_Latn"] = raw["mya_Mymr_to_eng_Latn"]
    cell["items"] = len(pairs)
    return cell


def _score_burmese_san(loaded: dict, rows: list[dict], weights_dir: Path) -> dict:
    items = loaded["items"]
    prompts = loaded.get("prompts")
    cell = burmese_san_model_cell()
    predictions = {str(row["id"]): row for row in rows if "id" in row}
    groups = burmese_san.grouped(items)
    task_scores: dict[str, float | None] = {}
    for spec_name in burmese_san.TASK_NAMES:
        group = groups[spec_name]
        expected = burmese_san.EXPECTED[spec_name]
        metric = cell["tasks"][spec_name]["metric"]
        ready = burmese_san.task_ready(spec_name, group)
        if not ready:
            task_cell = gate(expected, len(group), None, metric)
            task_cell["expected_items"] = expected
            cell["tasks"][spec_name] = task_cell
            task_scores[spec_name] = None
            continue
        if not prompts:
            task_cell = empty_cell(metric)
            task_cell["expected_items"] = expected
            task_cell["reason"] = "Burmese prompt templates are absent"
            cell["tasks"][spec_name] = task_cell
            task_scores[spec_name] = None
            continue
        if spec_name == "mt":
            raw, reason = _metricx_score(group, predictions, weights_dir)
        else:
            covered = all(item["id"] in predictions for item in group)
            raw = burmese_san.score_task(spec_name, group, predictions) if covered else None
            if not covered:
                reason = "predictions do not cover every item"
            elif spec_name == "as" and raw is None:
                reason = "XL-Sum multilingual ROUGE-L is not available"
            else:
                reason = None
        task_cell = gate(expected, len(group), raw, metric)
        task_cell["expected_items"] = expected
        if reason and task_cell["score"] is None:
            task_cell["reason"] = reason
        cell["tasks"][spec_name] = task_cell
        task_scores[spec_name] = task_cell["score"]
    macro = macro_average(
        task_scores, burmese_san.TASK_NAMES, task_scores.get("mt")
    )
    if macro is None or len(items) != BURMESE_SAN_ITEMS:
        cell["status"] = NOT_MEASURED
        cell["score"] = None
        cell["reason"] = (
            "macro average is empty until all seven tasks have a 0-100 score"
        )
        return cell
    cell["status"] = "measured"
    cell["score"] = macro
    cell["items"] = len(items)
    return cell


def _metricx_score(group: list[dict], predictions: dict[str, dict], weights_dir: Path):
    if not metricx.scale_is_declared():
        return None, "MetricX-24 scale is not the declared normalization"
    checkpoint = metricx.find_checkpoint(weights_dir)
    if checkpoint is None:
        return None, "MetricX-24 checkpoint is absent"
    if any(item["id"] not in predictions for item in group):
        return None, "predictions do not cover every item"
    examples = [
        {
            "source": item["input"],
            "candidate": str(predictions[item["id"]].get("prediction", "")),
            "reference": item["label"],
        }
        for item in group
    ]
    try:
        return metricx.normalized_mean(examples, checkpoint), None
    except (OSError, RuntimeError, ValueError, ImportError, KeyError):
        return None, "MetricX-24 checkpoint could not be scored"


def _load_task(task_id: str, data_dir: Path):
    if task_id == "belebele":
        return belebele.load(data_dir)
    if task_id == "flores":
        return flores.load(data_dir)
    if task_id == "burmese_san":
        return burmese_san.load(data_dir)
    if task_id == "gsm8k":
        return gsm8k.load(data_dir)
    if task_id == "humaneval":
        return humaneval.load(data_dir)
    raise ValueError(f"unknown task {task_id}")


def _blank(task_id: str, reason: str) -> dict:
    if task_id == "flores":
        cell = flores_cell()
    elif task_id == "burmese_san":
        cell = burmese_san_model_cell()
    elif task_id == "belebele":
        cell = empty_cell("accuracy")
    elif task_id == "gsm8k":
        cell = empty_cell("exact_match")
    elif task_id == "humaneval":
        cell = empty_cell("pass@1")
    else:
        cell = {"status": NOT_MEASURED, "score": None}
    cell["reason"] = reason
    cell["status"] = NOT_MEASURED
    cell["score"] = None
    return cell


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m burmeseai.eval")
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser("report", help="Write a score report. Does not run a model.")
    _add_dirs(report)
    report.add_argument("--out", type=Path, required=True)

    run = sub.add_parser("run", help="Generate predictions when local files exist.")
    _add_dirs(run)
    run.add_argument("--model", required=True, choices=[*MODEL_ORDER, "all"])
    run.add_argument("--task", required=True, choices=[*TASK_ORDER, "all"])
    run.add_argument("--out", type=Path)

    args = parser.parse_args(argv)
    if args.command == "report":
        payload = build_report(args.data_dir, args.weights_dir, args.predictions_dir)
        write_report(payload, args.out)
        json.dump(
            {"status": report_stdout_status(payload), "out": str(args.out)},
            sys.stdout,
        )
        sys.stdout.write("\n")
        return 0

    models = MODEL_ORDER if args.model == "all" else (args.model,)
    tasks = TASK_ORDER if args.task == "all" else (args.task,)
    results = []
    for model_id in models:
        for task_id in tasks:
            cell = run_task(
                model_id,
                task_id,
                args.data_dir,
                args.weights_dir,
                args.predictions_dir,
            )
            results.append({"model": model_id, "task": task_id, **cell})
    if args.out:
        payload = build_report(args.data_dir, args.weights_dir, args.predictions_dir)
        write_report(payload, args.out)
    json.dump(results, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


def report_stdout_status(report: dict) -> str:
    """Say not measured only when every task cell is still unmeasured."""
    statuses = []
    for task in report["tasks"].values():
        for cell in task["models"].values():
            statuses.append(cell["status"])
            for nested in cell.get("tasks", {}).values():
                if isinstance(nested, dict) and "status" in nested:
                    statuses.append(nested["status"])
    if statuses and all(status == NOT_MEASURED for status in statuses):
        return NOT_MEASURED
    if any(status == "measured" for status in statuses):
        return "measured"
    return "incomplete"


def _add_dirs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("runs"))


if __name__ == "__main__":
    sys.exit(main())

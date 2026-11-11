"""Report schema. Measured cells start empty and stay empty without a run."""

from __future__ import annotations

NOT_MEASURED = "not measured"
SCHEMA_VERSION = 1

MODEL_ORDER = (
    "qwen3-0.6b",
    "gemma4-e2b",
    "gemma4-e4b",
    "sea-lion-v4.5-e2b-it",
    "padauk-q8",
    "arc-1.0",
)

MODELS = {
    "qwen3-0.6b": {
        "name": "Qwen3-0.6B",
        "source": "Qwen/Qwen3-0.6B",
        "format": "hf",
        "device": "cpu",
        "role": "unmodified baseline",
    },
    "gemma4-e2b": {
        "name": "Gemma 4 E2B",
        "source": "Gemma 4 E2B",
        "format": "gguf",
        "device": "cpu",
        "role": "baseline",
    },
    "gemma4-e4b": {
        "name": "Gemma 4 E4B",
        "source": "Gemma 4 E4B",
        "format": "gguf",
        "device": "cpu",
        "role": "baseline",
    },
    "sea-lion-v4.5-e2b-it": {
        "name": "SEA-LION v4.5 E2B-IT",
        "source": "aisingapore/Gemma-SEA-LION-v4.5-E2B-IT",
        "format": "gguf",
        "device": "cpu",
        "role": "baseline",
    },
    "padauk-q8": {
        "name": "Padauk Q8",
        "source": "WYNN747/ai4burmese-padauk-gguf",
        "filename": "padauk-gemma-q8_0.gguf",
        "format": "gguf",
        "device": "cpu",
        "role": "baseline",
    },
    "arc-1.0": {
        "name": "Arc 1.0",
        "source": "Arc 1.0",
        "format": "",
        "device": "cpu",
        "role": "empty until a weight file exists",
    },
}

TASK_ORDER = ("belebele", "flores", "burmese_san", "gsm8k", "humaneval")

BELEBELE_ITEMS = 900
FLORES_PAIRS = 1012
BURMESE_SAN_ITEMS = 3920
GSM8K_ITEMS = 1319

BURMESE_SAN_TASKS = (
    {
        "task": "qa",
        "dataset": "Belebele",
        "license": "CC BY-NC 4.0",
        "items": 120,
        "metric": "accuracy",
    },
    {
        "task": "sa",
        "dataset": "GKLMIP-mya",
        "license": "Unknown",
        "items": 600,
        "metric": "accuracy",
    },
    {
        "task": "td",
        "dataset": "myHateSpeech",
        "license": "CC BY-NC-SA 4.0",
        "items": 400,
        "metric": "accuracy",
    },
    {
        "task": "cr",
        "dataset": "Balanced COPA",
        "license": "CC BY 4.0",
        "items": 400,
        "metric": "accuracy",
    },
    {
        "task": "nli",
        "dataset": "myXNLI",
        "license": "CC BY-NC 4.0",
        "items": 600,
        "metric": "accuracy",
    },
    {
        "task": "as",
        "dataset": "XL-Sum",
        "license": "CC BY-NC-SA 4.0",
        "items": 100,
        "metric": "rouge_l_char_f1",
    },
    {
        "task": "mt",
        "dataset": "FLORES+",
        "license": "CC BY-SA 4.0",
        "items": 1700,
        "metric": "metricx_24",
        "directions": {"eng_to_mya": 850, "mya_to_eng": 850},
    },
)

ARC_EMPTY = "empty until a weight file exists"
PADAUK_FILENAME = "padauk-gemma-q8_0.gguf"
WEIGHT_SUFFIXES = {".gguf", ".safetensors", ".bin", ".pt"}


def empty_cell(metric: str) -> dict:
    return {"status": NOT_MEASURED, "score": None, "metric": metric}


def flores_cell() -> dict:
    cell = empty_cell("chrf++")
    cell["eng_Latn_to_mya_Mymr"] = None
    cell["mya_Mymr_to_eng_Latn"] = None
    return cell


def burmese_san_model_cell() -> dict:
    tasks = {}
    for spec in BURMESE_SAN_TASKS:
        tasks[spec["task"]] = {
            "status": NOT_MEASURED,
            "score": None,
            "metric": spec["metric"],
            "expected_items": spec["items"],
        }
    return {
        "status": NOT_MEASURED,
        "score": None,
        "metric": "macro",
        "tasks": tasks,
    }


def measurement_rows() -> list[dict]:
    return [
        {
            "id": "resident_memory",
            "label": "Resident memory",
            "status": NOT_MEASURED,
            "value": None,
            "unit": "bytes",
        },
        {
            "id": "tokens_per_second",
            "label": "Tokens per second",
            "status": NOT_MEASURED,
            "value": None,
            "unit": "tokens/sec",
        },
        {
            "id": "apple_silicon",
            "label": "Apple Silicon",
            "status": NOT_MEASURED,
            "value": None,
            "unit": None,
        },
        {
            "id": "mlx",
            "label": "MLX",
            "status": NOT_MEASURED,
            "value": None,
            "unit": None,
        },
        {
            "id": "android_8gb",
            "label": "8 GB Android-class device",
            "status": NOT_MEASURED,
            "value": None,
            "unit": None,
        },
        {
            "id": "colloquial",
            "label": "Colloquial and code-switching",
            "status": NOT_MEASURED,
            "value": None,
            "unit": None,
            "note": "No score until the file contains at least 20 human-written items.",
        },
    ]


def label_rows() -> dict:
    rules = {
        "belebele": (
            "accuracy at least 5.0 points above unmodified Qwen3-0.6B and at least 30.0"
        ),
        "flores": (
            "each direction at least 2.0 chrF++ above unmodified Qwen3-0.6B"
        ),
        "burmese_san": (
            "macro average at least the unmodified Qwen3-0.6B macro average"
        ),
        "gsm8k": "drop versus unmodified Qwen3-0.6B at most 3.0 points",
        "humaneval": "drop versus unmodified Qwen3-0.6B at most 3.0 points",
    }
    return {
        task: {"status": NOT_MEASURED, "rule": rule} for task, rule in rules.items()
    }


def empty_report(frozen_ids: tuple[str, ...]) -> dict:
    """Skeleton report. Every model cell is unmeasured and every score is null."""
    belebele_models = {
        model_id: empty_cell("accuracy") for model_id in MODEL_ORDER
    }
    flores_models = {model_id: flores_cell() for model_id in MODEL_ORDER}
    san_models = {model_id: burmese_san_model_cell() for model_id in MODEL_ORDER}
    gsm_models = {model_id: empty_cell("exact_match") for model_id in MODEL_ORDER}
    human_models = {model_id: empty_cell("pass@1") for model_id in MODEL_ORDER}
    return {
        "schema_version": SCHEMA_VERSION,
        "enable_thinking": False,
        "shots": 0,
        "decoding": "greedy",
        "samples": 1,
        "models": [_model_row(model_id, "absent") for model_id in MODEL_ORDER],
        "tasks": {
            "belebele": {
                "language": "mya_Mymr",
                "expected_items": BELEBELE_ITEMS,
                "choices": 4,
                "shots": 0,
                "enable_thinking": False,
                "metric": "accuracy",
                "chance": 25,
                "license_note": (
                    "Belebele GitHub README: test set CC-BY-SA 4.0, assembled "
                    "training set CC-BY-NC. BURMESE-SAN Table 6 lists Belebele "
                    "as CC BY-NC 4.0. Eval only."
                ),
                "models": belebele_models,
            },
            "flores": {
                "corpus": "facebook/flores",
                "split": "devtest",
                "configs": ["eng_Latn", "mya_Mymr"],
                "expected_pairs": FLORES_PAIRS,
                "metric": "chrf++",
                "directions": ["eng_Latn_to_mya_Mymr", "mya_Mymr_to_eng_Latn"],
                "shots": 0,
                "enable_thinking": False,
                "models": flores_models,
            },
            "burmese_san": {
                "expected_items": BURMESE_SAN_ITEMS,
                "task_count": 7,
                "release_license": "CC-BY-SA 4.0",
                "licenses": [dict(spec) for spec in BURMESE_SAN_TASKS],
                "macro_note": (
                    "Unweighted mean of the seven task scores on a 0-100 scale. "
                    "Empty until each task has a score from a run. "
                    "Machine translation uses a local MetricX-24 checkpoint and "
                    "the declared 0-100 normalization."
                ),
                "shots": 0,
                "enable_thinking": False,
                "models": san_models,
            },
            "gsm8k": {
                "split": "test",
                "expected_items": GSM8K_ITEMS,
                "metric": "exact_match",
                "shots": 0,
                "enable_thinking": False,
                "models": gsm_models,
            },
            "humaneval": {
                "metric": "pass@1",
                "frozen_task_ids": list(frozen_ids),
                "expected_items": len(frozen_ids),
                "shots": 0,
                "enable_thinking": False,
                "models": human_models,
            },
        },
        "measurements": measurement_rows(),
        "labels": label_rows(),
    }


def _model_row(model_id: str, weights: str) -> dict:
    spec = MODELS[model_id]
    row = {
        "id": model_id,
        "name": spec["name"],
        "source": spec["source"],
        "format": spec["format"],
        "device": spec["device"],
        "role": spec["role"],
        "enable_thinking": False,
        "weights": ARC_EMPTY if model_id == "arc-1.0" and weights != "present" else weights,
    }
    if "filename" in spec:
        row["filename"] = spec["filename"]
    return row

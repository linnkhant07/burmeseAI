"""MetricX-24 from a local checkpoint, on the SEA-HELM 0-100 scale.

The checkpoint is google/metricx-24-hybrid-xxl-v2p6-bfloat16. Nothing is
downloaded. SEA-HELM's translation metric declares the error bounds as
METRICX_MIN_SCORE, METRICX_MAX_SCORE = 25, 0 and maps each raw error with
normalize_score(raw, 25, 0) * 100. That is a higher-is-better score: the
native maximum of the error is 25, and the generative baseline on the
inverted unit interval is 0. A missing checkpoint, or a scale that is not
this declaration, does not produce a number.
"""

from __future__ import annotations

from pathlib import Path

# SEA-HELM seahelm_tasks/nlg/translation/translation_metric.py
METRICX_MIN_SCORE = 25.0
METRICX_MAX_SCORE = 0.0
GENERATIVE_BASELINE = 0.0
UNIT_MAXIMUM = 1.0
CHECKPOINT_NAME = "metricx-24-hybrid-xxl-v2p6-bfloat16"
EXTRA_ID_10 = 250089
_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt"}


def scale_is_declared() -> bool:
    """True only for the bounds the SEA-HELM translation metric declares."""
    return (
        METRICX_MIN_SCORE == 25.0
        and METRICX_MAX_SCORE == 0.0
        and GENERATIVE_BASELINE == 0.0
        and UNIT_MAXIMUM == 1.0
    )


def normalize_metricx(raw_error: float) -> float:
    """Map one MetricX error onto a higher-is-better 0-100 score.

    This is SeaHelmMetric.normalize_score(raw, 25, 0) * 100. The first step
    inverts the error by its native maximum. The second step applies the
    generative baseline of 0 on that unit interval.
    """
    if not scale_is_declared():
        raise RuntimeError("MetricX-24 scale is not the declared normalization")
    if METRICX_MIN_SCORE == METRICX_MAX_SCORE:
        raise RuntimeError("MetricX-24 scale is not the declared normalization")
    inverted = (raw_error - METRICX_MIN_SCORE) / (METRICX_MAX_SCORE - METRICX_MIN_SCORE)
    unit = max(inverted, 0.0)
    return max((unit - GENERATIVE_BASELINE) / (UNIT_MAXIMUM - GENERATIVE_BASELINE), 0.0) * 100.0


def find_checkpoint(weights_dir: Path) -> Path | None:
    candidates = (
        weights_dir / CHECKPOINT_NAME,
        weights_dir / "google" / CHECKPOINT_NAME,
    )
    for path in candidates:
        if _is_checkpoint(path):
            return path
    return None


def _is_checkpoint(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    return any(
        child.is_file()
        and child.suffix.lower() in _WEIGHT_SUFFIXES
        and child.stat().st_size > 0
        for child in path.rglob("*")
    )


def normalized_mean(examples: list[dict], checkpoint: Path) -> float:
    """Score source/candidate/reference rows. Prediction-file side fields are ignored."""
    if not scale_is_declared():
        raise RuntimeError("MetricX-24 scale is not the declared normalization")
    if not examples:
        raise RuntimeError("MetricX-24 has no items")
    raw_errors = _raw_errors(examples, checkpoint)
    if len(raw_errors) != len(examples):
        raise RuntimeError("MetricX-24 did not score every item")
    values = [normalize_metricx(error) for error in raw_errors]
    return sum(values) / len(values)


def _prompt(example: dict) -> str:
    return (
        f"source: {example['source']} candidate: {example['candidate']} "
        f"reference: {example['reference']}"
    )


def _raw_errors(examples: list[dict], checkpoint: Path) -> list[float]:
    import torch
    from transformers import AutoTokenizer, MT5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = MT5ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True)
    model.to("cpu")
    model.eval()
    errors: list[float] = []
    batch_size = 4
    texts = [_prompt(example) for example in examples]
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        encoded = {key: value.to("cpu") for key, value in encoded.items()}
        with torch.no_grad():
            encoder_outputs = model.get_encoder()(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
            decoder_input_ids = torch.zeros(
                (encoded["input_ids"].size(0), 1),
                dtype=torch.long,
            )
            decoder_outputs = model.get_decoder()(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=encoded["attention_mask"],
            )
            logits = model.lm_head(decoder_outputs.last_hidden_state)
            if logits.size(-1) <= EXTRA_ID_10:
                raise RuntimeError("MetricX-24 logit index is outside the vocabulary")
            predictions = torch.clamp(logits[:, 0, EXTRA_ID_10], 0, 25)
        errors.extend(float(value) for value in predictions.tolist())
    return errors

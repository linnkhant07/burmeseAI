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


MAX_INPUT_LENGTH = 1024


def drop_trailing_eos(
    input_ids: list[int], attention_mask: list[int]
) -> tuple[list[int], list[int]]:
    """Drop the final token the way metricx24/predict.py drops EOS."""
    if not input_ids or not attention_mask or len(input_ids) != len(attention_mask):
        raise RuntimeError("MetricX-24 input has no trailing token to drop")
    trimmed_ids = input_ids[:-1]
    trimmed_mask = attention_mask[:-1]
    if not trimmed_ids or not any(trimmed_mask):
        raise RuntimeError("MetricX-24 input is empty after dropping EOS")
    return trimmed_ids, trimmed_mask


def tied_rescale_factor(tie_word_embeddings: bool, d_model: int | None) -> float:
    """mT5 multiplies the decoder state by d_model ** -0.5 when the head is tied."""
    if not tie_word_embeddings:
        return 1.0
    if not isinstance(d_model, int) or d_model <= 0:
        raise RuntimeError("MetricX-24 tied embeddings have no d_model")
    return float(d_model) ** -0.5


def _raw_errors(examples: list[dict], checkpoint: Path) -> list[float]:
    import torch
    from transformers import AutoTokenizer, MT5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = MT5ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True)
    model.to("cpu")
    model.eval()
    scale = tied_rescale_factor(
        bool(getattr(model.config, "tie_word_embeddings", False)),
        getattr(model.config, "d_model", None),
    )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("MetricX-24 tokenizer has no pad id")
    errors: list[float] = []
    batch_size = 4
    texts = [_prompt(example) for example in examples]
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        rows = [_tokenize_without_eos(tokenizer, text) for text in batch]
        width = max(len(ids) for ids, _mask in rows)
        input_rows = []
        mask_rows = []
        for ids, mask in rows:
            pad = width - len(ids)
            input_rows.append(ids + [pad_id] * pad)
            mask_rows.append(mask + [0] * pad)
        input_ids = torch.tensor(input_rows, dtype=torch.long)
        attention_mask = torch.tensor(mask_rows, dtype=torch.long)
        with torch.no_grad():
            encoder_outputs = model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            decoder_input_ids = torch.zeros(
                (input_ids.size(0), 1),
                dtype=torch.long,
            )
            decoder_outputs = model.get_decoder()(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
            )
            sequence_output = decoder_outputs.last_hidden_state
            if scale != 1.0:
                sequence_output = sequence_output * scale
            logits = model.lm_head(sequence_output)
            if logits.size(-1) <= EXTRA_ID_10:
                raise RuntimeError("MetricX-24 logit index is outside the vocabulary")
            predictions = torch.clamp(logits[:, 0, EXTRA_ID_10], 0, 25)
        errors.extend(float(value) for value in predictions.tolist())
    return errors


def _tokenize_without_eos(tokenizer, text: str) -> tuple[list[int], list[int]]:
    encoded = tokenizer(
        text,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,
    )
    input_ids = [int(token) for token in encoded["input_ids"]]
    attention_mask = [int(bit) for bit in encoded["attention_mask"]]
    return drop_trailing_eos(input_ids, attention_mask)

"""Local generation. Thinking is off. GGUF files run with no GPU layers.

Nothing is downloaded. A missing runtime or a bad weight file raises, and the
driver then leaves the score empty.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from burmeseai.eval.schema import MODELS, PADAUK_FILENAME, WEIGHT_SUFFIXES


def call_chat_template(tokenizer, messages):
    """Pass enable_thinking false. A template that rejects the flag raises."""
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        enable_thinking=False,
    )


def gguf_template_kwargs(parameters) -> dict:
    """Require a parameter that can carry enable_thinking false."""
    accepts = "chat_template_kwargs" in parameters or any(
        item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()
    )
    if not accepts:
        raise RuntimeError("the GGUF chat call cannot take enable_thinking")
    return {"chat_template_kwargs": {"enable_thinking": False}}


def find_weights(model_id: str, weights_dir: Path) -> Path | None:
    root = weights_dir / model_id
    if not root.is_dir():
        return None
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in WEIGHT_SUFFIXES
        and path.stat().st_size > 0
    ]
    if model_id == "padauk-q8":
        named = [path for path in files if path.name == PADAUK_FILENAME]
        return sorted(named)[0] if named else None
    if not files:
        return None
    gguf = sorted(path for path in files if path.suffix.lower() == ".gguf")
    if gguf:
        return gguf[0]
    if (root / "config.json").is_file():
        return root
    return None


def make_generator(model_id: str, weight_path: Path, max_new_tokens: int = 512):
    spec = MODELS[model_id]
    if weight_path.is_file() and weight_path.suffix.lower() == ".gguf":
        return _gguf_generator(weight_path, max_new_tokens)
    if spec["format"] == "gguf":
        raise RuntimeError("GGUF weights were expected")
    return _hf_generator(weight_path, max_new_tokens)


def _gguf_generator(weight_path: Path, max_new_tokens: int):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("llama.cpp runtime is not installed") from exc

    llm = Llama(
        model_path=str(weight_path),
        n_gpu_layers=0,
        n_ctx=4096,
        verbose=False,
    )
    signature = inspect.signature(llm.create_chat_completion)

    def generate(prompt: str, *, enable_thinking: bool) -> str:
        if enable_thinking:
            raise ValueError("enable_thinking must be false")
        call = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_new_tokens,
        }
        call.update(gguf_template_kwargs(signature.parameters))
        result = llm.create_chat_completion(**call)
        return str(result["choices"][0]["message"]["content"])

    return generate


def _hf_generator(weight_path: Path, max_new_tokens: int):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers runtime is not installed") from exc

    tokenizer = AutoTokenizer.from_pretrained(weight_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(weight_path, local_files_only=True)
    model.to("cpu")
    model.eval()

    def generate(prompt: str, *, enable_thinking: bool) -> str:
        if enable_thinking:
            raise ValueError("enable_thinking must be false")
        messages = [{"role": "user", "content": prompt}]
        encoded = call_chat_template(tokenizer, messages)
        if hasattr(encoded, "input_ids"):
            inputs = {key: value.to("cpu") for key, value in encoded.items()}
            input_length = inputs["input_ids"].shape[-1]
        else:
            inputs = {"input_ids": encoded.to("cpu")}
            input_length = encoded.shape[-1]
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        new_tokens = output[0][input_length:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generate

"""CPU smoke and the one-L4 training entry.

``python -m burmeseai.train --smoke`` runs two AdamW steps on a hand-written
fixture. The fixture is resized with a per-piece mean. That command does not
import Modal and does not read ``MODAL_TOKEN_ID`` or ``MODAL_TOKEN_SECRET``.

``modal run burmeseai/train.py`` is the GPU entry: one L4, 2 CPU cores, and
16 GiB. It exits with status 2 when either token variable is missing. It does
not request a GPU when ``artifacts/fertility.json`` is missing or a workspace
usage budget of $20 cannot be set. An environment budget and a post-credit
spend limit are different controls and are not accepted here. Preemption
reloads the latest checkpoint. One resume is allowed, and that resume does
not start a second run from step zero.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

FERTILITY_PATH = Path("artifacts/fertility.json")
WORKSPACE_USAGE_BUDGET_USD = 20
WORKSPACE_BUDGET_SETTING = "workspace-budget"
RESUME_RETRY_CAP = 1

GPU = "L4"
CPU_CORES = 2
MEMORY_GIB = 16
MEMORY_MIB = MEMORY_GIB * 1024
RUN_HOURS = 18
TIMEOUT_SECONDS = RUN_HOURS * 60 * 60
CHECKPOINT_DIR = "/checkpoints"
VOLUME_NAME = "burmeseai-checkpoints"

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
SEQUENCE_LENGTH = 512
BATCH_SIZE = 4
MAX_NEW_TOKENS = 10_000
ENGLISH_REPLAY_FRACTION = 0.0
MERGE_CPT = 0.5
MERGE_CHAT = 0.5
LORA_RANK = 16
LORA_ALPHA = 32
LORA_LEARNING_RATE = 1e-4
SMOKE_STEPS = 2
SMOKE_DIM = 8

L4_USD_PER_SEC = 0.000222
CPU_USD_PER_CORE_SEC = 0.0000131
MEMORY_USD_PER_GIB_SEC = 0.00000222

TOKEN_VARIABLES = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")

# Hand-written fixture. These strings are not a corpus download.
FIXTURE_LINES = (
    "water",
    "heat",
    "health",
    "quake",
    "hello",
    "river",
    "fever",
    "shade",
    "drink",
    "rest",
    "clean",
    "wound",
    "storm",
    "flood",
    "night",
    "child",
    "elder",
    "clinic",
    "street",
    "power",
    "blackout",
    "warning",
    "shelter",
    "boil",
    "မြန်မာ",
    "ရေ",
    "အပူ",
    "ကျန်းမာ",
    "ငလျင်",
    "အရိပ်",
    "သောက်",
    "အနား",
    "သန့်",
    "ဒဏ်ရာ",
    "မုန်တိုင်း",
    "ရေလွှမ်း",
    "ည",
    "ကလေး",
    "ဆေးရုံ",
    "လမ်း",
    "မီး",
    "သတိ",
)


def missing_token_names(env: Mapping[str, str]) -> list[str]:
    return [name for name in TOKEN_VARIABLES if not env.get(name)]


def missing_token_message(missing: Sequence[str]) -> str:
    if len(missing) == 1:
        return f"{missing[0]} is missing; not requesting a GPU"
    return "MODAL_TOKEN_ID or MODAL_TOKEN_SECRET is missing; not requesting a GPU"


def mean_init_row(embedding: Sequence[Sequence[float]], piece_ids: Sequence[int]) -> list[float]:
    """Mean of the source-piece rows for one new token.

    Yamaguchi et al. initialize each added row as the mean of the pieces that
    already spell that string. An empty piece list is not a defined mean.
    """

    if len(piece_ids) == 0:
        raise ValueError("a new token needs at least one source piece")
    width = len(embedding)
    if width == 0:
        raise ValueError("embedding is empty")
    dim = len(embedding[0])
    total = [0.0] * dim
    for piece_id in piece_ids:
        if piece_id < 0 or piece_id >= width:
            raise ValueError(f"piece id {piece_id} is outside vocab size {width}")
        row = embedding[piece_id]
        if len(row) != dim:
            raise ValueError("embedding rows have different widths")
        for index, value in enumerate(row):
            total[index] += float(value)
    count = float(len(piece_ids))
    return [value / count for value in total]


def mean_init_resize(
    input_embeddings: Sequence[Sequence[float]],
    new_token_pieces: Sequence[Sequence[int]],
    output_embeddings: Sequence[Sequence[float]] | None = None,
    *,
    tie_word_embeddings: bool,
) -> tuple[list[list[float]], list[list[float]] | None]:
    """Append mean-initialized rows.

    The caller reads ``tie_word_embeddings`` from the loaded model before
    calling this. When the flag is true, the input matrix and the output head
    are one matrix, and the same object is returned for both. When it is
    false, each new input row is the mean of the input pieces and each new
    output row is the mean of the output pieces.
    """

    if tie_word_embeddings:
        if output_embeddings is not None and output_embeddings is not input_embeddings:
            raise ValueError("tied embeddings share one matrix")
        grown = [list(map(float, row)) for row in input_embeddings]
        for pieces in new_token_pieces:
            grown.append(mean_init_row(input_embeddings, pieces))
        return grown, grown

    if output_embeddings is None:
        raise ValueError("untied embeddings need an output matrix")
    if len(output_embeddings) != len(input_embeddings):
        raise ValueError("input and output vocab sizes differ")
    grown_in = [list(map(float, row)) for row in input_embeddings]
    grown_out = [list(map(float, row)) for row in output_embeddings]
    for pieces in new_token_pieces:
        grown_in.append(mean_init_row(input_embeddings, pieces))
        grown_out.append(mean_init_row(output_embeddings, pieces))
    return grown_in, grown_out


def _zeros_like(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[0.0] * len(row) for row in matrix]


def sequence_loss_and_grad(
    embedding: Sequence[Sequence[float]],
    sequences: Sequence[Sequence[int]],
) -> tuple[float, list[list[float]]]:
    """Mean next-token cross entropy for a tied embedding, plus its gradient."""

    if not sequences:
        raise ValueError("sequences are empty")
    vocab = len(embedding)
    dim = len(embedding[0])
    grad = _zeros_like(embedding)
    total_loss = 0.0
    predictions = 0
    for tokens in sequences:
        if len(tokens) < 2:
            raise ValueError("a sequence needs at least two tokens")
        for position in range(len(tokens) - 1):
            source = tokens[position]
            target = tokens[position + 1]
            if source < 0 or source >= vocab or target < 0 or target >= vocab:
                raise ValueError("token id is outside the embedding")
            hidden = embedding[source]
            logits = [0.0] * vocab
            for token_id, row in enumerate(embedding):
                score = 0.0
                for dim_index in range(dim):
                    score += hidden[dim_index] * row[dim_index]
                logits[token_id] = score
            peak = max(logits)
            shifted = [math.exp(score - peak) for score in logits]
            partition = math.fsum(shifted)
            loss = (peak + math.log(partition)) - logits[target]
            total_loss += loss
            predictions += 1
            probs = [value / partition for value in shifted]
            probs[target] -= 1.0
            hidden_grad = [0.0] * dim
            for token_id, row in enumerate(embedding):
                scale = probs[token_id]
                if scale == 0.0:
                    continue
                grad_row = grad[token_id]
                for dim_index in range(dim):
                    grad_row[dim_index] += scale * hidden[dim_index]
                    hidden_grad[dim_index] += scale * row[dim_index]
            source_grad = grad[source]
            for dim_index in range(dim):
                source_grad[dim_index] += hidden_grad[dim_index]
    scale = 1.0 / predictions
    for row in grad:
        for index, value in enumerate(row):
            row[index] = value * scale
    return total_loss * scale, grad


class AdamW:
    """AdamW over one or more matrices of floats, updated in place."""

    def __init__(
        self,
        parameters: Sequence[list[list[float]]],
        *,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        betas: tuple[float, float] = ADAM_BETAS,
        eps: float = ADAM_EPS,
    ) -> None:
        self.parameters = parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        self.moment1 = [_zeros_like(matrix) for matrix in parameters]
        self.moment2 = [_zeros_like(matrix) for matrix in parameters]

    def step(self, gradients: Sequence[Sequence[Sequence[float]]]) -> None:
        self.step_count += 1
        bias1 = 1.0 - self.beta1**self.step_count
        bias2 = 1.0 - self.beta2**self.step_count
        for matrix, grad, moment1, moment2 in zip(
            self.parameters, gradients, self.moment1, self.moment2
        ):
            for row, grad_row, moment1_row, moment2_row in zip(matrix, grad, moment1, moment2):
                for index, grad_value in enumerate(grad_row):
                    moment1_row[index] = self.beta1 * moment1_row[index] + (1.0 - self.beta1) * grad_value
                    moment2_row[index] = (
                        self.beta2 * moment2_row[index] + (1.0 - self.beta2) * grad_value * grad_value
                    )
                    corrected1 = moment1_row[index] / bias1
                    corrected2 = moment2_row[index] / bias2
                    update = corrected1 / (math.sqrt(corrected2) + self.eps)
                    update += self.weight_decay * row[index]
                    row[index] -= self.lr * update


def _init_row(index: int, dim: int) -> list[float]:
    return [((index + 1) * (dim_index + 3) % 11) / 10.0 - 0.5 for dim_index in range(dim)]


def load_fixture() -> dict:
    """Build the hand-written fixture and the tie flag the smoke reads."""

    chars: list[str] = []
    seen: set[str] = set()
    for line in FIXTURE_LINES:
        for char in line:
            if char not in seen:
                seen.add(char)
                chars.append(char)
    index_of = {char: index for index, char in enumerate(chars)}
    embedding = [_init_row(index, SMOKE_DIM) for index in range(len(chars))]
    new_words = ("heat", "ရေ")
    pieces = [tuple(index_of[char] for char in word) for word in new_words]
    sequences: list[list[int]] = []
    for line in FIXTURE_LINES:
        ids = [index_of[char] for char in line]
        if len(ids) >= 2:
            sequences.append(ids)
    return {
        "tie_word_embeddings": True,
        "input_embeddings": embedding,
        "output_embeddings": embedding,
        "new_token_text": new_words,
        "new_token_pieces": pieces,
        "sequences": sequences[:BATCH_SIZE],
    }


def _format_row(row: Sequence[float]) -> str:
    return " ".join(f"{value:.8f}" for value in row)


def run_smoke() -> dict:
    """Two CPU AdamW steps on the resized fixture. No Modal import."""

    modal_present_before = "modal" in sys.modules
    loaded = load_fixture()
    tie_word_embeddings = bool(loaded["tie_word_embeddings"])
    if tie_word_embeddings:
        output = loaded["input_embeddings"]
    else:
        output = loaded["output_embeddings"]
    resized, resized_output = mean_init_resize(
        loaded["input_embeddings"],
        loaded["new_token_pieces"],
        output,
        tie_word_embeddings=tie_word_embeddings,
    )
    vocab_before = len(loaded["input_embeddings"])
    init_rows = [list(row) for row in resized[vocab_before:]]
    optimizer = AdamW([resized], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    losses: list[float] = []
    for _ in range(SMOKE_STEPS):
        loss, grad = sequence_loss_and_grad(resized, loaded["sequences"])
        optimizer.step([grad])
        losses.append(loss)
    modal_imported = (not modal_present_before) and ("modal" in sys.modules)
    lines = [
        "--- smoke ---",
        "device: cpu",
        f"steps: {SMOKE_STEPS}",
        "optimizer: AdamW",
        "learning_rate: 2e-5",
        f"weight_decay: {WEIGHT_DECAY}",
        f"tie_word_embeddings: {str(tie_word_embeddings).lower()}",
        "mean_init: per-piece",
        f"vocab_before: {vocab_before}",
        f"vocab_after: {len(resized)}",
    ]
    for text, pieces, row in zip(loaded["new_token_text"], loaded["new_token_pieces"], init_rows):
        lines.append(f"new_token: {text}")
        lines.append(f"pieces: {len(pieces)}")
        lines.append(f"init_row: {_format_row(row)}")
    for number, loss in enumerate(losses, start=1):
        lines.append(f"step: {number}")
        lines.append(f"loss: {loss:.8f}")
    lines.append(f"modal_imported: {str(modal_imported).lower()}")
    lines.append("--- end smoke ---")
    print("\n".join(lines))
    return {
        "losses": losses,
        "tie_word_embeddings": tie_word_embeddings,
        "vocab_before": vocab_before,
        "vocab_after": len(resized),
        "init_rows": init_rows,
        "embedding": resized,
        "modal_imported": modal_imported,
        "same_matrix": resized_output is resized,
    }


def plan_resume(attempt: int, checkpoint: Path | None, cap: int = RESUME_RETRY_CAP) -> str:
    """Choose start, resume, or stop.

    Attempt 0 may start from step zero when no checkpoint exists. A later
    attempt resumes from the latest checkpoint. If that checkpoint is gone,
    or the attempt is past the cap, the run stops instead of starting over.
    """

    if attempt < 0:
        raise ValueError("attempt must be >= 0")
    if cap < 0:
        raise ValueError("cap must be >= 0")
    if attempt > cap:
        return "stop"
    if checkpoint is not None:
        return "resume"
    if attempt == 0:
        return "start"
    return "stop"


def latest_checkpoint(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    found: list[tuple[int, Path]] = []
    for path in root.glob("checkpoint-*.json"):
        suffix = path.stem.removeprefix("checkpoint-")
        if not suffix.isdigit():
            continue
        found.append((int(suffix), path))
    if not found:
        return None
    found.sort()
    return found[-1][1]


def read_checkpoint(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "step" not in data:
        raise ValueError(f"checkpoint missing step: {path}")
    return data


def write_checkpoint(root: Path, step: int, payload: Mapping | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    body = {"step": step}
    if payload:
        body.update(payload)
    path = root / f"checkpoint-{step:06d}.json"
    path.write_text(json.dumps(body) + "\n", encoding="utf-8")
    return path


def record_attempt(root: Path) -> int:
    """Return this invocation's attempt index and persist the next one."""

    root.mkdir(parents=True, exist_ok=True)
    path = root / "attempts.json"
    if path.is_file():
        attempt = int(json.loads(path.read_text(encoding="utf-8"))["attempts"])
    else:
        attempt = 0
    path.write_text(json.dumps({"attempts": attempt + 1}) + "\n", encoding="utf-8")
    return attempt


def _commit_volume(volume) -> bool:
    if volume is None:
        return True
    volume.commit()
    return True


def train_on_volume(root: Path, volume=None) -> dict:
    """Resume bookkeeping shared by the L4 function.

    This reads the latest checkpoint when one exists. It does not download
    weights and it does not write a loss it did not compute. ``training_started``
    stays false until a measured training loop runs.
    """

    attempt = record_attempt(root)
    try:
        committed = _commit_volume(volume)
    except Exception as exc:
        return {
            "attempt": attempt,
            "action": "stop",
            "reason": f"checkpoint volume commit failed: {exc}",
            "training_started": False,
            "checkpoint": None,
        }
    if not committed:
        return {
            "attempt": attempt,
            "action": "stop",
            "reason": "checkpoint volume commit failed",
            "training_started": False,
            "checkpoint": None,
        }
    checkpoint = latest_checkpoint(root)
    action = plan_resume(attempt, checkpoint, RESUME_RETRY_CAP)
    result = {
        "attempt": attempt,
        "action": action,
        "training_started": False,
        "checkpoint": None if checkpoint is None else checkpoint.name,
    }
    if action == "resume" and checkpoint is not None:
        loaded = read_checkpoint(checkpoint)
        result["resumed_step"] = int(loaded["step"])
        result["reason"] = "loaded latest checkpoint"
        return result
    if action == "start":
        result["reason"] = "no checkpoint yet; weights are not on the volume"
        return result
    if checkpoint is None:
        result["reason"] = "resume without a checkpoint; not starting from step zero"
    else:
        result["reason"] = "resume cap reached"
    return result


def estimated_cost_usd(seconds: float, cores: float = CPU_CORES, memory_gib: float = MEMORY_GIB) -> float:
    """Rate-card estimate. This is not an invoice and is not a measured run."""

    if seconds < 0:
        raise ValueError("seconds must be >= 0")
    return seconds * (
        L4_USD_PER_SEC + cores * CPU_USD_PER_CORE_SEC + memory_gib * MEMORY_USD_PER_GIB_SEC
    )


def next_batch_size(current: int, out_of_memory: bool) -> int:
    if current < 1:
        raise ValueError("batch size must be >= 1")
    if out_of_memory and current > 1:
        return current // 2
    return current


def lora_enabled(permissive_instruction_sources: int) -> bool:
    return permissive_instruction_sources >= 1


def merge_rows(
    continued: Sequence[Sequence[float]],
    chat: Sequence[Sequence[float]],
    continued_fraction: float = MERGE_CPT,
) -> list[list[float]]:
    """Element-wise mix. The first recorded fraction is half and half."""

    if len(continued) != len(chat):
        raise ValueError("mix matrices differ in length")
    chat_fraction = 1.0 - continued_fraction
    mixed: list[list[float]] = []
    for left, right in zip(continued, chat):
        if len(left) != len(right):
            raise ValueError("mix rows differ in width")
        mixed.append(
            [continued_fraction * float(a) + chat_fraction * float(b) for a, b in zip(left, right)]
        )
    return mixed


def fertility_file_ready(path: Path) -> bool:
    return path.is_file()


def workspace_usage_budget_can_be_set() -> bool:
    """True only when the workspace usage budget is set to $20 and confirmed.

    The public settings call is the attempt. Modal currently accepts only the
    image builder version and the default environment there, so the call
    fails and this returns false. An environment budget and a spend limit are
    different controls and are not used.
    """

    import modal

    try:
        workspace = modal.Workspace.from_context()
        billing = getattr(workspace, "billing", None)
        setter = getattr(billing, "set_usage_budget", None) if billing is not None else None
        if setter is None:
            setter = getattr(workspace, "set_usage_budget", None)
        if setter is not None:
            setter(WORKSPACE_USAGE_BUDGET_USD)
        else:
            workspace.settings.set(WORKSPACE_BUDGET_SETTING, str(WORKSPACE_USAGE_BUDGET_USD))
        listed = workspace.settings.list()
    except Exception:
        return False
    recorded = getattr(listed, "workspace_budget", None)
    return recorded == WORKSPACE_USAGE_BUDGET_USD or recorded == str(WORKSPACE_USAGE_BUDGET_USD)


def l4_resources() -> dict:
    return {
        "gpu": GPU,
        "cpu": CPU_CORES,
        "memory_mib": MEMORY_MIB,
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_resume_retries": RESUME_RETRY_CAP,
        "region": None,
        "nonpreemptible": False,
    }


def run_gated_launch(
    *,
    environ: Mapping[str, str] | None = None,
    fertility_path: Path | None = None,
    budget_set=None,
    launch=None,
) -> int:
    """Apply the launch gates. The default launch is the only GPU request."""

    env = environ if environ is not None else _os_environ()
    missing = missing_token_names(env)
    if missing:
        print(missing_token_message(missing))
        return 2
    path = fertility_path if fertility_path is not None else FERTILITY_PATH
    if not fertility_file_ready(path):
        print(f"fertility file missing: {path}; not starting a GPU")
        return 1
    setter = budget_set if budget_set is not None else workspace_usage_budget_can_be_set
    if not setter():
        print("workspace usage budget of $20 cannot be set; not starting a GPU")
        return 1
    resources = l4_resources()
    if launch is None:
        launch = _launch_l4
    launch(resources)
    return 0


def _os_environ() -> Mapping[str, str]:
    import os

    return os.environ


_APP = None
_L4 = None


def train_on_l4() -> dict:
    """L4 body. Reload the volume, then resume or stop. No invented loss."""

    import modal

    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    try:
        volume.reload()
    except Exception:
        pass
    result = train_on_volume(Path(CHECKPOINT_DIR), volume)
    print(json.dumps(result, sort_keys=True))
    return result


def launch() -> None:
    raise SystemExit(run_gated_launch(launch=_remote_l4))


def _remote_l4(resources: Mapping) -> None:
    """The only call that asks Modal for the L4."""

    del resources
    if _L4 is None:
        build_modal_app()
    _L4.remote()


def _launch_l4(resources: Mapping) -> None:
    build_modal_app()
    _remote_l4(resources)


def build_modal_app():
    """Register the L4 function. Decorating it does not start a GPU."""

    global _APP, _L4
    if _APP is not None:
        return _APP
    import modal

    app = modal.App("burmeseai-train")
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    registered = app.function(
        gpu=GPU,
        cpu=CPU_CORES,
        memory=MEMORY_MIB,
        timeout=TIMEOUT_SECONDS,
        retries=modal.Retries(max_retries=RESUME_RETRY_CAP, initial_delay=1.0),
        volumes={CHECKPOINT_DIR: volume},
        serialized=True,
    )(train_on_l4)
    app.local_entrypoint()(launch)
    _APP = app
    _L4 = registered
    return app


def _invoked_by_modal_cli() -> bool:
    return Path(sys.argv[0]).name == "modal"


def _refuse_modal_cli_without_tokens() -> None:
    if not _invoked_by_modal_cli():
        return
    import os

    missing = missing_token_names(os.environ)
    if not missing:
        build_modal_app()
        return
    print(missing_token_message(missing))
    raise SystemExit(2)


_refuse_modal_cli_without_tokens()


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--smoke" in args:
        run_smoke()
        return 0
    return run_gated_launch()


if __name__ == "__main__":
    raise SystemExit(main())

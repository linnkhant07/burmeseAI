# Reproducibility

This repository has no training run, no downloaded corpus, and no model weights. A measured cell stays empty until a command in this repository prints the value. The words smoke and target name the kind of check. They are not scores.

Scope, rates, and the source list are in [PLAN.md](PLAN.md).

## Environment

Python 3.10 or newer. Continuous integration uses Ubuntu and CPython 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -c "import burmeseai"
pytest
```

The workflow checks out the tree, installs the `dev` extra, imports `burmeseai`, and runs `pytest`.

Importing `burmeseai` does not import sibling modules. Zawgyi conversion, fertility, training, retrieval, evaluation, and the app can be absent from the tree. A caller imports a module when that module is needed.

## Pins

The files below are not in git. A later run names these same inputs unless the report records a change.

| Input | Pin |
| --- | --- |
| App and training base | `Qwen/Qwen3-0.6B`, Apache-2.0. Product calls and eval scripts pass `enable_thinking` false. The chat template default is true. |
| Zawgyi | `github.com/arckara/zawgyi` at `e66860c30b616227b070ab12cb9e995dd99917fb`, Node 20, function `convertLine`. The license at that commit is Apache-2.0. Keep that notice. |
| Fertility corpus | `facebook/flores` devtest, configs `eng_Latn` and `mya_Mymr`, 1,012 pairs. `openlanguagedata/flores_plus` lists the same split sizes and is not interchangeable. A switch has to be named in the report. |
| External fertility cell | Petrov, La Malfa, Torr, and Bibi, NeurIPS 2023, Table 1, ChatGPT/GPT-4 column, Burmese 11.70, on their sample of about 2,000 FLORES-200 sentences, tokenizer `cl100k_base`. This is the only external fertility number a cell may hold. |

## Ignored paths

Git ignores `data/`, `weights/`, `artifacts/`, `checkpoints/`, `*.gguf`, `*.safetensors`, `*.bin`, and `*.pt`. A corpus download, a weight file, a GGUF, a safetensors file, and a checkpoint stay on disk and out of the history.

## Commands owned by the other modules

This package shell does not implement those modules. The entry points below are the ones the plan names. When the module file is missing, `import burmeseai` still succeeds.

- `python -m burmeseai.fertility`. A missing corpus writes the status `baseline not measured`.
- `python -m burmeseai.train --smoke`. Two CPU optimizer steps on the fixture. The smoke does not import Modal.
- `modal run burmeseai/train.py`. Exit status 2 when `MODAL_TOKEN_ID` or `MODAL_TOKEN_SECRET` is unset. The command does not request a GPU when the fertility file is missing, or when a workspace usage budget of $20 cannot be set. Resume loads the latest checkpoint. The retry cap is one resume.
- Evaluation scripts for Belebele (900 four-choice questions, 0-shot, thinking off), FLORES chrF++ in both directions, all 3,920 BURMESE-SAN items, GSM8K, and a frozen HumanEval subset. The same driver names unmodified Qwen3-0.6B, Gemma 4 E2B, Gemma 4 E4B, SEA-LION v4.5 E2B-IT, Padauk Q8, and an Arc path that stays empty until a weight file exists. Absent data writes `not measured`. Predictions are written only by a run.

## Training budget, when a GPU run is allowed

One Modal L4, 2 CPU cores, 16 GiB of host memory. At the published list rates recorded in the plan, 18 hours at that request is $18.385056 before retries and egress. The operator sets the workspace usage budget to $20 and does not launch when that cap cannot be set. A cost note in a run log is arithmetic from the rate card. It is not an invoice. Billing uses the max of the request and the measured usage, so the request stays at 16 GiB.

## How to fill a cell

1. Run the command that owns the cell.
2. Copy the value that command printed.
3. Leave every other cell as it was.

A missing corpus, a missing weight file, or a skipped check stays empty, or it keeps the status string that command prints (`baseline not measured`, `not measured`). Do not estimate a cell from a paper, a model card, or a previous draft.

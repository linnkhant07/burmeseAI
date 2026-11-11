# On-device Burmese assistant

Plan only. This repository has no training run, no downloaded corpus, and no model weights. A cell is either a named external citation or empty. Nothing in an empty cell is a result from this repository.

Decision: the first artifact is an offline app on unmodified `Qwen/Qwen3-0.6B` (Apache-2.0), with thinking turned off, mixed-line Zawgyi conversion, and a refusal when safety retrieval misses. A later training run uses that same checkpoint on one Modal L4, and only after a fertility file exists. The operator sets the workspace usage budget to $20 and does not launch if that cap cannot be set. At the published base rates, 18 hours on one L4 with 2 cores and 16 GiB is $18.385056 before retries and egress. Gemma 4 E2B, Gemma 4 E4B, SEA-LION v4.5 E2B-IT, and `WYNN747/ai4burmese-padauk-gguf` are same-harness baselines. Arc 1.0 is not a dependency. This plan has no quality-per-watt target and no date by which a trained checkpoint must ship.

The owner note at `internal/research-plan.md` was checked and is not binding. Section 10 records what was opened, what was not, and where later reading disagreed with the first draft of this plan.

## 1. Product scope

Ship a small Burmese and English assistant that runs with the network off.

The first shipped artifact is that app on the unmodified post-trained checkpoint `Qwen/Qwen3-0.6B`. `apply_chat_template` defaults `enable_thinking` to true. The product call passes false. Mixed-line Zawgyi conversion runs before the model sees the string. If the safety index is empty or the top score is below the cutoff, the app refuses and shows the disclaimer. A trained checkpoint is published with whatever it measured. It replaces the base in the app only when the report labels it a success.

- Laptop first. The 180-minute slice ships the network-off process and leaves resident memory and tokens per second empty until a process has actually run.
- Offline translation in both directions, English and Burmese, from the model that is loaded. No separate NLLB weights.
- Zawgyi-to-Unicode through `convertLine` in `github.com/arckara/zawgyi`, pinned at commit `e66860c30b616227b070ab12cb9e995dd99917fb` (the HEAD read on 11 November 2026). The repository LICENSE at that commit is Apache License 2.0. The build keeps that notice.
- Local retrieval over earthquake, heat, and health text. In the 180-minute slice the paragraphs are written for this project, and each chunk carries the license this project chooses. An external page is added only after that page's license is written into the chunk. An answer that states a safety fact quotes a chunk id.
- English chat stays on the unmodified checkpoint until a trained checkpoint passes the English label in section 7.

Out of scope for the first build: accounts, servers, speech, images, and any tool that needs a network during a reply. Weights and corpora stay out of git. Apple Silicon, MLX, and an 8 GB Android-class device are empty measurement rows. They are not part of the 180 minutes. Padauk's Q8 file is outside the 8 GB product target (section 3).

Why offline is the product:

- Freedom House, *Freedom on the Net 2026*, Myanmar, scores the country Not Free at 9/100. The abridged report says telecommunications infrastructure continued to sustain damage after the 2021 coup, the military kept control of major providers, and the coverage period included localized internet shutdowns plus blocking of news sites and anticensorship tools. <https://freedomhouse.org/country/myanmar/freedom-net/2026>
- DVB, dated 7 October 2026, says the Yangon and Mandalay utilities announced the rotation on Sunday to start Monday. The caption records a blackout in Yangon's Thaketa Township on 5 October. Power in those two cities is four hours on and four hours off. Ayeyarwady Region is two hours on and six hours off. The article attributes the rationing to heat and a strained grid, and it ties longer outages to the March 2025 earthquake. <https://english.dvb.no/rolling-blackouts-across-major-cities-as-heatwave-and-power-grid-collapse-intensify/>

A model that calls a remote API is down whenever the shutdown or the blackout is. The on-device copy has to answer from local weights and local files.

## 2. Architecture

One process. No network socket is opened at reply time.

1. Zawgyi gate, per span. The 180-minute build calls `convertLine` from the pinned `arckara/zawgyi` commit, under Node 20, and keeps the Apache notice. The tool splits a line on whitespace, scores each span with the Google detector, and chooses keep or convert. A Burmese line with no spaces is one span, and the call then collapses to a line decision. The original string is kept beside the Unicode string. Threshold 0.95 on the whole line is the comparison the hand-built fixture is meant to beat, not the product path. Their README reports, and this plan does not reproduce, 93.3% exact match versus 1.0% for whole-line detection at 0.95 on "a Zawgyi run inside a Unicode line", and 77.8% versus 0.3% on scattered Zawgyi words, on a 11,049-case bench built from 2,500 Burmese Wikipedia sentences. That bench text is CC BY-SA 4.0 and stays out of training.
2. Router, keyword and script based. Three lanes: safety (earthquake, heat, health), translation, general chat.
3. Safety lane. BM25 over overlapping Unicode grapheme n-grams of width 3, 4, and 5. Return the top spans with ids. The model may only verbalize a span that is in the result. An empty index refuses. A score below the cutoff refuses. Recall at 5 of 0.80 on a future 20-question set is a success bar for when that set exists. It is not a blocker for the first app.
4. Translation lane. The loaded model, prompted, English to Burmese or Burmese to English.
5. Chat lane. The first app loads unmodified `Qwen/Qwen3-0.6B` with `enable_thinking` false. A trained checkpoint replaces that file only after section 7 labels it a success.
6. Measurements. The network-off entry point exists in the 180-minute slice. Resident memory and tokens per second stay empty until a real process writes them. Empty rows are reserved for Apple Silicon, MLX, and an 8 GB Android-class device. Those rows are not promised in this slice. A trained-checkpoint GGUF export sits after the slice.

The in-repo Zawgyi fixture is a few hand-built mixed lines. Expected Unicode is stored after a local run of the pinned `convertLine`. Those rows must pass the span path.

## 3. Base model

Train, when the fertility gate and the budget cap allow it, `Qwen/Qwen3-0.6B`.

The model card README lists `license: apache-2.0`, 0.6B parameters, 0.44B non-embedding, 28 layers, and a context of 32,768: <https://huggingface.co/Qwen/Qwen3-0.6B>. The same card lists the base model as `Qwen/Qwen3-0.6B-Base`, so `Qwen/Qwen3-0.6B` is the post-trained chat checkpoint. Official `config.json` lists `vocab_size` 151936 and `tie_word_embeddings` true, and `max_position_embeddings` 40960. The README context and the config length are both recorded. The product does not need either limit resolved beyond that record. The build still reads `tokenizer.json` before any resize, because a config vocab can include unused slots. `Qwen/Qwen3-1.7B` is Apache-2.0, 1.7B parameters, 1.4B non-embedding, 28 layers, README context 32,768, and the same config pattern of `vocab_size` 151936, `tie_word_embeddings` true, and `max_position_embeddings` 40960: <https://huggingface.co/Qwen/Qwen3-1.7B>. The 1.7B model is considered only after this 0.6B run has measured numbers and only under a new budget.

`apply_chat_template` defaults `enable_thinking` to true. Every product call and every eval script passes false, and the script writes that flag into its output.

Why this base:

- The license is Apache-2.0.
- The parameter count can sit on a laptop. Gemma 4 E2B is 2.3B effective parameters and 5.1B with embeddings. That embedding table is the wrong storage budget for an 8 GB device.
- Petrov et al., NeurIPS 2023, show that an English-heavy byte-level BPE is an expensive way to write Burmese. Whether this tokenizer needs an extension is a measurement, made before any GPU launch (section 4).

Landscape:

| Model | What the source says | Role here |
| --- | --- | --- |
| Qwen3-0.6B and 1.7B | Apache-2.0. Post-trained chat checkpoints of the matching `-Base` models. README context 32,768. `config.json`: `vocab_size` 151936, `tie_word_embeddings` true, `max_position_embeddings` 40960. | 0.6B is the app and the only training base. 1.7B waits on a new budget. |
| Gemma 3 | Model card: 270M, 1B, 4B, 12B, 27B; text and image in, text out; 140+ languages. The Gemma Terms of Use list Gemma 3 and point Gemma 4 at a separate license. <https://ai.google.dev/gemma/docs/core/model_card_3> <https://ai.google.dev/gemma/terms> | Not the training base. |
| Gemma 4 E2B, E4B, and 26B A4B | Model card: Apache 2.0, and the license URL canonicalizes to <https://ai.google.dev/gemma/apache_2>, titled Apache License 2.0. E2B is 2.3B effective (5.1B with embeddings). E4B is 4.5B effective (8B with embeddings). Vocabulary 262K. E2B and E4B context 128K, with text, image, and audio. 26B A4B is a mixture of experts: 25.2B total parameters, 3.8B active, 8 active experts out of 128 plus one shared expert, 256K context, text and image, no audio. <https://ai.google.dev/gemma/docs/core/model_card_4> | E2B and E4B are eval baselines, from local GGUF on CPU when the file is present. They are not loaded onto the L4 job. 26B A4B is the row Arckara's teacher line names. That teacher is outside this $20 run. |
| SEA-LION v4.5 E2B-IT | Card updated 2026-05-19: post-trained from `google/gemma-4-E2B-it`, logit distillation from `google/gemma-4-31B-it` on SEA-Instruct, languages listed include Burmese, same tokenizer as Gemma 4 E2B. The repository LICENSE file is MIT. The card's `license_link` points at the Gemma 4 license page, which resolves to Apache License 2.0. The card says the model has not been aligned for safety. <https://huggingface.co/aisingapore/Gemma-SEA-LION-v4.5-E2B-IT> | Eval baseline on CPU when the file is local. Same size objection as E2B. |
| Padauk GGUF | `WYNN747/ai4burmese-padauk` is a LoRA whose card names base model `unsloth/gemma-4-E4B-it` and license string `gemma`. `WYNN747/ai4burmese-padauk-gguf` is public and ungated, architecture `gemma4`, context length 131072, one file `padauk-gemma-q8_0.gguf`. The model API reports `totalFileSize` 7,972,725,088 bytes (7.42 GiB). The card says standalone llama.cpp at or above `b8751` is required because of Gemma 4 shared keys and values. The card publishes no data mix and no scores. | Shipped same-harness baseline, run on CPU when the file is local. This Q8 file is not the 8 GB product model. |
| MyanmarGPT | The project README states a 128M model and a 1.42B MyanmarGPT-Big. The Hugging Face card's license field is `creativeml-openrail-m`, and the card body requires a citation. <https://huggingface.co/jojo-ai-mst/MyanmarGPT> | Off the training path. |
| Arckara Arc 1.0 | Model page: open weights expected December 2026, 4B effective, LoRA on Gemma 4 E4B, still in training. Teacher line "Gemma 4 26B", which matches the 26B A4B row above. Their Belebele read of 900 Burmese questions, unreproduced here: Gemma 4 E4B 77.3, Gemma 4 E2B 60.4, SEA-LION v4.5 E2B 59.9. No script, shot count, or thinking flag is on that page. <https://arckara.com/models> | Empty harness row until a weight file exists. The path no-ops. Do not wait. |

## 4. Tokenizer extension

Published fertility used here is one external cell. Petrov, La Malfa, Torr, and Bibi, *Language Model Tokenizers Introduce Unfairness Between Languages*, NeurIPS 2023, <https://arxiv.org/abs/2305.15425>, measure a premium: tokens in a language divided by tokens in the English parallel text. The proceedings table lists Burmese at 16.89 under GPT-2/RoBERTa, 11.70 under ChatGPT/GPT-4, and a dash under FlanT5. Shan in the ChatGPT/GPT-4 column is 15.05. The paper names `cl100k_base` as the tokenizer of the then-current ChatGPT and GPT-4, and it uses about 2,000 FLORES-200 sentences. Arckara's essay cites this as about 11.7 times: <https://arckara.com/research/myanmar-ai-gap>. Arckara's mT5 "about 1.6 times" summary is unused. A clean Burmese mT5 cell was not recovered from the extracted PDF.

The owner note's fertility premiums are unreproduced. They are not written into the table below or into any result cell.

Public fertility table. Every measured cell starts empty. The script fills a cell only when it has printed that premium. Corpus pin: `facebook/flores` devtest, configs `eng_Latn` and `mya_Mymr`, 1,012 pairs. `openlanguagedata/flores_plus` also lists 997 dev and 1,012 devtest, and the sentences can differ. A switch to `flores_plus` has to be named in the report.

| Tokenizer | External | Measured premium, this repository |
| --- | --- | --- |
| `cl100k_base` | Petrov Table 1, ChatGPT/GPT-4 column, Burmese 11.70, on their sample of about 2,000 sentences. This is the only external fertility cell. | empty |
| `o200k_base` | empty | empty |
| Unmodified Qwen3-0.6B | empty | empty |
| Extended Qwen3-0.6B | empty | empty until the gate extends |
| Gemma 4 E2B tokenizer | empty | empty until those files are local |

Gate, before any GPU launch. The CPU script writes the unmodified Qwen3-0.6B premium on the pinned split, or it writes "baseline not measured" when the files are absent. Rules:

- If the measured Qwen3-0.6B premium is under 3.0, skip vocabulary extension.
- If it is 3.0 or higher, a CPU step after that file exists and before `modal run` trains an auxiliary byte-level BPE, adds the most frequent tokens that are absent from the Qwen vocabulary, and stops at 10,000. The count is the config field `max_new_tokens`, and the value in that field is at most 10,000. There is no separate "stop when the curve flattens" rule. The cap of 10,000 is the default in Yamaguchi et al., who cite Tejaswi et al. for no significant gains past 10,000 tokens on their setups. It is a cap, not a claim that 10,000 is the Burmese optimum.
- Success bars for an extended tokenizer: premium at most 3.0, and at most 0.60 times the measured unmodified Qwen premium, on the same pinned pairs. A miss is a row in the report. The checkpoint and the table are still published.
- Initialization follows Yamaguchi et al.: each new row is the mean of the source pieces for that string, on the input embedding and on the output head. Official `config.json` sets `tie_word_embeddings` true, so those two matrices are one tied matrix until the loaded checkpoint says otherwise. The build reads the loaded model before resize.
- The new tokenizer is saved next to the full checkpoint. It is not committed.
- The 180-minute milestone does not fit a BPE over MADLAD. That milestone resizes and mean-initializes the fixture only.

## 5. Data plan

No corpus, tokenizer file larger than a fixture, or weight file is committed. The ignore list covers `data/`, `weights/`, `*.gguf`, `*.safetensors`, `*.bin`, `*.pt`, and checkpoints. The CPU fixture is a few dozen short strings written by hand, plus the mixed-line Zawgyi lines.

A data-mix card ships beside any checkpoint. Each trained source gets the license notice from the snapshot that was downloaded, a document or token count, a synthetic count, a teacher name, the filters, and the number of native reviewers. The first run may say synthetic count 0, teacher none, and native reviewers none. The MADLAD notice is in that card, not only in a log line. Zero synthetic rows and no teacher is an acceptable first card.

Continued pretraining, in this order, when the GPU run happens:

1. `allenai/MADLAD-400`, language `my`, streamed. Kudugunta et al. used `google/myanmar-tools` and converted documents whose Zawgyi probability was above 50%. Appendix Table 9 lists Myanmar (Burmese), script Mymr: 176.5K noisy documents, 172.4K clean documents, 1.3B noisy characters, 1.3B clean characters. Those last two cells are characters, not tokens. The Hugging Face card says that for Myanmar script the clean split matches the noisy split, because whitespace sentence scores were invalid, and that the card's license line is CC-BY-4.0. The paper's datasheet says the hosted release is ODC-BY plus the Common Crawl terms. The mix card records the notice on the snapshot that was downloaded. A second pass uses the same pinned `convertLine`, not a whole-line 0.95 decision. Drop leftovers that fail a Burmese-script ratio check.
2. English replay fraction is 0 until the run config names a source that is present and whose license is neither non-commercial nor share-alike. CC-BY-SA English is not that source. The fraction is a field in the config, and it stays 0 while the field is empty.
3. CulturaX `my` stays unused until a Hugging Face token exists and the mC4 and OSCAR terms are written down. The dataset card lists 865,575 Burmese documents and 882,606,546 tokens (their tokenizer), says the license follows mC4 and OSCAR, and its download example sets `use_auth_token=True`. <https://huggingface.co/datasets/uonlp/CulturaX>. The owner note's 6.46 GB and 2.58 GB community repacks were not checked and are not inputs.

Translation and instruction:

- FLORES-200 is CC-BY-SA 4.0. The Hugging Face `facebook/flores` card lists dev 997 and devtest 1,012. Dev and devtest are evaluation only. The first training mix does not include them.
- NLLB-MD is CC-BY-NC 4.0. It is excluded. NLLB-SEED is CC-BY-SA 4.0 and stays out of the first mix with the other share-alike parallel sets.
- Belebele and BURMESE-SAN are evaluation only (section 7).
- Aya Burmese is 472 rows, as counted by Yamaguchi et al. The license was not re-opened here. Zero Aya rows enter training until that license is read, recorded on the mix card, and is neither non-commercial nor share-alike.
- WikiHow-MY, the translated instruction mix of about 5,000 rows, ShweYon, myX-Tokenizer, and alphaedge trimmed vocabularies were not opened. They are not inputs.

Safety text is separate from the training mix. It is not used as supervised targets. The model quotes a chunk or refuses.

Web share, checked on the Common Crawl language table, crawl CC-MAIN-2026-34: Burmese (`mya`) 0.0164%, Thai 0.3740%, Vietnamese 1.0750%, English 40.4526%. <https://commoncrawl.github.io/cc-crawl-statistics/plots/languages>. Arckara cites the same 0.0164% figure for that crawl. A later column on the same table, CC-MAIN-2026-43, lists `mya` at 0.0157%.

## 6. Training plan

Provider: Modal. One command for the full run. The CPU smoke needs no credential.

Credential the later run needs, both values, as documented by Modal (`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` are the token pair; `.modal.toml` is the other place the client reads them): <https://modal.com/docs/reference/modal.config>. If either variable is unset, the full command exits with status 2 and does not request a GPU.

```
python -m burmeseai.train --smoke
modal run burmeseai/train.py
```

The smoke command runs two optimizer steps on the fixture, on CPU, including the fixture-scale resize and per-token mean init. It writes a smoke block. It does not import Modal and does not read the token variables.

An L4 function cannot set `nonpreemptible`. Preemption restarts the same input. The function loads the latest volume checkpoint and continues, or it exits without requesting a GPU when no checkpoint exists. The retry cap is one resume. That resume cannot start a second 18-hour run from step zero. The operator sets the workspace usage budget to $20 before launch. The command does not start if that cap cannot be set. A post-credit spend limit is a different control and is not this compute cap. Starter lists $30 of monthly compute credits, so a spend limit of $20 would not hold usage at $20 while those credits remain. This plan does not count on the credits. The maximum budget on a new account can depend on prior charges. If the dashboard will not accept $20, there is no launch.

Billing uses the max of the request and the measured usage, so a host-memory spike above the request raises the bill. The request stays at 16 GiB. No region pin. A pinned region is 1.15 to 1.75 times the base price.

Rates from <https://modal.com/pricing>:

| Resource | List rate | 18-hour charge at the request |
| --- | --- | --- |
| L4 GPU | $0.000222 / sec = $0.7992 / hour | $14.3856 |
| CPU, 2 cores | $0.0000131 / core / sec = $0.09432 / hour | $1.69776 |
| Memory, 16 GiB | $0.00000222 / GiB / sec = $0.127872 / hour | $2.301696 |
| Sum at 16 GiB | $1.021392 / hour | $18.385056 |

The same 18 hours at 24 GiB of host memory add $1.150848 and total $19.535904, about $19.54, before retries and egress. At 32 GiB they add $2.301696 and total $20.686752, about $20.69, before retries and egress, which is already past the cap. Volume storage is $0.09 per GiB-month. Egress is $0.04 per GiB past the included allotment. The volume holds the resized full model and the tokenizer. It is removed only after that full checkpoint has been copied out.

Inside the 18 hours, in order:

1. Exit with no GPU if the fertility file is missing, if the workspace usage budget is not $20, or if either token variable is unset.
2. If the gate says extend: resize, mean-initialize, then continued pretraining. Sequence length 512, the length Yamaguchi et al. used. AdamW. Learning rate `2e-5` as the starting value, recorded in the config. Batch size 4 with gradient checkpointing. On an out-of-memory error the batch halves. The learning rate is a starting setting, not a tuned result. English replay uses the fraction in the config, which is 0 until a source is named.
3. Merge with the original chat checkpoint. The mix ratio is a required config field. The first recorded value is 0.5 continued-pretraining weights and 0.5 original chat weights, and that value is a starting setting, not a tuned result. Save the resized full model and the tokenizer. An adapter alone drops the new embeddings and the pretraining.
4. Run LoRA on the attention and MLP projections, rank 16, alpha 32, learning rate `1e-4`, only when the mix card lists at least one permissive instruction source. If it lists none, skip LoRA and still save the merged checkpoint. LoRA, when it runs, uses the time left after the merge and at most 2 hours.
5. Write tokens seen, wall clock, and a cost note from the rates above. The cost note is an estimate from the rate card, not an invoice. On preemption, step 2 or 3 reloads the latest checkpoint under the retry cap.

Yamaguchi et al., <https://arxiv.org/abs/2412.11704>, trained 500M tokens after adding 10K tokens, on Qwen2.5 7B and Llama 3.1 8B. Their Qwen3 runs are Qwen3 14B on Amharic, Bengali, and Telugu only. 500M tokens is not this run's budget. This run stops on the clock and logs how many tokens it managed. There is no teacher pass. Gemma 4 26B A4B does not fit the $20 cap.

Full Belebele, FLORES, BURMESE-SAN, GSM8K, and the frozen HumanEval subset are not part of this reservation. They are scripts, run later, on CPU for the large GGUF baselines.

## 7. Eval plan

Three labels stay separate: external citation, baseline or trained measurement in this repository, and smoke. Empty cells stay empty. The first shipped artifact does not wait on the numeric bars. Those bars label a checkpoint as a success or a miss. A miss is published with its measurements and does not replace the unmodified base in the app. The 1.7B fallback stays behind a new budget.

The 3.0-point English bar is stricter than the GSM8K drops of up to 6 points that Yamaguchi et al. report for their recovered Qwen3 14B models. The bar is a label, not a calibrated promise.

Fertility success, only after the script has filled the Qwen cell: extended premium at most 3.0 and at most 0.60 times that measured Qwen premium. A miss still publishes the table.

| Check | External figure | This repository | Smoke | How a trained checkpoint is labeled |
| --- | --- | --- | --- | --- |
| Fertility premium on the pinned `facebook/flores` devtest, 1,012 pairs | Petrov Table 1, ChatGPT/GPT-4, Burmese 11.70, about 2,000 sentences. No other external fertility number. | empty | Fixture counts only, once the smoke exists. A missing corpus writes "baseline not measured". | Success when the extended premium is at most 3.0 and at most 0.60 times the measured Qwen premium. Otherwise a miss. The table is published either way. |
| Belebele `mya_Mymr`, 900 four-choice questions, 0-shot, `enable_thinking` false. Chance is 25. Eval only. | Arckara's page, unreproduced, and with no script or shot count: E4B 77.3, E2B 60.4, SEA-LION v4.5 E2B 59.9. | empty until the script runs | None. Not a smoke fixture. | Success when accuracy is at least 5.0 points above the unmodified Qwen3-0.6B and at least 30.0. The output records the shot count and the thinking flag. |
| FLORES devtest chrF++, both directions, pinned split | None adopted | empty until the script runs | None | Success when each direction is at least 2.0 chrF++ above the unmodified model. |
| BURMESE-SAN, 3,920 samples, 7 tasks. Eval only. | Paper point estimates: Gemini 2.5 Pro 72.35, GPT-5 66.46, GPT-4o (2024-11-20) 51.61, ERNIE 4.5 54.68, Llama 3.3 70B 23.07. The paper prints a ± on the table cells. This plan does not copy those ± values. <https://arxiv.org/abs/2602.18788> | empty until the script runs | None | Success when the macro average is at least the unmodified model's macro average. The report names each Table 6 license, including sentiment's Unknown. |
| GSM8K exact match, and a frozen HumanEval subset | None adopted | empty until the script runs | None | Success when the absolute drop versus the unmodified model is at most 3.0 points on each. This label is stricter than the drops Yamaguchi et al. report. |
| Safety retrieval | None | empty. The 20 questions do not exist yet. | The 180-minute slice checks that an empty index refuses. | When a 20-question set with known chunk ids exists, success is recall at 5 of at least 0.80. That bar is not a blocker for the first app. Every safety fact in an answer carries a chunk id. A miss refuses. |
| Mixed-line Zawgyi fixture | Their README's 93.3%, 1.0%, 77.8%, and 0.3% stay external and unreproduced. | Expected Unicode stored after a local `convertLine` run | Those hand-built rows | The span path matches the stored Unicode on every fixture row. Whole-line 0.95 is the comparison, not the implementation. |
| Resident memory and tokens per second | None | empty until a real process writes them | empty | No numeric speed bar. The product requirement is a network-off process. |
| Apple Silicon, MLX, 8 GB Android-class device | None | empty | empty | Not part of this slice. |
| Colloquial and code-switching | None. BURMESE-SAN's sources are a retranslated benchmark. Calling that benchmark formal-only is a reading of those sources, not a label printed in the paper. | No score until the file contains at least 20 human-written items. The items are eval-only. | None | No numeric target. |

Same driver, after the 180-minute slice, accepts unmodified Qwen3-0.6B, Gemma 4 E2B, Gemma 4 E4B, SEA-LION v4.5 E2B-IT, Padauk's Q8 GGUF, and an Arc 1.0 path that no-ops until a weight file is present. Predictions are written only by a run. Large GGUF baselines run on CPU when the file is local. They are not part of the L4 function.

Belebele license conflict, unresolved: the Belebele GitHub README says the test set is CC-BY-SA 4.0 and the assembled training set is CC-BY-NC. BURMESE-SAN Table 6 lists Belebele as CC BY-NC 4.0. This plan does not train on Belebele either way. BURMESE-SAN's own release is CC-BY-SA 4.0. Table 6 also lists myXNLI as CC BY-NC 4.0, myHateSpeech and XL-Sum as CC BY-NC-SA 4.0, FLORES+ as CC BY-SA 4.0, Balanced COPA as CC BY 4.0, and GKLMIP-mya sentiment as Unknown.

Scripts that score the full sets are required before any of those cells can be filled. Writing and running them is outside the 180 minutes and outside the $20 reservation.

## 8. Milestones

About three hours after this plan is approved. The GPU command is not part of those three hours. It waits for the token pair, the fertility file, and a workspace usage budget of $20 that the dashboard has accepted.

| Minutes | Deliverable |
| --- | --- |
| 30 | Package layout and ignore rules for data and weights. |
| 35 | Mixed-line Zawgyi wrapper: pinned `convertLine`, Node 20, Apache notice, and the hand-built fixture. |
| 30 | Fertility script for the columns in section 4, including a path that writes "baseline not measured". |
| 35 | Fixture-scale resize, per-token mean init, and `python -m burmeseai.train --smoke` for two CPU steps. |
| 30 | `modal run burmeseai/train.py`: exit 2 when a token is missing, exit with no GPU when the $20 usage budget cannot be set, load the latest checkpoint, retry cap of one resume. |
| 20 | Report schema with the empty baseline rows, including Padauk and the empty Arc path, and the safety refusal when the index is empty. |

Total 180 minutes.

After that slice, and still not on the $20 reservation: the CPU BPE fit once a fertility file exists and the gate says extend; the data-mix card; the full Belebele, FLORES, BURMESE-SAN, GSM8K, and frozen HumanEval scripts; a trained-checkpoint GGUF; MLX; a phone build. The colloquial file publishes no score until it holds at least 20 human-written items.

## 9. Risks

- Continued pretraining of the post-trained chat checkpoint can damage chat. Yamaguchi et al. document that, and their recovered Qwen3 14B models still showed GSM8K drops of up to 6 points. The English label here is a drop of at most 3.0 points. A miss is published and stays out of the app. The countermeasure inside this budget is the merge with the original chat checkpoint, plus English replay only after a named permissive source exists. Replay starts at fraction 0.
- The first mix may list no permissive instruction source. LoRA is skipped in that case. The merged full checkpoint is still saved. The app the user has in the meantime is the unmodified 0.6B model.
- Whole-line Zawgyi detection at 0.95 is the method their bench says fails on mixed lines. The product path is `convertLine`. A line with no spaces is still one span.
- Formal and colloquial Burmese are different. The colloquial and code-switching block has no score until 20 human-written items exist, and those items stay out of training.
- Instruction data is scarce. 472 Aya rows, if the license clears, will not by themselves make a chat model.
- Belebele's license is in conflict across sources. Training on it is out.
- MADLAD's card and the paper name different attribution licenses. The mix card records the notice from the snapshot that was downloaded.
- CC-BY-SA parallel text stays out of the first mix.
- Safety answers can harm someone if the model improvises. An empty index and a low score both refuse. The disclaimer says the pack is not emergency care.
- Preemption restarts the same input. Without a checkpoint load, that restart would repeat training from step zero. The function exits instead when no checkpoint exists, and the retry cap blocks a second full run. An L4 function cannot turn preemption off.
- Billing follows max(request, usage). A memory spike above 16 GiB costs more. 32 GiB for 18 hours is about $20.69 before retries and egress.
- A region pin, or a larger GPU, breaks the $18.385056 estimate. The code requests an L4 only.
- If the workspace will not accept a $20 usage budget, the run does not start. A spend limit that applies only after credits is not a substitute.
- Arc 1.0 may appear later and post higher Belebele numbers on a 4B model. Their teacher line "Gemma 4 26B" matches the card's 26B A4B mixture of experts, 25.2B total and 3.8B active. That teacher is outside this budget. Padauk is already a public Gemma 4 E4B LoRA in GGUF, with no published scores, and the Q8 file is 7.42 GiB.
- The Qwen premium might already be under 3.0. The gate then skips extension. The checkpoint and the empty-or-filled table are still the record.
- Power cuts also hit anyone training locally in Myanmar. Training stays on Modal. Inference stays on the device.
- This slice does not measure watts, and it does not promise a ship date.

## 10. Source check

Verified by opening the page, the paper, or the file:

- Common Crawl language table, CC-MAIN-2026-34, `mya` 0.0164%. Later crawl CC-MAIN-2026-43, `mya` 0.0157%.
- Arckara research page <https://arckara.com/research/myanmar-ai-gap>: the 0.0164% claim and the "about 11.7 times" claim, the latter attributed there to Petrov et al.
- Petrov et al. 2023, Table 1: Burmese 16.89, 11.70, and a dash. The 11.70 cell is the ChatGPT/GPT-4 column. Shan in that column is 15.05. The paper names `cl100k_base` and uses about 2,000 FLORES-200 sentences.
- Freedom House Freedom on the Net 2026, Myanmar, 9/100 Not Free, abridged, as summarized in section 1. The page does not itself give the shutdown counts quoted elsewhere from Access Now. Those counts were not opened.
- DVB, dated 7 October 2026: Sunday announcement for a Monday start, Thaketa caption on 5 October, the two rotation schedules, heat, and the March 2025 earthquake, as summarized in section 1.
- Qwen3-0.6B and Qwen3-1.7B README cards, and official `config.json`: Apache-2.0, the parameter lists in section 3, `vocab_size` 151936, `tie_word_embeddings` true, README context 32,768, `max_position_embeddings` 40960. `Qwen/Qwen3-0.6B` lists base model `Qwen/Qwen3-0.6B-Base`. The chat template defaults `enable_thinking` to true.
- Gemma 3 model card and the Gemma Terms of Use. Gemma 4 model card, including the 26B A4B mixture-of-experts row (25.2B total, 3.8B active). The Gemma 4 license page <https://ai.google.dev/gemma/apache_2> is titled Apache License 2.0. This plan did not diff that page byte for byte against the Apache Software Foundation's canonical file.
- SEA-LION v4.5 E2B-IT card and repository LICENSE: MIT text in the LICENSE file, `license_link` to the Gemma 4 license page, distillation from Gemma 4 31B, Burmese included among the fine-tuning languages, Gemma 4 E2B tokenizer.
- Padauk cards: LoRA base `unsloth/gemma-4-E4B-it`, license string `gemma`, GGUF architecture `gemma4`, context 131072, file `padauk-gemma-q8_0.gguf`, API `totalFileSize` 7,972,725,088 bytes, llama.cpp at or above `b8751`. The card publishes no mix and no scores. The Q8 file was not re-downloaded for this plan.
- MyanmarGPT project README: 128M and 1.42B MyanmarGPT-Big. Hugging Face license field `creativeml-openrail-m`, body requires a citation.
- Arckara model page: December 2026, Belebele 77.3 / 60.4 / 59.9, teacher line "Gemma 4 26B", Gemma 4 E4B as the base they are training. Those Belebele scores are their page's figures and were not reproduced.
- `github.com/arckara/zawgyi` at `e66860c30b616227b070ab12cb9e995dd99917fb`: Apache License 2.0, `convertLine`, Node 20. The README bench percentages in section 2 were read there and were not reproduced.
- CulturaX card, Burmese row 865,575 documents and 882,606,546 tokens, license defers to mC4 and OSCAR, download example uses `use_auth_token=True`.
- MADLAD-400 paper Table 9 and the Zawgyi paragraph; HF card CC-BY-4.0 line and the note that Myanmar-script clean data matches noisy data.
- FLORES licenses: FLORES-200 and NLLB-SEED CC-BY-SA 4.0, NLLB-MD CC-BY-NC 4.0. `facebook/flores` dev 997 and devtest 1,012. `flores_plus` lists the same split sizes and is not interchangeable without a named switch.
- Belebele GitHub README: test set CC-BY-SA 4.0, assembled training set CC-BY-NC, 900 questions, code `mya_Mymr`.
- BURMESE-SAN HTML: 7 tasks, 3,920 samples, release CC-BY-SA 4.0, Table 6 source licenses, and the point estimates quoted in section 7.
- Yamaguchi et al., arXiv 2412.11704: +10K tokens, mean of source pieces, sequence length 512, 500M-token CPT on Qwen2.5 7B and Llama 3.1 8B, Qwen3 14B only on Amharic, Bengali, and Telugu, 472 Aya Burmese rows, MADLAD Burmese 176.5K articles and 1.3B characters in Table 20 (the body also says 172k documents), GSM8K drops of up to 6 points on their recovered Qwen3 14B models.
- `google/myanmar-tools` README: detection, 0.05 and 0.95 threshold heuristics. The product path does not use whole-line 0.95 as the converter.
- Modal pricing page and Modal config docs: the rates in section 6 and the token variable names. An L4 function cannot set `nonpreemptible`. Preemption restarts the same input. The workspace usage budget is the control that caps usage before credits. A spend limit applies after credits and is not the same control. Billing uses max(request, usage).

Not verified, so not used as inputs and not placed in result cells:

- The owner note's fertility premiums, including per-sentence token counts. Petrov 11.70 is the only external fertility cell.
- Community corpus sizes 6.46 GB and 2.58 GB.
- ShweYon, myX-Tokenizer, alphaedge vocabulary trims, WikiHow-MY, and the translated instruction mix of about 5,000 rows.
- myWord. The index uses grapheme n-grams of width 3, 4, and 5.
- Access Now shutdown totals and the SEA-HELM numbers quoted by Arckara.
- Aya's license text.
- A SEA-LION model at 27B. The card opened here is the E2B distilled from `google/gemma-4-31B-it`.
- Arckara's Belebele 77.3 / 60.4 / 59.9 as anything but that page's figures.
- The Zawgyi README bench percentages, beyond their presence in that README.
- Padauk's training mix, teacher, and any quality number.
- Whether `tokenizer.json` length equals `vocab_size` 151936. The build checks the file before resize.
- CulturaX upstream mC4 and OSCAR terms, until they are written down next to a Hugging Face token.
- Modal budget behavior on an account whose maximum cap depends on prior charges. The launch rule covers that case by refusing to start.

Disagreements with the owner note, and corrections to the first draft of this plan:

- The note's fertility premiums are not a reason to start GPU work, and they are not result cells. The size, the Apache-2.0 license, and a fertility script are. SEA-LION is a multilingual distillation that includes Burmese.
- Mean init is the per-piece mean Yamaguchi et al. specify. The tied embedding matrix is what `tie_word_embeddings: true` implies.
- Their 500M-token CPT was run on 7B and 8B models. This budget cannot copy it, and their Qwen3 experiments skipped Burmese.
- Belebele as CC BY-NC matches BURMESE-SAN Table 6 and conflicts with the Belebele repository README.
- A teacher at 26B A4B or 31B does not fit the $20 run. The first draft of this plan said "Gemma 4 26B" was absent from Google's card. That was wrong. The card lists 26B A4B. The teacher is still outside this run. A SEA-LION model at 27B was not on the card that was opened.
- The first artifact is the offline app on the unmodified 0.6B checkpoint. Numeric bars label a later checkpoint. They do not gate that first app.
- Whole-line Zawgyi detection is the wrong product path for mixed lines. `convertLine` is the path this slice calls.

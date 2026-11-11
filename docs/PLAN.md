# On-device Burmese assistant

Plan only. This repository has no training run, no downloaded corpus, and no model weights. Figures below are either copied from a named source or marked as a target or an unmeasured cell. Nothing here is a result from this repository.

Decision: train `Qwen/Qwen3-0.6B` (Apache-2.0) on Modal, one Nvidia L4, hard stop 18 hours. At the published base rates that reservation is $18.39 before egress. The workspace spend limit is $20. Gemma 4 E2B, Gemma 4 E4B, and SEA-LION v4.5 E2B-IT stay eval baselines. Arc 1.0 is unpublished, so it is not a dependency. Zawgyi conversion and safety answers stay outside the language model: a deterministic converter, then local retrieval with a citation.

The owner note at `internal/research-plan.md` was checked and is not binding. Section 10 records what was opened, what was not, and where the note and the sources differ.

## 1. Product scope

Ship a small Burmese and English assistant that runs with the network off.

- Laptop first. A phone build is in scope only after a quantized checkpoint has a measured memory figure on the laptop.
- Offline translation in both directions, English and Burmese.
- Zawgyi-to-Unicode on every input and on every training document, using Google `myanmar-tools`, before the model sees the string.
- Local retrieval over a small pack of Burmese earthquake, heat, and health safety pages. Each chunk stores its title, source URL, and license. An answer that states a safety fact quotes a chunk id. If retrieval misses, the assistant refuses and shows the disclaimer. It does not invent medical or emergency instructions from model weights.
- English chat remains available. A Burmese adaptation that wrecks English is a failed run.

Out of scope for the first build: accounts, servers, speech, images, and any tool that needs a network during a reply. Weights and corpora stay out of git.

Why offline is the product, not a preference:

- Freedom House, *Freedom on the Net 2026*, Myanmar, scores the country Not Free at 9/100. The abridged report says telecommunications infrastructure continued to sustain damage after the 2021 coup, the military kept control of major providers, and the coverage period included localized internet shutdowns plus blocking of news sites and anticensorship tools. <https://freedomhouse.org/country/myanmar/freedom-net/2026>
- DVB, 7 October 2026, reports rolling blackouts from 6 October 2026: Yangon and Mandalay on a four-hours-on, four-hours-off rotation, and harsher two-hours-on, six-hours-off cuts in Ayeyarwady Region, during a heatwave and a strained grid. The same article ties longer outages to the March 2025 earthquake. <https://english.dvb.no/rolling-blackouts-across-major-cities-as-heatwave-and-power-grid-collapse-intensify/>

A model that calls a remote API is down whenever the shutdown or the blackout is. The on-device copy has to answer from local weights and local files.

## 2. Architecture

One process. No network socket is opened at reply time.

1. Zawgyi gate. `google/myanmar-tools` returns a Zawgyi probability. Automatic conversion uses threshold 0.95, the project README's setting for the case where a false Zawgyi flag is costly (fewer than 1% of Unicode strings wrongly flagged, on their stated heuristic). Conversion uses the same project's Zawgyi-to-Unicode rules, or ICU transform `Zawgyi-my` where the Python client has detection only. The original string is kept beside the Unicode string so a bad conversion can be reversed by the user.
2. Router, keyword and script based, not a second model. Three lanes: safety (earthquake, heat, health), translation, general chat.
3. Safety lane. BM25 over Unicode grapheme clusters in the local pack. Burmese does not use spaces between words, so the index does not depend on a word segmenter. Return the top spans with ids. The model may only verbalize a span that is in the result. Below the score cutoff, refuse.
4. Translation lane. The same small model, prompted, English to Burmese or Burmese to English. No separate NLLB weights in the product (see data licenses).
5. Chat lane. `Qwen3-0.6B` after vocabulary extension, continued pretraining, and a short LoRA. Thinking mode is off for the product build so a phone-sized context is not spent on a hidden trace.
6. Packaging. After a real training run, export a GGUF for laptop CPU and record resident memory and tokens per second in the eval report. Those two numbers do not exist yet.

The safety pack is data, not code, and it is still small enough to read. It is not committed until each page's license allows an offline copy. Until that pack exists, the safety lane refuses every safety question and the eval fixture is the only index.

## 3. Base model

Train `Qwen/Qwen3-0.6B`.

Verified from the model card README front matter and overview (`license: apache-2.0`, 0.6B parameters, 0.44B non-embedding, 28 layers, context 32,768): <https://huggingface.co/Qwen/Qwen3-0.6B>. `Qwen/Qwen3-1.7B` is the same license and the same context length, with 1.7B parameters and 1.4B non-embedding: <https://huggingface.co/Qwen/Qwen3-1.7B>. The 1.7B model is a later candidate if the 0.6B eval gate fails. It is not the model this $20 run trains. A converted-checkpoint card lists vocabulary size 151,936 for the 0.6B tokenizer; the official card does not. The build confirms the size from `tokenizer.json` before any resize.

Why this base:

- The license is Apache-2.0, which matches a weight release we can actually ship.
- The parameter count is the one that can plausibly sit on a laptop and, after quantization, on a phone. Gemma 4 E2B is 2.3B effective parameters and 5.1B with embeddings (Google model card). That embedding table is the wrong storage budget for the first phone build.
- Petrov et al., NeurIPS 2023, show that an English-heavy byte-level BPE is an expensive way to write Burmese. Qwen3 is the model whose tokenizer we will measure and, if the premium is high, extend. The measurement is a gate, not a completed result (section 4).

Landscape, from the pages opened:

| Model | What the source says | Role here |
| --- | --- | --- |
| Qwen3-0.6B and 1.7B | Apache-2.0. Dense causal LMs. 32,768 context. | 0.6B is the training base. 1.7B is the fallback after a failed gate. |
| Gemma 3 | Model card: 270M, 1B, 4B, 12B, 27B; text and image in, text out; 140+ languages. The Gemma Terms of Use list Gemma 3 and point Gemma 4 at a separate license. <https://ai.google.dev/gemma/docs/core/model_card_3> <https://ai.google.dev/gemma/terms> | Baseline family only if a checkpoint is already local. The terms are a worse fit than Apache-2.0 for a weight release. |
| Gemma 4 E2B and E4B | Model card: Apache 2.0. E2B 2.3B effective (5.1B with embeddings), E4B 4.5B effective (8B with embeddings), 262K vocabulary, 128K context, text, image, and audio. <https://ai.google.dev/gemma/docs/core/model_card_4> | Eval baselines. Too heavy, and already a large vocabulary, so they are the wrong place to spend the $20. |
| SEA-LION v4.5 E2B-IT | Model card, updated 2026-05-19: MIT license, post-trained from `google/gemma-4-E2B-it`, logit distillation from `google/gemma-4-31B-it` on SEA-Instruct, languages listed include Burmese. Same tokenizer as Gemma 4 E2B. The card says the model has not been aligned for safety. <https://huggingface.co/aisingapore/Gemma-SEA-LION-v4.5-E2B-IT> | Strong open baseline for Burmese eval. Same size objection as E2B. Safety alignment is our retrieval layer, which this card does not provide. |
| MyanmarGPT | `jojo-ai-mst/MyanmarGPT` is a GPT-2 Burmese model; the card lists about 0.1B parameters and a citation requirement. The project README describes a 128M model and a 1.42B `MyanmarGPT-Big` under CreativeML OpenRAIL-M. <https://huggingface.co/jojo-ai-mst/MyanmarGPT> | Historical Burmese LM. Not the base. |
| Arckara Arc 1.0 | Arckara's model page: open weights expected December 2026, 4B effective, LoRA on Gemma 4 E4B, still in training. Their pre-training Belebele read of 900 Burmese questions: Gemma 4 E4B 77.3, Gemma 4 E2B 60.4, SEA-LION v4.5 E2B 59.9. <https://arckara.com/models> | Do not wait. Those three scores are Arckara's measurements, used as external reference points, not as our baselines. |

## 4. Tokenizer extension

Published fertility (verified):

- Petrov, La Malfa, Torr, and Bibi, *Language Model Tokenizers Introduce Unfairness Between Languages*, NeurIPS 2023. <https://arxiv.org/abs/2305.15425>. They measure a premium, tokens in a language divided by tokens in the English FLORES-200 parallel text. In the extracted Table 1, the Burmese row is 16.89, 11.70, and a dash. The column header groups GPT-2/RoBERTa, then ChatGPT/GPT-4, then FlanT5. The paper names `cl100k_base` as the tokenizer of the then-current ChatGPT and GPT-4, and it says Shan reaches about 15 times on that tokenizer, which matches the 15.05 in the second group. Burmese in that same group is 11.70. Arckara reports this as about 11.7 times English: <https://arckara.com/research/myanmar-ai-gap>.
- The same paper says byte-level models still miss parity, and that Burmese, Dzongkha, and Shan need more than four times as many Unicode characters as Yue Chinese for the same FLORES text. Subword pieces that cover Burmese clusters are the remedy they argue for. mT5-style multilingual tokenizers are described by Arckara, citing Petrov, as about 1.6 times. That 1.6 is Arckara's summary; it was not re-read as a table cell in this pass.

Not verified, from the owner note, and not used as facts:

- `cl100k_base` 11.62 times on FLORES-200 devtest only (1,012 sentences). Petrov's 11.70 is on their FLORES-200 sample of about 2,000 sentences. The two numbers can both be honest. This plan does not adopt 11.62 until our script prints it.
- Qwen3-0.6B 8.93 times (243.7 Burmese tokens per sentence, 27.3 English, 1.51 tokens per Burmese character).
- Gemma 4 E2B-it and SEA-LION v4.5 E2B-IT 2.41 times. The shared tokenizer is verified, so one measured premium would apply to both. The number 2.41 is not.
- GPT-4o `o200k_base` 3.16 times.

Gate, before any GPU launch: a CPU script tokenizes the FLORES-200 devtest English and Burmese sides (1,012 aligned sentences; split sizes from the public FLORES card, 997 dev and 1,012 devtest) with `cl100k_base` and with the unmodified Qwen3-0.6B tokenizer. It writes the premium. Rules:

- If the Qwen3-0.6B premium is under 3.0, skip vocabulary extension and spend the GPU time on continued pretraining plus LoRA.
- If it is 3.0 or higher, train a byte-level BPE on the Unicode training sample, add up to 10,000 tokens that are absent from the Qwen vocabulary, and stop earlier if a held-out fertility curve flattens. The cap of 10,000 is the default in Yamaguchi et al., who cite Tejaswi et al. for "no significant gains when adding more than 10K tokens" on their setups. It is a cap, not a claim that 10,000 is the Burmese optimum.
- Initialization follows Yamaguchi et al., section on vocabulary expansion: each new row is the mean of the embeddings, and of the output-head rows, of the source tokens that the current tokenizer produces for that string. That is the per-token mean from Yao et al. 2021 as they apply it, not one global mean of the whole matrix.
- Resize both the input embedding and the output head. Confirm whether they are tied before assuming a single matrix.
- The new tokenizer is saved next to the checkpoint. It is not committed.

## 5. Data plan

No corpus, tokenizer file larger than a fixture, or weight file is committed. The later build's ignore list covers `data/`, `weights/`, `*.gguf`, `*.safetensors`, `*.bin`, `*.pt`, and checkpoints. The CPU fixture is a few dozen short strings written by hand.

Continued pretraining, in this order:

1. `allenai/MADLAD-400`, language `my`, streamed. Kudugunta et al. used `google/myanmar-tools` and converted documents whose Zawgyi probability was above 50%. Appendix Table 9 lists Myanmar (Burmese), script Mymr: 176.5K noisy documents, 172.4K clean documents, 1.3B noisy characters, 1.3B clean characters. Those last two cells are characters, not tokens. The Hugging Face card says that for Myanmar script the clean split matches the noisy split, because whitespace sentence scores were invalid, and that the card's license line is CC-BY-4.0. The paper's datasheet says the hosted release is ODC-BY plus the Common Crawl terms. Both notices are attribution licenses; the build records whichever file is on the snapshot it downloads and keeps the Common Crawl terms. Run Zawgyi detection again at threshold 0.95. Drop leftovers that fail a Burmese-script ratio check.
2. English replay: 1 sequence in 10 is English text from a permissive source already on disk. The fraction is a starting setting, written into the run config.
3. CulturaX `my` is the reserve corpus, not the first download. The dataset card lists 865,575 Burmese documents and 882,606,546 tokens (their tokenizer), 0.01% of CulturaX, and says the license follows mC4 and OSCAR. <https://huggingface.co/datasets/uonlp/CulturaX>. Confirm those upstream terms before a single CulturaX row is trained on. The owner note's 6.46 GB and 2.58 GB community repacks were not checked and are not inputs.

Translation and instruction:

- FLORES-200 is CC-BY-SA 4.0 (Facebook Research FLORES README). Dev and devtest are evaluation only.
- NLLB-MD is CC-BY-NC 4.0 on that same README. It is excluded. NLLB-SEED is CC-BY-SA 4.0 and is held for a later translation stage, after the share-alike obligation is written down. The first GPU run does not use it.
- Belebele and BURMESE-SAN are evaluation only (section 7).
- Aya Burmese is 472 rows, as counted by Yamaguchi et al. in the Aya dataset. The license was not re-opened here. Zero Aya rows enter training until that license is read and recorded.
- The owner note's WikiHow-MY, `burmese_instruction_tuning`, ShweYon vocabulary, myX-Tokenizer, and alphaedge trimmed vocabularies were not opened. They are not inputs. Trimming a vocabulary cannot add Burmese pieces the tokenizer never had; extension is the operation this plan uses when the fertility gate says so.

Safety pack, separate from the training mix: earthquake, heat, and health pages whose license allows offline copies. Chunks carry title, URL, and license. The pack is not used as supervised targets. The model quotes it or refuses.

Web share that motivates the corpus work, checked on the Common Crawl language table, crawl CC-MAIN-2026-34: Burmese (`mya`) 0.0164%, Thai 0.3740%, Vietnamese 1.0750%, English 40.4526%. <https://commoncrawl.github.io/cc-crawl-statistics/plots/languages>. Arckara cites the same 0.0164% figure for that crawl. A later column on the same table, CC-MAIN-2026-43, lists `mya` at 0.0157%.

## 6. Training plan

Provider: Modal. One command for the full run. The CPU smoke needs no credential.

Credential the later run needs, both values, as documented by Modal (`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` are the API token pair; `.modal.toml` is the other place the client reads them): <https://modal.com/docs/reference/modal.config>. If either variable is unset, the full command exits with status 2 and does not request a GPU.

```
python -m burmeseai.train --smoke
modal run burmeseai/train.py
```

The smoke command trains two optimizer steps on the tiny fixture, on CPU, and writes a smoke block (loss values and the fixture fertility counts). It does not import Modal and does not read the token variables.

The Modal command is one preemptible Function, because the non-preemptible CPU and memory rates are 3 times list price and would leave the $20 cap. Resources requested:

- 1 Nvidia L4
- 2 physical CPU cores
- 16 GiB memory
- timeout 18 hours
- no region pin (a pinned region is 1.15 to 1.75 times the base price on the pricing page)

Rates from <https://modal.com/pricing>, read as per-second list prices:

| Resource | List rate | 18-hour charge at the request |
| --- | --- | --- |
| L4 GPU | $0.000222 / sec = $0.7992 / hour | $14.3856 |
| CPU, 2 cores | $0.0000131 / core / sec = $0.09432 / hour | $1.69776 |
| Memory, 16 GiB | $0.00000222 / GiB / sec = $0.127872 / hour | $2.301696 |
| Sum | $1.021392 / hour | $18.385056 |

That is 18 GPU-hours and $18.39 of reserved compute at base rates. Volume storage is $0.09 per GiB-month; a checkpoint volume is deleted after the adapter is downloaded. Egress is $0.04 per GiB past the plan's included allotment. The operator sets a Modal workspace spend limit of $20. That limit is the stop that still works if a preempted Function restarts. Starter workspaces list $30 of monthly compute credits; this plan does not count on them.

Inside the 18 hours, in order:

1. Refuse to start if the fertility baseline file is missing.
2. If the gate says extend: resize, mean-initialize, then continued pretraining on the streamed Burmese sample with English replay. Sequence length 512, the length Yamaguchi et al. used. AdamW. Learning rate `2e-5` as the starting value for this 0.6B run, recorded in the config. Batch size 4 with gradient checkpointing; on an out-of-memory error the batch halves and the run continues. The learning rate is a starting setting, not a tuned result.
3. At 16 hours, or earlier if the stream ends, switch to LoRA on the attention and MLP projections, rank 16, alpha 32, learning rate `1e-4`, for the remaining time and at most 2 hours. Supervised rows are whatever permissive instruction mix was cleared in section 5. If fewer than 30 minutes remain, skip LoRA and save the continued-pretraining checkpoint.
4. Write tokenizer, adapter, tokens seen, wall clock, and a cost note computed from the rates above. The cost note is an estimate from the rate card, not an invoice.

Yamaguchi et al., <https://arxiv.org/abs/2412.11704>, trained 500M tokens after adding 10K tokens, on Qwen2.5 7B and Llama 3.1 8B, and their Qwen3 runs did not include Burmese. 500M tokens is not this run's budget. This run stops on the clock. It logs how many tokens it managed.

There is no teacher-model pass. A 31B teacher does not fit the $20 cap or the three-hour build.

## 7. Eval plan

Three columns stay separate in every report the harness writes: baseline (unmodified `Qwen3-0.6B`, same script), smoke (fixture or two-step loss only), target (the bar for the trained checkpoint). Empty cells stay empty. External numbers are cited in their own column and are never copied into ours.

| Check | External figure we may quote | Baseline in this repo | Smoke result | Target |
| --- | --- | --- | --- | --- |
| Fertility premium, FLORES-200 devtest, 1,012 pairs, MY tokens / EN tokens | Petrov Table 1, ChatGPT/GPT-4 column, Burmese 11.70 on their FLORES sample. Owner 11.62, 8.93, 2.41, 3.16 are not adopted. | Not measured | Fixture counts only, once the smoke exists | Extended tokenizer at most 0.60 times the measured Qwen3-0.6B premium. If that measured premium is above 5, the extended premium is also at most 5.0. |
| Belebele `mya_Mymr`, 900 four-choice questions. Chance is 25. Eval only. | Arckara, their own run: Gemma 4 E4B 77.3, E2B 60.4, SEA-LION v4.5 E2B 59.9. <https://arckara.com/models> | Not measured | None. The set is not a smoke fixture. | At least 5.0 points above the unmodified Qwen3-0.6B, and at least 30.0. |
| FLORES-200 devtest chrF++, both directions | None adopted | Not measured | None | Each direction at least 2.0 chrF++ above the unmodified model. |
| BURMESE-SAN, 3,920 samples, 7 tasks. Eval only. | Paper HTML: Gemini 2.5 Pro 72.35, GPT-5 66.46, GPT-4o (2024-11-20) 51.61, ERNIE 4.5 54.68, Llama 3.3 70B 23.07. <https://arxiv.org/abs/2602.18788> | Not measured | None | Macro average at least the unmodified model's macro average. |
| English GSM8K exact match, and a frozen HumanEval subset | None adopted | Not measured | None | Absolute drop of at most 3.0 points versus the unmodified model on each. |
| Safety retrieval, 20 questions written with known chunk ids, earthquake, heat, health | None | Not measured. The questions do not exist yet. | None | Recall at 5 of at least 0.80 on that future set. Every safety fact in an answer carries a chunk id. A miss refuses. |
| Zawgyi fixture | `myanmar-tools` README gives the 0.95 threshold heuristic. No accuracy was copied from them. | The converter is the reference | Exact match on the fixture once it exists | 100% of fixture rows match the reference converter |
| Laptop runtime | None | Not measured | Not measured | Report resident memory and tokens per second. No numeric speed target until a smoke has produced one. The product requirement is a CPU process on an 8 GB laptop with the network off. |

Belebele license conflict, unresolved: the Belebele GitHub README says the test set is CC-BY-SA 4.0 and that the assembled training set is CC-BY-NC. BURMESE-SAN Table 6 lists Belebele as CC BY-NC 4.0. The owner note says CC BY-NC. This plan does not train on Belebele either way. BURMESE-SAN itself says its release is CC-BY-SA 4.0, while several of its source tasks are NC or NC-SA (Table 6). Eval only, and the score report names the license of each task.

Regression and Burmese gains are both required. A checkpoint that misses any target in the last column is not the release candidate. The 1.7B fallback is considered only after this 0.6B run has real baseline and trained numbers, and only under a new budget.

## 8. Milestones

About three hours of build time after this plan is approved. The GPU command is not part of those three hours. It waits for the token pair and for the fertility file.

| Minutes | Deliverable |
| --- | --- |
| 45 | Package layout, ignore rules for data and weights, Zawgyi wrapper at threshold 0.95, and a short fixture whose conversion matches `myanmar-tools`. |
| 40 | Fertility script for `cl100k_base` and `Qwen3-0.6B` on FLORES-200 devtest when the files are local. If they are absent, the script writes "baseline not measured" and still prints fixture counts under the smoke label. |
| 40 | Vocabulary extension up to the 10,000 cap, per-token mean init, and `python -m burmeseai.train --smoke` for two CPU steps. |
| 35 | `modal run burmeseai/train.py` with the L4 request, the 18-hour timeout, and the exit-2 credential check. |
| 20 | Eval report with the three blocks in section 7, plus a local BM25 index over a handful of licensed safety paragraphs and a refusal when the index is empty. |

Total 180 minutes. GGUF export and a phone build wait until a trained checkpoint exists. They are not squeezed into this slice.

## 9. Risks

- English and instruction following can drop after vocabulary expansion. Yamaguchi et al. document that risk for chat models and spend the rest of their method on recovering it. Our countermeasure inside this budget is English replay plus the GSM8K and HumanEval gates, not a second full-size model.
- Zawgyi left in the corpus teaches the model the wrong encoding. Detection at 0.95 and a second pass over MADLAD are the controls. Short strings are ambiguous; the README says identical-looking strings can score anywhere, so the fixture includes consonants with few diacritics.
- Formal and colloquial Burmese are both in web text. The safety pack should be the register we are willing to quote. Web CPT will be noisier.
- Instruction data is scarce. 472 Aya rows, if the license clears, will not by themselves make a chat model. The first release is allowed to be a weak chat model with a strong retrieval path.
- Belebele's license is in conflict across sources. Training on it is forbidden until that is settled, which this plan does not expect to do.
- MADLAD's HF card and the paper name different attribution licenses. The run logs the notice it actually received.
- CC-BY-SA translation sets (FLORES-200, NLLB-SEED) are kept out of the first training mix so the weight license stays the Apache-2.0 of the base plus our changes.
- Safety answers can harm someone if the model improvises. Refusal on a retrieval miss is the default. The disclaimer says the pack is not emergency care.
- Modal preemption can restart the Function. Checkpoints and the $20 workspace limit bound the loss. A non-preemptible retry at 3 times the rate is a budget breach and is not allowed.
- A region pin, or an accidental larger GPU, breaks the $18.39 estimate. The code requests an L4 only.
- Arc 1.0 may appear in December 2026 and post higher Belebele numbers on a 4B model. That does not change the laptop and phone constraint, and their page's teacher line names a "Gemma 4 26B", which is not a size on Google's Gemma 4 model card (E2B, E4B, 12B Unified, 31B Dense).
- Fertility of Qwen3 might already be moderate. The gate exists so we do not spend the run on an extension the data does not support.
- Power cuts also hit anyone fine-tuning locally in Myanmar. Training stays on Modal. Inference stays on the device.

## 10. Source check

Verified by opening the page or the paper:

- Common Crawl language table, CC-MAIN-2026-34, `mya` 0.0164%. Later crawl CC-MAIN-2026-43, `mya` 0.0157%.
- Arckara research page <https://arckara.com/research/myanmar-ai-gap>: the 0.0164% claim and the "about 11.7 times" claim, the latter attributed there to Petrov et al.
- Petrov et al. 2023, Table 1 as extracted: Burmese 11.70 in the ChatGPT/GPT-4 column group. Full PDF table layout was reconstructed from the extracted text and from the paper's statement that Shan is about 15 times on that tokenizer.
- Freedom House Freedom on the Net 2026, Myanmar, 9/100 Not Free, abridged, shutdowns and damaged infrastructure as summarized in section 1. The page does not itself give the shutdown counts Arckara quotes from Access Now. Those counts were not opened.
- DVB, 7 October 2026, rotational blackouts and the heatwave, as summarized in section 1.
- Qwen3-0.6B and Qwen3-1.7B cards, Apache-2.0 and the parameter lists in section 3.
- Gemma 3 model card and the Gemma Terms of Use. Gemma 4 model card, Apache 2.0, sizes and 262K vocabulary. The full Gemma 4 license text was not compared clause by clause with the Apache 2.0 text.
- SEA-LION v4.5 E2B-IT card: MIT, distillation from Gemma 4 31B, Burmese included among the fine-tuning languages, Gemma 4 E2B tokenizer.
- MyanmarGPT card and the MinSiThu project README, sizes and licenses in section 3.
- Arckara model page: December 2026, Belebele 77.3 / 60.4 / 59.9, Gemma 4 E4B as the base they are training.
- CulturaX card, Burmese row 865,575 documents and 882,606,546 tokens, license defers to mC4 and OSCAR.
- MADLAD-400 paper Table 9 and the Zawgyi paragraph; HF card CC-BY-4.0 line and the note that Myanmar-script clean data matches noisy data.
- FLORES README licenses: FLORES-200 and NLLB-SEED CC-BY-SA 4.0, NLLB-MD CC-BY-NC 4.0. Dev 997 and devtest 1,012 from a public FLORES-200 card.
- Belebele GitHub README: test set CC-BY-SA 4.0, assembled training set CC-BY-NC.
- BURMESE-SAN HTML: 7 tasks, 3,920 samples, release CC-BY-SA 4.0, Table 6 source licenses, and the scores quoted in section 7.
- Yamaguchi et al., arXiv 2412.11704: +10K tokens, mean of source pieces, 500M-token CPT, 472 Aya Burmese rows, MADLAD Burmese 176.5K articles and 1.3B characters, Qwen3 experiments on Amharic, Bengali, and Telugu only.
- `google/myanmar-tools` README: detection, 0.05 and 0.95 threshold heuristics, Zawgyi-to-Unicode in Java and JavaScript, ICU elsewhere.
- Modal pricing page and Modal config docs: the rates in section 6 and the token variable names.

Not verified, so not used as inputs or as baselines:

- Owner fertility figures 11.62, 8.93, 2.41, and 3.16, including the per-sentence token counts.
- Community corpus sizes 6.46 GB and 2.58 GB.
- ShweYon (+1,418 tokens), myX-Tokenizer, alphaedge Qwen3 vocabulary trims, WikiHow-MY, and the translated instruction mix of about 5,000 rows.
- myWord segmenter. The index uses grapheme clusters instead.
- Access Now shutdown totals and the SEA-HELM numbers quoted by Arckara.
- Aya's license text.
- A Gemma 4 model at 26B, and a SEA-LION model at 27B. The owner note names both as possible teachers. The SEA-LION card opened here is the E2B distilled from Gemma 4 31B.

Disagreements with the owner note:

- The note's Qwen3 and Gemma 4 fertility numbers are not a reason to start GPU work. The size, the Apache-2.0 license, and a fertility script are. SEA-LION is a multilingual distillation that includes Burmese, which is stronger than "Burmese-distilled" and weaker than a Burmese-only model.
- The note's flat "mean-init" is tightened to the per-piece mean Yamaguchi et al. specify.
- Their 500M-token CPT was run on 7B and 8B models. This budget cannot copy it, and their Qwen3 experiments skipped Burmese.
- Belebele as CC BY-NC matches BURMESE-SAN Table 6 and conflicts with the Belebele repository README.
- A 31B or 27B teacher, and MCP-style tools, do not fit the $20 run or the three-hour build. Local retrieval replaces the tool layer for safety.
- Arc's published teacher name ("Gemma 4 26B") and the note's "SEA-LION 27B" were not found on the model cards opened here.

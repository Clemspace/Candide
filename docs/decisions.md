# Candide pretraining — decision log and ground-truth index

Working log for the ERC ModERN Candide effort (Nov 2026). Each entry: date,
decision, the rule it was decided by, the evidence, and where the evidence
lives. Numbers here must match a file under `results/` or a W&B run; when a
number has no file yet, it is marked **[no file]** and listed in
"Evidence still to commit" at the end. This file is the source the reference
document (`docs/reference/`, Overleaf) is written from: prose may be
rewritten, numbers and dates may not be changed without a new entry.

Machines: obtic1 (data, `/data/pretrain_candide`), obtic-gpu1 (one A40 46 GB,
training). Code: `github.com/Clemspace/Candide`, branch `refactor/composition-architecture`.
Release policy: the corpus is closed and not released; trained models are
released by ERC ModERN.

---

## 1. Data

### 1.1 Extraction (TEI -> text) — fixed 2026-11-10
- `scripts/extract_tei.py` replaces the previous extraction script. Three bugs in the old one:
  (1) the tail of a skipped element was emitted twice; (2) nested `<p>` were emitted
  once per level; (3) press files were dropped when their first 2,000 characters matched
  another file ("prefix dedup"). (1)+(2) inflated `modern_canon` about 3.3x (76% of its
  text then removed as internal repetition). (3) was not a bug in intent: content dedup
  later shows ~50% of the press is near-duplicate; prefix dedup is now off by default and
  replaced by content dedup (1.3).
- Known limit, not fixed: extraction keeps `<p>` only, so verse in `<l>` is dropped.
  **[to document: share of verse lost per sub-corpus]**
- Tests: `tests/scripts/test_extract_tei.py`.

### 1.2 Inventory and period policy — 2026-11-10
- `scripts/inventory_corpus.py`: TEI header date/genre/author, chars, words,
  function-word language guess per file -> `inventory.csv`, `inventory_summary.json`.
- Policy ("vintage LLM" reading horizon): French; published <= 1800; no lower bound;
  undated files kept. Exclusion = language != fr, or language score < 0.6, or year > 1800:
  19,614 files listed, 15,319 of them among the selected files. Post-1800 was ~1% of
  the corpus (posthumous / 19th-c. / modern editions). Latin: deferred to v3.
- Evidence: `/data/pretrain_candide/inventory_v2/inventory_summary.json`, `exclude_v2g.txt` **[no file in repo]**.

### 1.3 Deduplication v2g — 2026-11-10
- `scripts/dedup_corpus.py`: content-defined sampling of 200-byte windows (rolling hash,
  O(n)); a file is a near-duplicate when >= 80% of its sampled windows occur in earlier
  files (priority: canon, press, dicos, pamphlets, lettres, archive, merged last);
  internal repetition >= 0.3 drops a file; a correction pass restores files whose only
  source was dropped. Clean validation per sub-corpus: shared <= 2%, 0.5% of kept chars.
- Result (`dedup_v2g/dedup_summary.json` **[no file in repo]**):

| subcorpus | files | chars (M) | kept | near-dup | val files |
|---|---|---|---|---|---|
| modern_canon | 2,788 | 381 | 97.1% | 0.0% | 13 |
| modern_press | 55,701 | 6,339 | 49.1% | 50.6% | 163 |
| modern_dicos | 84 | 269 | 100.0% | 0.0% | 3 |
| modern_pamphlets | 21,208 | 599 | 99.7% | 0.0% | 129 |
| modern_lettres | 39,361 | 84 | 61.7% | 0.0% | 70 |
| modern_archive | 2,379 | 521 | 89.0% | 10.1% | 16 |
| modern_merged | 30,148 | 6,951 | 89.7% | 9.0% | 121 |
| total | 151,669 | 15,143 | 73.3% | 25.7% | 515 |

  (kept + near-dup < 100% where files were excluded by 1.2.)
- `modern_merged` = Glenn Roe's newer corpus with external sources (incl. Hugging Face
  historical data): deduplicated last so that it only adds what the v1 tiers lack.

### 1.4 Tokenizer v2g — 2026-11-10
- The v1 tokenizer is lossy (whitespace/punctuation pre-split + WordPiece decoder:
  "L'Encyclopédie" -> "L ' Encyclopédie"): a blocker for any released model.
- v2g: byte-level BPE, lossless (100% round-trip on validation files), French elision
  pre-split (" l'", " qu'", " lorsqu'"...), NFC only, digits split, vocabulary 32,132
  including 133 specials (pad 0, unk 1, bos 2, eos 3 kept from v1; reserved 4-131;
  [MASK] 132). Trained on 18,237 files / 2,003M chars with a 10% share of OCR-noisy text
  (`modern_archive`), following P.-C. Langlais's 90/10 suggestion. Noisy shares 0 / 0.1 /
  0.3 gave indistinguishable evaluation: the language filter had already removed most
  noisy files. Evidence: `tokenizer_v2g/tokenizer.meta.json`, tokenizer eval JSON **[no file in repo]**.
- Script and tests: `scripts/train_tokenizer.py`, `tests/scripts/test_train_tokenizer.py`.

### 1.5 Shards v2g — 2026-11-10
- `scripts/prepare_shards.py --manifest dedup_v2g/dedup_manifest.csv`: train 103,910 files
  -> **2,863M tokens** (29 shards); val 515 files -> 14.9M tokens.
- Encyclopédie (separate shard set, `shards_enc`): ARTFL encoding, 17 volumes,
  **74,199 articles, 20.0M words**; train 26.5M tokens (vols 1-16), val 1.5M (vol 17,
  held out to measure the cooldown). Source bug found and handled: two `<index>` tags
  per article left unclosed, which made parsers nest the article inside metadata (78% of
  volume 1 came out empty); `<index>` metadata is saved per article in
  `ENC_volumeNN.articles.tsv` (head, type, author, attribution, domain, POS).
  `encyclopediana.xml` (a different work) is excluded.
- Canon cooldown set (`shards_canon`): `modern_canon` files of the v2g manifest. **[token count pending]**
- Unique training tokens for sizing: 2,863M + 26.5M = **2.89B**.

### 1.6 Duplication audit of the v1 shards — 2026-11-09
- `scripts/dup_audit.py` (50-token k-grams, 1/32 content-defined sample): 36% of
  training 50-grams repeated; top repeat (Gospel of Nicodemus) ~470x; the v1 validation
  shard was 55% internally repeated. This invalidated the v1 validation loss as a
  decision metric (-> 2.3). Evidence: dup_audit JSON **[no file in repo]**.

## 2. Architecture pilot (GPT+ vs anGPT)

- Arms: GPT+ = pre-norm RMSNorm, GQA, SwiGLU, QK-norm, untied head (Qwen3-like);
  anGPT = approximately normalized transformer (LERP residuals, unit-ball weight rows).
  Pilot shape 8L x 768, 118.6M params, v1 tokenizer and v1 shards.
- Rule (fixed before stage 2): adopt anGPT only if better by > 2x the seed spread at
  comparable wall-clock; otherwise GPT+ (the release format).

### 2.1 Stage 1b LR sweep (100M tokens, v1 validation) — 2026-11-08/09
| arm | LR | final val loss |
|---|---|---|
| GPT+ | 6e-4 / **1.2e-3** / 2.4e-3 | 3.650 / **3.556** / 3.580 |
| anGPT | 2e-3 / 4e-3 / **8e-3** / 1.6e-2 | 3.682 / 3.646 / **3.580** / 3.736 |
- Predictions written before the sweep were wrong (all of them). Evidence: W&B group `lr-sweep` **[export not in repo]**.
- 2026-11-07: global window shuffle replaced sequential shard reading; it removed a loss
  collapse at step ~340 and cut anGPT's clipped steps from 37-79% to 7-10%.

### 2.2 Stage 2 (300M tokens, seeds 1 and 2) — 2026-11-09/10
- On the v1 validation: anGPT lower by ~0.28. On fresh text (memorization probe,
  bucket "seen once"): GPT+ lower by 0.18 (s1) and 0.17 (s2).
- Prediction written before stage 2 (anGPT 0.15 nats lower): wrong in sign on held-out text.
- The GPT+ validation plateau/rise after ~2k steps is explained by memorizing corpus
  duplicates (1.6) evaluated on a duplicated validation shard.
- Evidence: probe JSONs (`scripts/memorization_probe.py`) **[no file in repo]**, W&B group stage 2 **[export not in repo]**.

### 2.3 Decision on clean validation — 2026-11-10
- Deferral recorded 2026-11-09 because of 1.6; rule fixed 2026-11-10: same 2x-spread rule
  on a clean validation sample (`shards_v1f/val`, deduplicated, v1 tokenizer to match the checkpoints).
- `results/2026-11-10/rescore_v1f.json`: GPT+ 3.6349 / 3.6362; anGPT 3.8046 / 3.7994.
  Gap 0.166 nats vs threshold ~0.010. **GPT+ adopted.**
- Caveat: LRs were selected on the contaminated v1 validation.

## 3. Optimizers (code, 2026-11-09; commit a3081d3)
- Muon (Newton-Schulz quintic, Nesterov, match_rms scaling, AdamW inside for
  embeddings/head/1-D), AdEMAMix (paper update, alpha/beta3 warmups), AdamW/SGD as torch
  subclasses (fused on CUDA). Correctness tests: `tests/training/optimizers/`.

## 4. Full run v2g (in progress)

### 4.1 Sizing — 2026-11-10
- `scripts/size_model.py` (Muennighoff et al. 2023 data-constrained law, Eq. 17; only the
  N/D split is used; constants fitted on C4). Inputs: 2.89B unique tokens, 48 A40-hours,
  15.7 TFLOP/s achieved (118.6M params at 22k tok/s, measured before the server reboot).
  Output: N ~ 149M, D ~ 3.03B tokens = 1.03 epochs -> **compute-bound, not data-bound**.
  Optimum flat: half/double params cost +0.044 / +0.078 nats (C4 law).
- Shape: 18L x 768, 12 heads (4 KV), SwiGLU d_ff 2304, tied embeddings, 148.6M params.
- Provisional: max_steps 46,305 and keep_steps, to be reset from measured throughput.

### 4.2 Schedule and data mix — 2026-11-10
- WSD, 1000 warmup steps, 1-sqrt cooldown over the last 20% (Hägele et al. 2024),
  `pre_cooldown.pt` kept as branch point for scaling-law cooldowns.
- Mix (`configs/full/gptplus_v2g.yaml`): stable {main 0.988, enc 0.012}; cooldown
  {main 0.80, canon 0.15, enc 0.05}. Expected passes: Encyclopédie ~1.1 + ~1.15,
  canon ~0.8 + ~1.0, main ~1.0 (Muennighoff: repetition harmless up to ~4 epochs).
  Annealing on high-quality text: MiniCPM, Llama 3, OLMo 2.
- Extra validation: Encyclopédie vol 17 (`val_enc`), canon (`val_canon`).

### 4.3 LR check (AdamW) — started 2026-11-10 ~15:30
- 100M tokens each, full-run shape and mix, LR 6e-4 and 1.2e-3. Rule: lower final val
  loss wins; if gap < 0.01 nats take 6e-4. **[result pending]**

### 4.4 Muon check and throughput — planned 2026-11-10 evening
- `scripts/bench_throughput.py` (AdamW/Muon x compile x micro-batch). Muon at LR 6e-4
  and 1.2e-3 with settings identical to 4.3. Rule: Muon adopted iff its best final loss
  beats AdamW's best by > 0.01 nats and its per-step overhead is < 10%.
  Literature: Muon/SOAP ~1.4x over tuned AdamW at 0.1B, ~1.1x at 1.2B (Wen et al. 2025). **[pending]**

## 5. Open questions and missing pieces (for the reference document)
- Cross-tokenizer comparison: v2g losses are not comparable with v1 losses (tokenizer,
  validation set, shape all changed). Needed: bits per character on identical documents.
- Evaluation chapter: downstream tasks for 18c French (e.g. POS on FreEM), comparison
  with existing French historical models; generated samples (illustration only).
- Release: extraction/memorization test before release, threshold to agree with partners;
  Hugging Face conversion (our RoPE rotates interleaved pairs: q/k rows need a per-head
  permutation; check with a logits-equality test).
- Encoder (ModernBERT-like, `configs/encoder_modernbert18c.yaml`): LR not tuned; run after the decoder.
- Corpus: verse lost to `<p>`-only extraction; noisy French files kept out; dictionary
  editions; Latin (v3).
- Data-constrained regime (if compute grows ~10x, e.g. SCAI access): high weight decay,
  ensembles + distillation (Kim et al. 2025), diffusion LMs for many epochs (Prabhudesai et al. 2025).

## Evidence still to commit (copy into `results/` and reference from the text)
- obtic1 `/data/pretrain_candide/`: `inventory_v2/inventory_summary.json`,
  `dedup_v2g/dedup_summary.json` (and v1f), `tokenizer_v2g/tokenizer.meta.json`,
  tokenizer eval JSON, `extraction_stats_*.json` (v1f, merged), `exclude_v2g.txt` (counts only if large).
- GPU server: `extracted_enc/extraction_stats_encyclopedie.json`, dup_audit JSON,
  probe JSONs (s1, s2), W&B exports of stage 1b and stage 2 (CSV), LR check, bench.

# Candide probe

An inference and inspection tool for checkpoints written by
`ramanujan/training/train_candide18c.py`. It reads each model's architecture from the
config pickled in the checkpoint and rebuilds it with the trainer's own `setup_model`
(loaded strictly, no `strict=False`). The same tool therefore opens the 29L GPT+ full run,
its branched cooldowns and the pilot anGPT arms without any change.

```
# on obtic-gpu1
PYTHONPATH=. python -m candide_probe --runs runs/ \
    --tokenizer /data/pretrain_candide/tokenizer/tokenizer.json

# on the laptop
ssh -N -L 7860:127.0.0.1:7860 obtic-gpu1     # then open http://127.0.0.1:7860
```

The server uses only the standard library plus `torch` and `tokenizers`. It listens on
127.0.0.1 by default. `--runs` takes run directories or `.pt` files; every
`*/checkpoints/*.pt` below them is listed. Up to `--max-loaded` models (default 3) stay in
memory.

## Views

| view | what it shows | what its numbers are |
|---|---|---|
| Complete | a continuation of a prompt; each token is shaded by its surprisal, with the alternatives on hover; you can click an alternative to continue from it | the model's own probabilities (temperature 1, no truncation); temperature, top-k, top-p and the repetition penalty only choose the token. The prompt is scored in the same forward pass |
| Score & diff | per-token surprisal of a passage for the selected checkpoint and the ticked ones; Δ view, cumulative Δ, largest gains and losses, Δ summed by token | windows, stride and beginning-of-text token of `scripts/score_text.py`: the total is `score_ids`' total |
| Lens | logit lens: the next token read from the residual stream after each block | the model's unembedding (final norm + tied head for GPT+, head × logit scale for anGPT). The last row is checked against the output |
| Attention | per head and block: attention to the previous token, to the first token (sink), to itself, and relative entropy. Full pattern of one head; the text shaded by where one position attends | weights recomputed explicitly inside every `scaled_dot_product_attention` call of one fp32 forward pass |
| Probes | candidates: which continuation each checkpoint prefers. Probe sets: JSON lines of pairs (`a`, `b`) or choices (`prompt`, `choices`, `answer`), summarised by `group` | `score_text.score_text` throughout: pairs are its `--pairs` mode |

Probe-set format:

```
{"id": "oit-1", "group": "-oit/-ait", "a": "Il étoit tard.", "b": "Il était tard."}
{"id": "d-1", "group": "dated", "prompt": "…", "choices": [" …", " …"], "answer": 0}
```

## Saved results

Nothing is written unless you click Save. A save writes
`results/<date>/probe_<kind>_<name>_<time>.json`, which holds the request, the full
response, each checkpoint's path, step, run and W&B id, the tokenizer, the code commit and
whether the working tree had uncommitted changes. That is what decisions.md asks of a
number. Single passages in Score are illustrations; evaluation numbers come from
`scripts/score_text.py` on a document set.

## Checks

```
PYTHONPATH=. python -m pytest tests/probe                  # tiny GPT+ and anGPT models, CPU
PYTHONPATH=. python -m candide_probe --selftest runs/full/candide-v2g-224m/checkpoints/final.pt
PYTHONPATH=. python -m candide_probe --demo                # interface only: random numbers
```

`--selftest` runs on the real checkpoint, at the precision the probe uses, and prints the
margins:

- the score total against `score_text`
- the lens last layer against the output
- the logits with attention recorded against the fused kernel
- greedy decoding with the KV cache against full recomputation

## Limits

- One tokenizer per server. A checkpoint records its vocabulary size, not the tokenizer
  that made its shards. v1 and v2g have the same size, so a pilot checkpoint (v1) passes
  the size check with the v2g tokenizer and gives wrong text. The side panel shows
  `data.shards_dir` so you can check.
- Lens: first 256 tokens. Attention: first 160 tokens.
- Read-only: no ablation or activation patching.

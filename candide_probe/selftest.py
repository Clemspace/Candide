"""Checks on a real checkpoint, on the machine that serves it (GPU, bf16):

    PYTHONPATH=. python -m candide_probe --selftest runs/full/candide-v2g-224m/checkpoints/final.pt

The unit tests (tests/probe/) run tiny models in fp32 on CPU. This runs the same checks on
the checkpoint you will look at, at the precision the probe uses, and prints the margins:

  score      per-token losses summed by the probe against score_text.score_ids (several windows)
  lens       last layer of the logit lens against the model output
  attention  logits with attention recorded against logits without (fp32)
  cache      greedy decoding with the KV cache against full recomputation (bf16 on GPU may
             drift after many tokens; reported, not failed)
"""
from __future__ import annotations

import time

SAMPLE = ("Il n'eſt point de bonheur ſans vertu. Les philoſophes ont ſoutenu que la raiſon humaine "
          "eſt capable de connoître les loix de la nature, & que l'homme, éclairé par elle, doit "
          "chercher dans ſes devoirs la ſource de ſa félicité. Le roi de France avoit déclaré la guerre ; "
          "les troupes ſeroient miſes en campagne ſi la paix ne ſe faiſoit point avant l'hiver. ")


def run(ckpt: str, tokenizer: str, device: str) -> int:
    import torch
    from .checkpoints import Registry
    from .engine import Engine, _AttentionRecorder, _ScorerBackend, _logits
    from .text import CandideTokenizer, score_text_module

    tok = CandideTokenizer(tokenizer)
    reg = Registry([ckpt], device=device, max_loaded=1)
    refs = reg.refs()
    if len(refs) != 1:
        print(f"expected one checkpoint at {ckpt}, found {len(refs)}")
        return 2
    cid = refs[0].id
    e = Engine(reg, tok)
    rows, failed = [], False

    def row(name, value, limit, hard=True):
        nonlocal failed
        ok = value <= limit
        failed |= hard and not ok
        rows.append((name, value, limit, "ok" if ok else ("FAIL" if hard else "note")))

    t0 = time.time()
    meta = e.meta(cid)
    loaded = reg.get(cid)
    print(f"{cid}: step {meta['step']}, {meta['model'].get('arch')} {meta['model'].get('n_layers')}L x "
          f"{meta['model'].get('d_model')}, {loaded.meta['n_params_built'] / 1e6:.1f}M params, "
          f"loaded in {time.time() - t0:.1f}s on {device}")
    if not meta["compatible"]:
        print("tokenizer incompatible:", "; ".join(meta["warnings"]))
        return 2
    row("params: built - read from checkpoint", abs(loaded.meta["n_params_built"] - meta["n_params"]), 0)

    text = SAMPLE * 12                                       # > 1024 tokens: several windows
    res = e.score([cid], text)
    ours = res["runs"][cid]
    S = score_text_module()
    ids = tok.encode(res["text"])
    backend = _ScorerBackend(loaded.model, tok, ours["max_len"], cid)
    ref_nll, ref_n = S.score_ids(backend, ids, ours["max_len"], ours["stride"], batch_size=8, device=device)
    print(f"score: {len(ids)} tokens, {ours['bits_per_byte']:.4f} bits/byte (a repeated paragraph: the model "
          f"copies after the first pass), {ref_n} scored by score_text")
    row("score: |probe - score_text| / total", abs(ours["nll_nats"] - ref_nll) / ref_nll, 1e-5)
    row("score: |probe - score_text| nats per token", abs(ours["nll_nats"] - ref_nll) / max(1, ref_n), 1e-5)
    row("score: tokens scored differ", abs(ours["n_scored"] - ref_n), 0)

    lens = e.lens(cid, SAMPLE)
    row("lens: last layer vs output, max |dp| (fp32)", lens["check"]["last_layer_vs_model_max_abs_prob_diff"], 1e-4)

    x = torch.tensor([[tok.bos_id] + tok.encode(SAMPLE)[:120]], device=device)
    with torch.no_grad():
        a = _logits(loaded.model(x)).float()
        with _AttentionRecorder(loaded.model) as rec:
            b = _logits(loaded.model(x)).float()
    row("attention: recorded vs fused, max |dlogit|", float((a - b).abs().max()), 1e-2)
    row("attention: blocks recorded vs n_layers", abs(len(rec.weights) - len(loaded.model.blocks)), 0)

    g = e.generate(cid, "Le roi de France avoit déclaré", greedy=True, max_new_tokens=48, stop_at_eos=False)
    seq = list(g["prompt_ids"])
    agree = 0
    for st in g["run"]["generated"]:
        with torch.no_grad(), (torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda")
                               else torch.autocast("cpu", enabled=False)):
            nxt = int(_logits(loaded.model(torch.tensor([seq], device=device)))[0, -1].argmax())
        if nxt != st["id"]:
            break
        agree += 1
        seq.append(nxt)
    n = len(g["run"]["generated"])
    print(f"generate: {g['tokens_per_second']} tok/s with cache; greedy text: {g['text'][:100]!r}")
    row(f"cache: greedy tokens before divergence (of {n}), shortfall", n - agree, 0, hard=False)

    print()
    print(f"{'check':<46}{'value':>12}{'limit':>10}  result")
    for name, value, limit, status in rows:
        print(f"{name:<46}{value:>12.3g}{limit:>10.3g}  {status}")
    print("\nall hard checks passed" if not failed else "\nSOME CHECKS FAILED")
    return 1 if failed else 0

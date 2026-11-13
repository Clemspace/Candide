"""Inference engine of the probe: scoring, completion, candidates, logit lens, attention.

Every number that has a counterpart in scripts/score_text.py is computed by it or with
its windows, so the probe and the scorer agree:

  * score: the per-token losses of a text, read through score_text.windows (same
    max_len, stride and beginning-of-text token). Their sum is the total that
    score_text.score_ids returns for the same text.
  * pairs and candidates: score_text.score_text itself, through a backend adapter.

Probabilities reported for generated tokens are those of the model (temperature 1, no
truncation); temperature, top-k, top-p and the repetition penalty only decide which token
is drawn. The logit lens applies the model's own unembedding (final norm and head for
GPT+, head and logit scale for anGPT) to the residual stream after each block. Attention
weights are recomputed explicitly inside every call to scaled_dot_product_attention during
one forward pass, so they are read from the model as it runs, whatever the architecture.
"""
from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .checkpoints import Registry
from .errors import ProbeError
from .text import CandideTokenizer, check_compatible, prepare, score_text_module

LN2 = math.log(2.0)
LENS_MAX_TOKENS = 256
ATTN_MAX_TOKENS = 160


# ── Helpers ──────────────────────────────────────────────────────────────────

def _precision(device: str):
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _logits(out) -> torch.Tensor:
    if isinstance(out, dict):
        return out["logits"]
    if isinstance(out, tuple):
        return out[0]
    return out


def unembed(model, h: torch.Tensor) -> torch.Tensor:
    """The model's own map from the residual stream to logits."""
    if getattr(model, "final_norm", None) is not None:
        return model.lm_head(model.final_norm(h))
    if hasattr(model, "logit_scale"):
        return model.lm_head(h) * model.logit_scale().to(h.dtype)
    return model.lm_head(h)


def _entropy(logp: torch.Tensor) -> torch.Tensor:
    return -(logp.exp() * logp).sum(-1)


def _top(logp: torch.Tensor, k: int):
    v, i = torch.topk(logp, k, dim=-1)
    return v.exp(), i


class _ScorerBackend:
    """score_text's backend interface over a loaded Candide model."""

    def __init__(self, model, tokenizer: CandideTokenizer, max_len: int, name: str):
        self.model, self.tok = model, tokenizer
        self.bos_id, self.max_len = tokenizer.bos_id, max_len
        self.name, self.tokenizer_name = f"candide:{name}", tokenizer.path

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def logits(self, ids, attention_mask):
        return _logits(self.model(ids))


# ── Attention capture ────────────────────────────────────────────────────────

class _AttentionRecorder:
    """Replaces F.scaled_dot_product_attention for one forward pass by the explicit
    computation, and keeps the weights of batch row 0, per block."""

    def __init__(self, model):
        self.model = model
        self.layer = -1
        self.weights: dict[int, torch.Tensor] = {}
        self._handles = []
        self._orig = None

    def __enter__(self):
        for i, block in enumerate(self.model.blocks):
            self._handles.append(block.register_forward_pre_hook(self._at(i)))
        self._orig = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = self._sdpa
        return self

    def __exit__(self, *exc):
        F.scaled_dot_product_attention = self._orig
        for h in self._handles:
            h.remove()
        return False

    def _at(self, i):
        def hook(module, args):
            self.layer = i
        return hook

    def _sdpa(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        if k.shape[-3] != q.shape[-3]:                      # grouped heads not yet repeated
            rep = q.shape[-3] // k.shape[-3]
            k, v = k.repeat_interleave(rep, dim=-3), v.repeat_interleave(rep, dim=-3)
        L, S = q.shape[-2], k.shape[-2]
        scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
        scores = (q.float() @ k.float().transpose(-2, -1)) * scale
        if is_causal:
            keep = torch.ones(L, S, dtype=torch.bool, device=q.device).tril(diagonal=S - L)
            scores = scores.masked_fill(~keep, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask.float()
        w = torch.softmax(scores, dim=-1)
        if self.layer >= 0:
            self.weights[self.layer] = w[0].detach().to("cpu", torch.float32)
        return (w.to(v.dtype) @ v)


# ── Engine ───────────────────────────────────────────────────────────────────

@dataclass
class Context:
    model: object
    meta: dict
    max_len: int
    warnings: list


class Engine:
    def __init__(self, registry: Registry, tokenizer: CandideTokenizer):
        self.registry = registry
        self.tok = tokenizer
        self.device = registry.device

    # -- registry -------------------------------------------------------------

    def info(self) -> dict:
        return {"tokenizer": {"path": self.tok.path, "vocab_size": self.tok.vocab_size,
                              "byte_level": self.tok.byte_level, "bos_id": self.tok.bos_id,
                              "eos_id": self.tok.eos_id},
                "device": str(self.device), "torch": torch.__version__}

    def list_checkpoints(self, refresh: bool = False) -> list[dict]:
        refs = self.registry.refresh() if refresh else self.registry.refs()
        loaded = set(self.registry.loaded_ids())
        return [{"id": r.id, "run": r.run, "name": r.name, "step_from_name": r.step_from_name,
                 "size_mb": round(r.size_mb, 1), "loaded": r.id in loaded} for r in refs]

    def meta(self, ckpt_id: str) -> dict:
        meta = dict(self.registry.meta(ckpt_id))
        meta["warnings"] = check_compatible(self.tok, meta)
        meta["compatible"] = not meta["warnings"]
        meta["notes"] = ["The checkpoint records its vocabulary size, not the tokenizer that made its "
                         "shards; v1 and v2g have the same size. Check data.shards_dir."]
        return meta

    # -- model access ---------------------------------------------------------

    def context(self, ckpt_id: str) -> Context:
        loaded = self.registry.get(ckpt_id)
        problems = check_compatible(self.tok, loaded.meta)
        if problems:
            raise ProbeError(f"{ckpt_id}: " + "; ".join(problems))
        max_len = int(loaded.meta["model"].get("max_seq_len") or 1024)
        return Context(loaded.model, loaded.meta, max_len, [])

    def _bos(self, use_bos: bool) -> list[int]:
        if use_bos and self.tok.bos_id is not None:
            return [self.tok.bos_id]
        return []

    def _forward(self, model, ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), _precision(self.device):
            return _logits(model(ids)).float()

    # -- score ----------------------------------------------------------------

    def score(self, ckpt_ids: Sequence[str], text: str, *, clean: bool = True, bos: bool = True,
              topk: int = 5, max_len: Optional[int] = None, stride: Optional[int] = None) -> dict:
        """Per-token loss, rank and alternatives of a text, for one or more checkpoints."""
        if not ckpt_ids:
            raise ProbeError("no checkpoint selected")
        prepared = prepare(text, clean=clean)
        ids = self.tok.encode(prepared)
        if not ids:
            raise ProbeError("empty text")
        prefix = self._bos(bos)
        seq = prefix + ids
        n_bytes = len(prepared.encode("utf-8"))
        S = score_text_module()
        runs = {}
        with self.registry.lock:
            for cid in ckpt_ids:
                ctx = self.context(cid)
                L = min(max_len or ctx.max_len, ctx.max_len)
                st = min(stride or L // 2, L - 1)
                t0 = time.time()
                per = [None] * len(seq)
                for begin, end, first in S.windows(len(seq), L, st):
                    x = torch.tensor([seq[begin:end]], device=self.device)
                    logp = torch.log_softmax(self._forward(ctx.model, x)[0], dim=-1)   # [T, V]
                    rows = torch.arange(first - begin - 1, end - begin - 1, device=self.device)
                    tgt = x[0, first - begin:end - begin]
                    lp = logp[rows, tgt]
                    rank = (logp[rows] > lp[:, None]).sum(-1)
                    ent = _entropy(logp[rows])
                    p_top, i_top = _top(logp[rows], topk)
                    lp, rank, ent = lp.tolist(), rank.tolist(), ent.tolist()      # one copy to host
                    p_top, i_top = p_top.tolist(), i_top.tolist()
                    for j in range(len(lp)):
                        per[first + j] = {
                            "nll": -lp[j], "rank": rank[j] + 1, "entropy": ent[j],
                            "top": [[i, p] for i, p in zip(i_top[j], p_top[j])],
                        }
                tokens = per[len(prefix):]
                scored = [t for t in tokens if t is not None]
                nll = sum(t["nll"] for t in scored)
                runs[cid] = {
                    "step": ctx.meta["step"], "tokens": tokens,
                    "nll_nats": nll, "n_scored": len(scored),
                    "bits_per_byte": nll / LN2 / n_bytes if n_bytes else None,
                    "bits_per_token": nll / LN2 / len(scored) if scored else None,
                    "max_len": L, "stride": st, "seconds": round(time.time() - t0, 3),
                }
        return {"text": prepared, "n_bytes": n_bytes, "segments": self.tok.segments(ids),
                "labels": self._labels(runs), "runs": runs,
                "settings": {"clean_entities": clean, "bos_token": bool(prefix), "topk": topk}}

    def _labels(self, runs: dict) -> dict:
        ids = set()
        for r in runs.values():
            for t in r.get("tokens", []):
                if t:
                    ids.update(i for i, _ in t["top"])
            for g in r.get("generated", []):
                ids.update(i for i, _ in g["top"])
        return {str(i): self.tok.label(i) for i in sorted(ids)}

    # -- generate -------------------------------------------------------------

    def generate(self, ckpt_id: str, prompt: str = "", *, ids: Optional[Sequence[int]] = None,
                 clean: bool = True, bos: bool = True, max_new_tokens: int = 120,
                 greedy: bool = False, temperature: float = 0.8, top_k: int = 0, top_p: float = 0.95,
                 repetition_penalty: float = 1.0, seed: Optional[int] = None, topn: int = 8,
                 stop_at_eos: bool = True) -> dict:
        """Continue a prompt, recording for each new token the model's own distribution."""
        if ids is not None:
            prompt_ids = [int(i) for i in ids]
            prepared = None
        else:
            prepared = prepare(prompt, clean=clean)
            prompt_ids = self._bos(bos) + self.tok.encode(prepared)
        if not prompt_ids:
            raise ProbeError("empty prompt: enable the beginning-of-text token or write a prompt")
        seed = int(seed) if seed is not None else int(time.time() * 1000) % (2 ** 31)
        with self.registry.lock:
            ctx = self.context(ckpt_id)
            max_new = max(1, min(int(max_new_tokens), ctx.max_len - 1))
            cropped = max(0, len(prompt_ids) + max_new - ctx.max_len)
            work = prompt_ids[cropped:]
            gen = torch.Generator(device=self.device).manual_seed(seed)
            t0 = time.time()
            prompt_stats, steps = self._decode(ctx.model, work, max_new, greedy, temperature, top_k, top_p,
                                               repetition_penalty, gen, topn, stop_at_eos)
            elapsed = time.time() - t0
        new_ids = [s["id"] for s in steps]
        run = {"step": ctx.meta["step"], "generated": steps, "prompt_tokens": prompt_stats}
        return {
            "ckpt": ckpt_id, "seed": seed, "prompt_ids": work, "cropped": cropped,
            "prompt_segments": self.tok.segments(work),
            "segments": self.tok.segments(new_ids), "text": self.tok.decode(new_ids),
            "prepared_prompt": prepared,
            "run": run, "labels": self._labels({"r": run}),
            "seconds": round(elapsed, 3), "tokens_per_second": round(len(steps) / elapsed, 1) if elapsed else None,
            "settings": {"greedy": greedy, "temperature": temperature, "top_k": top_k, "top_p": top_p,
                         "repetition_penalty": repetition_penalty, "max_new_tokens": max_new,
                         "bos_token": bool(self._bos(bos)) if ids is None else None},
        }

    def _decode(self, model, prompt_ids, max_new, greedy, temperature, top_k, top_p, rep, gen, topn, stop_at_eos):
        device = self.device
        x = torch.tensor([prompt_ids], device=device)
        with torch.no_grad(), _precision(device):
            out = model(x, use_cache=True)
        logits_all, past = out if isinstance(out, tuple) else (out, None)
        logits_all = logits_all.float()

        # the prompt itself, scored from the same forward pass
        logp_prompt = torch.log_softmax(logits_all[0, :-1], dim=-1)
        prompt_stats = [None]
        if len(prompt_ids) > 1:
            tgt = x[0, 1:]
            lp = logp_prompt.gather(-1, tgt[:, None])[:, 0]
            rank = (logp_prompt > lp[:, None]).sum(-1) + 1
            for a, b in zip(lp.tolist(), rank.tolist()):
                prompt_stats.append({"nll": -a, "rank": b})

        seq = list(prompt_ids)
        steps = []
        last = logits_all[0, -1]
        for _ in range(max_new):
            logp = torch.log_softmax(last, dim=-1)
            choose = last.clone()
            if rep != 1.0 and seq:
                prev = torch.tensor(sorted(set(seq)), device=device)
                vals = choose[prev]
                choose[prev] = torch.where(vals < 0, vals * rep, vals / rep)
            if greedy:
                nxt = int(choose.argmax())
            else:
                choose = choose / max(float(temperature), 1e-6)
                if top_k and top_k > 0:
                    kth = torch.topk(choose, min(int(top_k), choose.numel())).values[-1]
                    choose = choose.masked_fill(choose < kth, float("-inf"))
                if top_p < 1.0:
                    srt, idx = torch.sort(choose, descending=True)
                    probs = torch.softmax(srt, dim=-1)
                    drop = probs.cumsum(-1) - probs > top_p          # keep the token that crosses p
                    srt = srt.masked_fill(drop, float("-inf"))
                    choose = torch.full_like(choose, float("-inf")).scatter(0, idx, srt)
                nxt = int(torch.multinomial(torch.softmax(choose, dim=-1), 1, generator=gen))
            p_top, i_top = _top(logp, topn)
            steps.append({
                "id": nxt, "nll": -float(logp[nxt]), "rank": int((logp > logp[nxt]).sum()) + 1,
                "entropy": float(_entropy(logp)),
                "top": [[int(i), float(p)] for i, p in zip(i_top.tolist(), p_top.tolist())],
            })
            seq.append(nxt)
            if stop_at_eos and nxt == self.tok.eos_id:
                break
            if len(steps) == max_new:
                break
            step_in = torch.tensor([[nxt]], device=device)
            with torch.no_grad(), _precision(device):
                if past is not None:
                    out = model(step_in, past_key_values=past, use_cache=True)
                    last_logits, past = out
                else:
                    last_logits = _logits(model(torch.tensor([seq], device=device)))
            last = last_logits.float()[0, -1]
        return prompt_stats, steps

    # -- candidates and pairs -------------------------------------------------

    def candidates(self, ckpt_ids: Sequence[str], prompt: str, choices: Sequence[str], *,
                   clean: bool = True, max_len: Optional[int] = None, stride: Optional[int] = None) -> dict:
        """How likely each continuation is after a prompt, scored as score_text scores text.

        conditional nats = nll(prompt + choice) - nll(prompt): the prompt term is common to
        all choices, so the ranking is that of the whole strings. When the tokens of the
        prompt are not a prefix of those of prompt + choice (a token spans the boundary),
        the row is flagged: the split is then not a clean conditional.
        """
        choices = [c for c in choices if c is not None and c != ""]
        if not choices:
            raise ProbeError("no candidate continuations")
        S = score_text_module()
        p_prep = prepare(prompt, clean=clean)
        p_ids = self.tok.encode(p_prep)
        out = {"prompt": p_prep, "choices": [], "runs": {}}
        for c in choices:
            full = prepare(prompt + c, clean=clean)
            f_ids = self.tok.encode(full)
            out["choices"].append({"text": c, "n_bytes": len(full.encode()) - len(p_prep.encode()),
                                   "n_tokens": len(f_ids) - len(p_ids),
                                   "boundary_merge": f_ids[:len(p_ids)] != p_ids})
        with self.registry.lock:
            for cid in ckpt_ids:
                ctx = self.context(cid)
                L = min(max_len or ctx.max_len, ctx.max_len)
                kw = dict(max_len=L, stride=min(stride or L // 2, L - 1), batch_size=8,
                          device=self.device, clean=clean)
                backend = _ScorerBackend(ctx.model, self.tok, L, cid)
                base = S.score_text(backend, prompt, **kw).nll_nats if p_prep else 0.0
                rows = []
                for c, info in zip(choices, out["choices"]):
                    s = S.score_text(backend, prompt + c, **kw)
                    cond = s.nll_nats - base
                    rows.append({"nll_total": s.nll_nats, "nll_cond": cond,
                                 "bits_per_byte_cond": cond / LN2 / info["n_bytes"] if info["n_bytes"] > 0 else None})
                order = sorted(range(len(rows)), key=lambda i: rows[i]["nll_total"])
                for r, i in enumerate(order):
                    rows[i]["rank"] = r + 1
                out["runs"][cid] = {"step": ctx.meta["step"], "rows": rows}
        return out

    def pairs(self, ckpt_ids: Sequence[str], pairs: Sequence[dict], *, clean: bool = True,
              max_len: Optional[int] = None, stride: Optional[int] = None) -> dict:
        """score_text --pairs, for several checkpoints: which of a and b the model prefers."""
        S = score_text_module()
        out = {"runs": {}}
        with self.registry.lock:
            for cid in ckpt_ids:
                ctx = self.context(cid)
                L = min(max_len or ctx.max_len, ctx.max_len)
                kw = dict(max_len=L, stride=min(stride or L // 2, L - 1), batch_size=8,
                          device=self.device, clean=clean)
                backend = _ScorerBackend(ctx.model, self.tok, L, cid)
                rows = []
                for n, row in enumerate(pairs):
                    a = S.score_text(backend, row["a"], **kw)
                    b = S.score_text(backend, row["b"], **kw)
                    rows.append({"id": row.get("id", n), "nll_a": a.nll_nats, "nll_b": b.nll_nats,
                                 "bits_per_byte_a": a.bits_per_byte, "bits_per_byte_b": b.bits_per_byte,
                                 "prefers_a_total": a.nll_nats < b.nll_nats,
                                 "prefers_a_per_byte": a.bits_per_byte < b.bits_per_byte})
                n_rows = max(1, len(rows))
                out["runs"][cid] = {
                    "step": ctx.meta["step"], "rows": rows,
                    "prefers_a_total": sum(r["prefers_a_total"] for r in rows) / n_rows,
                    "prefers_a_per_byte": sum(r["prefers_a_per_byte"] for r in rows) / n_rows,
                }
        return out

    # -- internals ------------------------------------------------------------

    def _ids_for_internals(self, text: str, clean: bool, bos: bool, cap: int):
        prepared = prepare(text, clean=clean)
        ids = self._bos(bos) + self.tok.encode(prepared)
        if len(ids) < 2:
            raise ProbeError("need at least two tokens")
        truncated = max(0, len(ids) - cap)
        return prepared, ids[:cap], truncated

    def lens(self, ckpt_id: str, text: str, *, clean: bool = True, bos: bool = True, topn: int = 3) -> dict:
        """Logit lens: the next-token prediction read from the residual stream after each block."""
        with self.registry.lock:
            ctx = self.context(ckpt_id)
            prepared, ids, truncated = self._ids_for_internals(text, clean, bos, min(LENS_MAX_TOKENS, ctx.max_len))
            model = ctx.model
            resid: list[torch.Tensor] = []
            hooks = [model.blocks[0].register_forward_pre_hook(lambda m, a: resid.append(a[0].detach()))]
            for b in model.blocks:
                hooks.append(b.register_forward_hook(
                    lambda m, a, o: resid.append((o[0] if isinstance(o, tuple) else o).detach())))
            x = torch.tensor([ids], device=self.device)
            try:
                final = self._forward(model, x)[0]
            finally:
                for h in hooks:
                    h.remove()
            T = len(ids)
            tgt = torch.tensor(ids[1:] + [ids[-1]], device=self.device)
            layers = []
            with torch.no_grad():
                for l, h in enumerate(resid):
                    with _precision(self.device):
                        lg = unembed(model, h[0]).float()
                    logp = torch.log_softmax(lg, dim=-1)
                    lp = logp.gather(-1, tgt[:, None])[:, 0]
                    rank = (logp > lp[:, None]).sum(-1) + 1
                    p_top, i_top = _top(logp, topn)
                    p_target, rank_target = lp.exp().tolist(), rank.tolist()
                    p_target[-1] = rank_target[-1] = None      # the last position has no next token
                    layers.append({
                        "name": "embed" if l == 0 else f"L{l}",
                        "p_target": p_target, "rank_target": rank_target,
                        "entropy": _entropy(logp).tolist(),
                        "top": [[[int(i), float(p)] for i, p in zip(ii, pp)]
                                for ii, pp in zip(i_top.tolist(), p_top.tolist())],
                    })
                gap = float((torch.log_softmax(final, -1).exp()
                             - torch.log_softmax(unembed(model, resid[-1][0]).float(), -1).exp()).abs().max())
        labels = {}
        for layer in layers:
            for row in layer["top"]:
                for i, _ in row:
                    labels.setdefault(str(i), self.tok.label(i))
        return {"ckpt": ckpt_id, "step": ctx.meta["step"], "text": prepared, "ids": ids,
                "segments": self.tok.segments(ids), "n_positions": T, "truncated": truncated,
                "layers": layers, "labels": labels,
                "check": {"last_layer_vs_model_max_abs_prob_diff": gap}}

    def attention(self, ckpt_id: str, text: str, *, clean: bool = True, bos: bool = True,
                  layer: Optional[int] = None) -> dict:
        """Per-head summaries for every block, and full patterns for one block."""
        with self.registry.lock:
            ctx = self.context(ckpt_id)
            prepared, ids, truncated = self._ids_for_internals(text, clean, bos, min(ATTN_MAX_TOKENS, ctx.max_len))
            x = torch.tensor([ids], device=self.device)
            with torch.no_grad(), _AttentionRecorder(ctx.model) as rec:
                ctx.model(x)                      # fp32: the recorded weights are the exact ones
            weights = rec.weights
        if not weights:
            raise ProbeError("no attention call was recorded for this architecture")
        T = len(ids)
        summary = []
        for l in sorted(weights):
            w = weights[l]                                   # [H, T, T]
            ent = -(w.clamp_min(1e-12).log() * w).sum(-1)    # [H, T]
            norm = torch.log(torch.arange(1, T + 1, dtype=torch.float32))
            prev = torch.stack([w[:, t, t - 1] for t in range(1, T)], -1).mean(-1) if T > 1 else torch.zeros(w.shape[0])
            first = w[:, 1:, 0].mean(-1) if T > 1 else w[:, :, 0].mean(-1)
            self_ = torch.diagonal(w, dim1=-2, dim2=-1)[:, 1:].mean(-1) if T > 1 else torch.ones(w.shape[0])
            summary.append({
                "layer": l,
                "entropy_rel": (ent[:, 1:] / norm[1:]).mean(-1).tolist() if T > 1 else [0.0] * w.shape[0],
                "prev_token": prev.tolist(), "first_token": first.tolist(), "self": self_.tolist(),
            })
        pick = layer if layer is not None and layer in weights else max(weights)
        return {"ckpt": ckpt_id, "step": ctx.meta["step"], "text": prepared, "ids": ids,
                "segments": self.tok.segments(ids), "truncated": truncated,
                "n_layers": len(weights), "n_heads": int(weights[pick].shape[0]),
                "summary": summary, "layer": pick,
                "patterns": [[[round(v, 4) for v in row] for row in head] for head in weights[pick].tolist()]}


__all__ = ["Engine", "ProbeError", "unembed", "LENS_MAX_TOKENS", "ATTN_MAX_TOKENS"]

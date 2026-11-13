"""A stand-in engine with random numbers, for working on the interface without a GPU.

    python -m candide_probe --demo

Same request and response shapes as engine.Engine; no PyTorch, no tokenizer file, no
checkpoint. Every view is labelled DEMO in the interface. Nothing it returns means anything.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
import time
import unicodedata
from typing import Sequence

from .errors import ProbeError

_PIECE = re.compile(r" ?[^\W\d_]+'?| ?\d| ?[^\w\s]|\s+", re.UNICODE)
LN2 = math.log(2.0)


class DemoTokenizer:
    path = "demo"
    vocab_size = 32132
    bos_id, eos_id = 2, 3
    byte_level = True

    def __init__(self):
        self.pieces: dict[int, str] = {2: "<|bos|>", 3: "<|eos|>"}
        self.ids: dict[str, int] = {}

    def _id(self, piece: str) -> int:
        if piece not in self.ids:
            h = int(hashlib.md5(piece.encode()).hexdigest(), 16) % 32000 + 133
            while h in self.pieces and self.pieces[h] != piece:
                h = 133 + (h - 132) % 31999
            self.ids[piece], self.pieces[h] = h, piece
        return self.ids[piece]

    def encode(self, text: str) -> list[int]:
        out = []
        for m in _PIECE.finditer(text):
            p = m.group(0)
            while len(p) > 6:                       # long words in several pieces, like BPE
                out.append(self._id(p[:4]))
                p = p[4:]
            out.append(self._id(p))
        return out

    def decode(self, ids) -> str:
        return "".join(self.pieces.get(int(i), "") for i in ids if int(i) not in (2, 3))

    def label(self, i: int) -> str:
        return self.pieces.get(int(i), f"<{i}>")

    def segments(self, ids) -> list[dict]:
        out = []
        for i in ids:
            i = int(i)
            special = i in (2, 3)
            t = self.label(i)
            out.append({"id": i, "text": t, "n_bytes": 0 if special else len(t.encode()),
                        "special": special, "partial": False})
        return out


_VOCAB = [" le", " la", " de", " et", " qui", " que", " roi", " étoit", " avoit", " ſon", " France", ",",
          ".", " les", " des", " en", " un", " une", " Dieu", " point", " peuple", " liberté", " raiſon",
          " auſſi", " ſur", " par", " cette", " ont", " ſont", " fort", " bien", " lui", " nous", " vous"]


def _rng(*parts) -> random.Random:
    return random.Random(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest())


def _prepare(text: str, clean: bool = True) -> str:
    if clean:
        text = re.sub(r"&\s*nbsp\s*;", " ", text, flags=re.I)
    return unicodedata.normalize("NFC", text)


class DemoEngine:
    demo = True

    def __init__(self, n_layers: int = 29, n_heads: int = 12):
        self.tok = DemoTokenizer()
        for p in _VOCAB:
            self.tok._id(p)
        self.n_layers, self.n_heads = n_layers, n_heads
        self.device = "demo"
        steps = [24000, 48000, 62000, 71000, 76000]
        self._ckpts = [{"id": f"full/candide-v2g-224m/step_{s}", "run": "full/candide-v2g-224m",
                        "name": f"step_{s}", "step": s} for s in steps]
        self._ckpts.insert(4, {"id": "full/candide-v2g-224m/pre_cooldown", "run": "full/candide-v2g-224m",
                               "name": "pre_cooldown", "step": 76000})
        self._ckpts.append({"id": "full/candide-v2g-224m/final", "run": "full/candide-v2g-224m",
                            "name": "final", "step": 95000})
        self._ckpts.append({"id": "full-cool62k/final", "run": "full-cool62k", "name": "final", "step": 77500})

    # -- registry-like ----------------------------------------------------------

    def list_checkpoints(self, refresh: bool = False) -> list[dict]:
        return [{"id": c["id"], "run": c["run"], "name": c["name"], "step_from_name": c["step"],
                 "size_mb": 2692.4, "loaded": False} for c in self._ckpts]

    def meta(self, ckpt_id: str) -> dict:
        c = next((c for c in self._ckpts if c["id"] == ckpt_id), None)
        if c is None:
            raise KeyError(f"unknown checkpoint {ckpt_id!r}")
        step = c["step"]
        cs = 62000 if c["run"] == "full-cool62k" else 76000
        max_steps = 77500 if c["run"] == "full-cool62k" else 95000
        phase = "cooldown" if step >= cs else "stable"
        lr = 1.0 if step < cs else (1 - math.sqrt((step - cs) / (max_steps - cs))) if max_steps > cs else 0.0
        return {
            "id": ckpt_id, "run": c["run"], "name": c["name"], "path": f"/demo/{ckpt_id}.pt", "size_mb": 2692.4,
            "step": step, "n_params": 224_300_000, "vocab_size": 32132,
            "model": {"arch": "gpt", "qk_norm": True, "n_layers": self.n_layers, "d_model": 768,
                      "n_heads": self.n_heads, "n_kv_heads": 4, "d_ff": 2304, "max_seq_len": 1024,
                      "tie_word_embeddings": True, "norm_type": "rms", "ffn_type": "swiglu",
                      "attention_type": "gqa", "use_rope": True, "rope_theta": 10000.0},
            "train": {"schedule": "wsd", "max_steps": max_steps, "warmup_steps": 1000, "cooldown_start": cs,
                      "phase": phase, "lr_peak": 2.4e-3, "lr_factor": round(lr, 4), "optimizer": "muon",
                      "tokens_per_step": 65536, "tokens_seen": step * 65536},
            "data": {"shards_dir": "/data/pretrain_candide/shards_v2g", "mix": {"main": 0.994, "enc": 0.006},
                     "cooldown_mix": {"main": 0.9, "canon": 0.075, "enc": 0.025}},
            "wandb_id": "qq2gik3r" if c["run"] != "full-cool62k" else None, "has_optimizer": True,
            "warnings": ["DEMO: random numbers, no model loaded"],
        }

    def _quality(self, ckpt_id: str) -> float:
        step = self.meta(ckpt_id)["step"]
        return 0.6 + 0.4 * min(1.0, step / 95000) + (0.25 if "final" in ckpt_id else 0.0)

    def _token_stats(self, ckpt_id: str, ids: Sequence[int], pos: int, q: float, topk: int) -> dict:
        r = _rng(ckpt_id.split("/")[-1][:4], ids[max(0, pos - 3):pos + 1], pos)
        nll = max(0.01, r.expovariate(q / 2.4))
        rank = 1 if nll < 0.7 else 1 + int(r.expovariate(1 / (4 + 30 * nll)))
        alts = [self.tok._id(p) for p in r.sample(_VOCAB, topk)]
        if rank <= topk:
            alts[rank - 1] = ids[pos]
        ps = sorted((r.random() ** 2 for _ in range(topk)), reverse=True)
        tot = sum(ps) / min(0.95, math.exp(-nll) + 0.3)
        return {"nll": nll, "rank": rank, "entropy": r.uniform(1.0, 5.0),
                "top": [[a, p / tot] for a, p in zip(alts, ps)]}

    def _labels(self, *groups) -> dict:
        ids = set()
        for g in groups:
            for t in g:
                if t:
                    ids.update(i for i, _ in t["top"])
        return {str(i): self.tok.label(i) for i in ids}

    # -- endpoints ------------------------------------------------------------

    def score(self, ckpt_ids, text, *, clean=True, bos=True, topk=5, max_len=None, stride=None) -> dict:
        if not ckpt_ids:
            raise ProbeError("no checkpoint selected")
        prepared = _prepare(text, clean)
        ids = self.tok.encode(prepared)
        if not ids:
            raise ProbeError("empty text")
        n_bytes = len(prepared.encode())
        runs, all_tokens = {}, []
        for cid in ckpt_ids:
            q = self._quality(cid)
            toks = [self._token_stats(cid, ids, i, q, topk) if (bos or i) else None for i in range(len(ids))]
            nll = sum(t["nll"] for t in toks if t)
            n = sum(1 for t in toks if t)
            runs[cid] = {"step": self.meta(cid)["step"], "tokens": toks, "nll_nats": nll, "n_scored": n,
                         "bits_per_byte": nll / LN2 / n_bytes, "bits_per_token": nll / LN2 / n,
                         "max_len": 1024, "stride": 512, "seconds": 0.01}
            all_tokens += toks
        return {"text": prepared, "n_bytes": n_bytes, "segments": self.tok.segments(ids),
                "labels": self._labels(all_tokens), "runs": runs,
                "settings": {"clean_entities": clean, "bos_token": bos, "topk": topk, "demo": True}}

    def generate(self, ckpt_id, prompt="", *, ids=None, clean=True, bos=True, max_new_tokens=120,
                 greedy=False, temperature=0.8, top_k=0, top_p=0.95, repetition_penalty=1.0,
                 seed=None, topn=8, stop_at_eos=True) -> dict:
        seed = int(seed) if seed is not None else int(time.time() * 1000) % (2 ** 31)
        prompt_ids = list(ids) if ids is not None else ([2] if bos else []) + self.tok.encode(_prepare(prompt, clean))
        if not prompt_ids:
            raise ProbeError("empty prompt")
        q = self._quality(ckpt_id)
        r = random.Random(seed if not greedy else 0)
        seq, steps = list(prompt_ids), []
        for n in range(min(int(max_new_tokens), 400)):
            piece = r.choice(_VOCAB) if r.random() > 0.15 else "."
            nxt = self.tok._id(piece)
            st = self._token_stats(ckpt_id, seq + [nxt], len(seq), q, topn)
            st["id"] = nxt
            steps.append(st)
            seq.append(nxt)
        prompt_stats = [None] + [{"nll": self._token_stats(ckpt_id, prompt_ids, i, q, 1)["nll"],
                                  "rank": self._token_stats(ckpt_id, prompt_ids, i, q, 1)["rank"]}
                                 for i in range(1, len(prompt_ids))]
        run = {"step": self.meta(ckpt_id)["step"], "generated": steps, "prompt_tokens": prompt_stats}
        new_ids = [s["id"] for s in steps]
        return {"ckpt": ckpt_id, "seed": seed, "prompt_ids": prompt_ids, "cropped": 0,
                "prompt_segments": self.tok.segments(prompt_ids), "segments": self.tok.segments(new_ids),
                "text": self.tok.decode(new_ids), "prepared_prompt": prompt, "run": run,
                "labels": self._labels(steps), "seconds": 0.4, "tokens_per_second": len(steps) / 0.4,
                "settings": {"greedy": greedy, "temperature": temperature, "top_k": top_k, "top_p": top_p,
                             "repetition_penalty": repetition_penalty, "max_new_tokens": max_new_tokens,
                             "demo": True}}

    def candidates(self, ckpt_ids, prompt, choices, *, clean=True, max_len=None, stride=None) -> dict:
        choices = [c for c in choices if c]
        if not choices:
            raise ProbeError("no candidate continuations")
        out = {"prompt": prompt, "choices": [{"text": c, "n_bytes": len(c.encode()),
                                              "n_tokens": len(self.tok.encode(c)), "boundary_merge": False}
                                             for c in choices], "runs": {}}
        for cid in ckpt_ids:
            r = _rng(cid, prompt)
            rows = [{"nll_total": 40 + r.uniform(0, 8), "nll_cond": r.uniform(1, 12)} for _ in choices]
            for row, ch in zip(rows, out["choices"]):
                row["bits_per_byte_cond"] = row["nll_cond"] / LN2 / max(1, ch["n_bytes"])
            order = sorted(range(len(rows)), key=lambda i: rows[i]["nll_total"])
            for k, i in enumerate(order):
                rows[i]["rank"] = k + 1
            out["runs"][cid] = {"step": self.meta(cid)["step"], "rows": rows}
        return out

    def pairs(self, ckpt_ids, pairs, *, clean=True, max_len=None, stride=None) -> dict:
        out = {"runs": {}}
        for cid in ckpt_ids:
            rows = []
            for n, p in enumerate(pairs):
                r = _rng(cid, p.get("a"), p.get("b"))
                a, b = r.uniform(20, 40), r.uniform(20, 40) + (2 if self._quality(cid) > 1 else 0)
                rows.append({"id": p.get("id", n), "nll_a": a, "nll_b": b,
                             "bits_per_byte_a": a / LN2 / max(1, len(p["a"])),
                             "bits_per_byte_b": b / LN2 / max(1, len(p["b"])),
                             "prefers_a_total": a < b, "prefers_a_per_byte": a < b})
            k = max(1, len(rows))
            out["runs"][cid] = {"step": self.meta(cid)["step"], "rows": rows,
                                "prefers_a_total": sum(x["prefers_a_total"] for x in rows) / k,
                                "prefers_a_per_byte": sum(x["prefers_a_per_byte"] for x in rows) / k}
        return out

    def lens(self, ckpt_id, text, *, clean=True, bos=True, topn=3) -> dict:
        ids = ([2] if bos else []) + self.tok.encode(_prepare(text, clean))
        ids = ids[:256]
        T, L = len(ids), self.n_layers
        layers = []
        for l in range(L + 1):
            depth = l / L
            r = _rng(ckpt_id, l)
            p_t, rk, top = [], [], []
            for t in range(T):
                emerge = _rng(ckpt_id, ids[max(0, t - 2):t + 2], t).uniform(0.3, 1.1)
                p = max(0.0, min(0.98, (depth - emerge + 0.35) * 1.6)) if t < T - 1 else None
                p_t.append(p)
                rk.append(None if p is None else (1 if p > 0.35 else 1 + int(r.expovariate(0.02 + p))))
                guess = ids[t + 1] if (t < T - 1 and p and p > 0.35) else self.tok._id(r.choice(_VOCAB))
                top.append([[guess, max(p or 0, 0.05)], [self.tok._id(r.choice(_VOCAB)), 0.04],
                            [self.tok._id(r.choice(_VOCAB)), 0.02]][:topn])
            layers.append({"name": "embed" if l == 0 else f"L{l}", "p_target": p_t, "rank_target": rk,
                           "entropy": [r.uniform(1, 8) * (1 - 0.6 * depth) for _ in range(T)], "top": top})
        labels = {str(i): self.tok.label(i) for layer in layers for row in layer["top"] for i, _ in row}
        return {"ckpt": ckpt_id, "step": self.meta(ckpt_id)["step"], "text": text, "ids": ids,
                "segments": self.tok.segments(ids), "n_positions": T, "truncated": 0, "layers": layers,
                "labels": labels, "check": {"last_layer_vs_model_max_abs_prob_diff": 0.0, "demo": True}}

    def attention(self, ckpt_id, text, *, clean=True, bos=True, layer=None) -> dict:
        ids = (([2] if bos else []) + self.tok.encode(_prepare(text, clean)))[:160]
        T, H, L = len(ids), self.n_heads, self.n_layers
        pick = L - 1 if layer is None else max(0, min(L - 1, int(layer)))
        summary = []
        for l in range(L):
            r = _rng(ckpt_id, "attn", l)
            summary.append({"layer": l, "entropy_rel": [r.uniform(0.1, 0.9) for _ in range(H)],
                            "prev_token": [r.random() ** 3 for _ in range(H)],
                            "first_token": [r.random() ** 0.5 * (0.3 + l / L) for _ in range(H)],
                            "self": [r.random() ** 4 for _ in range(H)]})
        patterns = []
        for h in range(H):
            r = _rng(ckpt_id, "pat", pick, h)
            kind = r.choice(["prev", "first", "broad"])
            head = []
            for t in range(T):
                w = [0.0] * T
                if kind == "prev" and t > 0:
                    w[t - 1], w[0] = 0.8, 0.2
                elif kind == "first" or t == 0:
                    w[0] = 1.0
                else:
                    xs = [r.random() for _ in range(t + 1)]
                    s = sum(xs)
                    w[:t + 1] = [x / s for x in xs]
                head.append([round(v, 4) for v in w])
            patterns.append(head)
        return {"ckpt": ckpt_id, "step": self.meta(ckpt_id)["step"], "text": text, "ids": ids,
                "segments": self.tok.segments(ids), "truncated": 0, "n_layers": L, "n_heads": H,
                "summary": summary, "layer": pick, "patterns": patterns}

    def info(self) -> dict:
        return {"tokenizer": {"path": "demo", "vocab_size": 32132, "byte_level": True, "bos_id": 2, "eos_id": 3},
                "device": "demo", "torch": None}


__all__ = ["DemoEngine", "DemoTokenizer"]

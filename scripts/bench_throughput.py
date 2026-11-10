#!/usr/bin/env python3
"""
Training throughput of a train_candide18c config on this GPU (random tokens, no data loading).

Each combination of optimizer x torch.compile x micro-batch runs `--warmup`
untimed steps (compilation happens there), then `--steps` timed optimizer
steps at the config's tokens per step (gradient accumulation is rescaled so
micro-batch x accumulation stays constant). Reports tokens/s, peak memory and
achieved TFLOP/s, counting 6 N per token plus attention (12 L d T per token,
PaLM appendix B); an untied input embedding is a lookup and is not counted.

    PYTHONPATH=. python scripts/bench_throughput.py --config configs/full/gptplus_v2g.yaml \\
        --optimizer adamw muon --compile 0 1 --micro-batch 8 16 32 --out results/2026-11-10/bench.json

The model, optimizer and loss are the training script's own (setup_model,
setup_optimizer, same autocast), so the numbers transfer to real runs up to
data-loading stalls, which the training log shows separately.
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from ramanujan.training import train_candide18c as T


def flops_per_token(model: nn.Module, cfg) -> float:
    n = sum(p.numel() for p in model.parameters())
    if not cfg.model.tie_word_embeddings:
        n -= cfg.model.vocab_size * cfg.model.d_model           # input embedding: a lookup, no matmul
    attn = 12 * cfg.model.n_layers * cfg.model.d_model * cfg.data.sequence_length
    return 6 * n + attn


def bench(cfg, optimizer: str, compile_: bool, micro: int, steps: int, warmup: int) -> dict:
    total = cfg.training.batch_size * cfg.training.gradient_accumulation_steps
    if total % micro:
        return {"error": f"micro-batch {micro} does not divide {total}"}
    accum = total // micro
    device = torch.device("cuda")
    torch.manual_seed(0)
    cfg.training.optimizer = optimizer
    model = T.setup_model(cfg.model).to(device)
    opt = T.setup_optimizer(model, cfg.training, arch=cfg.model.arch)
    run = torch.compile(model) if compile_ else model
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    crit = nn.CrossEntropyLoss()
    seq, vocab = cfg.data.sequence_length, cfg.model.vocab_size
    x = torch.randint(0, vocab, (micro, seq + 1), device=device)
    inp, tgt = x[:, :-1].contiguous(), x[:, 1:].contiguous()

    def step():
        for _ in range(accum):
            with autocast:
                logits = run(inp)
                loss = crit(logits.view(-1, vocab), tgt.view(-1)) / accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()
    try:
        t_compile = time.perf_counter()
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        t_compile = time.perf_counter() - t_compile
        t0 = time.perf_counter()
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    except torch.cuda.OutOfMemoryError:
        return {"error": "out of memory"}
    finally:
        peak = torch.cuda.max_memory_allocated() / 2**30
    tok_s = steps * total * seq / dt
    return {"tokens_per_sec": round(tok_s), "sec_per_step": round(dt / steps, 3),
            "tflops": round(tok_s * flops_per_token(model, cfg) / 1e12, 1),
            "peak_mem_gib": round(peak, 1), "warmup_sec": round(t_compile, 1), "accum": accum}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--optimizer", nargs="+", default=["adamw"], choices=["adamw", "muon"])
    p.add_argument("--compile", nargs="+", type=int, default=[0], choices=[0, 1])
    p.add_argument("--micro-batch", nargs="+", type=int, default=[8])
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    torch.backends.cuda.matmul.allow_tf32 = True

    rows = []
    print(f"{'optimizer':<9}{'compile':>8}{'micro':>6}{'accum':>6}{'tok/s':>9}{'s/step':>8}"
          f"{'TFLOP/s':>9}{'mem GiB':>9}{'warmup s':>10}", flush=True)
    for opt, comp, micro in itertools.product(args.optimizer, args.compile, args.micro_batch):
        cfg = T.load_config(args.config)
        r = {"optimizer": opt, "compile": bool(comp), "micro_batch": micro,
             **bench(cfg, opt, bool(comp), micro, args.steps, args.warmup)}
        rows.append(r)
        if "error" in r:
            print(f"{opt:<9}{comp:>8}{micro:>6}  {r['error']}", flush=True)
        else:
            print(f"{opt:<9}{comp:>8}{micro:>6}{r['accum']:>6}{r['tokens_per_sec']:>9,}{r['sec_per_step']:>8}"
                  f"{r['tflops']:>9}{r['peak_mem_gib']:>9}{r['warmup_sec']:>10}", flush=True)
        torch.cuda.empty_cache()
        torch._dynamo.reset()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"config": args.config, "gpu": torch.cuda.get_device_name(),
                                              "rows": rows}, indent=2))


if __name__ == "__main__":
    main()

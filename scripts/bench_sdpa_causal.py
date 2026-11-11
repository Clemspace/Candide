"""Does the attention kernel run faster when causality is declared than with an explicit mask?

The decoder always passes an explicit causal mask to F.scaled_dot_product_attention.
This times the kernel alone, forward and backward, both ways, at the shapes of the full
run. It changes no model code: if the two timings are close, there is nothing to gain and
the model stays as it is.

    python scripts/bench_sdpa_causal.py                 # micro-batch 16, 12 heads, 1024 x 64
"""
import argparse
import time

import torch
import torch.nn.functional as F


def timed(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--layers", type=int, default=29)
    p.add_argument("--accum", type=int, default=4, help="micro-batches per optimizer step")
    a = p.parse_args()

    shape = (a.batch, a.heads, a.seq, a.head_dim)
    q, k, v = (torch.randn(shape, device="cuda", requires_grad=True) for _ in range(3))
    mask = torch.ones(a.seq, a.seq, dtype=torch.bool, device="cuda").tril()[None, None]

    def attend(**kw):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return F.scaled_dot_product_attention(q, k, v, **kw)

    def step(**kw):
        attend(**kw).float().sum().backward()

    with torch.no_grad():
        diff = (attend(attn_mask=mask).float() - attend(is_causal=True).float()).abs().max().item()
    ms_mask = timed(lambda: step(attn_mask=mask), a.iters, 5)
    ms_causal = timed(lambda: step(is_causal=True), a.iters, 5)
    per_step = (ms_mask - ms_causal) * a.layers * a.accum / 1e3

    print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, shape {shape}")
    print(f"max |difference| between the two outputs: {diff:.2e}")
    print(f"explicit mask : {ms_mask:8.2f} ms per call (forward + backward)")
    print(f"is_causal=True: {ms_causal:8.2f} ms per call")
    print(f"difference over one optimizer step ({a.layers} layers x {a.accum} micro-batches): {per_step:+.3f} s")


if __name__ == "__main__":
    main()

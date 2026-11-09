#!/usr/bin/env python3
"""
Size a decoder from the corpus and the compute budget.

    PYTHONPATH=. python scripts/size_model.py \
        --unique-tokens 1.748e9 --gpu-hours 72 \
        --measured-params 118.6e6 --measured-tokens-per-sec 50000 \
        --vocab-size 32132

Achieved FLOP/s come from a measured run of the same code on the same GPU
(6 x params x tokens/s), or directly from --flops-per-sec. Prints the
allocation, the shape, and a YAML `model:` section plus `training.max_steps`.
See ramanujan/utils/sizing.py for the method and its caveats.
"""

import argparse

import yaml

from ramanujan.utils.sizing import flops_per_sec_from_run, size_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unique-tokens", type=float, required=True, help="unique training tokens in the corpus")
    p.add_argument("--gpu-hours", type=float, required=True, help="training budget in GPU-hours")
    p.add_argument("--flops-per-sec", type=float, default=None, help="achieved FLOP/s on the GPU")
    p.add_argument("--measured-params", type=float, default=None, help="params of a measured run")
    p.add_argument("--measured-tokens-per-sec", type=float, default=None, help="its tokens/s")
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--tokens-per-step", type=int, default=65536)
    p.add_argument("--tolerance", type=float, default=0.0,
                   help="accept the smallest model within this many nats of the predicted optimum")
    p.add_argument("--target-aspect", type=float, default=32.0, help="d_model / n_layers")
    p.add_argument("--untied", action="store_true", help="separate input and output embeddings")
    args = p.parse_args()

    if args.flops_per_sec is None:
        if args.measured_params is None or args.measured_tokens_per_sec is None:
            p.error("give --flops-per-sec, or --measured-params and --measured-tokens-per-sec")
        args.flops_per_sec = flops_per_sec_from_run(args.measured_params, args.measured_tokens_per_sec)

    r = size_model(args.unique_tokens, args.gpu_hours, args.flops_per_sec, args.vocab_size,
                   tokens_per_step=args.tokens_per_step, tolerance=args.tolerance,
                   tie=not args.untied, target_aspect=args.target_aspect)
    a, s = r.allocation, r.shape
    print(f"Budget        {r.gpu_hours:.0f} GPU-h x {r.flops_per_sec/1e12:.1f} TFLOP/s = {a.compute:.2e} FLOPs")
    print(f"Corpus        {r.unique_tokens/1e9:.3f}B unique tokens")
    print(f"Allocation    N = {a.params/1e6:.0f}M params, D = {a.tokens/1e9:.2f}B tokens ({a.epochs:.2f} epochs)")
    print(f"Flatness      half the params: {a.loss_half_params - a.loss:+.3f} nats; "
          f"double: {a.loss_double_params - a.loss:+.3f} nats (C4-fitted law)")
    print(f"Shape         {s.n_layers} layers x {s.d_model}, {s.n_heads} heads ({s.n_kv_heads} KV), "
          f"d_ff {s.d_ff}, {'tied' if s.tie_word_embeddings else 'untied'} embeddings, "
          f"aspect {s.aspect:.0f}, {s.params/1e6:.1f}M params")
    print(f"Training      {r.max_steps:,} steps x {r.tokens_per_step:,} tokens = "
          f"{r.max_steps * r.tokens_per_step / 1e9:.2f}B tokens "
          f"({r.max_steps * r.tokens_per_step / r.unique_tokens:.2f} epochs), ~{r.est_hours:.0f} h")
    print("Caveat        scaling constants fitted on English web text (C4); only the N/D split is used.\n")
    print(yaml.safe_dump({"model": s.model_section(args.vocab_size),
                          "training": {"max_steps": r.max_steps}}, sort_keys=False))


if __name__ == "__main__":
    main()

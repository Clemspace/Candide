#!/usr/bin/env python3
"""
Re-score trained checkpoints on a (new) validation set, without retraining.

Evaluates each checkpoint on the fixed validation sample of a shard directory
(the same FixedWindowDataset as training: val_windows windows drawn with
val_seed) and reports the token-weighted loss.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/rescore_checkpoints.py \\
        --shards /data/pretrain_candide/shards_v2 \\
        --run gptplus-s1=runs/pilot/gptplus-1.2e-3-s1/checkpoints/final.pt \\
        --run angpt-s1=runs/pilot/angpt-8e-3-s1/checkpoints/final.pt \\
        --out results/rescore_v2.json
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("memorization_probe", _here / "memorization_probe.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


@torch.no_grad()
def val_loss(model, loader, device) -> dict:
    total, tokens = 0.0, 0
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" \
        else torch.autocast("cpu", enabled=False)
    for batch in loader:
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        with amp:
            logits = model(x)
        total += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1),
                                 ignore_index=-100, reduction="sum").item()
        tokens += int((y != -100).sum())
    loss = total / max(1, tokens)
    return {"loss": loss, "perplexity": math.exp(min(loss, 20)), "tokens": tokens}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", required=True, help="dir with val/ shards")
    p.add_argument("--run", action="append", required=True, help="name=checkpoint.pt")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--val-windows", type=int, default=1600)
    p.add_argument("--val-seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", default="rescore.json")
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)   # fail-safe before long work

    from ramanujan.data.loaders.shard_loader import FixedWindowDataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = FixedWindowDataset(Path(args.shards) / "val", args.seq_len, args.val_windows, args.val_seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    results = {"shards": args.shards, "val_windows": len(ds), "val_seed": args.val_seed, "runs": {}}
    for spec in args.run:
        name, path = spec.split("=", 1)
        model = probe.load_run(path, device)
        results["runs"][name] = val_loss(model, loader, device)
        print(f"{name:<20} loss {results['runs'][name]['loss']:.4f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()

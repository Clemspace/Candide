#!/usr/bin/env python3
"""
Export W&B runs (full-resolution history, system metrics, config, summary)
to CSV files that can be shared for offline analysis.

Usage (on the server, in the candide venv, after `wandb login`):
    python scripts/export_wandb.py --group pilot-lr-sweep
    python scripts/export_wandb.py --group pilot-lr-sweep --group pilot-confirm --out runs/export

Writes to --out (default ./wandb_export/):
    runs.csv      one row per run: name, id, state, group, tags, runtime,
                  flattened config, final summary metrics
    history.csv   every logged training/eval row, all runs, long format
    system.csv    GPU utilisation / memory / power samples, all runs
then zips the folder to <out>.zip for upload.
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import wandb


def flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def summary_dict(run):
    """Run summary as a plain dict, across wandb versions."""
    for attr in ("summary_metrics", "_summary"):
        v = getattr(run, attr, None)
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError:
                v = None
        if isinstance(v, dict) and v:
            return v
    raw = getattr(run.summary, "_json_dict", None)
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict):
        return raw
    try:
        return dict(run.summary)
    except Exception:
        return {}


def system_history(run):
    """System metrics (GPU util/memory); stream name differs across versions."""
    for stream in ("system", "events"):
        try:
            s = run.history(stream=stream, samples=100_000, pandas=True)
            if s is not None and not s.empty:
                return s
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="clemspace/candide-18c")
    ap.add_argument("--group", action="append", default=None,
                    help="W&B group(s) to export; repeatable. Default: pilot-lr-sweep")
    ap.add_argument("--include-smoke", action="store_true",
                    help="Also export runs named smoke-*")
    ap.add_argument("--out", default="wandb_export")
    args = ap.parse_args()
    groups = args.group or ["pilot-lr-sweep"]

    api = wandb.Api(timeout=60)
    runs = api.runs(args.project, filters={"group": {"$in": groups}})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_rows, hist, sysm = [], [], []

    for run in runs:
        if run.name.startswith("smoke-") and not args.include_smoke:
            continue
        print(f"{run.name:<32} {run.state:<10} {run.id}")

        raw_summary = summary_dict(run)
        summary = {k: v for k, v in raw_summary.items()
                   if not k.startswith("_") and isinstance(v, (int, float, str, bool))}
        run_rows.append({
            "name": run.name, "id": run.id, "state": run.state, "group": run.group,
            "tags": ";".join(run.tags), "created_at": run.created_at,
            "runtime_s": raw_summary.get("_runtime"),
            **{f"config.{k}": v for k, v in flatten(
                json.loads(run.config) if isinstance(run.config, str) else run.config).items()},
            **{f"summary.{k}": v for k, v in summary.items()},
        })

        h = pd.DataFrame(list(run.scan_history()))          # full resolution, no sampling
        if not h.empty:
            h.insert(0, "run", run.name)
            hist.append(h)

        s = system_history(run)
        if s is not None:
            s.insert(0, "run", run.name)
            sysm.append(s)

    if not run_rows:
        raise SystemExit(f"No runs found in {args.project} for groups {groups}")

    pd.DataFrame(run_rows).to_csv(out / "runs.csv", index=False)
    if hist:
        pd.concat(hist, ignore_index=True).to_csv(out / "history.csv", index=False)
    if sysm:
        pd.concat(sysm, ignore_index=True).to_csv(out / "system.csv", index=False)

    archive = shutil.make_archive(str(out), "zip", root_dir=out)
    print(f"\n{len(run_rows)} runs -> {archive}")


if __name__ == "__main__":
    main()
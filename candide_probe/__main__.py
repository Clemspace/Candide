"""Command line: python -m candide_probe [--runs DIR ...] [--tokenizer PATH] [--demo]."""
import argparse
import sys
from pathlib import Path

from .text import REPO

DEFAULT_TOKENIZER = "/data/pretrain_candide/tokenizer/tokenizer.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m candide_probe", description=__doc__)
    ap.add_argument("--runs", nargs="+", default=["runs"],
                    help="run directories or checkpoint files; every */checkpoints/*.pt below is listed")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="tokenizer.json of the checkpoints (v2g)")
    ap.add_argument("--host", default="127.0.0.1", help="default: local only (reach it with ssh -L)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    ap.add_argument("--max-loaded", type=int, default=3, help="models kept in memory at once")
    ap.add_argument("--results-dir", default=str(REPO / "results"), help="where saved results go (<date>/probe_*.json)")
    ap.add_argument("--demo", action="store_true", help="random numbers, no model: to work on the interface")
    ap.add_argument("--selftest", metavar="CKPT", help="check the engine on one checkpoint and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        import torch
        from .selftest import run
        return run(args.selftest, args.tokenizer, args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    from .server import App, serve
    if args.demo:
        from .demo import DemoEngine
        engine = DemoEngine()
        print("DEMO mode: random numbers, no model loaded")
    else:
        import torch
        from .checkpoints import Registry
        from .engine import Engine
        from .text import CandideTokenizer
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not Path(args.tokenizer).exists():
            print(f"tokenizer not found: {args.tokenizer} (use --tokenizer)", file=sys.stderr)
            return 2
        tok = CandideTokenizer(args.tokenizer)
        registry = Registry(args.runs, device=device, max_loaded=args.max_loaded)
        engine = Engine(registry, tok)
        refs = registry.refs()
        print(f"tokenizer {args.tokenizer}: {tok.vocab_size} entries, byte-level {tok.byte_level}")
        print(f"device {device}; {len(refs)} checkpoints under {', '.join(args.runs)}")
        for r in refs:
            print(f"  {r.id}  ({r.size_mb:,.0f} MB)")
        if not refs:
            print("  none found: pass --runs <dir containing <run>/checkpoints/*.pt>", file=sys.stderr)

    app = App(engine, Path(args.results_dir), demo=args.demo)
    httpd = serve(app, args.host, args.port)
    print(f"Candide probe on http://{args.host}:{args.port}  (code {app.commit or '?'}"
          f"{', uncommitted changes' if app.dirty else ''})")
    if args.host in ("127.0.0.1", "localhost"):
        print(f"from your laptop: ssh -N -L {args.port}:127.0.0.1:{args.port} obtic-gpu1, then open "
              f"http://127.0.0.1:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""HTTP server of the probe: one page and a small JSON API, on the standard library.

Every computed response gets a record id and is kept in memory (last 64). POST /api/save
writes a kept record under results/<date>/ with everything needed to reproduce it: the
request, the checkpoints (path, step, run, W&B id), the tokenizer, the code commit, the
device. The page saves only when asked; nothing is written on its own.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import threading
import traceback
import uuid
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .checkpoints import dirty, git_commit
from .errors import ProbeError
from .text import REPO

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 8 * 1024 * 1024


class Records:
    def __init__(self, keep: int = 64):
        self.keep, self.items, self.lock = keep, OrderedDict(), threading.Lock()

    def add(self, kind: str, request: dict, response: dict) -> str:
        rid = uuid.uuid4().hex[:10]
        with self.lock:
            self.items[rid] = {"kind": kind, "request": request, "response": response,
                               "time": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
            while len(self.items) > self.keep:
                self.items.popitem(last=False)
        return rid

    def get(self, rid: str) -> Optional[dict]:
        with self.lock:
            return self.items.get(rid)


def _ckpt_ids(req: dict) -> list[str]:
    ids = req.get("ckpts") or ([req["ckpt"]] if req.get("ckpt") else [])
    if not ids:
        raise ProbeError("no checkpoint selected")
    return [str(i) for i in ids]


def parse_probeset(text: str) -> list[dict]:
    """JSON lines: {"a", "b"} rows are pairs, {"prompt", "choices"[, "answer"]} rows are choices."""
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ProbeError(f"line {n}: not JSON ({e.msg})")
        if "a" in row and "b" in row:
            row["_kind"] = "pair"
        elif "prompt" in row and isinstance(row.get("choices"), list) and len(row["choices"]) >= 2:
            row["_kind"] = "choice"
        else:
            raise ProbeError(f"line {n}: expected {{\"a\", \"b\"}} or {{\"prompt\", \"choices\": [...]}}")
        row.setdefault("id", f"{row['_kind']}-{n}")
        rows.append(row)
    if not rows:
        raise ProbeError("no rows")
    return rows


def _answer_index(row: dict) -> Optional[int]:
    ans = row.get("answer")
    if ans is None:
        return None
    if isinstance(ans, int):
        return ans
    try:
        return row["choices"].index(ans)
    except ValueError:
        raise ProbeError(f"{row['id']}: answer {ans!r} is not one of the choices")


def run_probeset(engine, ckpts: list[str], rows: list[dict], clean: bool = True) -> dict:
    """Pairs through engine.pairs (score_text's pairs mode), choices through engine.candidates."""
    pairs = [r for r in rows if r["_kind"] == "pair"]
    choices = [r for r in rows if r["_kind"] == "choice"]
    out_rows = {r["id"]: {"id": r["id"], "kind": r["_kind"], "group": r.get("group"),
                          **({"a": r["a"], "b": r["b"]} if r["_kind"] == "pair" else
                             {"prompt": r["prompt"], "choices": r["choices"], "answer": _answer_index(r)}),
                          "by_ckpt": {}} for r in rows}
    if pairs:
        res = engine.pairs(ckpts, pairs, clean=clean)
        for cid, run in res["runs"].items():
            for r, row in zip(pairs, run["rows"]):
                out_rows[r["id"]]["by_ckpt"][cid] = {
                    "nll_a": row["nll_a"], "nll_b": row["nll_b"],
                    "bits_per_byte_a": row["bits_per_byte_a"], "bits_per_byte_b": row["bits_per_byte_b"],
                    "prefers_a_total": row["prefers_a_total"], "prefers_a_per_byte": row["prefers_a_per_byte"]}
    for r in choices:
        res = engine.candidates(ckpts, r["prompt"], r["choices"], clean=clean)
        ans = out_rows[r["id"]]["answer"]
        out_rows[r["id"]]["boundary_merge"] = [c["boundary_merge"] for c in res["choices"]]
        for cid, run in res["runs"].items():
            ranks = [x["rank"] for x in run["rows"]]
            entry = {"ranks": ranks, "nll_cond": [x["nll_cond"] for x in run["rows"]],
                     "bits_per_byte_cond": [x["bits_per_byte_cond"] for x in run["rows"]]}
            if ans is not None:
                entry["answer_rank"] = ranks[ans]
                entry["correct"] = ranks[ans] == 1
            out_rows[r["id"]]["by_ckpt"][cid] = entry

    summary = {}
    for cid in ckpts:
        groups: dict[str, dict] = {}
        for row in out_rows.values():
            res = row["by_ckpt"].get(cid)
            if res is None:
                continue
            for g in ("all", row.get("group")):
                if g is None:
                    continue
                s = groups.setdefault(str(g), {"pairs": 0, "prefers_a_total": 0, "prefers_a_per_byte": 0,
                                               "choices": 0, "with_answer": 0, "correct": 0, "answer_rank_sum": 0})
                if row["kind"] == "pair":
                    s["pairs"] += 1
                    s["prefers_a_total"] += res["prefers_a_total"]
                    s["prefers_a_per_byte"] += res["prefers_a_per_byte"]
                else:
                    s["choices"] += 1
                    if "answer_rank" in res:
                        s["with_answer"] += 1
                        s["correct"] += res["correct"]
                        s["answer_rank_sum"] += res["answer_rank"]
        for s in groups.values():
            if s["pairs"]:
                s["share_prefers_a_total"] = s["prefers_a_total"] / s["pairs"]
                s["share_prefers_a_per_byte"] = s["prefers_a_per_byte"] / s["pairs"]
            if s["with_answer"]:
                s["accuracy"] = s["correct"] / s["with_answer"]
                s["mean_answer_rank"] = s["answer_rank_sum"] / s["with_answer"]
        summary[cid] = groups
    return {"rows": list(out_rows.values()), "summary": summary,
            "n_pairs": len(pairs), "n_choices": len(choices)}


class App:
    def __init__(self, engine, results_dir: Path, demo: bool = False):
        self.engine, self.results_dir, self.demo = engine, Path(results_dir), demo
        self.records = Records()
        self.commit, self.dirty = git_commit(REPO), dirty(REPO)

    # -- API ------------------------------------------------------------------

    def info(self) -> dict:
        return {**self.engine.info(), "demo": self.demo, "commit": self.commit, "dirty": self.dirty,
                "results_dir": str(self.results_dir)}

    def compute(self, kind: str, req: dict) -> dict:
        e = self.engine
        clean, bos = bool(req.get("clean", True)), bool(req.get("bos", True))
        if kind == "score":
            out = e.score(_ckpt_ids(req), str(req.get("text", "")), clean=clean, bos=bos,
                          topk=int(req.get("topk", 5)))
        elif kind == "generate":
            out = e.generate(_ckpt_ids(req)[0], str(req.get("prompt", "")), ids=req.get("ids"),
                             clean=clean, bos=bos, max_new_tokens=int(req.get("max_new_tokens", 120)),
                             greedy=bool(req.get("greedy", False)), temperature=float(req.get("temperature", 0.8)),
                             top_k=int(req.get("top_k", 0)), top_p=float(req.get("top_p", 0.95)),
                             repetition_penalty=float(req.get("repetition_penalty", 1.0)),
                             seed=req.get("seed"), topn=int(req.get("topn", 8)),
                             stop_at_eos=bool(req.get("stop_at_eos", True)))
        elif kind == "candidates":
            out = e.candidates(_ckpt_ids(req), str(req.get("prompt", "")),
                               [str(c) for c in req.get("choices", [])], clean=clean)
        elif kind == "probeset":
            rows = parse_probeset(str(req.get("jsonl", "")))
            out = run_probeset(e, _ckpt_ids(req), rows, clean=clean)
        elif kind == "lens":
            out = e.lens(_ckpt_ids(req)[0], str(req.get("text", "")), clean=clean, bos=bos)
        elif kind == "attention":
            layer = req.get("layer")
            out = e.attention(_ckpt_ids(req)[0], str(req.get("text", "")), clean=clean, bos=bos,
                              layer=int(layer) if layer is not None else None)
        else:
            raise ProbeError(f"unknown kind {kind!r}")
        out["record_id"] = self.records.add(kind, req, out)
        return out

    def save(self, req: dict) -> dict:
        rec = self.records.get(str(req.get("record_id", "")))
        if rec is None:
            raise ProbeError("record not found (the server keeps the last 64 results): run it again")
        name = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(req.get("name") or rec["kind"])).strip("-") or rec["kind"]
        today = dt.date.today().isoformat()
        out_dir = self.results_dir / today
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%H%M%S")
        path = out_dir / f"probe_{rec['kind']}_{name}_{stamp}.json"
        ckpts = rec["request"].get("ckpts") or [rec["request"].get("ckpt")]
        metas = {}
        for c in ckpts:
            if not c:
                continue
            try:
                m = self.engine.meta(c)
                metas[c] = {k: m.get(k) for k in ("path", "run", "name", "step", "wandb_id", "n_params", "model", "train")}
            except Exception as ex:                       # keep the result even if metadata fails
                metas[c] = {"error": str(ex)}
        doc = {
            "tool": "candide_probe", "kind": rec["kind"], "computed_at": rec["time"],
            "saved_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "code_commit": self.commit, "code_dirty": self.dirty, "demo": self.demo,
            "info": self.engine.info(), "checkpoints": metas,
            "request": rec["request"], "response": {k: v for k, v in rec["response"].items() if k != "record_id"},
        }
        path.write_text(json.dumps(_finite(doc), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        try:
            shown = str(path.relative_to(REPO))
        except ValueError:
            shown = str(path)
        return {"path": shown}


def _finite(obj):
    """NaN and infinities are not JSON: send them as null."""
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "candide-probe/1"

        def log_message(self, fmt, *args):            # one short line per API call
            if self.path.startswith("/api/"):
                print(f"[probe] {self.command} {self.path.split('?')[0]} {args[1] if len(args) > 1 else ''}", flush=True)

        def _send(self, status: int, body: bytes, ctype: str):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200):
            self._send(status, json.dumps(_finite(obj), ensure_ascii=False, allow_nan=False, default=str).encode(),
                       "application/json; charset=utf-8")

        def _error(self, e: Exception):
            if isinstance(e, (ProbeError, KeyError)):
                msg = e.args[0] if e.args else str(e)
                self._json({"error": str(msg)}, HTTPStatus.BAD_REQUEST)
            else:
                traceback.print_exc()
                self._json({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=6)},
                           HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                if url.path == "/api/info":
                    return self._json(app.info())
                if url.path == "/api/checkpoints":
                    return self._json(app.engine.list_checkpoints(**({"refresh": True} if q.get("refresh") else {})))
                if url.path == "/api/meta":
                    return self._json(app.engine.meta(q["id"]))
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as e:
                self._error(e)

        def do_POST(self):
            url = urlparse(self.path)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    raise ProbeError("request too large")
                req = json.loads(self.rfile.read(n) or b"{}")
                if url.path == "/api/save":
                    return self._json(app.save(req))
                if url.path.startswith("/api/"):
                    return self._json(app.compute(url.path[len("/api/"):], req))
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as e:
                self._error(e)

    return Handler


def serve(app: App, host: str, port: int):
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    return httpd


__all__ = ["App", "serve", "parse_probeset", "run_probeset", "Records"]

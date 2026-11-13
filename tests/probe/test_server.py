"""Server side of the probe, without PyTorch: discovery, probe sets, records, HTTP round trip.

The demo engine stands in for the model; what is tested is the plumbing around it.
"""
import json
import threading
import urllib.request
from pathlib import Path

import pytest

from candide_probe import checkpoints as C
from candide_probe.demo import DemoEngine
from candide_probe.errors import ProbeError
from candide_probe.server import App, Records, parse_probeset, run_probeset, serve
from candide_probe.text import _byte_decoder


def test_discover_orders_steps_and_names_runs(tmp_path):
    ck = tmp_path / "full" / "candide-v2g-224m" / "checkpoints"
    ck.mkdir(parents=True)
    for n in ("final", "step_9000", "pre_cooldown", "step_10000", "step_800"):
        (ck / f"{n}.pt").write_bytes(b"x")
    (ck / "step_5.pt.tmp").write_bytes(b"x")                       # half-written: ignored
    other = tmp_path / "cool62k" / "checkpoints"
    other.mkdir(parents=True)
    (other / "final.pt").write_bytes(b"x")
    (tmp_path / "loose.pt").write_bytes(b"x")                       # not under checkpoints/: ignored
    refs = C.discover([tmp_path])
    ids = [r.id for r in refs]
    assert ids == ["cool62k/final", "full/candide-v2g-224m/step_800", "full/candide-v2g-224m/step_9000",
                   "full/candide-v2g-224m/step_10000", "full/candide-v2g-224m/pre_cooldown",
                   "full/candide-v2g-224m/final"]
    assert C.discover([ck / "final.pt"])[0].id == "candide-v2g-224m/final"
    assert C.discover([tmp_path / "missing"]) == []


def test_byte_decoder_is_a_bijection_on_bytes():
    table = _byte_decoder()
    assert len(table) == 256 and sorted(table.values()) == list(range(256))
    assert table["Ġ"] == 32 and table["A"] == 65


def test_parse_probeset():
    rows = parse_probeset('# comment\n{"a": "x", "b": "y", "group": "g"}\n\n'
                          '{"prompt": "p", "choices": [" a", " b"], "answer": " b"}\n')
    assert [r["_kind"] for r in rows] == ["pair", "choice"]
    assert rows[0]["id"] == "pair-2" and rows[1]["id"] == "choice-4"
    with pytest.raises(ProbeError, match="line 1"):
        parse_probeset("{not json}")
    with pytest.raises(ProbeError, match="expected"):
        parse_probeset('{"prompt": "p", "choices": ["only one"]}')
    with pytest.raises(ProbeError):
        parse_probeset("\n# nothing\n")


class StubEngine:
    """Fixed losses: pair a wins when its text is shorter; choice ranks follow text length."""

    def pairs(self, ckpts, pairs, clean=True):
        runs = {}
        for c in ckpts:
            rows = [{"nll_a": len(p["a"]), "nll_b": len(p["b"]), "bits_per_byte_a": 1.0, "bits_per_byte_b": 1.0,
                     "prefers_a_total": len(p["a"]) < len(p["b"]), "prefers_a_per_byte": False} for p in pairs]
            runs[c] = {"rows": rows}
        return {"runs": runs}

    def candidates(self, ckpts, prompt, choices, clean=True):
        order = sorted(range(len(choices)), key=lambda i: len(choices[i]))
        rank = {i: r + 1 for r, i in enumerate(order)}
        return {"choices": [{"boundary_merge": False} for _ in choices],
                "runs": {c: {"rows": [{"rank": rank[i], "nll_cond": float(len(ch)), "bits_per_byte_cond": 1.0}
                                      for i, ch in enumerate(choices)]} for c in ckpts}}


def test_run_probeset_summaries():
    rows = parse_probeset("\n".join(json.dumps(r) for r in [
        {"group": "oit", "a": "étoit", "b": "était!"},          # a shorter: preferred
        {"group": "oit", "a": "avoit!!", "b": "avait"},         # b preferred
        {"group": "date", "prompt": "p", "choices": [" a", " bbb"], "answer": 0},     # rank 1
        {"group": "date", "prompt": "p", "choices": [" a", " bbb"], "answer": " bbb"},  # rank 2
        {"prompt": "p", "choices": [" a", " b"]},              # no answer: not in accuracy
    ]))
    res = run_probeset(StubEngine(), ["A", "B"], rows)
    s = res["summary"]["A"]
    assert s["oit"]["pairs"] == 2 and s["oit"]["share_prefers_a_total"] == 0.5
    assert s["date"]["accuracy"] == 0.5 and s["date"]["mean_answer_rank"] == 1.5
    assert s["all"]["pairs"] == 2 and s["all"]["choices"] == 3 and s["all"]["with_answer"] == 2
    assert res["n_pairs"] == 2 and res["n_choices"] == 3
    with pytest.raises(ProbeError, match="not one of the choices"):
        run_probeset(StubEngine(), ["A"], parse_probeset('{"prompt": "p", "choices": [" a", " b"], "answer": " c"}'))


def test_records_keep_the_last_n():
    r = Records(keep=2)
    a, b, c = (r.add("score", {}, {"n": i}) for i in range(3))
    assert r.get(a) is None and r.get(c)["response"] == {"n": 2}


@pytest.fixture
def server(tmp_path):
    app = App(DemoEngine(n_layers=4, n_heads=3), tmp_path / "results", demo=True)
    httpd = serve(app, "127.0.0.1", 0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", tmp_path
    httpd.shutdown()
    httpd.server_close()


def _call(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_round_trip_and_save(server):
    base, tmp = server
    st, page = 200, urllib.request.urlopen(base + "/").read().decode()
    assert "Candide" in page and "<script>" in page
    st, info = _call(base + "/api/info")
    assert st == 200 and info["demo"] is True
    st, ckpts = _call(base + "/api/checkpoints")
    cid = ckpts[-2]["id"]
    st, meta = _call(base + "/api/meta?id=" + cid)
    assert meta["step"] == ckpts[-2]["step_from_name"]
    for kind, body in [("score", {"text": "Il étoit tard."}), ("generate", {"prompt": "Le roi", "seed": 1}),
                       ("lens", {"text": "Il étoit tard."}), ("attention", {"text": "Il étoit tard.", "layer": 2}),
                       ("candidates", {"prompt": "Il", "choices": [" étoit", " était"]}),
                       ("probeset", {"jsonl": '{"a": "étoit", "b": "était"}'})]:
        st, res = _call(base + "/api/" + kind, {"ckpts": [cid], **body})
        assert st == 200, (kind, res)
        assert res["record_id"]
    st, saved = _call(base + "/api/save", {"record_id": res["record_id"], "name": "oit pairs/v1"})
    assert st == 200
    path = tmp / "results" / Path(saved["path"]).parent.name / Path(saved["path"]).name
    doc = json.loads(path.read_text())
    assert path.name.startswith("probe_probeset_oit-pairs-v1_")
    assert doc["kind"] == "probeset" and doc["request"]["jsonl"] and doc["checkpoints"][cid]["step"]
    assert "record_id" not in doc["response"] and doc["demo"] is True


def test_http_errors_are_json(server):
    base, _ = server
    st, res = _call(base + "/api/score", {"ckpts": [], "text": "x"})
    assert st == 400 and "checkpoint" in res["error"]
    st, res = _call(base + "/api/meta?id=nope")
    assert st == 400 and "unknown" in res["error"]
    st, res = _call(base + "/api/save", {"record_id": "missing"})
    assert st == 400
    st, res = _call(base + "/api/nothing", {})
    assert st == 400

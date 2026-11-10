"""Source-level deduplication and the manifest mode of prepare_shards."""

import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # lets multiprocessing pickle its functions
    spec.loader.exec_module(mod)
    return mod


DD = _load("dedup_corpus")
PS = DD.ps

WORDS = ("roi lettre amour dieu peuple raison liberté nature esprit siècle loi homme femme "
         "ville cour gloire vertu honneur philosophe histoire guerre paix temps monde").split()


def text(seed, n=600):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) + str(rng.randrange(1000)) for _ in range(n))


# ── Hashing ───────────────────────────────────────────────────────────────────

def test_window_hash_matches_direct_polynomial():
    data = b"abcdefghij"
    h = DD.window_hashes(data, 4)
    direct = []
    for i in range(len(data) - 3):
        v = 0
        for ch in data[i:i + 4]:
            v = (v * DD.BASE + ch) % 2 ** 64
        direct.append(v)
    assert [int(x) for x in h] == direct


def test_window_hash_is_position_independent():
    block = text(1, 50).encode()
    data = b"xx" + block + b"yyyyy" + block
    h = DD.window_hashes(data, 32)
    a, b = 2, 2 + len(block) + 5
    n = len(block) - 31
    assert np.array_equal(h[a:a + n], h[b:b + n])


def test_blocked_hashing_equals_whole():
    data = text(9, 3000).encode()
    whole = DD.window_hashes(data, 64)
    whole = whole[DD.sample_mask(whole, 2)]
    assert np.array_equal(DD.sampled_window_hashes(data, 64, 2, block=1000), whole)


def test_normalize_cleans_and_folds():
    assert DD.normalize("Le  Roi&nbsp;de\nFrance &amp; Navarre") == "le roi de france & navarre".encode()


def test_clean_text_entities():
    assert PS.clean_text("a&nbsp;b & nbsp ; c &#233; &lt;x&gt;") == "a b   c é <x>"


# ── Decisions ─────────────────────────────────────────────────────────────────

def _signatures(texts, k=64, shift=1):
    hs, ds = [], []
    for i, t in enumerate(texts):
        h = DD.window_hashes(DD.normalize(t), k)
        h = h[DD.sample_mask(h, shift)]
        hs.append(h)
        ds.append(np.full(len(h), i, dtype=np.int64))
    return np.concatenate(hs), np.concatenate(ds)


def test_near_duplicates_internal_repetition_and_restoration():
    A, C, U = text(1), text(2), text(3)
    block = text(4, 150)
    docs = [
        A,                                   # 0 keep (first copy)
        A + " fin",                          # 1 near_dup of 0
        C,                                   # 2 keep (unique)
        A[: len(A) // 2] + " " + U,          # 3 keep (half shared)
        " ".join([block] * 6),               # 4 internal_rep
        block,                               # 5 near_dup of 4 at first, restored (4 dropped)
    ]
    h, d = _signatures(docs)
    r = DD.analyse(h, d, len(docs), max_containment=0.8, max_intra_dup=0.3)
    assert [DD.DECISIONS[int(x)] for x in r["decision"]] == \
        ["keep", "near_dup", "keep", "keep", "internal_rep", "keep"]
    assert r["containment"][1] > 0.95 and r["duplicate_of"][1] == 0
    assert 0.3 < r["containment"][3] < 0.7
    assert r["intra_dup"][4] > 0.7 and r["intra_dup"][0] < 0.01
    assert r["restored"][5]
    assert r["shared"][2] < 0.01 and r["shared"][3] > 0.3


def test_choose_val_matches_mix_and_avoids_shared_files():
    sub = ["a"] * 200 + ["b"] * 200
    n_chars = np.full(400, 10_000)
    decision = np.zeros(400, dtype=np.int8)
    shared = np.zeros(400)
    shared[:100] = 0.5                       # half of "a" shares text: not eligible
    is_val = DD.choose_val(sub, n_chars, decision, shared, val_fraction=0.05,
                           max_shared=0.02, min_chars=2000, seed=0)
    assert not is_val[:100].any()
    assert is_val[:200].sum() == 10 and is_val[200:].sum() == 10


def test_choose_val_skips_files_that_overshoot():
    sub = ["a"] * 50
    n_chars = np.array([10_000_000] + [1_000] * 49)
    is_val = DD.choose_val(sub, n_chars, np.zeros(50, dtype=np.int8), np.zeros(50),
                           val_fraction=0.01, max_shared=0.02, min_chars=500, seed=0)
    assert not is_val[0]


# ── End to end + prepare_shards manifest mode ─────────────────────────────────

@pytest.fixture
def extracted(tmp_path, monkeypatch):
    monkeypatch.setattr(PS, "SUBCORPORA", {"modern_canon": 1.0, "modern_lettres": 1.0})
    canon, lettres = tmp_path / "modern_canon", tmp_path / "modern_lettres"
    canon.mkdir(), lettres.mkdir()
    for i in range(30):
        (canon / f"c{i:02d}.txt").write_text(text(100 + i), encoding="utf-8")
    for i in range(30):
        (lettres / f"l{i:02d}.txt").write_text(text(200 + i), encoding="utf-8")
    (lettres / "l99_copy_of_c00.txt").write_text(text(100) + " copie", encoding="utf-8")
    return tmp_path


def test_end_to_end_manifest(extracted, tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["dedup_corpus.py", "--extracted-dir", str(extracted),
                                      "--out-dir", str(out), "--workers", "1", "--k", "64",
                                      "--sample-shift", "1", "--val-fraction", "0.1",
                                      "--val-min-chars", "100"])
    DD.main()
    rows = list(csv.DictReader(open(out / "dedup_manifest.csv", encoding="utf-8")))
    by_name = {Path(r["path"]).name: r for r in rows}
    assert by_name["l99_copy_of_c00.txt"]["decision"] == "near_dup"
    assert Path(by_name["l99_copy_of_c00.txt"]["duplicate_of"]).name == "c00.txt"
    assert by_name["c00.txt"]["decision"] == "keep"
    assert {r["split"] for r in rows if r["decision"] == "keep"} == {"train", "val"}
    summary = json.loads((out / "dedup_summary.json").read_text())
    assert summary["subcorpora"]["modern_lettres"]["near_dup_files"] == 1

    train, val = PS.manifest_split(out / "dedup_manifest.csv", seed=0)
    assert len(train) + len(val) == 60 and len(val) > 0
    assert not set(train) & set(val)
    assert Path(by_name["l99_copy_of_c00.txt"]["path"]) not in set(train) | set(val)


def test_select_files_reproduces_v1_selection(tmp_path, monkeypatch):
    weights = {"modern_canon": 1.0, "modern_archive": 0.15}
    monkeypatch.setattr(PS, "SUBCORPORA", weights)
    for name in weights:
        (tmp_path / name).mkdir()
        for i in range(40):
            (tmp_path / name / f"{i:03d}.txt").write_text("x")
    # the v1 loop, verbatim
    random.seed(42)
    v1 = []
    for name, w in weights.items():
        files = sorted((tmp_path / name).glob("*.txt"))
        if w < 1.0:
            files = random.sample(files, max(1, int(len(files) * w)))
        v1.extend(files)
    random.shuffle(v1)
    n_val = max(1, int(len(v1) * 0.005))
    v1_train, v1_val = v1[n_val:], v1[:n_val]

    train, val = PS.legacy_split(PS.select_files(tmp_path, 42))
    assert train == v1_train and val == v1_val


def test_exclude_list(extracted, tmp_path, monkeypatch):
    excl = tmp_path / "exclude.txt"
    excl.write_text("# non-French files\n/some/where/c00.tei\nl05\n", encoding="utf-8")
    out = tmp_path / "out_excl"
    monkeypatch.setattr(sys, "argv", ["dedup_corpus.py", "--extracted-dir", str(extracted),
                                      "--out-dir", str(out), "--workers", "1", "--k", "64",
                                      "--sample-shift", "1", "--val-fraction", "0.1",
                                      "--val-min-chars", "100", "--exclude-list", str(excl)])
    DD.main()
    rows = {Path(r["path"]).name: r for r in csv.DictReader(open(out / "dedup_manifest.csv", encoding="utf-8"))}
    assert rows["c00.txt"]["decision"] == "excluded" and rows["l05.txt"]["decision"] == "excluded"
    assert rows["c00.txt"]["split"] == ""
    # c00 is excluded, so its copy is no longer a near-duplicate of anything kept
    assert rows["l99_copy_of_c00.txt"]["decision"] == "keep"
    s = json.loads((out / "dedup_summary.json").read_text())
    assert s["subcorpora"]["modern_canon"]["excluded_files"] == 1


def test_v2_sources_keep_v1_selection_and_add_merged(tmp_path, monkeypatch):
    monkeypatch.setattr(PS, "SUBCORPORA", {"modern_canon": 1.0, "modern_archive": 0.15})
    monkeypatch.setattr(PS, "SUBCORPORA_V2", {"modern_canon": 1.0, "modern_archive": 0.15,
                                              "modern_merged": 1.0})
    for name in ("modern_canon", "modern_archive", "modern_merged"):
        (tmp_path / name).mkdir()
        for i in range(40):
            (tmp_path / name / f"{i:03d}.txt").write_text("x")
    v1 = PS.select_files(tmp_path, 42, "v1")
    v2 = PS.select_files(tmp_path, 42, "v2")
    assert v2[: len(v1)] == v1                               # same v1 files, same order
    assert [s for s, _ in v2[len(v1):]] == ["modern_merged"] * 40   # merged last

"""Tokenizer training: lossless round-trip, French elision, specials, noisy share."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
_spec = importlib.util.spec_from_file_location("train_tokenizer", SCRIPTS / "train_tokenizer.py")
TT = importlib.util.module_from_spec(_spec)
sys.modules["train_tokenizer"] = TT
_spec.loader.exec_module(TT)

CLEAN = ("L'Encyclopédie, ou Dictionnaire raisonné des sciences, des arts et des métiers. "
         "Qu'en dit-on ? Lorsqu'il vint, l'homme dit : « c'est la raison ». ") * 40
NOISY = "Lc rui de Fraiice eft daus la ville , & il ue veut pas q le peuple fe plaigne. " * 40


@pytest.fixture
def manifest(tmp_path):
    rows = []
    for i in range(10):
        for sub, text in (("modern_canon", CLEAN), ("modern_archive", NOISY)):
            f = tmp_path / f"{sub}_{i}.txt"
            f.write_text(text, encoding="utf-8")
            rows.append({"priority": len(rows), "subcorpus": sub, "path": str(f),
                         "n_chars": len(text), "n_windows": 0, "intra_dup": 0, "containment": 0,
                         "duplicate_of": "", "shared": 0, "decision": "keep",
                         "split": "val" if i >= 8 else "train"})
    path = tmp_path / "manifest.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def test_sample_files_respects_noisy_share(manifest):
    rows = TT.read_manifest(str(manifest))
    budget = 6 * len(CLEAN)
    picked = TT.sample_files(rows, "train", {"modern_archive"}, 0.25, budget, seed=0)
    noisy = sum(int(r["n_chars"]) for r in picked if r["subcorpus"] == "modern_archive")
    clean = sum(int(r["n_chars"]) for r in picked if r["subcorpus"] != "modern_archive")
    # each budget is met by whole files: reached, overshot by less than one file
    assert 0.25 * budget <= noisy < 0.25 * budget + len(NOISY)
    assert 0.75 * budget <= clean < 0.75 * budget + len(CLEAN)
    assert all(r["split"] == "train" for r in picked)
    none = TT.sample_files(rows, "train", {"modern_archive"}, 0.0, 8 * len(CLEAN), seed=0)
    assert all(r["subcorpus"] != "modern_archive" for r in none)


def test_train_roundtrip_specials_and_elision(manifest, tmp_path, monkeypatch):
    pytest.importorskip("tokenizers")
    out = tmp_path / "tok" / "tokenizer.json"
    monkeypatch.setattr(sys, "argv", ["tt", "train", "--manifest", str(manifest), "--out", str(out),
                                      "--vocab-size", "400", "--noisy-share", "0.1",
                                      "--sample-chars", "1e6", "--min-frequency", "1"])
    TT.main()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(out))
    for s in ["L'Encyclopédie, ou Dictionnaire raisonné des sciences. Qu'en dit-on ?",
              "lorsqu'il vint aujourd'hui, l’homme dit : « 1746 ».\n\n  Fin",
              "la ſcience  &  les arts — 漢字 \t é"]:
        assert tok.decode(tok.encode(s).ids) == s
    assert [tok.token_to_id(s) for s in TT.DEFAULT_SPECIALS] == [0, 1, 2, 3, 4]
    pieces = [tok.decode([i]) for i in tok.encode(" l'homme").ids]
    assert pieces[0] == " l'"                                  # elision kept as one pre-token
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["vocab_size"] <= 400 and 0 < meta["noisy_share_actual"] < 0.2


def test_old_tokenizer_specials_keep_their_ids(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.models import WordLevel
    old = tokenizers.Tokenizer(WordLevel({"<|bos|>": 0, "<|eos|>": 1, "a": 2}, unk_token="<|bos|>"))
    old.add_special_tokens(["<|bos|>", "<|eos|>"])
    path = tmp_path / "old.json"
    old.save(str(path))
    specials = TT.special_tokens(str(path))
    assert specials[:2] == ["<|bos|>", "<|eos|>"] and "[MASK]" in specials


def test_evaluate_reports_metrics(manifest, tmp_path, monkeypatch):
    pytest.importorskip("tokenizers")
    out = tmp_path / "tok.json"
    monkeypatch.setattr(sys, "argv", ["tt", "train", "--manifest", str(manifest), "--out", str(out),
                                      "--vocab-size", "400", "--noisy-share", "0.1",
                                      "--sample-chars", "1e6", "--min-frequency", "1"])
    TT.main()
    rep = TT.evaluate({"t": str(out)}, TT.read_manifest(str(manifest)), {"modern_archive"}, 10, 0)["t"]
    assert rep["roundtrip_exact_rate"] == 1.0
    assert rep["chars_per_token_clean"] > 1 and rep["chars_per_token_noisy"] > 1
    assert 0 <= rep["vocab_share_mostly_noisy"] < 1

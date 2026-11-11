"""Test set built from the validation split of a dedup manifest."""
import csv
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "make_testset", Path(__file__).resolve().parents[2] / "scripts" / "make_testset.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

COLS = ["priority", "subcorpus", "path", "n_chars", "n_windows", "intra_dup",
        "containment", "duplicate_of", "shared", "decision", "split"]


def _corpus(tmp_path):
    files = {"canon/candide.txt": ("keep", "val"), "canon/zadig.txt": ("keep", "train"),
             "press/gazette.txt": ("keep", "val"), "press/copie.txt": ("near_dup", ""),
             "press/sans_date.txt": ("keep", "val")}
    rows = []
    for i, (rel, (decision, split)) in enumerate(files.items()):
        p = tmp_path / "extracted" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"texte de {rel}\n" * 50, encoding="utf-8")
        rows.append([i, rel.split("/")[0], str(p), 0, 0, 0, 0, "", 0, decision, split])
    manifest = tmp_path / "dedup_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)
    inventory = tmp_path / "inventory.csv"
    with open(inventory, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["corpus", "source", "path", "date", "year", "genre", "author", "title"])
        w.writerow(["canon", "x", "/tei/canon/candide.xml", "1759", "1759", "conte", "Voltaire", "Candide"])
        w.writerow(["press", "x", "/tei/press/gazette.xml", "1687", "1687", "", "", "Gazette"])
        w.writerow(["press", "x", "/tei/press/sans_date.xml", "", "", "", "", ""])
    return manifest, inventory


def test_only_kept_validation_files_with_metadata(tmp_path):
    manifest, inventory = _corpus(tmp_path)
    out = tmp_path / "t" / "val.jsonl"
    assert M.main(["--manifest", str(manifest), "--inventory", str(inventory), "--out", str(out)]) == 0
    recs = {r["id"]: r for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}
    assert set(recs) == {"canon/candide.txt", "press/gazette.txt", "press/sans_date.txt"}
    c = recs["canon/candide.txt"]
    assert (c["year"], c["decade"], c["century"], c["author"]) == (1759, "1750s", "1700s", "Voltaire")
    assert c["text"].startswith("texte de canon/candide.txt") and c["n_chars"] == len(c["text"])
    assert (recs["press/gazette.txt"]["decade"], recs["press/gazette.txt"]["century"]) == ("1680s", "1600s")
    assert recs["press/sans_date.txt"]["decade"] == "undated" and recs["press/sans_date.txt"]["year"] is None


def test_without_inventory_and_with_a_cap(tmp_path):
    manifest, _ = _corpus(tmp_path)
    out = tmp_path / "val.jsonl"
    assert M.main(["--manifest", str(manifest), "--out", str(out), "--max-chars", "200"]) == 0
    recs = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 3 and all(r["decade"] == "undated" for r in recs)
    assert all(r["n_chars"] <= 200 and r["text"].startswith("texte de ") for r in recs)


def test_period_labels():
    assert M.period("1759") == ("1750s", "1700s") and M.period(1600) == ("1600s", "1600s")
    assert M.period(1800) == ("1800s", "1800s") and M.period(1799) == ("1790s", "1700s")
    assert M.period("") == ("undated", "undated") and M.period(None) == ("undated", "undated")

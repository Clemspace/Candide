"""TEI text extraction: the three fixes of 2026-11-10 and v1 behaviour otherwise."""

import importlib.util
import sys
from pathlib import Path

import pytest

etree = pytest.importorskip("lxml.etree")

_path = Path(__file__).resolve().parents[2] / "scripts" / "extract_tei.py"
_spec = importlib.util.spec_from_file_location("extract_tei", _path)
EX = importlib.util.module_from_spec(_spec)
sys.modules["extract_tei"] = EX
_spec.loader.exec_module(EX)


def paragraphs(xml: str):
    root = etree.fromstring(xml.encode())
    return EX.extract_paragraphs(EX.find_body(root))


def test_tail_after_skipped_element_is_emitted_once():
    out = paragraphs('<TEI><text><body><p>Alpha <pb n="2"/>beta <note>note</note> gamma.</p>'
                     '</body></text></TEI>')
    assert out == ["Alpha beta gamma."]


def test_nested_paragraph_is_emitted_once():
    out = paragraphs("<TEI><text><body><p>Outer <p>inner</p> end.</p><p>Next.</p></body></text></TEI>")
    assert out == ["Outer inner end.", "Next."]


def test_inline_markup_and_skipped_content():
    out = paragraphs("<TEI><text><body><head>Chapitre I</head>"
                     "<p>Le <hi>roi</hi> dit<ref>[1]</ref> : non.</p></body></text></TEI>")
    assert out == ["Le roi dit : non."]


def test_header_is_not_text(tmp_path):
    f = tmp_path / "a.xml"
    f.write_text("<TEI><teiHeader><fileDesc><title>Titre</title></fileDesc></teiHeader>"
                 "<text><body><p>" + "Texte du corps. " * 5 + "</p></body></text></TEI>",
                 encoding="utf-8")
    text, meta = EX.extract_file(f, "modern_canon")
    assert "Titre" not in text and text.startswith("Texte du corps.")
    assert meta["title"] == "Titre"


def test_pamphlet_fragments_without_root(tmp_path):
    f = tmp_path / "p.xml"
    f.write_text('<?xml version="1.0"?><teiHeader><title>T</title></teiHeader>'
                 "<text><body><p>" + "Libelle. " * 10 + "</p></body></text>", encoding="utf-8")
    text, _ = EX.extract_file(f, "modern_pamphlets")
    assert text.startswith("Libelle.")


def test_press_prefix_dedup_off_by_default(tmp_path, monkeypatch):
    src = tmp_path / "corpora" / "modern_press"
    src.mkdir(parents=True)
    masthead = "GAZETTE DE FRANCE. " * 120               # > 2,000 identical leading chars
    for i in range(3):
        (src / f"issue{i}.tei").write_text(
            f"<TEI><text><body><p>{masthead}</p><p>Nouvelles du jour {i}.</p></body></text></TEI>",
            encoding="utf-8")
    monkeypatch.setattr(EX, "BASE_DIR", tmp_path / "corpora")
    monkeypatch.setattr(EX, "OUT_DIR", tmp_path / "out")
    cfg = {"extensions": [".tei"], "recursive": True}
    assert EX.process_subcorpus("modern_press", cfg)["files_extracted"] == 3
    assert EX.process_subcorpus("modern_press", cfg, dry_run=True,
                                press_prefix_dedup=True)["files_extracted"] == 1


def test_encyclopedie_pattern_skips_encyclopediana(tmp_path):
    for name in ("ENC_volume01.xml", "ENC_volume17.xml", "encyclopediana.xml"):
        (tmp_path / name).write_text("<TEI><text><body><p>" + "x " * 100 + "</p></body></text></TEI>")
    cfg = EX.SUBCORPORA["encyclopedie"]
    files = EX.find_files(tmp_path, cfg["extensions"], cfg["recursive"], cfg.get("pattern"))
    assert [f.name for f in files] == ["ENC_volume01.xml", "ENC_volume17.xml"]


def test_manifest_from_dir(tmp_path):
    spec = importlib.util.spec_from_file_location("manifest_from_dir", _path.parent / "manifest_from_dir.py")
    MF = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MF)
    for v in ("ENC_volume01", "ENC_volume17"):
        (tmp_path / f"{v}.txt").write_text("texte " * 50, encoding="utf-8")
    rows = MF.build(tmp_path, "encyclopedie", ["ENC_volume17"])
    assert [(Path(r["path"]).stem, r["split"], r["decision"]) for r in rows] == \
        [("ENC_volume01", "train", "keep"), ("ENC_volume17", "val", "keep")]

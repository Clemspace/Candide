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


ENC_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<TEI.2><teiHeader><fileDesc><titleStmt><title>Encyclopédie, tome 1</title></titleStmt></fileDesc></teiHeader>
<text><body><p>Front matter.</p></body></text></TEI.2>
<div1 n="1" id="0:2" vol="1">
<index type="head" value="Title Page"/>
<index type="author" value="unsigned"/>
<milestone unit="volume" n="1"/>
<pb n="1:NA5" facs="V1/ENC_1-NA5.jpeg"/>
<head>Title Page</head>
<p>
<center>
<b><i>ENCYCLOPÉDIE</i>,<br/>
DICTIONNAIRE RAISONNÉ<br/>
DES SCIENCES</b>.
</center>
</p>
<p><hd>
Mis en ordre &amp; publié par M. <i>DIDEROT</i>,
de l'Académie Royale.
</hd>
</p>
<p>
<center>TOME PREMIER.</center>
<gap type="image"/>
<p>
A PARIS,<br/>
Chez BRIASSON.
</div1>
<!-- a comment that is not text -->
<div1 n="2" id="0:3" vol="1">
<index type="head" value="ABACO"/>
<head>ABACO</head>
<p>ABACO, s. m. (<i>Arithm.</i>) mot italien. <ref target="x">Voyez</ref> <sc>Abaque</sc>.</p>
<p>Second paragraph,
wrapped over two lines.</p>
</div1>
</body></text></TEI.2>
"""


def test_encyclopedie_articles_outside_body_and_malformed(tmp_path):
    f = tmp_path / "ENC_volume01.xml"
    f.write_text(ENC_SAMPLE, encoding="utf-8")
    text, meta = EX.extract_articles_file(f)
    assert meta == {"articles": 2, "div1_in_raw": 2, "articles_failed": 0}
    assert "Front matter" not in text                       # the <body> is not an article
    title, abaco = text.split("\n\nABACO\n\n")
    assert title.startswith("Title Page\n\nENCYCLOPÉDIE,\nDICTIONNAIRE RAISONNÉ\nDES SCIENCES.")
    assert "Mis en ordre & publié par M. DIDEROT, de l'Académie Royale." in title
    assert title.endswith("TOME PREMIER.\n\nA PARIS,\nChez BRIASSON.")   # unclosed <p> kept
    assert abaco == ("ABACO, s. m. (Arithm.) mot italien. Voyez Abaque.\n\n"
                     "Second paragraph, wrapped over two lines.")
    for noise in ("unsigned", "NA5", "comment", "Title Page\nTitle Page"):
        assert noise not in text


def test_encyclopedie_uses_article_mode(tmp_path, monkeypatch):
    src = tmp_path / "corpora" / "encyclopedie"
    src.mkdir(parents=True)
    (src / "ENC_volume01.xml").write_text(ENC_SAMPLE, encoding="utf-8")
    monkeypatch.setattr(EX, "BASE_DIR", tmp_path / "corpora")
    monkeypatch.setattr(EX, "OUT_DIR", tmp_path / "out")
    stats = EX.process_subcorpus("encyclopedie", EX.SUBCORPORA["encyclopedie"])
    assert stats["files_extracted"] == 1 and stats["articles"] == 2 and stats["div1_in_raw"] == 2
    out = (tmp_path / "out" / "encyclopedie" / "ENC_volume01.txt").read_text(encoding="utf-8")
    assert out.startswith("Title Page") and out.endswith("wrapped over two lines.")

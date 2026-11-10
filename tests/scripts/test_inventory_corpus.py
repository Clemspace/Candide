"""Corpus inventory: TEI header parsing, body text, language guess, summary."""

import importlib.util
import sys
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[2] / "scripts" / "inventory_corpus.py"
_spec = importlib.util.spec_from_file_location("inventory_corpus", _path)
INV = importlib.util.module_from_spec(_spec)
sys.modules["inventory_corpus"] = INV
_spec.loader.exec_module(INV)

TEI = """<?xml version="1.0" encoding="utf-8"?>
<TEI.2>
<teiHeader>
  <modernFileDesc>
   <author>
    <arkBnf>
     https://data.bnf.fr/ark:/12148/cb119124328
    </arkBnf>
    <bnfName>
     Nicolas Lenglet Du Fresnoy (1674-1755)
    </bnfName>
   </author>
   <title>
    <titleOriginal>
     Recueil de romans historiques. ...
    </titleOriginal>
   </title>
   <date>
    1746
   </date>
   <genre>
    Romans
   </genre>
  </modernFileDesc>
  <fileDesc><sourceDesc><bibl><imprint><date>1746-47</date></imprint></bibl></sourceDesc></fileDesc>
</teiHeader>
<text><body><p>{body}</p></body></text>
</TEI.2>
"""

FR = ("Le roi de France est dans la ville avec les seigneurs de la cour, et il ne veut pas "
      "que le peuple se plaigne de ses ministres qui sont dans une grande peine. ") * 20
EN = ("The king of England was in the city with all the lords of his court, and he would not "
      "have it that the people complained of their ministers. ") * 20
LA = ("Rex est in urbe cum principibus, et non vult ut populus de ministris queratur, quod "
      "enim sunt in magna poena, sed tamen nec timent. ") * 20


def test_header_fields():
    h = INV.parse_header(TEI.format(body=FR))
    assert h["year"] == 1746 and h["date"] == "1746"
    assert h["genre"] == "Romans"
    assert h["author"] == "Nicolas Lenglet Du Fresnoy (1674-1755)"
    assert h["title"] == "Recueil de romans historiques. ..."
    assert h["ark"].endswith("cb119124328")


def test_header_falls_back_to_source_date():
    xml = TEI.replace("<date>\n    1746\n   </date>", "")
    assert INV.parse_header(xml)["year"] == 1746


def test_body_text_strips_tags_and_entities():
    t = INV.body_text(TEI.format(body="Bonjour&nbsp;<hi>madame</hi> &amp; monsieur"))
    assert t == "Bonjour madame & monsieur"          # &nbsp; becomes a plain space
    assert "Lenglet" not in t                              # header excluded


@pytest.mark.parametrize("text,lang", [(FR, "fr"), (EN, "en"), (LA, "la")])
def test_language_guess(text, lang):
    got, score = INV.guess_language(text)
    assert got == lang and score > 0.6


def test_language_guess_empty():
    assert INV.guess_language("1746 12 ...") == ("", 0.0)


def test_source_prefix():
    assert INV.source_of("modern_ecco_0613900107.tei") == "ecco"
    assert INV.source_of("modern_frantext18_tlf.389.tei") == "frantext18"
    assert INV.source_of("other.tei") == ""


def test_end_to_end(tmp_path, monkeypatch, capsys):
    root = tmp_path / "modern"
    (root / "modern_merged").mkdir(parents=True)
    (root / "modern_merged" / "modern_ecco_1.tei").write_text(TEI.format(body=EN), encoding="utf-8")
    (root / "modern_merged" / "modern_ecco_2.tei").write_text(TEI.format(body=FR), encoding="utf-8")
    (root / "modern_merged" / "modern_x_3.tei").write_text(
        TEI.format(body=FR).replace("1746", "1650"), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["inv", "--root", str(root), "--corpora", "modern_merged",
                                      "--out", str(tmp_path / "out"), "--workers", "1"])
    INV.main()
    import json
    s = json.loads((tmp_path / "out" / "inventory_summary.json").read_text())
    c = s["corpora"]["modern_merged"]
    assert c["files"] == 3
    assert 0.3 < c["lang_share"]["en"] < 0.4
    assert 0.6 < c["in_period_share"] < 0.7               # the 1650 file is out of period

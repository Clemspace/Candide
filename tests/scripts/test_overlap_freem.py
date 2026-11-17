"""Overlap of FreEM_LPM test passages with the corpus: normalization and matching."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("overlap_freem", SCRIPTS / "overlap_freem.py")
ov = importlib.util.module_from_spec(spec)
sys.modules["overlap_freem"] = ov
spec.loader.exec_module(ov)

PLAY = ("Ah ! Monsieur, c’est un fils naturel ; je l'ai appris d'une voix certaine, "
        "et je ne sçaurois vous le cacher plus long-temps, quoi qu'il m'en coûte.")


def _tsv(text):
    toks = text.replace(",", " ,").replace(";", " ;").replace("!", " !").replace(".", " .").split()
    return "token\tlemma\tPOS\n" + "".join(f"{t}\t_\tX\n" for t in toks)


def test_squeeze_folds_spacing_case_punctuation_accents_and_period_variants():
    a = ov.squeeze("C\u2019e\u017ft  un Fils ,naturel. Il a ve\u0301cu &amp; joui")
    b = ov.squeeze("c'est un fils naturel ; il a uecu & ioui")
    assert a == b == "cestvnfilsnatvrelilavecviovi".encode()


def test_control_passages_are_found(tmp_path):
    text = ("Lorem " * 50) + PLAY * 8 + (" ipsum" * 50)
    src = tmp_path / "src.txt"
    src.write_text(text, encoding="utf-8")
    ctrl = ov.control_passages([str(src)], 1, tmp_path / "ctrl", chars=300)
    assert len(ctrl) == 1
    targets, per_file = ov.test_windows(ctrl, k=40)
    path, hits, n = ov.file_hits(str(src), k=40, targets=targets)
    s = ov.summarize(per_file, {path: set(hits.tolist())}, tmp_path / "ctrl")
    assert s["files"]["control_00.tsv"]["share"] == 1.0 and n > 0


def test_unreadable_file_is_reported_not_counted_as_no_overlap(tmp_path):
    path, hits, n = ov.file_hits(str(tmp_path / "gone.txt"), k=40, targets=np.zeros(0, np.uint64))
    assert hits is None and n == 0


def test_tokenized_passage_matches_running_text(tmp_path):
    t = tmp_path / "test" / "18_orig"
    t.mkdir(parents=True)
    (t / "play.tsv").write_text(_tsv(PLAY), encoding="utf-8")
    (t / "other.tsv").write_text(_tsv("Rien de commun avec le corpus, absolument rien du tout ici."),
                                 encoding="utf-8")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Préface.\n\n" + PLAY + "\n\nSuite de la pièce.", encoding="utf-8")

    targets, per_file = ov.test_windows(sorted(t.glob("*.tsv")), k=40)
    path, hits, _ = ov.file_hits(str(corpus), k=40, targets=targets)
    s = ov.summarize(per_file, {path: set(hits.tolist())}, tmp_path / "test")
    assert s["files"]["18_orig/play.tsv"]["share"] == 1.0
    assert s["files"]["18_orig/play.tsv"]["top_sources"][0][0] == str(corpus)
    assert s["files"]["18_orig/other.tsv"]["share"] == 0.0
    play, other = s["files"]["18_orig/play.tsv"], s["files"]["18_orig/other.tsv"]
    assert s["groups"]["18_orig"]["share"] == play["windows"] / (play["windows"] + other["windows"])


def test_member_against_sorted_targets():
    t = np.array([3, 7, 11], dtype=np.uint64)
    h = np.array([1, 3, 8, 11, 20], dtype=np.uint64)
    assert ov.member(h, t).tolist() == [False, True, False, True, False]
    assert ov.member(h, np.zeros(0, np.uint64)).tolist() == [False] * 5


def test_corpus_files_reads_kept_train_rows(tmp_path):
    m = tmp_path / "m.csv"
    m.write_text("priority,subcorpus,path,n_chars,n_windows,intra_dup,containment,duplicate_of,shared,decision,split\n"
                 "0,c,/a.txt,1,1,0,0,,0,keep,train\n1,c,/b.txt,1,1,0,0,,0,keep,val\n"
                 "2,c,/c.txt,1,1,0,1,/a.txt,0,near_dup,\n", encoding="utf-8")
    (tmp_path / "enc").mkdir()
    (tmp_path / "enc" / "v1.txt").write_text("x", encoding="utf-8")
    assert ov.corpus_files(m, [tmp_path / "enc"]) == ["/a.txt", str(tmp_path / "enc" / "v1.txt")]

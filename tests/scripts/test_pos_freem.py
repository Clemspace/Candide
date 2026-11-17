"""FreEM_LPM tagging harness: data reading, alignment, pooling, summaries; watcher helpers."""

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pf = _load("pos_freem")
pw = _load("pos_watch")

TSV = "token\tlemma\tPOS\tmorph\nLe\tle\tDETdef\t_\nRoi\troi\tNOMcom\t_\nvient\tvenir\tVERcjg\t_\n.\t.\tPONfrt\t_\n" \
      "Il\til\tPROper\t_\nrit\trire\tVERcjg\t_\n\nFin\tfin\tNOMcom\t_\n"


def test_read_tsv_splits_on_punctuation_and_blank_lines(tmp_path):
    f = tmp_path / "a.tsv"
    f.write_text(TSV, encoding="utf-8")
    sents, info = pf.read_tsv(f)
    assert [s.words for s in sents] == [["Le", "Roi", "vient", "."], ["Il", "rit"], ["Fin"]]
    assert sents[0].tags == ["DETdef", "NOMcom", "VERcjg", "PONfrt"]
    assert info == {"words": 7, "skipped_lines": 0}


def test_read_tsv_caps_sentence_length_and_skips_bad_lines(tmp_path):
    f = tmp_path / "a.tsv"
    f.write_text("form\tPOS\n" + "".join(f"m{i}\tNOMcom\n" for i in range(5)) + "broken\n\tNOMcom\n",
                 encoding="utf-8")
    sents, info = pf.read_tsv(f, max_words=2)
    assert [len(s.words) for s in sents] == [2, 2, 1]
    assert info["skipped_lines"] == 2


def test_read_tsv_requires_a_pos_column(tmp_path):
    f = tmp_path / "a.tsv"
    f.write_text("token\tlemma\nle\tle\n", encoding="utf-8")
    with pytest.raises(ValueError, match="POS"):
        pf.read_tsv(f)


def test_tagset_marks_unseen_tags():
    ts = pf.TagSet([pf.Sentence(["a", "b"], ["X", "Y"])])
    assert len(ts) == 2 and ts.ids(["Y", "Z", "X"]) == [1, -1, 0]


def test_align_words_with_specials():
    ids, widx = pf.align_words([[10, 11], [12], [13, 14, 15]], bos=2, eos=3)
    assert ids == [2, 10, 11, 12, 13, 14, 15, 3]
    assert widx == [-1, 0, 0, 1, 2, 2, 2, -1]
    with pytest.raises(ValueError):
        pf.align_words([[1], []], None, None)


def test_group_by_word_fills_empty_words_with_unk():
    assert pf.group_by_word([5, 6, 7], [0, 0, 2], 3, unk=0) == [[5, 6], [0], [7]]


def test_split_to_fit_halves_until_short_enough():
    words, tags = list("abcdefgh"), list("TTTTTTTT")
    parts = pf.split_to_fit(words, tags, lambda ws: 2 + 2 * len(ws), max_len=6)
    assert [w for w, _ in parts] == [["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]]
    assert sum(len(t) for _, t in parts) == 8


def test_test_groups_and_century_summary(tmp_path):
    for d in ("Data_outOfDomain_original/theatre_original/18_orig",
              "Data_outOfDomain_original/varia_original/18_orig",
              "Data_outOfDomain_normalised/theatre_normalised/19"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "x.tsv").write_text(TSV, encoding="utf-8")
    groups = pf.test_groups(tmp_path)
    assert len(groups) == 3
    acc = {"Data_outOfDomain_original/theatre_original/18_orig": (9, 10),
           "Data_outOfDomain_original/varia_original/18_orig": (27, 30),
           "Data_outOfDomain_normalised/theatre_normalised/19": (4, 5)}
    s = pf.century_summary(acc)
    assert s["original/18"] == {"acc": 0.9, "words": 40}
    assert s["normalised/19"]["acc"] == pytest.approx(0.8)


def test_result_names():
    assert pw.result_name("candide:/runs/x/checkpoints/step_5000.pt") == "step_5000"
    assert pw.result_name("hf:pjox/dalembert") == "baseline_pjox__dalembert"


def test_candidates_in_training_order_and_stability(tmp_path):
    names = ["step_1000.pt", "step_2000.pt", "step_3000.pt", "pre_cooldown.pt", "final.pt"]
    now = time.time()
    for i, n in enumerate(names):
        (tmp_path / n).write_bytes(b"")
        os.utime(tmp_path / n, (now - 1000 + i, now - 1000 + i))
    got = [p.name for p in pw.candidates(tmp_path, [1000, 2000, 5000])]
    assert got == ["step_1000.pt", "step_2000.pt", "pre_cooldown.pt", "final.pt"]
    assert pw.is_stable(tmp_path / "final.pt", 60)
    (tmp_path / "fresh.pt").write_bytes(b"")
    assert not pw.is_stable(tmp_path / "fresh.pt", 60)
    assert not pw.is_stable(tmp_path / "missing.pt", 0)


def test_headline_metrics():
    r = {"test_centuries": {"original/17": {"acc": 0.9}, "original/18": {"acc": 0.8},
                            "normalised/19": {"acc": 0.7}},
         "dev": [{"dev_acc": 0.5}, {"dev_acc": 0.95}]}
    m = pw.headline(r)
    assert m["pos/original_mean"] == pytest.approx(0.85)
    assert m["pos/dev_final"] == 0.95 and m["pos/normalised/19"] == 0.7


def test_passage_key_matches_both_spellings():
    a = pf.passage_key("Data_outOfDomain_original/theatre_original/18_orig/DIDEROT_FILSNATUREL_1757_orig.tsv")
    b = pf.passage_key("Data_outOfDomain_normalised/theatre_normalised/18/DIDEROT_FILSNATUREL_1757.tsv")
    c = pf.passage_key("Data_outOfDomain_original/varia_original/18_orig/DIDEROT_FILSNATUREL_1757_orig.tsv")
    assert a == b == "theatre/18/DIDEROT_FILSNATUREL_1757" and c != a


def test_clean_groups_drop_read_passages_in_both_spellings():
    overlap = {"files": {
        "Data_outOfDomain_original/theatre_original/18_orig/A_orig.tsv": {"share": 0.9},
        "Data_outOfDomain_original/theatre_original/18_orig/B_orig.tsv": {"share": 0.01}}}
    excluded = pf.contaminated(overlap, 0.05)
    per_file = {"Data_outOfDomain_original/theatre_original/18_orig/A_orig.tsv": (50, 100),
                "Data_outOfDomain_original/theatre_original/18_orig/B_orig.tsv": (90, 100),
                "Data_outOfDomain_normalised/theatre_normalised/18/A.tsv": (60, 100),
                "Data_outOfDomain_normalised/theatre_normalised/18/B.tsv": (95, 100)}
    g = pf.clean_groups(per_file, excluded)
    assert g == {"Data_outOfDomain_original/theatre_original/18_orig": (90, 100),
                 "Data_outOfDomain_normalised/theatre_normalised/18": (95, 100)}
    s = pf.century_summary(g)
    assert s["original/18"]["acc"] == 0.9 and s["normalised/18"]["acc"] == 0.95


def test_headline_includes_clean_scores():
    r = {"test_centuries": {"original/18": {"acc": 0.8}}, "dev": [{"dev_acc": 0.9}],
         "test_clean": {"centuries": {"original/17": {"acc": 0.7}, "original/18": {"acc": 0.9}}}}
    m = pw.headline(r)
    assert m["pos_clean/original_mean"] == pytest.approx(0.8) and m["pos_clean/original/18"] == 0.9

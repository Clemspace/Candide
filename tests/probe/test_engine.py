"""Model side of the probe, on tiny models built by the trainer (CPU, a few seconds).

What must hold for the numbers of the probe to be trusted:
  * a checkpoint is rebuilt from its own pickled config and gives the same logits;
  * per-token losses add up to the total score_text.score_ids returns for the same text;
  * token bytes add up to the text: bits per byte are attributed without loss;
  * cached generation follows the uncached model exactly (greedy) and reports its losses;
  * the last layer of the logit lens is the model output;
  * recording attention does not change the forward pass, and the weights are causal rows
    that sum to one.
"""
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")

from candide_probe import checkpoints as C                      # noqa: E402
from candide_probe.engine import Engine, _AttentionRecorder, _ScorerBackend  # noqa: E402
from candide_probe.errors import ProbeError                     # noqa: E402
from candide_probe.text import CandideTokenizer, score_text_module  # noqa: E402

TEXT = ("Candide, chaſſé du château de Thunder-ten-tronckh, marcha longtemps ſans ſavoir où. "
        "Il étoit une fois un jeune homme qui avoit les mœurs les plus douces. ")
CORPUS = [TEXT * 3, "Le roi de France avoit déclaré la guerre. Les troupes ſeroient miſes en campagne.",
          "Monſieur, j'ai l'honneur de vous informer que la lettre m'eſt parvenue. 1789, 1750."]


def _tokenizer(path):
    import importlib.util
    from tokenizers import pre_tokenizers, trainers
    spec = importlib.util.spec_from_file_location("train_tokenizer", C.Path(__file__).resolve().parents[2] / "scripts" / "train_tokenizer.py")
    tt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tt)
    tok = tt.build_tokenizer()
    trainer = trainers.BpeTrainer(vocab_size=420, special_tokens=tt.special_tokens(None),
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(CORPUS, trainer=trainer)
    tok.save(str(path))
    return path


def _checkpoint(run_dir, vocab, arch="gpt", step=1234, seed=0, max_len=64):
    T = C.trainer_module()
    cfg = T.FullConfig()
    for k, v in dict(vocab_size=vocab, d_model=64, n_layers=3, n_heads=4, n_kv_heads=2, d_ff=128,
                     max_seq_len=max_len, dropout=0.0, arch=arch, qk_norm=(arch == "gpt"),
                     tie_word_embeddings=True).items():
        setattr(cfg.model, k, v)
    for k, v in dict(max_steps=2000, warmup_steps=100, schedule="wsd", cooldown_frac=0.2,
                     learning_rate=1e-3, min_lr=1e-4, batch_size=2, gradient_accumulation_steps=2).items():
        setattr(cfg.training, k, v)
    cfg.data.sequence_length = max_len
    torch.manual_seed(seed)
    model = T.setup_model(cfg.model)
    with torch.no_grad():                       # move away from init so the test is not trivial
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    out = run_dir / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"step_{step}.pt"
    torch.save({"model": model.state_dict(), "optimizer": {}, "step": step, "config": cfg.__dict__}, path)
    return path, model.eval()


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("probe")
    tok = CandideTokenizer(_tokenizer(root / "tokenizer.json"))
    gpt_path, gpt = _checkpoint(root / "runs" / "gpt", tok.vocab_size, "gpt", step=1234)
    an_path, an = _checkpoint(root / "runs" / "angpt", tok.vocab_size, "angpt", step=1900, seed=1)
    reg = C.Registry([root / "runs"], device="cpu", max_loaded=2)
    return {"root": root, "tok": tok, "reg": reg, "engine": Engine(reg, tok),
            "models": {"gpt/step_1234": gpt, "angpt/step_1900": an}}


IDS = ["gpt/step_1234", "angpt/step_1900"]


def test_discovery_and_metadata(setup):
    reg = setup["reg"]
    assert sorted(r.id for r in reg.refs()) == sorted(IDS)
    meta = reg.meta("gpt/step_1234")
    assert meta["step"] == 1234 and meta["model"]["n_layers"] == 3 and meta["model"]["arch"] == "gpt"
    assert meta["train"]["phase"] == "stable" and meta["train"]["cooldown_start"] == 1600
    assert meta["train"]["lr_factor"] == 1.0
    an = reg.meta("angpt/step_1900")
    assert an["train"]["phase"] == "cooldown" and 0 < an["train"]["lr_factor"] < 1
    loaded = reg.get("gpt/step_1234")
    assert loaded.meta["n_params"] == loaded.meta["n_params_built"]       # tied head counted once


@pytest.mark.parametrize("cid", IDS)
def test_rebuilt_model_gives_the_same_logits(setup, cid):
    x = torch.tensor([setup["tok"].encode(TEXT)[:40]])
    with torch.no_grad():
        a = setup["models"][cid](x)
        b = setup["reg"].get(cid).model(x)
    a = a[0] if isinstance(a, tuple) else a
    b = b[0] if isinstance(b, tuple) else b
    assert torch.equal(a, b)


@pytest.mark.parametrize("cid", IDS)
@pytest.mark.parametrize("bos", [True, False])
def test_score_adds_up_to_score_text(setup, cid, bos):
    e, tok = setup["engine"], setup["tok"]
    text = TEXT * 2                                  # several 64-token windows
    res = e.score([cid], text, bos=bos, topk=3)
    run = res["runs"][cid]
    ids = tok.encode(res["text"])
    assert len(ids) > 64 and len(run["tokens"]) == len(ids)
    assert run["n_scored"] == len(ids) - (0 if bos else 1)
    S = score_text_module()
    backend = _ScorerBackend(setup["reg"].get(cid).model, tok, 64, cid)
    if not bos:
        backend.bos_id = None
    nll, scored = S.score_ids(backend, ids, 64, 32, batch_size=4, device="cpu")
    assert scored == run["n_scored"]
    assert run["nll_nats"] == pytest.approx(nll, rel=1e-5)
    assert run["bits_per_byte"] == pytest.approx(nll / math.log(2) / res["n_bytes"], rel=1e-5)
    for t in run["tokens"]:
        if t:
            assert t["rank"] >= 1 and len(t["top"]) == 3
            assert t["top"][0][1] >= t["top"][1][1] >= t["top"][2][1]
            if t["rank"] == 1:
                assert t["nll"] == pytest.approx(-math.log(t["top"][0][1]), abs=1e-4)


def test_token_bytes_add_up_to_the_text(setup):
    tok = setup["tok"]
    assert tok.byte_level
    for text in CORPUS + ["Ça, l'œuvre — « dit-il » ; 1789 ſ ß 漢字 🙂"]:
        ids = tok.encode(text)
        segs = tok.segments(ids)
        assert sum(s["n_bytes"] for s in segs) == len(text.encode("utf-8"))
        assert "".join(s["text"] for s in segs) == text
        assert b"".join(tok.token_bytes(i) for i in ids) == text.encode("utf-8")


@pytest.mark.parametrize("cid", IDS)
def test_greedy_generation_follows_the_uncached_model(setup, cid):
    e, model = setup["engine"], setup["reg"].get(cid).model
    res = e.generate(cid, "Le roi de France", greedy=True, max_new_tokens=20, stop_at_eos=False, topn=4)
    seq = list(res["prompt_ids"])
    for step in res["run"]["generated"]:
        with torch.no_grad():
            out = model(torch.tensor([seq]))
        logp = torch.log_softmax((out[0] if isinstance(out, tuple) else out)[0, -1].float(), -1)
        assert step["id"] == int(logp.argmax())
        assert step["nll"] == pytest.approx(-float(logp[step["id"]]), abs=1e-4)
        assert step["rank"] == 1
        seq.append(step["id"])
    assert len(res["run"]["prompt_tokens"]) == len(res["prompt_ids"])


def test_sampling_is_reproducible_and_top_k_1_is_greedy(setup):
    e = setup["engine"]
    a = e.generate(IDS[0], "Il étoit", seed=7, max_new_tokens=15, stop_at_eos=False)
    b = e.generate(IDS[0], "Il étoit", seed=7, max_new_tokens=15, stop_at_eos=False)
    assert [s["id"] for s in a["run"]["generated"]] == [s["id"] for s in b["run"]["generated"]]
    g = e.generate(IDS[0], "Il étoit", greedy=True, max_new_tokens=15, stop_at_eos=False)
    k1 = e.generate(IDS[0], "Il étoit", top_k=1, seed=3, max_new_tokens=15, stop_at_eos=False)
    assert [s["id"] for s in g["run"]["generated"]] == [s["id"] for s in k1["run"]["generated"]]


def test_fork_from_ids_and_context_crop(setup):
    e = setup["engine"]
    first = e.generate(IDS[0], "Il étoit", greedy=True, max_new_tokens=5, stop_at_eos=False)
    ids = first["prompt_ids"] + [first["run"]["generated"][0]["id"]]
    fork = e.generate(IDS[0], ids=ids, greedy=True, max_new_tokens=4, stop_at_eos=False)
    assert fork["prompt_ids"] == ids
    assert [s["id"] for s in fork["run"]["generated"]] == [s["id"] for s in first["run"]["generated"][1:5]]
    long = e.generate(IDS[0], TEXT * 3, greedy=True, max_new_tokens=10, stop_at_eos=False)
    assert long["cropped"] > 0 and len(long["prompt_ids"]) + 10 == 64


@pytest.mark.parametrize("cid", IDS)
def test_lens_last_layer_is_the_model_output(setup, cid):
    res = setup["engine"].lens(cid, TEXT)
    assert len(res["layers"]) == 3 + 1
    assert res["check"]["last_layer_vs_model_max_abs_prob_diff"] < 1e-5
    for layer in res["layers"]:
        assert len(layer["p_target"]) == res["n_positions"] and layer["p_target"][-1] is None


@pytest.mark.parametrize("cid", IDS)
def test_attention_recording_is_exact(setup, cid):
    model, tok = setup["reg"].get(cid).model, setup["tok"]
    x = torch.tensor([[tok.bos_id] + tok.encode(TEXT)[:40]])
    with torch.no_grad():
        ref = model(x)
        with _AttentionRecorder(model) as rec:
            got = model(x)
    ref = ref[0] if isinstance(ref, tuple) else ref
    got = got[0] if isinstance(got, tuple) else got
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5)
    assert sorted(rec.weights) == [0, 1, 2]
    for w in rec.weights.values():
        assert w.shape == (4, 41, 41)
        assert torch.allclose(w.sum(-1), torch.ones(4, 41), atol=1e-5)
        assert float(torch.triu(w, diagonal=1).abs().max()) == 0.0
    res = setup["engine"].attention(cid, TEXT, layer=1)
    assert res["layer"] == 1 and len(res["summary"]) == 3 and res["n_heads"] == 4


def test_candidates_and_pairs_use_score_text(setup):
    e, tok = setup["engine"], setup["tok"]
    S = score_text_module()
    res = e.candidates(IDS, "Il étoit", [" tard.", " une fois"])
    backend = _ScorerBackend(setup["reg"].get(IDS[0]).model, tok, 64, IDS[0])
    kw = dict(max_len=64, stride=32, batch_size=8, device="cpu")
    base = S.score_text(backend, "Il étoit", **kw).nll_nats
    full = S.score_text(backend, "Il étoit tard.", **kw).nll_nats
    row = res["runs"][IDS[0]]["rows"][0]
    assert row["nll_cond"] == pytest.approx(full - base, rel=1e-6)
    assert sorted(r["rank"] for r in res["runs"][IDS[0]]["rows"]) == [1, 2]
    pr = e.pairs(IDS, [{"a": "Il étoit tard.", "b": "Il était tard."}])
    r = pr["runs"][IDS[0]]["rows"][0]
    assert r["prefers_a_total"] == (r["nll_a"] < r["nll_b"])


def test_vocabulary_mismatch_is_refused(setup, tmp_path):
    _checkpoint(tmp_path / "runs" / "other", setup["tok"].vocab_size + 7, "gpt", step=10)
    reg = C.Registry([tmp_path / "runs"], device="cpu")
    e = Engine(reg, setup["tok"])
    assert e.meta("other/step_10")["compatible"] is False
    with pytest.raises(ProbeError):
        e.score(["other/step_10"], "Il étoit")


def test_registry_keeps_at_most_max_loaded(setup):
    reg = C.Registry([setup["root"] / "runs"], device="cpu", max_loaded=1)
    reg.get(IDS[0])
    reg.get(IDS[1])
    assert reg.loaded_ids() == [IDS[1]]

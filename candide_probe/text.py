"""Text side of the probe: preparation as in scoring, the tokenizer, and token-to-byte maps.

Text is prepared exactly as scripts/score_text.py prepares it (HTML entities cleaned as
in prepare_shards, then NFC), so a passage scored here and by the scorer gives the same
tokens. Each token is mapped to the bytes it encodes, so per-token losses add up to bits
per byte and the display never splits a character: a token that ends inside a multi-byte
character shows nothing, and the token that completes it shows the whole character.

No PyTorch here.
"""
from __future__ import annotations

import codecs
import importlib.util
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

REPO = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def score_text_module():
    """scripts/score_text.py as a module (same idiom as the other scripts)."""
    spec = importlib.util.spec_from_file_location("score_text", REPO / "scripts" / "score_text.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prepare(text: str, clean: bool = True) -> str:
    return score_text_module().prepare(text, clean=clean)


@lru_cache(maxsize=1)
def _byte_decoder() -> dict:
    """Inverse of the GPT-2 bytes_to_unicode table used by byte-level BPE."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class CandideTokenizer:
    """A tokenizer.json with what the probe needs: ids, specials, and bytes per token."""

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer
        self.path = str(path)
        self.tok = Tokenizer.from_file(self.path)
        self.vocab_size = self.tok.get_vocab_size()
        self.bos_id = self.tok.token_to_id("<|bos|>")
        self.eos_id = self.tok.token_to_id("<|eos|>")
        self.special_ids = set(self.tok.get_added_tokens_decoder())
        spec = json.loads(self.tok.to_str())
        self.byte_level = _is_byte_level(spec.get("decoder")) or _is_byte_level(spec.get("pre_tokenizer"))
        self.model_type = (spec.get("model") or {}).get("type")

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=True)

    def piece(self, token_id: int) -> str:
        return self.tok.id_to_token(int(token_id)) or f"<{token_id}>"

    def token_bytes(self, token_id: int) -> bytes:
        """The bytes a token stands for in the text (b'' for special tokens)."""
        token_id = int(token_id)
        if token_id in self.special_ids:
            return b""
        piece = self.tok.id_to_token(token_id)
        if piece is None:
            return b""
        if self.byte_level:
            table = _byte_decoder()
            if all(c in table for c in piece):
                return bytes(table[c] for c in piece)
        return self.tok.decode([token_id], skip_special_tokens=False).encode("utf-8")

    def label(self, token_id: int) -> str:
        """A short human-readable form of one token, for lists of alternatives."""
        token_id = int(token_id)
        if token_id in self.special_ids:
            return self.piece(token_id)
        return self.token_bytes(token_id).decode("utf-8", errors="replace")

    def segments(self, ids: Sequence[int]) -> list[dict]:
        """Display segments, one per token: {'id', 'text', 'n_bytes', 'special', 'partial'}."""
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        out = []
        for i in ids:
            i = int(i)
            if i in self.special_ids:
                out.append({"id": i, "text": self.piece(i), "n_bytes": 0, "special": True, "partial": False})
                continue
            b = self.token_bytes(i)
            t = dec.decode(b)
            out.append({"id": i, "text": t, "n_bytes": len(b), "special": False, "partial": bool(b) and t == ""})
        tail = dec.decode(b"", final=True)
        if tail and out:
            out[-1]["text"] += tail
        return out


def _is_byte_level(component: Optional[dict]) -> bool:
    if not component:
        return False
    if component.get("type") == "ByteLevel":
        return True
    for key in ("decoders", "pretokenizers"):
        if any(_is_byte_level(c) for c in component.get(key) or []):
            return True
    return False


def check_compatible(tokenizer: CandideTokenizer, meta: dict) -> list[str]:
    """Problems that make this tokenizer wrong for a checkpoint (empty list = none found).

    The checkpoint records its vocabulary size (from the embedding) but not the tokenizer
    that made its shards: a tokenizer of another size is refused (the one in
    /data/pretrain_candide/tokenizer/ has 32,133 entries, v2g has 32,132), one of the same
    size cannot be told apart from the checkpoint alone.
    """
    problems = []
    vocab = meta.get("vocab_size")
    if vocab is not None and int(vocab) != tokenizer.vocab_size:
        problems.append(f"tokenizer has {tokenizer.vocab_size} entries, checkpoint expects {vocab}")
    return problems


__all__ = ["prepare", "CandideTokenizer", "check_compatible", "score_text_module", "REPO"]

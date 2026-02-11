"""
French Phoneme Inventory
========================

IPA-based phoneme inventory for French TTS.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class PhonemeCategory(Enum):
    VOWEL_ORAL = "vowel_oral"
    VOWEL_NASAL = "vowel_nasal"
    SEMIVOWEL = "semivowel"
    STOP = "stop"
    FRICATIVE = "fricative"
    NASAL = "nasal"
    LIQUID = "liquid"
    SPECIAL = "special"


FRENCH_PHONEMES: Dict[PhonemeCategory, List[str]] = {
    PhonemeCategory.VOWEL_ORAL: ['i', 'e', 'ɛ', 'a', 'ɑ', 'ɔ', 'o', 'u', 'y', 'ø', 'œ', 'ə'],
    PhonemeCategory.VOWEL_NASAL: ['ɛ̃', 'ɑ̃', 'ɔ̃', 'œ̃'],
    PhonemeCategory.SEMIVOWEL: ['j', 'w', 'ɥ'],
    PhonemeCategory.STOP: ['p', 'b', 't', 'd', 'k', 'g'],
    PhonemeCategory.FRICATIVE: ['f', 'v', 's', 'z', 'ʃ', 'ʒ'],
    PhonemeCategory.NASAL: ['m', 'n', 'ɲ', 'ŋ'],
    PhonemeCategory.LIQUID: ['l', 'ʁ'],
    PhonemeCategory.SPECIAL: ['SIL', 'PAU', 'BRE', 'EUH', 'HUM'],
}

# Build flat list
PHONEME_LIST: List[str] = []
for cat in PhonemeCategory:
    PHONEME_LIST.extend(FRENCH_PHONEMES.get(cat, []))

PHONEME_TO_ID: Dict[str, int] = {p: i for i, p in enumerate(PHONEME_LIST)}
ID_TO_PHONEME: Dict[int, str] = {i: p for i, p in enumerate(PHONEME_LIST)}
N_PHONEMES: int = len(PHONEME_LIST)

# Phoneme sets
VOWELS = set(FRENCH_PHONEMES[PhonemeCategory.VOWEL_ORAL] + FRENCH_PHONEMES[PhonemeCategory.VOWEL_NASAL])
VOICED = {'b', 'd', 'g', 'v', 'z', 'ʒ', 'm', 'n', 'ɲ', 'ŋ', 'l', 'ʁ'} | VOWELS | set(FRENCH_PHONEMES[PhonemeCategory.SEMIVOWEL])
SPECIAL = set(FRENCH_PHONEMES[PhonemeCategory.SPECIAL])


def is_voiced(phoneme: str) -> bool:
    """Check if phoneme is voiced (has F0)."""
    return phoneme in VOICED


def get_category(phoneme: str) -> PhonemeCategory:
    """Get phoneme category."""
    for cat, phonemes in FRENCH_PHONEMES.items():
        if phoneme in phonemes:
            return cat
    return PhonemeCategory.SPECIAL


class PhonemeInventory:
    """Phoneme inventory for a language."""
    
    def __init__(self, phonemes: Dict[PhonemeCategory, List[str]], language: str = "fr"):
        self.language = language
        self.phonemes_by_category = phonemes
        self.phoneme_list = []
        for cat in PhonemeCategory:
            self.phoneme_list.extend(phonemes.get(cat, []))
        self.phoneme_to_id = {p: i for i, p in enumerate(self.phoneme_list)}
        self.id_to_phoneme = {i: p for i, p in enumerate(self.phoneme_list)}
        self.n_phonemes = len(self.phoneme_list)
    
    def to_ids(self, phonemes: List[str]) -> List[int]:
        unk = self.phoneme_to_id.get('SIL', 0)
        return [self.phoneme_to_id.get(p, unk) for p in phonemes]
    
    def to_phonemes(self, ids: List[int]) -> List[str]:
        return [self.id_to_phoneme.get(i, 'SIL') for i in ids]
    
    def __len__(self) -> int:
        return self.n_phonemes


FRENCH_INVENTORY = PhonemeInventory(FRENCH_PHONEMES, "fr")
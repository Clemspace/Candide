"""
Text Processing
===============

Text normalization and Grapheme-to-Phoneme (G2P) conversion for French.

Pipeline:
1. Normalize text (numbers, abbreviations, punctuation)
2. Convert to phonemes (IPA)
3. Add prosodic markers (pauses, emphasis)
"""

import re
import unicodedata
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from .phonemes import PHONEME_TO_ID, FRENCH_INVENTORY


@dataclass
class TextToken:
    """A token in processed text with metadata."""
    text: str
    phonemes: List[str]
    is_punctuation: bool = False
    pause_after: Optional[str] = None  # 'short', 'medium', 'long'


class FrenchNormalizer:
    """
    Text normalization for French TTS.
    
    Handles:
    - Unicode normalization
    - Number expansion (1, 2, 3 → un, deux, trois)
    - Abbreviation expansion (M. → Monsieur)
    - Punctuation normalization
    - Whitespace normalization
    """
    
    def __init__(self):
        # Number words
        self.units = [
            '', 'un', 'deux', 'trois', 'quatre', 'cinq',
            'six', 'sept', 'huit', 'neuf'
        ]
        self.teens = [
            'dix', 'onze', 'douze', 'treize', 'quatorze',
            'quinze', 'seize', 'dix-sept', 'dix-huit', 'dix-neuf'
        ]
        self.tens = [
            '', 'dix', 'vingt', 'trente', 'quarante',
            'cinquante', 'soixante', 'soixante-dix',
            'quatre-vingt', 'quatre-vingt-dix'
        ]
        
        # Common abbreviations
        self.abbreviations = {
            'M.': 'monsieur',
            'Mme': 'madame',
            'Mlle': 'mademoiselle',
            'Dr': 'docteur',
            'Pr': 'professeur',
            'Prof.': 'professeur',
            'etc.': 'et cetera',
            'ex.': 'par exemple',
            'cf.': 'confer',
            'c.-à-d.': "c'est-à-dire",
            'n°': 'numéro',
            'No': 'numéro',
            'St': 'saint',
            'Ste': 'sainte',
            'av.': 'avenue',
            'bd': 'boulevard',
            'pl.': 'place',
        }
    
    def normalize(self, text: str) -> str:
        """
        Normalize text for TTS processing.
        
        Args:
            text: Raw input text
            
        Returns:
            Normalized text
        """
        # Unicode normalization (NFC for composed characters)
        text = unicodedata.normalize('NFC', text)
        
        # Lowercase for consistency (French TTS typically lowercase)
        # Keep original for proper noun detection if needed
        text_lower = text.lower()
        
        # Expand abbreviations (case-insensitive matching)
        for abbr, expansion in self.abbreviations.items():
            pattern = re.compile(re.escape(abbr), re.IGNORECASE)
            text_lower = pattern.sub(expansion, text_lower)
        
        # Expand numbers
        text_lower = self._expand_numbers(text_lower)
        
        # Normalize punctuation
        text_lower = self._normalize_punctuation(text_lower)
        
        # Normalize whitespace
        text_lower = ' '.join(text_lower.split())
        
        return text_lower
    
    def _expand_numbers(self, text: str) -> str:
        """Expand numbers to words."""
        
        def replace_number(match):
            num_str = match.group(0)
            try:
                num = int(num_str)
                return self._number_to_words(num)
            except ValueError:
                return num_str
        
        # Match integers
        text = re.sub(r'\b\d+\b', replace_number, text)
        
        return text
    
    def _number_to_words(self, n: int) -> str:
        """Convert integer to French words."""
        if n < 0:
            return 'moins ' + self._number_to_words(-n)
        
        if n == 0:
            return 'zéro'
        
        if n < 10:
            return self.units[n]
        
        if n < 20:
            return self.teens[n - 10]
        
        if n < 100:
            tens_digit, units_digit = divmod(n, 10)
            
            # Special cases for French
            if tens_digit == 7:  # 70s
                if units_digit == 0:
                    return 'soixante-dix'
                elif units_digit == 1:
                    return 'soixante et onze'
                else:
                    return f'soixante-{self.teens[units_digit]}'
            
            if tens_digit == 8:  # 80s
                if units_digit == 0:
                    return 'quatre-vingts'
                else:
                    return f'quatre-vingt-{self.units[units_digit]}'
            
            if tens_digit == 9:  # 90s
                return f'quatre-vingt-{self.teens[units_digit]}'
            
            # Regular cases
            if units_digit == 0:
                return self.tens[tens_digit]
            elif units_digit == 1 and tens_digit in [2, 3, 4, 5, 6]:
                return f'{self.tens[tens_digit]} et un'
            else:
                return f'{self.tens[tens_digit]}-{self.units[units_digit]}'
        
        if n < 1000:
            hundreds, remainder = divmod(n, 100)
            if hundreds == 1:
                prefix = 'cent'
            else:
                prefix = f'{self.units[hundreds]} cent'
            
            if remainder == 0:
                return prefix + ('s' if hundreds > 1 else '')
            else:
                return f'{prefix} {self._number_to_words(remainder)}'
        
        if n < 10000:
            thousands, remainder = divmod(n, 1000)
            if thousands == 1:
                prefix = 'mille'
            else:
                prefix = f'{self._number_to_words(thousands)} mille'
            
            if remainder == 0:
                return prefix
            else:
                return f'{prefix} {self._number_to_words(remainder)}'
        
        # For larger numbers, just return digits (placeholder)
        return str(n)
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks."""
        # French quotes to nothing (will be in prosody)
        text = text.replace('«', '').replace('»', '')
        text = text.replace('"', '').replace('"', '').replace('"', '')
        
        # Dashes to pauses
        text = text.replace('—', ', ')
        text = text.replace('–', ', ')
        text = text.replace('...', '.')
        text = text.replace('…', '.')
        
        # Normalize multiple punctuation
        text = re.sub(r'[.!?]+', '.', text)
        text = re.sub(r'[,;:]+', ',', text)
        
        return text


class FrenchG2P:
    """
    Grapheme-to-Phoneme conversion for French.
    
    Two backends:
    1. phonemizer (espeak) - accurate, requires espeak installed
    2. Rule-based fallback - approximate, no dependencies
    """
    
    def __init__(self, backend: str = 'auto'):
        """
        Args:
            backend: 'phonemizer', 'rules', or 'auto' (try phonemizer first)
        """
        self.backend = backend
        self._phonemizer_available = False
        
        if backend in ['auto', 'phonemizer']:
            try:
                from phonemizer import phonemize
                from phonemizer.backend import EspeakBackend
                self._phonemize_fn = phonemize
                self._phonemizer_available = True
                if backend == 'auto':
                    self.backend = 'phonemizer'
            except ImportError:
                if backend == 'phonemizer':
                    raise ImportError(
                        "phonemizer not installed. "
                        "Install with: pip install phonemizer"
                    )
                self.backend = 'rules'
        
        # Rule-based conversion table
        self._init_rules()
    
    def _init_rules(self):
        """Initialize rule-based G2P tables."""
        # Multi-character patterns (order matters - longer first)
        self.patterns = [
            # Nasal vowels
            ('ain', 'ɛ̃'), ('aim', 'ɛ̃'), ('ein', 'ɛ̃'), ('eim', 'ɛ̃'),
            ('in', 'ɛ̃'), ('im', 'ɛ̃'), ('yn', 'ɛ̃'), ('ym', 'ɛ̃'),
            ('an', 'ɑ̃'), ('am', 'ɑ̃'), ('en', 'ɑ̃'), ('em', 'ɑ̃'),
            ('on', 'ɔ̃'), ('om', 'ɔ̃'),
            ('un', 'œ̃'), ('um', 'œ̃'),
            
            # Vowel combinations
            ('eau', 'o'), ('au', 'o'), ('aux', 'o'),
            ('ou', 'u'), ('oû', 'u'),
            ('ai', 'ɛ'), ('ei', 'ɛ'), ('ê', 'ɛ'), ('è', 'ɛ'),
            ('eu', 'ø'), ('œu', 'ø'),
            ('oi', 'wa'), ('oy', 'waj'),
            ('ui', 'ɥi'),
            
            # Consonant combinations
            ('ch', 'ʃ'), ('ph', 'f'), ('gn', 'ɲ'),
            ('qu', 'k'), ('gu', 'g'),
            ('ll', 'l'), ('ss', 's'), ('tt', 't'),
            ('mm', 'm'), ('nn', 'n'), ('pp', 'p'),
            ('cc', 'k'), ('ff', 'f'),
            
            # Accented vowels
            ('é', 'e'), ('ë', 'ɛ'), ('ï', 'i'), ('ü', 'y'),
            ('à', 'a'), ('â', 'ɑ'), ('ô', 'o'), ('î', 'i'), ('û', 'y'),
            ('ç', 's'),
        ]
        
        # Single character fallbacks
        self.char_map = {
            'a': 'a', 'b': 'b', 'c': 'k', 'd': 'd', 'e': 'ə',
            'f': 'f', 'g': 'g', 'h': '', 'i': 'i', 'j': 'ʒ',
            'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'o',
            'p': 'p', 'q': 'k', 'r': 'ʁ', 's': 's', 't': 't',
            'u': 'y', 'v': 'v', 'w': 'w', 'x': 'ks', 'y': 'i',
            'z': 'z'
        }
    
    def convert(self, text: str) -> List[str]:
        """
        Convert text to phoneme sequence.
        
        Args:
            text: Normalized French text
            
        Returns:
            List of IPA phoneme strings
        """
        if self.backend == 'phonemizer' and self._phonemizer_available:
            return self._convert_phonemizer(text)
        else:
            return self._convert_rules(text)
    
    def _convert_phonemizer(self, text: str) -> List[str]:
        """Use phonemizer backend."""
        phonemes_str = self._phonemize_fn(
            text,
            language='fr-fr',
            backend='espeak',
            strip=True,
            preserve_punctuation=True,
            punctuation_marks='.!?;:,'
        )
        
        # Split into individual phonemes
        phonemes = []
        for token in phonemes_str.split():
            if token in '.!?':
                phonemes.append('PAU')
            elif token in ',;:':
                phonemes.append('SIL')
            else:
                # espeak returns space-separated phonemes
                phonemes.append(token)
        
        return phonemes
    
    def _convert_rules(self, text: str) -> List[str]:
        """Rule-based conversion fallback."""
        text = text.lower()
        phonemes = []
        
        i = 0
        while i < len(text):
            # Skip whitespace
            if text[i].isspace():
                i += 1
                continue
            
            # Check punctuation
            if text[i] in '.!?':
                phonemes.append('PAU')
                i += 1
                continue
            elif text[i] in ',;:':
                phonemes.append('SIL')
                i += 1
                continue
            elif not text[i].isalpha():
                i += 1
                continue
            
            # Try multi-character patterns
            matched = False
            for pattern, phoneme in self.patterns:
                if text[i:i+len(pattern)] == pattern:
                    if phoneme:  # Some patterns map to empty (silent)
                        phonemes.append(phoneme)
                    i += len(pattern)
                    matched = True
                    break
            
            # Single character fallback
            if not matched:
                char = text[i]
                if char in self.char_map:
                    phoneme = self.char_map[char]
                    if phoneme:
                        phonemes.append(phoneme)
                i += 1
        
        return phonemes
    
    def to_ids(self, phonemes: List[str]) -> List[int]:
        """Convert phoneme strings to integer IDs."""
        return FRENCH_INVENTORY.to_ids(phonemes)
    
    def process(self, text: str) -> Tuple[List[str], List[int]]:
        """
        Full processing pipeline: text → phonemes → IDs.
        
        Returns:
            Tuple of (phoneme strings, phoneme IDs)
        """
        phonemes = self.convert(text)
        ids = self.to_ids(phonemes)
        return phonemes, ids


class TextProcessor:
    """
    Complete text processing pipeline for French TTS.
    
    Combines normalization and G2P conversion.
    """
    
    def __init__(self, g2p_backend: str = 'auto'):
        self.normalizer = FrenchNormalizer()
        self.g2p = FrenchG2P(backend=g2p_backend)
    
    def process(self, text: str) -> Dict:
        """
        Process text through full pipeline.
        
        Args:
            text: Raw input text
            
        Returns:
            Dict containing:
                - original: Original text
                - normalized: Normalized text
                - phonemes: List of phoneme strings
                - phoneme_ids: List of phoneme IDs
        """
        normalized = self.normalizer.normalize(text)
        phonemes, phoneme_ids = self.g2p.process(normalized)
        
        return {
            'original': text,
            'normalized': normalized,
            'phonemes': phonemes,
            'phoneme_ids': phoneme_ids
        }
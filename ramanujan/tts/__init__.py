"""
Candide TTS Extension
=====================

Text-to-Speech extension for the Candide framework.
Follows existing patterns from ramanujan/ for seamless integration.

Structure mirrors ramanujan/:
- tts/audio.py       -> Audio processing utilities
- tts/text.py        -> Text normalization, G2P
- tts/phonemes.py    -> Phoneme inventories
- models/components/tts/  -> TTS-specific components
- models/architectures/tts.py -> Full TTS architecture
- training/losses/tts/    -> TTS loss functions
- data/datasets/tts/      -> TTS datasets

Usage:
------
    from ramanujan.tts import AudioProcessor, FrenchG2P
    from ramanujan.models.components.tts import StyleSystem, ProsodySystem
    from ramanujan.models.architectures.tts import TTSModel
    from ramanujan.training.losses.tts import TTSLoss
"""

__version__ = "0.1.0"
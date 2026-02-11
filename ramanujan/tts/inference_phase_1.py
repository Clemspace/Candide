"""
TTS Phase 1 Inference Script
============================

Generate mel spectrograms from phonemes using trained model.

Usage:
    # Basic inference with phoneme IDs
    python inference_phase1.py --checkpoint ./checkpoints/siwis_phase1/checkpoint_best.pt
    
    # With text input (requires espeak-ng for G2P)
    python inference_phase1.py --checkpoint ./checkpoints/siwis_phase1/checkpoint_best.pt --text "Bonjour le monde"
    
    # Generate audio (requires vocoder)
    python inference_phase1.py --checkpoint ./checkpoints/siwis_phase1/checkpoint_best.pt --text "Bonjour" --vocoder hifigan

What this script does:
1. Loads trained TTS model
2. Converts text to phonemes (if text provided)
3. Generates mel spectrogram
4. Optionally converts to audio via vocoder
5. Saves outputs (mel, audio, visualizations)
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn.functional as F

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# PHONEME HANDLING
# =============================================================================

# SIWIS French phoneme set (simplified - matches preprocessing)
SIWIS_PHONEMES = [
    'SIL', 'UNK',
    'a', 'e', 'E', 'i', 'o', 'O', 'u', 'y', '@', '2', '9',
    'a~', 'e~', 'o~', '9~',
    'j', 'w', 'H',
    'p', 'b', 't', 'd', 'k', 'g',
    'f', 'v', 's', 'z', 'S', 'Z',
    'm', 'n', 'N', 'J',
    'l', 'R',
    'P2', 'Ohn', 'Oh', 'An', 'En', 'On', '@@',
]

PHONEME_TO_ID = {p: i for i, p in enumerate(SIWIS_PHONEMES)}
ID_TO_PHONEME = {i: p for p, i in PHONEME_TO_ID.items()}


def text_to_phonemes_simple(text: str) -> List[str]:
    """
    Simple French text to phoneme conversion.
    
    This is a BASIC fallback. For production, use espeak-ng or Phonemizer.
    """
    # Very simplified French G2P rules
    text = text.lower().strip()
    
    # Basic replacements
    replacements = [
        ('ou', 'u'),
        ('oi', 'wa'),
        ('au', 'o'),
        ('eau', 'o'),
        ('ai', 'E'),
        ('ei', 'E'),
        ('eu', '2'),
        ('oeu', '9'),
        ('an', 'a~'),
        ('en', 'a~'),
        ('am', 'a~'),
        ('em', 'a~'),
        ('in', 'e~'),
        ('im', 'e~'),
        ('ain', 'e~'),
        ('ein', 'e~'),
        ('on', 'o~'),
        ('om', 'o~'),
        ('un', '9~'),
        ('um', '9~'),
        ('ch', 'S'),
        ('gn', 'J'),
        ('qu', 'k'),
        ('gu', 'g'),
        ('ph', 'f'),
        ('th', 't'),
    ]
    
    for old, new in replacements:
        text = text.replace(old, new)
    
    # Character to phoneme
    char_to_phone = {
        'a': 'a', 'à': 'a', 'â': 'a',
        'e': '@', 'é': 'e', 'è': 'E', 'ê': 'E', 'ë': 'E',
        'i': 'i', 'î': 'i', 'ï': 'i',
        'o': 'o', 'ô': 'o',
        'u': 'y', 'û': 'y', 'ù': 'y',
        'y': 'i',
        'b': 'b', 'c': 'k', 'd': 'd', 'f': 'f', 'g': 'g',
        'h': '', 'j': 'Z', 'k': 'k', 'l': 'l', 'm': 'm',
        'n': 'n', 'p': 'p', 'r': 'R', 's': 's', 't': 't',
        'v': 'v', 'w': 'w', 'x': 'ks', 'z': 'z',
        ' ': 'SIL', '.': 'SIL', ',': 'SIL', '!': 'SIL', '?': 'SIL',
        "'": '', '-': '',
    }
    
    phones = []
    for char in text:
        if char in char_to_phone:
            p = char_to_phone[char]
            if p:
                phones.append(p)
        elif char.isalpha():
            phones.append('UNK')
    
    # Clean up consecutive silences
    cleaned = []
    for p in phones:
        if p == 'SIL' and cleaned and cleaned[-1] == 'SIL':
            continue
        cleaned.append(p)
    
    # Add start/end silence
    if not cleaned or cleaned[0] != 'SIL':
        cleaned.insert(0, 'SIL')
    if cleaned[-1] != 'SIL':
        cleaned.append('SIL')
    
    return cleaned


def text_to_phonemes_espeak(text: str, language: str = 'fr') -> List[str]:
    """
    Convert text to phonemes using espeak-ng.
    
    Requires: pip install phonemizer
              apt install espeak-ng
    """
    try:
        from phonemizer import phonemize
        from phonemizer.backend import EspeakBackend
        
        # Get IPA phonemes
        phonemes = phonemize(
            text,
            language=language,
            backend='espeak',
            strip=True,
            preserve_punctuation=True,
            with_stress=False,
        )
        
        # Convert IPA to SIWIS format
        # This is a simplified mapping
        ipa_to_siwis = {
            'a': 'a', 'ɑ': 'a', 'e': 'e', 'ɛ': 'E', 'i': 'i',
            'o': 'o', 'ɔ': 'O', 'u': 'u', 'y': 'y', 'ə': '@',
            'ø': '2', 'œ': '9',
            'ɑ̃': 'a~', 'ɛ̃': 'e~', 'ɔ̃': 'o~', 'œ̃': '9~',
            'j': 'j', 'w': 'w', 'ɥ': 'H',
            'p': 'p', 'b': 'b', 't': 't', 'd': 'd', 'k': 'k', 'g': 'g',
            'f': 'f', 'v': 'v', 's': 's', 'z': 'z', 'ʃ': 'S', 'ʒ': 'Z',
            'm': 'm', 'n': 'n', 'ŋ': 'N', 'ɲ': 'J',
            'l': 'l', 'ʁ': 'R', 'r': 'R',
            ' ': 'SIL', '.': 'SIL', ',': 'SIL',
        }
        
        phones = []
        i = 0
        while i < len(phonemes):
            # Check for 2-char sequences (nasals)
            if i + 1 < len(phonemes):
                digraph = phonemes[i:i+2]
                if digraph in ipa_to_siwis:
                    phones.append(ipa_to_siwis[digraph])
                    i += 2
                    continue
            
            char = phonemes[i]
            if char in ipa_to_siwis:
                phones.append(ipa_to_siwis[char])
            elif char.isalpha():
                phones.append('UNK')
            i += 1
        
        # Clean up
        cleaned = []
        for p in phones:
            if p == 'SIL' and cleaned and cleaned[-1] == 'SIL':
                continue
            cleaned.append(p)
        
        if not cleaned or cleaned[0] != 'SIL':
            cleaned.insert(0, 'SIL')
        if cleaned[-1] != 'SIL':
            cleaned.append('SIL')
        
        return cleaned
        
    except ImportError:
        print("phonemizer not installed, using simple G2P")
        return text_to_phonemes_simple(text)


def phonemes_to_ids(phonemes: List[str]) -> torch.Tensor:
    """Convert phoneme list to tensor of IDs."""
    ids = [PHONEME_TO_ID.get(p, PHONEME_TO_ID['UNK']) for p in phonemes]
    return torch.tensor(ids, dtype=torch.long)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(checkpoint_path: str, device: str = 'cuda') -> Tuple[torch.nn.Module, Dict]:
    """Load trained TTS model from checkpoint."""
    
    from ramanujan.models.architectures.tts_model import TTSModelPhase1, TTSConfigPhase1
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config
    config_dict = checkpoint.get('config', {})
    
    # Build model config
    model_config = TTSConfigPhase1(
        n_phonemes=config_dict.get('n_phonemes', 42),
        phoneme_dim=config_dict.get('phoneme_dim', 256),
        n_speakers=config_dict.get('n_speakers', 1),
        speaker_dim=config_dict.get('speaker_dim', 256),
        style_dim=config_dict.get('style_dim', 256),
        prosody_hidden_dim=config_dict.get('prosody_hidden_dim', 256),
        n_mels=config_dict.get('n_mels', 80),
        acoustic_hidden_channels=config_dict.get('acoustic_hidden_channels', 256),
        acoustic_channel_mults=tuple(config_dict.get('acoustic_channel_mults', [1, 2, 4])),
        dropout=0.0,  # No dropout at inference
    )
    
    model = TTSModelPhase1(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'best_val_loss': checkpoint.get('best_val_loss', 0),
    }
    
    print(f"Loaded model from epoch {info['epoch']}, step {info['global_step']}")
    print(f"Best validation loss: {info['best_val_loss']:.4f}")
    
    return model, info


# =============================================================================
# VOCODER
# =============================================================================

def load_vocoder(vocoder_type: str = 'griffin_lim', device: str = 'cuda'):
    """Load a vocoder for mel-to-audio conversion."""
    
    if vocoder_type == 'griffin_lim':
        # Simple Griffin-Lim (no neural network needed)
        return GriffinLimVocoder()
    
    elif vocoder_type == 'hifigan':
        try:
            # Try to load HiFi-GAN
            from speechbrain.inference.vocoders import HIFIGAN
            vocoder = HIFIGAN.from_hparams(
                source="speechbrain/tts-hifigan-ljspeech",
                savedir="pretrained_models/tts-hifigan-ljspeech"
            )
            return HiFiGANWrapper(vocoder)
        except ImportError:
            print("speechbrain not installed, falling back to Griffin-Lim")
            return GriffinLimVocoder()
    
    else:
        return GriffinLimVocoder()


class GriffinLimVocoder:
    """Simple Griffin-Lim vocoder (no neural network)."""
    
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, 
                 win_length: int = 1024, n_mels: int = 80,
                 sample_rate: int = 22050, n_iter: int = 60):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_iter = n_iter
        
        # Create mel filterbank for inversion
        import torchaudio.transforms as T
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            power=1.0,
        )
        
        self.griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_iter=n_iter,
        )
    
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert mel spectrogram to waveform.
        
        Args:
            mel: (batch, n_mels, T) log mel spectrogram
            
        Returns:
            waveform: (batch, samples)
        """
        # Convert from log mel to linear
        mel_linear = torch.exp(mel)
        
        # Approximate inverse mel filterbank
        # This is a rough approximation - proper inversion would use pseudo-inverse
        mel_basis = self.mel_spec.mel_scale.fb.T  # (n_mels, n_fft//2+1)
        mel_basis_pinv = torch.linalg.pinv(mel_basis)  # (n_fft//2+1, n_mels)
        
        # (batch, n_mels, T) @ (n_mels, n_fft//2+1).T -> (batch, n_fft//2+1, T)
        spec_linear = torch.matmul(mel_basis_pinv.T.to(mel.device), mel_linear)
        
        # Griffin-Lim
        waveforms = []
        for i in range(spec_linear.shape[0]):
            wav = self.griffin_lim(spec_linear[i])
            waveforms.append(wav)
        
        return torch.stack(waveforms)


class HiFiGANWrapper:
    """Wrapper for HiFi-GAN vocoder."""
    
    def __init__(self, vocoder):
        self.vocoder = vocoder
    
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel to waveform using HiFi-GAN."""
        waveforms = self.vocoder.decode_batch(mel)
        return waveforms.squeeze(1)


# =============================================================================
# INFERENCE
# =============================================================================

@torch.no_grad()
def synthesize(
    model: torch.nn.Module,
    phoneme_ids: torch.Tensor,
    speaker_id: int = 0,
    n_flow_steps: int = 10,
    temperature: float = 1.0,
    device: str = 'cuda',
) -> Dict[str, torch.Tensor]:
    """
    Synthesize mel spectrogram from phonemes.
    
    Args:
        model: Trained TTS model
        phoneme_ids: (seq_len,) phoneme indices
        speaker_id: Speaker index (0 for single speaker)
        n_flow_steps: Number of flow matching steps
        temperature: Sampling temperature
        device: Device to use
        
    Returns:
        Dictionary with 'mel', 'durations', 'f0', 'energy'
    """
    model.eval()
    
    # Prepare inputs
    phoneme_ids = phoneme_ids.unsqueeze(0).to(device)  # (1, S)
    speaker_id = torch.tensor([speaker_id], dtype=torch.long, device=device)
    
    # Generate
    outputs = model.generate(
        phoneme_ids=phoneme_ids,
        speaker_id=speaker_id,
        n_flow_steps=n_flow_steps,
        temperature=temperature,
    )
    
    return {k: v.squeeze(0).cpu() for k, v in outputs.items()}


def save_mel_plot(mel: np.ndarray, path: str, title: str = 'Mel Spectrogram'):
    """Save mel spectrogram visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xlabel('Frames')
    ax.set_ylabel('Mel bins')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Log magnitude')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved mel plot: {path}")


def save_prosody_plot(durations: np.ndarray, f0: np.ndarray, energy: np.ndarray,
                      phonemes: List[str], path: str):
    """Save prosody visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Durations
    ax = axes[0]
    ax.bar(range(len(durations)), durations, color='steelblue')
    ax.set_ylabel('Duration (frames)')
    ax.set_title('Predicted Durations')
    if len(phonemes) <= 30:
        ax.set_xticks(range(len(phonemes)))
        ax.set_xticklabels(phonemes, rotation=45, ha='right')
    
    # F0
    ax = axes[1]
    ax.plot(f0, color='darkorange', linewidth=1.5)
    ax.set_ylabel('F0 (Hz)')
    ax.set_title('Predicted F0 Contour')
    ax.set_xlabel('Frames')
    
    # Energy
    ax = axes[2]
    ax.plot(energy, color='forestgreen', linewidth=1.5)
    ax.set_ylabel('Energy')
    ax.set_title('Predicted Energy')
    ax.set_xlabel('Frames')
    
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved prosody plot: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TTS Phase 1 Inference")
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Input (one of these)
    parser.add_argument('--text', type=str, default=None,
                        help='Text to synthesize (French)')
    parser.add_argument('--phonemes', type=str, default=None,
                        help='Space-separated phonemes (e.g., "SIL b o~ Z u R SIL")')
    
    # Options
    parser.add_argument('--output_dir', type=str, default='./outputs/inference',
                        help='Output directory')
    parser.add_argument('--n_flow_steps', type=int, default=10,
                        help='Number of flow matching steps')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--vocoder', type=str, default='griffin_lim',
                        choices=['griffin_lim', 'hifigan', 'none'],
                        help='Vocoder for audio generation')
    parser.add_argument('--use_espeak', action='store_true',
                        help='Use espeak-ng for G2P (better quality)')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, info = load_model(args.checkpoint, args.device)
    
    # Get phonemes
    if args.phonemes:
        phonemes = args.phonemes.split()
        print(f"Using provided phonemes: {phonemes}")
    elif args.text:
        print(f"Converting text: '{args.text}'")
        if args.use_espeak:
            phonemes = text_to_phonemes_espeak(args.text)
        else:
            phonemes = text_to_phonemes_simple(args.text)
        print(f"Phonemes: {' '.join(phonemes)}")
    else:
        # Demo text
        demo_text = "Bonjour le monde"
        print(f"No input provided, using demo: '{demo_text}'")
        phonemes = text_to_phonemes_simple(demo_text)
        print(f"Phonemes: {' '.join(phonemes)}")
    
    # Convert to IDs
    phoneme_ids = phonemes_to_ids(phonemes)
    print(f"Phoneme IDs: {phoneme_ids.tolist()}")
    
    # Synthesize
    print(f"\nSynthesizing with {args.n_flow_steps} flow steps, temperature={args.temperature}")
    outputs = synthesize(
        model=model,
        phoneme_ids=phoneme_ids,
        n_flow_steps=args.n_flow_steps,
        temperature=args.temperature,
        device=args.device,
    )
    
    # Get outputs
    mel = outputs['mel'].numpy()
    durations = outputs['durations'].numpy()
    f0 = outputs.get('f0', torch.zeros(mel.shape[1])).numpy()
    energy = outputs.get('energy', torch.zeros(mel.shape[1])).numpy()
    
    print(f"\nGenerated mel shape: {mel.shape}")
    print(f"Total frames: {mel.shape[1]}")
    print(f"Duration: {mel.shape[1] * 256 / 22050:.2f} seconds (approx)")
    
    # Save outputs
    timestamp = Path(args.checkpoint).stem
    
    # Save mel
    mel_path = output_dir / f'mel_{timestamp}.npy'
    np.save(mel_path, mel)
    print(f"Saved mel: {mel_path}")
    
    # Save mel plot
    save_mel_plot(mel, output_dir / f'mel_{timestamp}.png', 
                  title=f"Generated Mel - {args.text or 'phonemes'}")
    
    # Save prosody plot
    save_prosody_plot(durations, f0, energy, phonemes,
                      output_dir / f'prosody_{timestamp}.png')
    
    # Generate audio if vocoder specified
    if args.vocoder != 'none':
        print(f"\nGenerating audio with {args.vocoder}...")
        try:
            vocoder = load_vocoder(args.vocoder, args.device)
            mel_tensor = torch.from_numpy(mel).unsqueeze(0).to(args.device)
            waveform = vocoder(mel_tensor)
            
            # Save audio
            import torchaudio
            audio_path = output_dir / f'audio_{timestamp}.wav'
            torchaudio.save(audio_path, waveform.cpu(), 22050)
            print(f"Saved audio: {audio_path}")
        except Exception as e:
            print(f"Vocoder failed: {e}")
            print("Mel spectrogram saved - you can use external vocoder")
    
    print("\n✓ Inference complete!")
    print(f"  Outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
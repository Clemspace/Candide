"""
SIWIS Dataset Preprocessing for TTS Phase 1
============================================

Processes the official SIWIS French Speech Synthesis Database zip file.
Extracts:
- Mel spectrograms (80 bins)
- F0 contours (via pyworld)  
- Energy contours
- Phoneme sequences and durations (from HTS labels!)

Usage:
    python preprocess_siwis_local.py \
        --zip_path /path/to/SiwisFrenchSpeechSynthesisDatabase.zip \
        --output_dir ./data/siwis_processed \
        --max_samples 1000  # optional, for testing
"""

import os
import sys
import re
import json
import argparse
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from tqdm import tqdm
import numpy as np
import io

# =============================================================================
# IMPORT CHECKS
# =============================================================================

print("Checking imports...")

try:
    import torch
    import torchaudio
    import torchaudio.transforms as T
    print(f"  torch: {torch.__version__}")
    print(f"  torchaudio: {torchaudio.__version__}")
except ImportError as e:
    print(f"  torch/torchaudio: FAILED - {e}")
    sys.exit(1)

try:
    import pyworld as pw
    print("  pyworld: OK")
    PYWORLD_AVAILABLE = True
except ImportError:
    print("  pyworld: NOT AVAILABLE (F0 extraction will use fallback)")
    PYWORLD_AVAILABLE = False

print()


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    f0_min: float = 50.0
    f0_max: float = 600.0
    
    @property
    def frame_shift_ms(self) -> float:
        return self.hop_length / self.sample_rate * 1000


# =============================================================================
# SIWIS PHONEME SET
# =============================================================================

# Phonemes extracted from SIWIS HTS labels
# Format in labels: phoneme appears in context like "x^prev-PHONE+next=next2"
SIWIS_PHONEMES = [
    # Special
    'SIL',      # Silence (pau, #, x)
    'UNK',      # Unknown
    
    # Vowels (oral)
    'a',        # as in "patte"
    'e',        # as in "été" (closed e)
    'E',        # as in "père" (open e) - sometimes written ɛ
    'i',        # as in "si"
    'o',        # as in "eau" (closed o)
    'O',        # as in "or" (open o) - sometimes written ɔ
    'u',        # as in "ou"
    'y',        # as in "tu"
    '@',        # schwa as in "le"
    '2',        # as in "deux" (ø)
    '9',        # as in "neuf" (œ)
    
    # Nasal vowels
    'a~',       # as in "an" (ɑ̃)
    'e~',       # as in "vin" (ɛ̃)  
    'o~',       # as in "on" (ɔ̃)
    '9~',       # as in "un" (œ̃)
    
    # Semi-vowels
    'j',        # as in "yeux"
    'w',        # as in "oui"
    'H',        # as in "huit" (ɥ)
    
    # Consonants - plosives
    'p',
    'b',
    't',
    'd',
    'k',
    'g',
    
    # Consonants - fricatives
    'f',
    'v',
    's',
    'z',
    'S',        # as in "chat" (ʃ)
    'Z',        # as in "je" (ʒ)
    
    # Consonants - nasals
    'm',
    'n',
    'N',        # as in "parking" (ŋ)
    'J',        # as in "agne" (ɲ)
    
    # Consonants - liquids
    'l',
    'R',        # French R (ʁ)
    
    # SIWIS-specific symbols I saw in the labels
    'P2',       # Appears to be a variant (maybe pause type 2?)
    'Ohn',      # Nasal variant?
    'Oh',       # Open o variant?
    'An',       # Nasal a?
    'En',       # Nasal e?
    'On',       # Nasal o?
    '@@',       # Double schwa or special marker
]

# Build phoneme to ID mapping
PHONEME_TO_ID = {p: i for i, p in enumerate(SIWIS_PHONEMES)}
N_PHONEMES = len(SIWIS_PHONEMES)

# Mapping for normalizing SIWIS phonemes
PHONEME_NORMALIZE = {
    'pau': 'SIL',
    '#': 'SIL', 
    'x': 'SIL',
    'sil': 'SIL',
    'sp': 'SIL',
    # Add more mappings as we discover them
}


# =============================================================================
# HTS LABEL PARSER
# =============================================================================

def parse_hts_label(label_content: str) -> List[Tuple[int, int, str]]:
    """
    Parse HTS label file content.
    
    HTS format: start_time end_time full_context
    Times are in 100ns units.
    
    The phoneme is extracted from the context string.
    Format: prev2^prev-PHONE+next=next2@...
    
    Returns: List of (start_frame, end_frame, phoneme)
    """
    phonemes = []
    
    for line in label_content.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) < 3:
            continue
        
        try:
            start_100ns = int(parts[0])
            end_100ns = int(parts[1])
            context = parts[2]
            
            # Extract phoneme from context
            # Format: prev2^prev-PHONE+next=next2@...
            # We need to find the part between - and +
            
            # First, get the part before @
            main_part = context.split('@')[0] if '@' in context else context
            
            # Now parse: prev^prev2-PHONE+next=next2
            # Find phoneme between - and +
            if '-' in main_part and '+' in main_part:
                after_minus = main_part.split('-')[1]
                phoneme = after_minus.split('+')[0]
            elif '-' in main_part:
                phoneme = main_part.split('-')[1].split('=')[0]
            else:
                phoneme = main_part
            
            # Normalize
            phoneme = PHONEME_NORMALIZE.get(phoneme, phoneme)
            
            # Convert time to seconds
            start_sec = start_100ns / 10_000_000
            end_sec = end_100ns / 10_000_000
            
            phonemes.append((start_sec, end_sec, phoneme))
            
        except (ValueError, IndexError) as e:
            continue
    
    return phonemes


def phonemes_to_ids_and_durations(
    phonemes: List[Tuple[float, float, str]],
    hop_length: int,
    sample_rate: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert phoneme list to IDs and frame-level durations.
    
    Args:
        phonemes: List of (start_sec, end_sec, phoneme)
        hop_length: Hop length in samples
        sample_rate: Audio sample rate
        
    Returns:
        phoneme_ids: (N,) array of phoneme IDs
        durations: (N,) array of durations in frames
    """
    frame_shift_sec = hop_length / sample_rate
    
    ids = []
    durations = []
    
    for start_sec, end_sec, phone in phonemes:
        # Get phoneme ID
        if phone in PHONEME_TO_ID:
            pid = PHONEME_TO_ID[phone]
        else:
            # Try to find a close match or use UNK
            pid = PHONEME_TO_ID.get('UNK', 1)
        
        # Calculate duration in frames
        dur_sec = end_sec - start_sec
        dur_frames = max(1, round(dur_sec / frame_shift_sec))
        
        ids.append(pid)
        durations.append(dur_frames)
    
    return np.array(ids, dtype=np.int64), np.array(durations, dtype=np.int64)


# =============================================================================
# AUDIO PROCESSING
# =============================================================================

class AudioProcessor:
    """Extract audio features for TTS training."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=1.0,
            norm='slaney',
            mel_scale='slaney'
        )
    
    def load_and_resample(self, wav_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Load wav from bytes and resample."""
        # Load from bytes
        waveform, sr = torchaudio.load(io.BytesIO(wav_bytes))
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.config.sample_rate:
            resampler = T.Resample(sr, self.config.sample_rate)
            waveform = resampler(waveform)
        
        return waveform.squeeze(0).numpy(), self.config.sample_rate
    
    def extract_mel(self, waveform: np.ndarray) -> np.ndarray:
        """Extract log mel spectrogram."""
        waveform_t = torch.from_numpy(waveform).float().unsqueeze(0)
        mel = self.mel_transform(waveform_t)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.squeeze(0).numpy()  # (n_mels, T)
    
    def extract_f0(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract F0 contour using WORLD."""
        if PYWORLD_AVAILABLE:
            waveform_f64 = waveform.astype(np.float64)
            f0, timeaxis = pw.harvest(
                waveform_f64,
                self.config.sample_rate,
                f0_floor=self.config.f0_min,
                f0_ceil=self.config.f0_max,
                frame_period=self.config.frame_shift_ms
            )
            f0 = pw.stonemask(waveform_f64, f0, timeaxis, self.config.sample_rate)
        else:
            # Fallback: zeros
            n_frames = len(waveform) // self.config.hop_length + 1
            f0 = np.zeros(n_frames)
        
        voiced = f0 > 0
        return f0.astype(np.float32), voiced
    
    def extract_energy(self, waveform: np.ndarray) -> np.ndarray:
        """Extract frame-level energy."""
        hop = self.config.hop_length
        win = self.config.win_length
        
        waveform_pad = np.pad(waveform, (win // 2, win // 2), mode='reflect')
        n_frames = (len(waveform_pad) - win) // hop + 1
        energy = np.zeros(n_frames, dtype=np.float32)
        
        for i in range(n_frames):
            start = i * hop
            frame = waveform_pad[start:start + win]
            energy[i] = np.sqrt(np.mean(frame ** 2) + 1e-8)
        
        return energy
    
    def process(self, wav_bytes: bytes) -> Dict[str, np.ndarray]:
        """Extract all features from wav bytes."""
        waveform, sr = self.load_and_resample(wav_bytes)
        
        mel = self.extract_mel(waveform)
        f0, voiced = self.extract_f0(waveform)
        energy = self.extract_energy(waveform)
        
        n_frames = mel.shape[1]
        
        # Align lengths
        f0 = np.pad(f0, (0, max(0, n_frames - len(f0))))[:n_frames]
        voiced = np.pad(voiced, (0, max(0, n_frames - len(voiced))))[:n_frames]
        energy = np.pad(energy, (0, max(0, n_frames - len(energy))))[:n_frames]
        
        return {
            'mel': mel.astype(np.float32),
            'f0': f0.astype(np.float32),
            'voiced': voiced,
            'energy': energy.astype(np.float32),
            'n_frames': n_frames,
            'duration_sec': len(waveform) / sr
        }


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_siwis_zip(
    zip_path: str,
    output_dir: str,
    max_samples: Optional[int] = None,
    parts: List[str] = ['part1', 'part2', 'part3'],
    max_duration_sec: float = 15.0,
):
    """
    Process SIWIS dataset from zip file.
    
    Args:
        zip_path: Path to SiwisFrenchSpeechSynthesisDatabase.zip
        output_dir: Output directory for processed files
        max_samples: Maximum samples to process (None = all)
        parts: Which parts to include (part1, part2, part3)
        max_duration_sec: Skip samples longer than this
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'train').mkdir(exist_ok=True)
    (output_dir / 'val').mkdir(exist_ok=True)
    
    config = AudioConfig()
    processor = AudioProcessor(config)
    
    print(f"Processing SIWIS from: {zip_path}")
    print(f"Output directory: {output_dir}")
    print(f"Parts: {parts}")
    print(f"Max duration: {max_duration_sec}s")
    if max_samples:
        print(f"Max samples: {max_samples}")
    print()
    
    # Collect all phonemes seen for analysis
    all_phonemes_seen: Set[str] = set()
    
    metadata = {
        'config': asdict(config),
        'phoneme_to_id': PHONEME_TO_ID,
        'n_phonemes': N_PHONEMES,
        'dataset': 'siwis',
        'samples': []
    }
    
    n_success = 0
    n_skipped = 0
    n_no_lab = 0
    n_too_long = 0
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Get list of all files
        all_files = zf.namelist()
        
        # Find all wav files in specified parts
        wav_files = []
        for f in all_files:
            if f.endswith('.wav'):
                for part in parts:
                    if f'/wavs/{part}/' in f:
                        wav_files.append(f)
                        break
        
        print(f"Found {len(wav_files)} wav files")
        
        if max_samples:
            wav_files = wav_files[:max_samples]
        
        for wav_path in tqdm(wav_files, desc="Processing"):
            try:
                # Extract sample name
                # e.g., SiwisFrenchSpeechSynthesisDatabase/wavs/part1/neut_parl_s01_0001.wav
                parts_split = wav_path.split('/')
                wav_name = parts_split[-1]  # neut_parl_s01_0001.wav
                part_name = parts_split[-2]  # part1
                sample_id = wav_name.replace('.wav', '')
                
                # Find corresponding lab file
                lab_path = wav_path.replace('/wavs/', '/labs/').replace('.wav', '.lab')
                
                if lab_path not in all_files:
                    n_no_lab += 1
                    continue
                
                # Find corresponding text file
                text_path = wav_path.replace('/wavs/', '/text/').replace('.wav', '.txt')
                text = ""
                if text_path in all_files:
                    text = zf.read(text_path).decode('utf-8', errors='replace').strip()
                
                # Load and process audio
                wav_bytes = zf.read(wav_path)
                features = processor.process(wav_bytes)
                
                # Skip if too long
                if features['duration_sec'] > max_duration_sec:
                    n_too_long += 1
                    continue
                
                # Parse HTS labels
                lab_content = zf.read(lab_path).decode('utf-8', errors='replace')
                phonemes = parse_hts_label(lab_content)
                
                if not phonemes:
                    n_skipped += 1
                    continue
                
                # Track phonemes seen
                for _, _, phone in phonemes:
                    all_phonemes_seen.add(phone)
                
                # Convert to IDs and durations
                phoneme_ids, durations = phonemes_to_ids_and_durations(
                    phonemes,
                    config.hop_length,
                    config.sample_rate
                )
                
                # Adjust durations to match mel length
                total_dur = durations.sum()
                n_frames = features['n_frames']
                
                if total_dur != n_frames:
                    # Scale durations proportionally
                    scale = n_frames / total_dur
                    durations = np.round(durations * scale).astype(np.int64)
                    durations = np.maximum(durations, 1)
                    
                    # Fix rounding errors
                    diff = n_frames - durations.sum()
                    if diff != 0:
                        durations[-1] = max(1, durations[-1] + diff)
                
                # Determine split (90/10)
                split = 'val' if n_success % 10 == 0 else 'train'
                
                # Save
                sample_name = f"{part_name}_{sample_id}"
                save_path = output_dir / split / f"{sample_name}.npz"
                
                np.savez_compressed(
                    save_path,
                    mel=features['mel'],
                    f0=features['f0'],
                    voiced=features['voiced'],
                    energy=features['energy'],
                    phoneme_ids=phoneme_ids,
                    durations=durations,
                    text=text
                )
                
                metadata['samples'].append({
                    'name': sample_name,
                    'split': split,
                    'n_frames': n_frames,
                    'n_phonemes': len(phoneme_ids),
                    'duration': features['duration_sec'],
                    'text': text[:100] if text else ''
                })
                
                n_success += 1
                
            except Exception as e:
                print(f"\n  Error processing {wav_path}: {e}")
                n_skipped += 1
                continue
    
    # Save metadata
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Save phoneme inventory
    with open(output_dir / 'phonemes_seen.txt', 'w') as f:
        for p in sorted(all_phonemes_seen):
            f.write(f"{p}\n")
    
    n_train = len([s for s in metadata['samples'] if s['split'] == 'train'])
    n_val = len([s for s in metadata['samples'] if s['split'] == 'val'])
    
    print()
    print("=" * 60)
    print("Processing complete!")
    print(f"  Success:    {n_success}")
    print(f"  Skipped:    {n_skipped}")
    print(f"  No labels:  {n_no_lab}")
    print(f"  Too long:   {n_too_long}")
    print(f"  Train:      {n_train}")
    print(f"  Val:        {n_val}")
    print(f"  Output:     {output_dir}")
    print()
    print(f"Unique phonemes seen: {len(all_phonemes_seen)}")
    print(f"  {sorted(all_phonemes_seen)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SIWIS French TTS dataset from zip"
    )
    parser.add_argument(
        '--zip_path',
        type=str,
        required=True,
        help='Path to SiwisFrenchSpeechSynthesisDatabase.zip'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/siwis_processed',
        help='Output directory'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Max samples to process (for testing)'
    )
    parser.add_argument(
        '--parts',
        type=str,
        nargs='+',
        default=['part1', 'part2', 'part3'],
        help='Parts to include (part1, part2, part3)'
    )
    parser.add_argument(
        '--max_duration',
        type=float,
        default=15.0,
        help='Max audio duration in seconds'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.zip_path):
        print(f"Error: Zip file not found: {args.zip_path}")
        sys.exit(1)
    
    process_siwis_zip(
        args.zip_path,
        args.output_dir,
        args.max_samples,
        args.parts,
        args.max_duration
    )


if __name__ == '__main__':
    main()
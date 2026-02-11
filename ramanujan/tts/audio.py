"""
Audio Processing
================

Utilities for audio loading and feature extraction.

Key features:
- Mel spectrogram extraction
- F0 (pitch) extraction
- Energy extraction
- Audio normalization

These features are the targets and conditioning signals for TTS training.
"""

import torch
from torch import Tensor
import numpy as np
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

# Optional audio libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


@dataclass
class AudioConfig:
    """
    Configuration for audio processing.
    
    These values should remain consistent between training and inference.
    Standard TTS settings are provided as defaults.
    """
    sample_rate: int = 22050          # Hz - standard for TTS
    n_fft: int = 1024                 # FFT window size
    hop_length: int = 256             # Samples between frames (~11.6ms)
    win_length: int = 1024            # Window length
    n_mels: int = 80                  # Number of mel bins
    fmin: float = 0.0                 # Minimum frequency for mel
    fmax: float = 8000.0              # Maximum frequency for mel
    
    # F0 extraction
    f0_min: float = 50.0              # Min pitch (Hz)
    f0_max: float = 800.0             # Max pitch (Hz)
    
    # Normalization
    mel_normalize: bool = True
    mel_mean: float = -4.0            # Approximate mean of log-mel
    mel_std: float = 4.0              # Approximate std of log-mel
    
    @property
    def frame_rate(self) -> float:
        """Frames per second."""
        return self.sample_rate / self.hop_length
    
    @property
    def frame_duration_ms(self) -> float:
        """Duration of one frame in milliseconds."""
        return 1000 * self.hop_length / self.sample_rate
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
            'n_mels': self.n_mels,
            'fmin': self.fmin,
            'fmax': self.fmax,
            'f0_min': self.f0_min,
            'f0_max': self.f0_max,
        }


# Default configuration
DEFAULT_AUDIO_CONFIG = AudioConfig()


class AudioProcessor:
    """
    Audio processing for TTS.
    
    Handles loading, mel extraction, F0 extraction, and energy extraction.
    Uses librosa or torchaudio depending on availability.
    
    Example:
        processor = AudioProcessor()
        features = processor.process_file("audio.wav")
        # features contains: mel, f0, voiced_mask, energy
    """
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or DEFAULT_AUDIO_CONFIG
        
        if not LIBROSA_AVAILABLE and not TORCHAUDIO_AVAILABLE:
            raise ImportError(
                "Either librosa or torchaudio required. "
                "Install with: pip install librosa torchaudio"
            )
        
        # Build mel filterbank for torchaudio
        if TORCHAUDIO_AVAILABLE:
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                n_mels=self.config.n_mels,
                f_min=self.config.fmin,
                f_max=self.config.fmax,
                norm='slaney',
                mel_scale='slaney'
            )
    
    def load_audio(
        self,
        path: Union[str, Path],
        normalize: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample to target sample rate.
        
        Args:
            path: Path to audio file
            normalize: Whether to normalize to [-1, 1]
            
        Returns:
            Tuple of (waveform, sample_rate)
        """
        path = str(path)
        
        if LIBROSA_AVAILABLE:
            waveform, sr = librosa.load(path, sr=self.config.sample_rate)
        elif TORCHAUDIO_AVAILABLE:
            waveform, sr = torchaudio.load(path)
            if sr != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
                waveform = resampler(waveform)
            waveform = waveform.numpy().squeeze()
            sr = self.config.sample_rate
        
        if normalize:
            max_val = np.abs(waveform).max()
            if max_val > 0:
                waveform = waveform / max_val
        
        return waveform, sr
    
    def compute_mel(self, waveform: np.ndarray) -> np.ndarray:
        """
        Compute mel spectrogram.
        
        Args:
            waveform: Audio samples (1D array)
            
        Returns:
            Log-mel spectrogram (n_mels, n_frames)
        """
        if LIBROSA_AVAILABLE:
            mel = librosa.feature.melspectrogram(
                y=waveform,
                sr=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                n_mels=self.config.n_mels,
                fmin=self.config.fmin,
                fmax=self.config.fmax
            )
            # Log scale
            mel = np.log(np.maximum(mel, 1e-5))
            
        elif TORCHAUDIO_AVAILABLE:
            waveform_t = torch.from_numpy(waveform).float().unsqueeze(0)
            mel = self._mel_transform(waveform_t)
            mel = torch.log(torch.clamp(mel, min=1e-5))
            mel = mel.squeeze().numpy()
        
        # Optional normalization
        if self.config.mel_normalize:
            mel = (mel - self.config.mel_mean) / self.config.mel_std
        
        return mel
    
    def extract_f0(
        self,
        waveform: np.ndarray,
        method: str = 'pyin'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract fundamental frequency (pitch).
        
        Args:
            waveform: Audio samples
            method: Extraction method ('pyin', 'yin', or 'crepe')
            
        Returns:
            Tuple of:
                f0: Pitch in Hz (0 for unvoiced frames)
                voiced_mask: Boolean mask of voiced frames
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required for F0 extraction")
        
        if method == 'pyin':
            f0, voiced_flag, _ = librosa.pyin(
                waveform,
                sr=self.config.sample_rate,
                fmin=self.config.f0_min,
                fmax=self.config.f0_max,
                hop_length=self.config.hop_length,
                frame_length=self.config.win_length
            )
        elif method == 'yin':
            f0 = librosa.yin(
                waveform,
                sr=self.config.sample_rate,
                fmin=self.config.f0_min,
                fmax=self.config.f0_max,
                hop_length=self.config.hop_length,
                frame_length=self.config.win_length
            )
            voiced_flag = (f0 > 0) & (f0 < self.config.f0_max)
        else:
            raise ValueError(f"Unknown F0 method: {method}. Use 'pyin' or 'yin'.")
        
        # Clean up
        f0 = np.nan_to_num(f0, nan=0.0)
        voiced_mask = np.asarray(voiced_flag, dtype=np.float32)
        f0 = f0 * voiced_mask  # Zero out unvoiced
        
        return f0, voiced_mask
    
    def extract_energy(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract frame-level energy (RMS).
        
        Args:
            waveform: Audio samples
            
        Returns:
            Energy contour (n_frames,)
        """
        if LIBROSA_AVAILABLE:
            energy = librosa.feature.rms(
                y=waveform,
                frame_length=self.config.win_length,
                hop_length=self.config.hop_length
            ).squeeze()
        else:
            # Numpy fallback
            n_frames = 1 + (len(waveform) - self.config.win_length) // self.config.hop_length
            energy = np.zeros(n_frames)
            for i in range(n_frames):
                start = i * self.config.hop_length
                end = start + self.config.win_length
                frame = waveform[start:end]
                energy[i] = np.sqrt(np.mean(frame ** 2))
        
        return energy
    
    def process_file(self, path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Process audio file and extract all features.
        
        Args:
            path: Path to audio file
            
        Returns:
            Dict containing:
                - waveform: Raw audio
                - mel: Log-mel spectrogram (n_mels, n_frames)
                - f0: F0 contour (n_frames,)
                - voiced_mask: Voicing mask (n_frames,)
                - energy: Energy contour (n_frames,)
                - n_frames: Number of frames
        """
        waveform, _ = self.load_audio(path)
        mel = self.compute_mel(waveform)
        f0, voiced_mask = self.extract_f0(waveform)
        energy = self.extract_energy(waveform)
        
        # Align lengths to mel
        n_frames = mel.shape[1]
        f0 = self._align_length(f0, n_frames)
        voiced_mask = self._align_length(voiced_mask, n_frames)
        energy = self._align_length(energy, n_frames)
        
        return {
            'waveform': waveform,
            'mel': mel,
            'f0': f0,
            'voiced_mask': voiced_mask,
            'energy': energy,
            'n_frames': n_frames
        }
    
    def _align_length(self, arr: np.ndarray, target_len: int) -> np.ndarray:
        """Align array length by padding or truncating."""
        if len(arr) >= target_len:
            return arr[:target_len]
        else:
            return np.pad(arr, (0, target_len - len(arr)))
    
    def denormalize_mel(self, mel: np.ndarray) -> np.ndarray:
        """Reverse mel normalization."""
        if self.config.mel_normalize:
            return mel * self.config.mel_std + self.config.mel_mean
        return mel
    
    def mel_to_audio(
        self,
        mel: np.ndarray,
        n_iter: int = 32
    ) -> np.ndarray:
        """
        Convert mel spectrogram to audio using Griffin-Lim.
        
        This is a simple vocoder for debugging. Use HiFi-GAN for quality.
        
        Args:
            mel: Log-mel spectrogram (n_mels, n_frames)
            n_iter: Griffin-Lim iterations
            
        Returns:
            Waveform
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required for Griffin-Lim")
        
        # Denormalize
        mel = self.denormalize_mel(mel)
        
        # Exp to get power spectrogram
        mel_power = np.exp(mel)
        
        # Invert mel to linear spectrogram
        mel_basis = librosa.filters.mel(
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax
        )
        mel_basis_inv = np.linalg.pinv(mel_basis)
        spec = np.maximum(1e-10, np.dot(mel_basis_inv, mel_power))
        
        # Griffin-Lim
        waveform = librosa.griffinlim(
            spec,
            n_iter=n_iter,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length
        )
        
        return waveform
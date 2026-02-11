"""
TTS Dataset
===========

Dataset classes for TTS training.
Follows ramanujan/data/datasets/ pattern.
"""

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass


@dataclass
class TTSSample:
    """Single TTS training sample metadata."""
    audio_path: str
    text: str
    speaker_id: int = 0
    accent_id: int = 0
    emotion_id: int = 0
    emotion_intensity: float = 0.5
    duration: Optional[float] = None  # Audio duration in seconds
    
    # Optional precomputed features
    phoneme_ids: Optional[List[int]] = None
    durations: Optional[List[int]] = None


class TTSDataset(Dataset):
    """
    Dataset for TTS training.
    
    Loads from a manifest file (JSON) with format:
    [
        {
            "audio_path": "/path/to/audio.wav",
            "text": "Bonjour",
            "speaker_id": 0,
            "accent_id": 0,
            "emotion_id": 0,
            ...
        },
        ...
    ]
    
    Args:
        manifest_path: Path to JSON manifest
        audio_processor: AudioProcessor instance
        text_processor: TextProcessor instance
        max_duration: Maximum audio duration in seconds
        min_duration: Minimum audio duration in seconds
        cache_features: Whether to cache processed features
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        audio_processor=None,
        text_processor=None,
        max_duration: float = 10.0,
        min_duration: float = 0.5,
        cache_features: bool = False
    ):
        self.manifest_path = Path(manifest_path)
        self.audio_processor = audio_processor
        self.text_processor = text_processor
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.cache_features = cache_features
        
        # Load manifest
        self.samples = self._load_manifest()
        
        # Feature cache
        self._cache = {} if cache_features else None
    
    def _load_manifest(self) -> List[TTSSample]:
        """Load and filter samples from manifest."""
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            sample = TTSSample(
                audio_path=item['audio_path'],
                text=item['text'],
                speaker_id=item.get('speaker_id', 0),
                accent_id=item.get('accent_id', 0),
                emotion_id=item.get('emotion_id', 0),
                emotion_intensity=item.get('emotion_intensity', 0.5),
                duration=item.get('duration'),
                phoneme_ids=item.get('phoneme_ids'),
                durations=item.get('durations')
            )
            
            # Filter by duration if available
            if sample.duration is not None:
                if sample.duration < self.min_duration or sample.duration > self.max_duration:
                    continue
            
            samples.append(sample)
        
        print(f"Loaded {len(samples)} samples from {self.manifest_path}")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns dict with:
            - phoneme_ids: (seq_len,) tensor
            - mel: (n_mels, frames) tensor
            - f0: (frames,) tensor
            - energy: (frames,) tensor
            - voiced_mask: (frames,) tensor
            - durations: (seq_len,) tensor (if available)
            - speaker_id, accent_id, emotion_id, emotion_intensity
        """
        # Check cache
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]
        
        sample = self.samples[idx]
        
        # Process text
        if sample.phoneme_ids is not None:
            phoneme_ids = sample.phoneme_ids
        elif self.text_processor is not None:
            result = self.text_processor.process(sample.text)
            phoneme_ids = result['phoneme_ids']
        else:
            raise ValueError("No phoneme_ids and no text_processor provided")
        
        # Process audio
        if self.audio_processor is not None:
            audio_features = self.audio_processor.process_file(sample.audio_path)
            mel = audio_features['mel']
            f0 = audio_features['f0']
            energy = audio_features['energy']
            voiced_mask = audio_features['voiced_mask']
        else:
            # Return placeholder if no audio processor
            mel = np.zeros((80, 100), dtype=np.float32)
            f0 = np.zeros(100, dtype=np.float32)
            energy = np.zeros(100, dtype=np.float32)
            voiced_mask = np.zeros(100, dtype=np.float32)
        
        # Get durations (from alignment or placeholder)
        if sample.durations is not None:
            durations = np.array(sample.durations, dtype=np.float32)
        else:
            # Placeholder: uniform distribution
            n_frames = mel.shape[1]
            n_phonemes = len(phoneme_ids)
            if n_phonemes > 0:
                base_dur = n_frames // n_phonemes
                durations = np.full(n_phonemes, base_dur, dtype=np.float32)
                remainder = n_frames - int(durations.sum())
                for i in range(abs(remainder)):
                    durations[i % n_phonemes] += 1 if remainder > 0 else -1
            else:
                durations = np.array([n_frames], dtype=np.float32)
        
        result = {
            'phoneme_ids': torch.tensor(phoneme_ids, dtype=torch.long),
            'mel': torch.tensor(mel, dtype=torch.float32),
            'f0': torch.tensor(f0, dtype=torch.float32),
            'energy': torch.tensor(energy, dtype=torch.float32),
            'voiced_mask': torch.tensor(voiced_mask, dtype=torch.bool),
            'durations': torch.tensor(durations, dtype=torch.float32),
            'speaker_id': torch.tensor(sample.speaker_id, dtype=torch.long),
            'accent_id': torch.tensor(sample.accent_id, dtype=torch.long),
            'emotion_id': torch.tensor(sample.emotion_id, dtype=torch.long),
            'emotion_intensity': torch.tensor(sample.emotion_intensity, dtype=torch.float32),
            'text': sample.text,  # For debugging
        }
        
        # Cache if enabled
        if self._cache is not None:
            self._cache[idx] = result
        
        return result


def tts_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
    """
    Collate function for TTS batches.
    
    Pads sequences to max length in batch.
    """
    # Find max lengths
    max_phoneme_len = max(item['phoneme_ids'].shape[0] for item in batch)
    max_frames = max(item['mel'].shape[1] for item in batch)
    
    batch_size = len(batch)
    n_mels = batch[0]['mel'].shape[0]
    
    # Initialize tensors
    phoneme_ids = torch.zeros(batch_size, max_phoneme_len, dtype=torch.long)
    phoneme_mask = torch.ones(batch_size, max_phoneme_len, dtype=torch.bool)  # True = padding
    mel = torch.zeros(batch_size, n_mels, max_frames)
    f0 = torch.zeros(batch_size, max_frames)
    energy = torch.zeros(batch_size, max_frames)
    voiced_mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)
    durations = torch.zeros(batch_size, max_phoneme_len)
    frame_mask = torch.ones(batch_size, max_frames, dtype=torch.bool)  # True = padding
    
    speaker_ids = torch.zeros(batch_size, dtype=torch.long)
    accent_ids = torch.zeros(batch_size, dtype=torch.long)
    emotion_ids = torch.zeros(batch_size, dtype=torch.long)
    emotion_intensities = torch.zeros(batch_size)
    
    # Fill tensors
    for i, item in enumerate(batch):
        p_len = item['phoneme_ids'].shape[0]
        f_len = item['mel'].shape[1]
        
        phoneme_ids[i, :p_len] = item['phoneme_ids']
        phoneme_mask[i, :p_len] = False
        
        mel[i, :, :f_len] = item['mel']
        f0[i, :f_len] = item['f0']
        energy[i, :f_len] = item['energy']
        voiced_mask[i, :f_len] = item['voiced_mask']
        frame_mask[i, :f_len] = False
        
        d_len = min(item['durations'].shape[0], max_phoneme_len)
        durations[i, :d_len] = item['durations'][:d_len]
        
        speaker_ids[i] = item['speaker_id']
        accent_ids[i] = item['accent_id']
        emotion_ids[i] = item['emotion_id']
        emotion_intensities[i] = item['emotion_intensity']
    
    return {
        'phoneme_ids': phoneme_ids,
        'phoneme_mask': phoneme_mask,
        'mel': mel,
        'f0': f0,
        'energy': energy,
        'voiced_mask': voiced_mask,
        'durations': durations,
        'frame_mask': frame_mask,
        'speaker_id': speaker_ids,
        'accent_id': accent_ids,
        'emotion_id': emotion_ids,
        'emotion_intensity': emotion_intensities,
    }


def create_tts_dataloader(
    manifest_path: str,
    audio_processor=None,
    text_processor=None,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataset_kwargs
) -> DataLoader:
    """Create a DataLoader for TTS training."""
    dataset = TTSDataset(
        manifest_path=manifest_path,
        audio_processor=audio_processor,
        text_processor=text_processor,
        **dataset_kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=tts_collate_fn,
        pin_memory=True
    )
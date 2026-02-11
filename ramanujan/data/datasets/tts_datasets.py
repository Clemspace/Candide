"""
TTS Dataset and DataLoader
==========================

PyTorch dataset for loading preprocessed TTS data.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class TTSDataset(Dataset):
    """
    Dataset for TTS training.
    
    Loads preprocessed .npz files containing:
        - mel: (n_mels, T) log mel spectrogram
        - f0: (T,) F0 contour
        - voiced: (T,) voiced mask
        - energy: (T,) energy contour
        - phoneme_ids: (S,) phoneme indices
        - durations: (S,) duration per phoneme in frames
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_frames: int = 1000,
        max_phonemes: int = 200,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_frames = max_frames
        self.max_phonemes = max_phonemes
        
        # Load metadata
        metadata_path = self.data_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            self.samples = [
                s for s in metadata['samples']
                if s['split'] == split
            ]
        else:
            # Fallback: list files directly
            split_dir = self.data_dir / split
            self.samples = [
                {'name': f.stem, 'split': split}
                for f in split_dir.glob('*.npz')
            ]
        
        print(f"TTSDataset [{split}]: {len(self.samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]
        split = sample_info.get('split', self.split)
        path = self.data_dir / split / f"{sample_info['name']}.npz"
        
        data = np.load(path, allow_pickle=True)
        
        # Load arrays
        mel = data['mel']  # (n_mels, T)
        f0 = data['f0']    # (T,)
        energy = data['energy']  # (T,)
        voiced = data['voiced']  # (T,)
        phoneme_ids = data['phoneme_ids']  # (S,)
        durations = data['durations']  # (S,)
        
        # Truncate if too long
        n_frames = mel.shape[1]
        if n_frames > self.max_frames:
            # Find a good cut point (at phoneme boundary)
            cumsum = np.cumsum(durations)
            cut_phone = np.searchsorted(cumsum, self.max_frames)
            if cut_phone < len(durations):
                cut_frame = int(cumsum[cut_phone - 1]) if cut_phone > 0 else self.max_frames
                
                mel = mel[:, :cut_frame]
                f0 = f0[:cut_frame]
                energy = energy[:cut_frame]
                voiced = voiced[:cut_frame]
                phoneme_ids = phoneme_ids[:cut_phone]
                durations = durations[:cut_phone]
                # Adjust last duration
                if len(durations) > 0:
                    durations[-1] = cut_frame - (int(cumsum[cut_phone - 2]) if cut_phone > 1 else 0)
        
        # Truncate phonemes if needed
        if len(phoneme_ids) > self.max_phonemes:
            phoneme_ids = phoneme_ids[:self.max_phonemes]
            durations = durations[:self.max_phonemes]
            # Recalculate frame count
            n_frames = int(np.sum(durations))
            mel = mel[:, :n_frames]
            f0 = f0[:n_frames]
            energy = energy[:n_frames]
            voiced = voiced[:n_frames]
        
        return {
            'mel': torch.from_numpy(mel).float(),           # (n_mels, T)
            'f0': torch.from_numpy(f0).float(),             # (T,)
            'energy': torch.from_numpy(energy).float(),     # (T,)
            'voiced': torch.from_numpy(voiced).bool(),      # (T,)
            'phoneme_ids': torch.from_numpy(phoneme_ids).long(),  # (S,)
            'durations': torch.from_numpy(durations).float(),     # (S,)
        }


def collate_tts_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for TTS batches.
    
    Pads sequences to max length in batch.
    Returns attention masks for variable-length handling.
    """
    batch_size = len(batch)
    
    # Get max lengths
    max_frames = max(b['mel'].shape[1] for b in batch)
    max_phones = max(b['phoneme_ids'].shape[0] for b in batch)
    n_mels = batch[0]['mel'].shape[0]
    
    # Initialize padded tensors
    mel = torch.zeros(batch_size, n_mels, max_frames)
    f0 = torch.zeros(batch_size, max_frames)
    energy = torch.zeros(batch_size, max_frames)
    voiced = torch.zeros(batch_size, max_frames, dtype=torch.bool)
    phoneme_ids = torch.zeros(batch_size, max_phones, dtype=torch.long)
    durations = torch.zeros(batch_size, max_phones)
    
    # Masks
    frame_mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)
    phone_mask = torch.zeros(batch_size, max_phones, dtype=torch.bool)
    
    # Lengths
    frame_lengths = torch.zeros(batch_size, dtype=torch.long)
    phone_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill tensors
    for i, b in enumerate(batch):
        n_frames = b['mel'].shape[1]
        n_phones = b['phoneme_ids'].shape[0]
        
        mel[i, :, :n_frames] = b['mel']
        f0[i, :n_frames] = b['f0']
        energy[i, :n_frames] = b['energy']
        voiced[i, :n_frames] = b['voiced']
        phoneme_ids[i, :n_phones] = b['phoneme_ids']
        durations[i, :n_phones] = b['durations']
        
        frame_mask[i, :n_frames] = True
        phone_mask[i, :n_phones] = True
        
        frame_lengths[i] = n_frames
        phone_lengths[i] = n_phones
    
    return {
        'mel': mel,                    # (B, n_mels, T)
        'f0': f0,                      # (B, T)
        'energy': energy,              # (B, T)
        'voiced': voiced,              # (B, T)
        'phoneme_ids': phoneme_ids,    # (B, S)
        'durations': durations,        # (B, S)
        'frame_mask': frame_mask,      # (B, T)
        'phone_mask': phone_mask,      # (B, S)
        'frame_lengths': frame_lengths,  # (B,)
        'phone_lengths': phone_lengths,  # (B,)
    }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    max_frames: int = 1000,
    max_phonemes: int = 200,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    train_dataset = TTSDataset(
        data_dir=data_dir,
        split='train',
        max_frames=max_frames,
        max_phonemes=max_phonemes,
    )
    
    val_dataset = TTSDataset(
        data_dir=data_dir,
        split='val',
        max_frames=max_frames,
        max_phonemes=max_phonemes,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_tts_batch,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_tts_batch,
        pin_memory=True,
    )
    
    return train_loader, val_loader


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    # Test with dummy data
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy data
        train_dir = Path(tmpdir) / 'train'
        train_dir.mkdir()
        
        for i in range(5):
            n_frames = random.randint(100, 500)
            n_phones = random.randint(10, 50)
            
            np.savez(
                train_dir / f'sample_{i:05d}.npz',
                mel=np.random.randn(80, n_frames).astype(np.float32),
                f0=np.random.rand(n_frames).astype(np.float32) * 200 + 100,
                energy=np.random.rand(n_frames).astype(np.float32),
                voiced=(np.random.rand(n_frames) > 0.3),
                phoneme_ids=np.random.randint(0, 40, n_phones),
                durations=np.full(n_phones, n_frames // n_phones),
            )
        
        # Test dataset
        dataset = TTSDataset(tmpdir, split='train')
        print(f"Dataset length: {len(dataset)}")
        
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Mel shape: {sample['mel'].shape}")
        print(f"Phoneme IDs shape: {sample['phoneme_ids'].shape}")
        
        # Test dataloader
        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate_tts_batch
        )
        
        batch = next(iter(loader))
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Batch mel shape: {batch['mel'].shape}")
        print(f"Batch phoneme_ids shape: {batch['phoneme_ids'].shape}")
        print(f"Frame lengths: {batch['frame_lengths']}")
        
        print("\n✓ Dataset test passed!")
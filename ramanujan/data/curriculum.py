"""
Curriculum learning utilities for Ramanujan Transformer.

This module provides curriculum learning strategies:
- ConfidenceBasedSampler: Sample based on model confidence
- DifficultyRankedDataset: Rank samples by difficulty
- CurriculumDataLoader: Progressive difficulty increase

Example:
    >>> from ramanujan.data import ConfidenceBasedSampler
    >>> 
    >>> sampler = ConfidenceBasedSampler(
        ...     dataset=train_dataset,
    ...     initial_keep_prob=0.5,
    ...     final_keep_prob=1.0
    ... )
    >>> 
    >>> loader = DataLoader(dataset, batch_sampler=sampler)
"""

import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from typing import Iterator, List, Optional, Tuple
import numpy as np


# ============================================================================
# CONFIDENCE-BASED SAMPLER
# ============================================================================

class ConfidenceBasedSampler(Sampler):
    """
    Sample based on model confidence/difficulty.
    
    Implements curriculum learning by gradually including harder examples.
    Starts with high-confidence examples and progressively adds more
    difficult ones.
    
    Args:
        dataset: Dataset to sample from
        initial_keep_prob: Initial probability of keeping samples (0-1)
        final_keep_prob: Final probability of keeping samples (0-1)
        num_epochs: Number of epochs to reach final_keep_prob
        batch_size: Batch size
        difficulties: Optional pre-computed difficulties per sample
    
    Example:
        >>> sampler = ConfidenceBasedSampler(
        ...     dataset=train_dataset,
        ...     initial_keep_prob=0.3,
        ...     final_keep_prob=1.0,
        ...     num_epochs=10,
        ...     batch_size=32
        ... )
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """
    
    def __init__(
        self,
        dataset: Dataset,
        initial_keep_prob: float = 0.5,
        final_keep_prob: float = 1.0,
        num_epochs: int = 10,
        batch_size: int = 32,
        difficulties: Optional[np.ndarray] = None
    ):
        self.dataset = dataset
        self.initial_keep_prob = initial_keep_prob
        self.final_keep_prob = final_keep_prob
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        # Store difficulties (lower = easier)
        if difficulties is not None:
            self.difficulties = difficulties
        else:
            # Initialize with random difficulties
            self.difficulties = np.random.rand(len(dataset))
        
        # Current epoch
        self.current_epoch = 0
        
        # Sort indices by difficulty (easy to hard)
        self.sorted_indices = np.argsort(self.difficulties)
    
    def set_epoch(self, epoch: int):
        """Set current epoch for curriculum progression."""
        self.current_epoch = epoch
    
    def _get_current_keep_prob(self) -> float:
        """Compute current keep probability based on epoch."""
        if self.current_epoch >= self.num_epochs:
            return self.final_keep_prob
        
        # Linear interpolation
        progress = self.current_epoch / self.num_epochs
        keep_prob = (
            self.initial_keep_prob + 
            (self.final_keep_prob - self.initial_keep_prob) * progress
        )
        
        return keep_prob
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches with curriculum sampling."""
        keep_prob = self._get_current_keep_prob()
        
        # Determine how many samples to keep
        num_keep = int(len(self.dataset) * keep_prob)
        
        # Keep easiest samples
        active_indices = self.sorted_indices[:num_keep].copy()
        
        # Shuffle active indices
        np.random.shuffle(active_indices)
        
        # Generate batches
        for i in range(0, len(active_indices), self.batch_size):
            batch = active_indices[i:i + self.batch_size]
            if len(batch) == self.batch_size:  # Only full batches
                yield batch.tolist()
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        keep_prob = self._get_current_keep_prob()
        num_keep = int(len(self.dataset) * keep_prob)
        return num_keep // self.batch_size


# ============================================================================
# DIFFICULTY COMPUTATION
# ============================================================================

def compute_sample_difficulties(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 32
) -> np.ndarray:
    """
    Compute difficulty scores for all samples in dataset.
    
    Difficulty is measured by model loss/perplexity.
    Higher loss = more difficult sample.
    
    Args:
        model: Trained model
        dataset: Dataset to evaluate
        device: Device for computation
        batch_size: Batch size for evaluation
    
    Returns:
        Array of difficulty scores (one per sample)
    
    Example:
        >>> difficulties = compute_sample_difficulties(model, dataset, device)
        >>> print(f"Easiest sample: {difficulties.argmin()}")
        >>> print(f"Hardest sample: {difficulties.argmax()}")
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    difficulties = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Compute loss per sample
            targets = input_ids[:, 1:]
            logits = logits[:, :-1, :]
            
            # Cross entropy per sample
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction='none'
            )
            
            # Average loss per sample
            loss = loss.view(input_ids.size(0), -1).mean(dim=1)
            
            difficulties.extend(loss.cpu().numpy())
    
    return np.array(difficulties)


# ============================================================================
# CURRICULUM DATA LOADER
# ============================================================================

class CurriculumDataLoader:
    """
    DataLoader with curriculum learning.
    
    Wraps a dataset and progressively increases difficulty over training.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        initial_keep_prob: Initial fraction of data to use
        final_keep_prob: Final fraction of data to use
        num_epochs: Number of epochs for curriculum
        difficulties: Optional pre-computed difficulties
        num_workers: Number of data loading workers
    
    Example:
        >>> loader = CurriculumDataLoader(
        ...     dataset=train_dataset,
        ...     batch_size=32,
        ...     initial_keep_prob=0.3,
        ...     num_epochs=10
        ... )
        >>> 
        >>> for epoch in range(10):
        ...     loader.set_epoch(epoch)
        ...     for batch in loader:
        ...         train_step(batch)
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        initial_keep_prob: float = 0.5,
        final_keep_prob: float = 1.0,
        num_epochs: int = 10,
        difficulties: Optional[np.ndarray] = None,
        num_workers: int = 0
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create sampler
        self.sampler = ConfidenceBasedSampler(
            dataset=dataset,
            initial_keep_prob=initial_keep_prob,
            final_keep_prob=final_keep_prob,
            num_epochs=num_epochs,
            batch_size=batch_size,
            difficulties=difficulties
        )
        
        # Create base loader
        self.loader = DataLoader(
            dataset,
            batch_sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def set_epoch(self, epoch: int):
        """Set current epoch for curriculum progression."""
        self.sampler.set_epoch(epoch)
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.loader)
    
    def __len__(self):
        """Number of batches."""
        return len(self.sampler)


# ============================================================================
# ADAPTIVE SAMPLING
# ============================================================================

class AdaptiveDifficultySampler(Sampler):
    """
    Adaptively sample based on recent model performance.
    
    Focuses on samples where the model is struggling,
    updating sampling probabilities based on recent losses.
    
    Args:
        dataset: Dataset to sample from
        batch_size: Batch size
        update_freq: How often to update sampling probabilities
        temperature: Temperature for softmax sampling (higher = more uniform)
    
    Example:
        >>> sampler = AdaptiveDifficultySampler(
        ...     dataset=train_dataset,
        ...     batch_size=32,
        ...     update_freq=100
        ... )
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        update_freq: int = 100,
        temperature: float = 1.0
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.temperature = temperature
        
        # Initialize uniform sampling probabilities
        self.sampling_probs = np.ones(len(dataset)) / len(dataset)
        
        # Track recent losses per sample
        self.recent_losses = np.zeros(len(dataset))
        self.loss_counts = np.zeros(len(dataset))
        
        self.step = 0
    
    def update_difficulties(self, indices: List[int], losses: np.ndarray):
        """
        Update difficulties based on recent losses.
        
        Args:
            indices: Sample indices
            losses: Loss values for samples
        """
        indices = np.array(indices)
        
        # Update recent losses
        self.recent_losses[indices] = (
            self.recent_losses[indices] * 0.9 + losses * 0.1
        )
        self.loss_counts[indices] += 1
        
        # Update sampling probabilities periodically
        if self.step % self.update_freq == 0:
            self._update_sampling_probs()
        
        self.step += 1
    
    def _update_sampling_probs(self):
        """Update sampling probabilities based on losses."""
        # Mask for samples that have been seen
        seen_mask = self.loss_counts > 0
        
        if seen_mask.sum() == 0:
            return
        
        # Compute probabilities (higher loss = higher probability)
        probs = np.zeros(len(self.dataset))
        probs[seen_mask] = self.recent_losses[seen_mask]
        
        # Apply temperature
        probs = probs / self.temperature
        
        # Softmax
        probs = np.exp(probs - probs.max())
        probs = probs / probs.sum()
        
        # Mix with uniform (exploration)
        uniform = np.ones(len(self.dataset)) / len(self.dataset)
        self.sampling_probs = 0.9 * probs + 0.1 * uniform
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches with adaptive sampling."""
        # Sample indices based on probabilities
        num_batches = len(self.dataset) // self.batch_size
        
        for _ in range(num_batches):
            indices = np.random.choice(
                len(self.dataset),
                size=self.batch_size,
                replace=False,
                p=self.sampling_probs
            )
            yield indices.tolist()
    
    def __len__(self) -> int:
        """Number of batches."""
        return len(self.dataset) // self.batch_size


# ============================================================================
# UTILITIES
# ============================================================================

def create_curriculum_schedule(
    num_epochs: int,
    initial_keep: float = 0.3,
    final_keep: float = 1.0,
    schedule_type: str = 'linear'
) -> List[float]:
    """
    Create curriculum schedule.
    
    Args:
        num_epochs: Number of epochs
        initial_keep: Initial keep probability
        final_keep: Final keep probability
        schedule_type: Type of schedule ('linear', 'exponential', 'step')
    
    Returns:
        List of keep probabilities for each epoch
    
    Example:
        >>> schedule = create_curriculum_schedule(10, 0.3, 1.0, 'linear')
        >>> print(schedule)  # [0.3, 0.37, 0.44, ..., 1.0]
    """
    if schedule_type == 'linear':
        return [
            initial_keep + (final_keep - initial_keep) * (i / (num_epochs - 1))
            for i in range(num_epochs)
        ]
    
    elif schedule_type == 'exponential':
        # Exponential growth
        alpha = np.log(final_keep / initial_keep) / (num_epochs - 1)
        return [
            initial_keep * np.exp(alpha * i)
            for i in range(num_epochs)
        ]
    
    elif schedule_type == 'step':
        # Step increases every num_epochs // 4
        step_size = (final_keep - initial_keep) / 4
        return [
            initial_keep + step_size * (i // (num_epochs // 4))
            for i in range(num_epochs)
        ]
    
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")


def visualize_curriculum_schedule(
    schedule: List[float],
    save_path: Optional[str] = None
):
    """
    Visualize curriculum schedule.
    
    Args:
        schedule: List of keep probabilities
        save_path: Optional path to save plot
    
    Example:
        >>> schedule = create_curriculum_schedule(10)
        >>> visualize_curriculum_schedule(schedule, 'curriculum.png')
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(schedule, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Keep Probability')
        plt.title('Curriculum Learning Schedule')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Schedule plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing curriculum.py module")
    print("="*70)
    
    # Create dummy dataset
    from torch.utils.data import TensorDataset
    
    input_ids = torch.randint(0, 1000, (1000, 128))
    dummy_dataset = TensorDataset(input_ids)
    
    # Convert to dict format
    class DictDataset:
        def __init__(self, tensor_dataset):
            self.data = tensor_dataset
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return {'input_ids': self.data[idx][0]}
    
    dataset = DictDataset(dummy_dataset)
    
    # Test ConfidenceBasedSampler
    print("\n1. Testing ConfidenceBasedSampler...")
    
    # Create difficulties
    difficulties = np.random.rand(len(dataset))
    
    sampler = ConfidenceBasedSampler(
        dataset=dataset,
        initial_keep_prob=0.3,
        final_keep_prob=1.0,
        num_epochs=10,
        batch_size=32,
        difficulties=difficulties
    )
    
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Batch size: {sampler.batch_size}")
    
    # Test different epochs
    for epoch in [0, 5, 10]:
        sampler.set_epoch(epoch)
        num_batches = len(sampler)
        keep_prob = sampler._get_current_keep_prob()
        print(f"   Epoch {epoch}: {num_batches} batches, keep_prob={keep_prob:.2f}")
    
    print(f"   ✅ ConfidenceBasedSampler working!")
    
    # Test iteration
    print("\n2. Testing sampler iteration...")
    sampler.set_epoch(0)
    batches = list(sampler)
    
    print(f"   Generated {len(batches)} batches")
    print(f"   First batch size: {len(batches[0])}")
    print(f"   ✅ Sampler iteration working!")
    
    # Test CurriculumDataLoader
    print("\n3. Testing CurriculumDataLoader...")
    
    loader = CurriculumDataLoader(
        dataset=dataset,
        batch_size=32,
        initial_keep_prob=0.3,
        final_keep_prob=1.0,
        num_epochs=10,
        difficulties=difficulties
    )
    
    # Test epoch progression
    for epoch in [0, 5, 10]:
        loader.set_epoch(epoch)
        print(f"   Epoch {epoch}: {len(loader)} batches")
    
    print(f"   ✅ CurriculumDataLoader working!")
    
    # Test AdaptiveDifficultySampler
    print("\n4. Testing AdaptiveDifficultySampler...")
    
    adaptive_sampler = AdaptiveDifficultySampler(
        dataset=dataset,
        batch_size=32,
        update_freq=10
    )
    
    # Simulate training
    batches = list(adaptive_sampler)
    print(f"   Generated {len(batches)} batches")
    
    # Update difficulties
    sample_indices = batches[0]
    sample_losses = np.random.rand(len(sample_indices))
    adaptive_sampler.update_difficulties(sample_indices, sample_losses)
    
    print(f"   Updated difficulties for {len(sample_indices)} samples")
    print(f"   ✅ AdaptiveDifficultySampler working!")
    
    # Test create_curriculum_schedule
    print("\n5. Testing create_curriculum_schedule...")
    
    schedule_linear = create_curriculum_schedule(10, 0.3, 1.0, 'linear')
    schedule_exp = create_curriculum_schedule(10, 0.3, 1.0, 'exponential')
    schedule_step = create_curriculum_schedule(10, 0.3, 1.0, 'step')
    
    print(f"   Linear schedule: {[f'{x:.2f}' for x in schedule_linear]}")
    print(f"   Exponential schedule: {[f'{x:.2f}' for x in schedule_exp]}")
    print(f"   Step schedule: {[f'{x:.2f}' for x in schedule_step]}")
    print(f"   ✅ create_curriculum_schedule working!")
    
    # Test with actual DataLoader
    print("\n6. Testing with DataLoader...")
    
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    batch = next(iter(loader))
    
    print(f"   Batch keys: {batch.keys()}")
    print(f"   Batch shape: {batch['input_ids'].shape}")
    print(f"   ✅ DataLoader integration working!")
    
    # Test visualization (if matplotlib available)
    print("\n7. Testing visualize_curriculum_schedule...")
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        visualize_curriculum_schedule(schedule_linear)
        print(f"   ✅ Visualization working!")
    except ImportError:
        print(f"   ⚠️  Matplotlib not available, skipping visualization")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    print("\nModule ready for use. Import with:")
    print("  from ramanujan.data.curriculum import ConfidenceBasedSampler")
    print("  from ramanujan.data.curriculum import CurriculumDataLoader")
    print("="*70)
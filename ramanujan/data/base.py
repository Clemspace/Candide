"""Base classes and protocols for data module."""

from typing import Protocol, Dict, Any, List, Optional, Callable, Union
from abc import ABC, abstractmethod
import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset


class DatasetProtocol(Protocol):
    """
    Protocol defining the interface all datasets must implement.
    
    This allows for type checking and ensures consistency across
    text, vision, audio, and multimodal datasets.
    """
    
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        ...
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary with modality-specific keys:
            - Text: {'input_ids': Tensor, 'labels': Tensor, 'attention_mask': Tensor}
            - Vision: {'images': Tensor, 'labels': Tensor}
            - Diffusion: {'images': Tensor, 'noise': Tensor, 'timesteps': Tensor}
        """
        ...
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        """
        Collate a batch of items.
        
        Args:
            batch: List of dictionaries from __getitem__
        
        Returns:
            Batched dictionary with stacked tensors
        """
        ...


class BaseDataset(ABC, TorchDataset):
    """
    Abstract base class for all datasets.
    
    Provides common functionality:
    - Transform pipeline
    - Metadata tracking
    - Consistent interface
    
    All datasets return Dict[str, Tensor] for consistency across modalities.
    
    Example:
        >>> class MyDataset(BaseDataset):
        ...     def __len__(self):
        ...         return 1000
        ...     
        ...     def __getitem__(self, idx):
        ...         return {'input_ids': torch.tensor([1, 2, 3])}
        ...     
        ...     def collate_fn(self, batch):
        ...         return {'input_ids': torch.stack([b['input_ids'] for b in batch])}
    """
    
    def __init__(
        self,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize base dataset.
        
        Args:
            transform: Optional transform to apply to each item
            cache_dir: Directory for caching processed data
            **kwargs: Additional arguments for subclasses
        """
        self.transform = transform
        self.cache_dir = cache_dir
        self.metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.
        
        Subclasses must implement this method.
        """
        pass
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        """
        Default collate function.
        
        Stacks all tensors in the batch. Subclasses can override
        for more sophisticated collation (e.g., padding).
        
        Args:
            batch: List of dictionaries from __getitem__
        
        Returns:
            Batched dictionary with stacked tensors
        """
        if not batch:
            return {}
        
        # Get all keys from first item
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            values = [item[key] for item in batch]
            
            # Stack if all are tensors
            if all(isinstance(v, Tensor) for v in values):
                collated[key] = torch.stack(values)
            else:
                # Keep as list if not all tensors
                collated[key] = values
        
        return collated
    
    def _apply_transform(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transform if provided."""
        if self.transform is not None:
            return self.transform(item)
        return item
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        return {
            'size': len(self),
            'type': self.__class__.__name__,
            **self.metadata
        }


class StreamingDataset(BaseDataset):
    """
    Base class for streaming datasets.
    
    Streaming datasets:
    - Don't load all data into memory
    - Can be infinite (keep iterating)
    - Used for large-scale training
    
    Example:
        >>> dataset = StreamingDataset(estimated_size=1_000_000)
        >>> len(dataset)  # Returns estimate, not exact
        1000000
    """
    
    def __init__(
        self,
        estimated_size: int = 10_000_000,
        **kwargs
    ):
        """
        Initialize streaming dataset.
        
        Args:
            estimated_size: Estimated number of samples (for progress bars)
            **kwargs: Arguments passed to BaseDataset
        """
        super().__init__(**kwargs)
        self._length = estimated_size
        self.metadata['streaming'] = True
        self.metadata['estimated_size'] = estimated_size
    
    def __len__(self) -> int:
        """Return estimated size."""
        return self._length
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item at index.
        
        For streaming datasets, idx might be ignored or used
        as a seed for random sampling.
        """
        pass


class TokenizedDataset(BaseDataset):
    """
    Base class for datasets that work with tokenized text.
    
    Provides common functionality for text datasets:
    - Tokenization tracking
    - Sequence length management
    - Padding/truncation
    
    Example:
        >>> dataset = TokenizedDataset(
        ...     max_length=1024,
        ...     pad_token_id=0
        ... )
    """
    
    def __init__(
        self,
        max_length: int = 1024,
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        stride: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize tokenized dataset.
        
        Args:
            max_length: Maximum sequence length
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            bos_token_id: Beginning-of-sequence token ID
            stride: Stride for sliding window chunking
            **kwargs: Arguments passed to BaseDataset
        """
        super().__init__(**kwargs)
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.stride = stride or max_length
        
        self.metadata.update({
            'max_length': max_length,
            'pad_token_id': pad_token_id,
            'stride': self.stride
        })
    
    def _pad_sequence(
        self,
        sequence: List[int],
        pad_to_length: Optional[int] = None
    ) -> List[int]:
        """
        Pad sequence to target length.
        
        Args:
            sequence: List of token IDs
            pad_to_length: Target length (default: self.max_length)
        
        Returns:
            Padded sequence
        """
        target_length = pad_to_length or self.max_length
        
        if len(sequence) >= target_length:
            return sequence[:target_length]
        
        return sequence + [self.pad_token_id] * (target_length - len(sequence))
    
    def _truncate_sequence(
        self,
        sequence: List[int],
        max_length: Optional[int] = None
    ) -> List[int]:
        """
        Truncate sequence to maximum length.
        
        Args:
            sequence: List of token IDs
            max_length: Maximum length (default: self.max_length)
        
        Returns:
            Truncated sequence
        """
        max_len = max_length or self.max_length
        return sequence[:max_len]
    
    def _chunk_sequence(
        self,
        sequence: List[int],
        chunk_size: Optional[int] = None,
        stride: Optional[int] = None,
        min_chunk_size: int = 32
    ) -> List[List[int]]:
        """
        Split sequence into chunks with sliding window.
        
        Args:
            sequence: List of token IDs
            chunk_size: Size of each chunk (default: self.max_length)
            stride: Stride between chunks (default: self.stride)
            min_chunk_size: Minimum chunk size to keep
        
        Returns:
            List of chunks
        """
        chunk_size = chunk_size or self.max_length
        stride = stride or self.stride
        
        chunks = []
        for i in range(0, len(sequence), stride):
            chunk = sequence[i:i + chunk_size]
            if len(chunk) >= min_chunk_size:
                chunks.append(chunk)
        
        return chunks


class DatasetConfig:
    """
    Configuration for dataset creation.
    
    Standardizes dataset configuration across different types.
    
    Example:
        >>> config = DatasetConfig(
        ...     name='wikitext',
        ...     split='train',
        ...     max_length=1024,
        ...     batch_size=32
        ... )
    """
    
    def __init__(
        self,
        name: str,
        split: str = 'train',
        max_length: int = 1024,
        batch_size: int = 32,
        num_workers: int = 4,
        streaming: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize dataset config.
        
        Args:
            name: Dataset name
            split: Dataset split (train/val/test)
            max_length: Maximum sequence length
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            streaming: Whether to use streaming
            cache_dir: Cache directory
            **kwargs: Additional dataset-specific arguments
        """
        self.name = name
        self.split = split
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.streaming = streaming
        self.cache_dir = cache_dir
        self.extra = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'split': self.split,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'streaming': self.streaming,
            'cache_dir': self.cache_dir,
            **self.extra
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DatasetConfig':
        """Create from dictionary."""
        return cls(**config_dict)
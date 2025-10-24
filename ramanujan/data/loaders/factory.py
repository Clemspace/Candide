"""DataLoader factory and creation utilities."""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from ..base import BaseDataset, DatasetConfig
from ..datasets.text import TextDataset, StreamingTextDataset
from .collation import get_collate_fn, text_collate


class WikiTextLoader:
    """
    WikiText dataset loader.
    
    Handles loading and preparing WikiText datasets (wikitext-2, wikitext-103).
    
    Example:
        >>> loader = WikiTextLoader(
        ...     dataset_name="wikitext-2-raw-v1",
        ...     tokenizer=my_tokenizer,
        ...     sequence_length=1024
        ... )
        >>> train_loader, val_loader = loader.get_dataloaders(batch_size=32)
    """
    
    def __init__(
        self,
        dataset_name: str = "wikitext-2-raw-v1",
        tokenizer = None,
        sequence_length: int = 1024,
        cache_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize WikiText loader.
        
        Args:
            dataset_name: WikiText dataset name
                          ('wikitext-2-raw-v1', 'wikitext-103-raw-v1')
            tokenizer: Tokenizer with encode() method
            sequence_length: Maximum sequence length
            cache_dir: Cache directory for datasets
            verbose: Print loading information
        """
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.cache_dir = cache_dir
        self.verbose = verbose
        
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        
        if verbose:
            print(f"📚 Loading {dataset_name}...")
            print(f"   Dataset: Salesforce/wikitext")
            print(f"   Config: {dataset_name}")
        
        # Import datasets
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets not installed. "
                "Install with: pip install datasets"
            )
        
        # Load dataset
        dataset = load_dataset(
            "Salesforce/wikitext",
            dataset_name,
            cache_dir=cache_dir,
        )
        
        if verbose:
            print(f"✅ Dataset loaded successfully:")
            print(f"   Train: {len(dataset['train'])} examples")
            print(f"   Validation: {len(dataset['validation'])} examples")
            print(f"   Test: {len(dataset['test'])} examples")
        
        self.train_data = dataset['train']
        self.val_data = dataset['validation']
        self.test_data = dataset['test']
    
    def get_dataloaders(
        self,
        batch_size: int = 8,
        num_workers: int = 4,
        shuffle_train: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation dataloaders.
        
        Args:
            batch_size: Batch size
            num_workers: Number of worker processes
            shuffle_train: Shuffle training data
            pin_memory: Pin memory for faster GPU transfer
            drop_last: Drop last incomplete batch
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Filter empty texts
        train_texts = [
            ex['text'] for ex in self.train_data
            if len(ex['text'].strip()) > 0
        ]
        val_texts = [
            ex['text'] for ex in self.val_data
            if len(ex['text'].strip()) > 0
        ]
        
        if self.verbose:
            print(f"\n📝 Creating datasets...")
            print(f"   Train texts: {len(train_texts)}")
            print(f"   Val texts: {len(val_texts)}")
        
        # Create datasets
        train_dataset = TextDataset(
            texts=train_texts,
            tokenizer=self.tokenizer,
            max_length=self.sequence_length,
            verbose=self.verbose,
        )
        
        val_dataset = TextDataset(
            texts=val_texts,
            tokenizer=self.tokenizer,
            max_length=self.sequence_length,
            verbose=self.verbose,
        )
        
        if self.verbose:
            print(f"\n✅ Datasets created:")
            print(f"   Train sequences: {len(train_dataset)}")
            print(f"   Val sequences: {len(val_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=lambda batch: text_collate(
                batch,
                pad_token_id=self.tokenizer.pad_token_id
                if hasattr(self.tokenizer, 'pad_token_id') else 0
            ),
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=lambda batch: text_collate(
                batch,
                pad_token_id=self.tokenizer.pad_token_id
                if hasattr(self.tokenizer, 'pad_token_id') else 0
            ),
        )
        
        return train_loader, val_loader


class FineWebLoader:
    """
    FineWeb-Edu dataset loader.
    
    Loads and streams the FineWeb-Edu dataset, a high-quality web text corpus.
    
    Example:
        >>> loader = FineWebLoader(
        ...     subset="sample-10BT",
        ...     tokenizer=my_tokenizer,
        ...     streaming=True
        ... )
        >>> train_loader, val_loader = loader.get_dataloaders(batch_size=32)
    """
    
    def __init__(
        self,
        subset: str = "sample-10BT",
        tokenizer = None,
        sequence_length: int = 1024,
        streaming: bool = True,
        cache_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize FineWeb-Edu loader.
        
        Args:
            subset: Dataset subset
                    ('sample-10BT', 'sample-100BT', 'sample-350BT', 'default' for full)
            tokenizer: Tokenizer with encode() method
            sequence_length: Maximum sequence length
            streaming: Whether to stream dataset (recommended for large subsets)
            cache_dir: Cache directory
            verbose: Print loading information
        """
        self.subset = subset
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.streaming = streaming
        self.cache_dir = cache_dir
        self.verbose = verbose
        
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        
        if verbose:
            print(f"🌐 Initializing FineWeb-Edu Loader")
            print(f"   Subset: {subset}")
            print(f"   Sequence length: {sequence_length}")
            print(f"   Streaming: {streaming}")
    
    def get_dataloaders(
        self,
        batch_size: int = 8,
        num_workers: int = 4,
        shuffle_train: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation dataloaders.
        
        Args:
            batch_size: Batch size
            num_workers: Number of worker processes (0 for streaming)
            shuffle_train: Shuffle training data (disabled for streaming)
            pin_memory: Pin memory for faster GPU transfer
            drop_last: Drop last incomplete batch
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Training dataset (streaming)
        train_dataset = StreamingTextDataset(
            dataset_name="HuggingFaceFW/fineweb-edu",
            dataset_config=self.subset,
            split="train",
            tokenizer=self.tokenizer,
            max_length=self.sequence_length,
            text_column="text",
            streaming=self.streaming,
            cache_dir=self.cache_dir,
            verbose=self.verbose,
        )
        
        # Validation dataset (small streaming subset)
        val_dataset = StreamingTextDataset(
            dataset_name="HuggingFaceFW/fineweb-edu",
            dataset_config=self.subset,
            split="train",  # Use train split but sample differently
            tokenizer=self.tokenizer,
            max_length=self.sequence_length,
            text_column="text",
            streaming=True,
            cache_dir=self.cache_dir,
            verbose=False,
        )
        
        # For streaming, adjust dataloader settings
        if self.streaming:
            shuffle_train = False
            num_workers = 0
        
        # Get pad token ID
        pad_token_id = (
            self.tokenizer.pad_token_id
            if hasattr(self.tokenizer, 'pad_token_id')
            else 0
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=lambda batch: text_collate(batch, pad_token_id=pad_token_id),
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=lambda batch: text_collate(batch, pad_token_id=pad_token_id),
        )
        
        if self.verbose:
            print(f"\n✅ DataLoaders created")
        
        return train_loader, val_loader


def create_dataloader(
    dataset: BaseDataset,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_type: str = 'default',
    **collate_kwargs
) -> DataLoader:
    """
    Create a DataLoader from a dataset.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        collate_type: Type of collate function
        **collate_kwargs: Additional arguments for collate function
    
    Returns:
        DataLoader instance
    
    Example:
        >>> dataset = TextDataset(texts, tokenizer)
        >>> loader = create_dataloader(
        ...     dataset,
        ...     batch_size=32,
        ...     collate_type='text'
        ... )
    """
    # Get collate function
    collate_fn = get_collate_fn(collate_type, **collate_kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def create_dataloader_from_config(
    dataset: BaseDataset,
    config: DatasetConfig,
) -> DataLoader:
    """
    Create a DataLoader from a DatasetConfig.
    
    Args:
        dataset: Dataset instance
        config: DatasetConfig instance
    
    Returns:
        DataLoader instance
    
    Example:
        >>> config = DatasetConfig(
        ...     name='wikitext',
        ...     batch_size=32,
        ...     num_workers=4
        ... )
        >>> loader = create_dataloader_from_config(dataset, config)
    """
    return create_dataloader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(config.split == 'train'),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=(config.split == 'train'),
        collate_type='text',
    )
"""
Data loaders for various datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path


class TextDataset(Dataset):
    """Base dataset for text data."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 1024,
        stride: Optional[int] = None,
    ):
        """
        Initialize text dataset.
        
        Args:
            texts: List of text strings
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            stride: Stride for sliding window (None = max_length)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length
        
        # Tokenize all texts and create sequences
        print(f"Tokenizing {len(texts)} texts...")
        self.sequences = []
        
        for text in texts:
            # Tokenize
            token_ids = self.tokenizer.encode(text)
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            
            # Create sequences with sliding window
            for i in range(0, len(token_ids), self.stride):
                seq = token_ids[i:i + max_length]
                if len(seq) >= 32:  # Minimum sequence length
                    self.sequences.append(seq)
        
        print(f"Created {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Pad if necessary
        if len(seq) < self.max_length:
            seq = seq + [self.tokenizer.pad_token_id] * (self.max_length - len(seq))
        
        input_ids = torch.tensor(seq, dtype=torch.long)
        
        # Labels are input_ids shifted by 1
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class StreamingTextDataset(Dataset):
    """Streaming dataset for large text corpora."""
    
    def __init__(
        self,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        split: str = "train",
        tokenizer = None,
        max_length: int = 1024,
        text_column: str = "text",
        streaming: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize streaming dataset.
        
        Args:
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration/subset
            split: Dataset split
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            text_column: Name of text column
            streaming: Whether to stream dataset
            cache_dir: Cache directory
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.streaming = streaming
        
        print(f"Loading {dataset_name}...")
        if dataset_config:
            print(f"  Config: {dataset_config}")
        print(f"  Split: {split}")
        print(f"  Streaming: {streaming}")
        
        # Load dataset
        self.dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=streaming,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        
        if streaming:
            self.length = 10_000_000  # Large number for streaming
            self.dataset_iter = iter(self.dataset)
            self.buffer = []
            self.buffer_size = 100
        else:
            self.length = len(self.dataset)
            print(f"✅ Loaded {self.length:,} examples")
        
        # Token buffer for creating sequences
        self.token_buffer = []
    
    def __len__(self):
        return self.length
    
    def _get_next_example(self):
        """Get next example from dataset."""
        if self.streaming:
            # Refill buffer if empty
            if not self.buffer:
                try:
                    for _ in range(self.buffer_size):
                        example = next(self.dataset_iter)
                        self.buffer.append(example)
                except StopIteration:
                    # Restart iterator
                    self.dataset_iter = iter(self.dataset)
                    example = next(self.dataset_iter)
                    self.buffer.append(example)
            
            return self.buffer.pop(0)
        else:
            import random
            idx = random.randint(0, len(self.dataset) - 1)
            return self.dataset[idx]
    
    def __getitem__(self, idx):
        """Get a tokenized sequence."""
        # Build up token buffer until we have enough
        while len(self.token_buffer) < self.max_length:
            example = self._get_next_example()
            text = example[self.text_column]
            
            # Tokenize
            tokens = self.tokenizer.encode(text)
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            
            self.token_buffer.extend(tokens)
        
        # Extract sequence
        input_ids = self.token_buffer[:self.max_length]
        self.token_buffer = self.token_buffer[self.max_length:]
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class WikiTextLoader:
    """WikiText dataset loader."""
    
    def __init__(
        self,
        dataset_name: str = "wikitext-2-raw-v1",
        vocab_size: int = 50257,
        sequence_length: int = 1024,
        tokenizer = None,
        cache_dir: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        
        print(f"Loading {dataset_name}...")
        print(f"  Dataset: Salesforce/wikitext")
        print(f"  Config: {dataset_name}")
        
        # Load datasets
        dataset = load_dataset(
            "Salesforce/wikitext",
            dataset_name,
            cache_dir=cache_dir,
        )
        
        print(f"Dataset loaded successfully:")
        print(f"  Train: {len(dataset['train'])} examples")
        print(f"  Validation: {len(dataset['validation'])} examples")
        print(f"  Test: {len(dataset['test'])} examples")
        
        self.train_dataset = dataset['train']
        self.eval_dataset = dataset['validation']
    
    def get_dataloaders(
        self,
        batch_size: int = 8,
        num_workers: int = 4,
        shuffle_train: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders."""
        
        # Filter empty texts
        train_texts = [ex['text'] for ex in self.train_dataset if len(ex['text'].strip()) > 0]
        eval_texts = [ex['text'] for ex in self.eval_dataset if len(ex['text'].strip()) > 0]
        
        print(f"Tokenizing dataset...")
        print(f"  Tokenizing {len(train_texts)} training texts...")
        
        train_dataset = TextDataset(
            train_texts,
            self.tokenizer,
            max_length=self.sequence_length,
        )
        
        print(f"  Tokenizing {len(eval_texts)} validation texts...")
        
        eval_dataset = TextDataset(
            eval_texts,
            self.tokenizer,
            max_length=self.sequence_length,
        )
        
        print(f"Tokenization complete:")
        print(f"  Train sequences: {len(train_dataset)}")
        print(f"  Eval sequences: {len(eval_dataset)}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        
        return train_loader, eval_loader


class FineWebLoader:
    """FineWeb-Edu dataset loader."""
    
    def __init__(
        self,
        subset: str = "sample-10BT",
        sequence_length: int = 1024,
        tokenizer = None,
        streaming: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize FineWeb-Edu loader.
        
        Args:
            subset: Dataset subset (sample-10BT, sample-100BT, sample-350BT, or 'default' for full)
            sequence_length: Maximum sequence length
            tokenizer: Tokenizer instance
            streaming: Stream dataset (recommended)
            cache_dir: Cache directory
        """
        self.subset = subset
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.streaming = streaming
        self.cache_dir = cache_dir
        
        print(f"Initializing FineWeb-Edu Loader")
        print(f"  Subset: {subset}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Streaming: {streaming}")
    
    def get_dataloaders(
        self,
        batch_size: int = 8,
        num_workers: int = 4,
        shuffle_train: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders."""
        
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
        )
        
        # Validation dataset (small non-streaming subset)
        # Take first 1000 examples for validation
        eval_dataset = StreamingTextDataset(
            dataset_name="HuggingFaceFW/fineweb-edu",
            dataset_config=self.subset,
            split="train",
            tokenizer=self.tokenizer,
            max_length=self.sequence_length,
            text_column="text",
            streaming=True,
            cache_dir=self.cache_dir,
        )
        
        # For streaming, disable shuffle and num_workers
        if self.streaming:
            shuffle_train = False
            num_workers = 0
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        
        print(f"\n✅ DataLoaders created")
        
        return train_loader, eval_loader
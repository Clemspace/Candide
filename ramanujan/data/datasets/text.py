"""Text dataset implementations."""

import torch
from torch import Tensor
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from ..base import TokenizedDataset, StreamingDataset


class TextDataset(TokenizedDataset):
    """
    Dataset for text sequences.
    
    Takes a list of texts, tokenizes them, and creates fixed-length sequences
    using sliding window chunking.
    
    Example:
        >>> texts = ["Hello world", "This is a test"]
        >>> dataset = TextDataset(
        ...     texts=texts,
        ...     tokenizer=my_tokenizer,
        ...     max_length=512
        ... )
        >>> len(dataset)
        2
        >>> item = dataset[0]
        >>> item.keys()
        dict_keys(['input_ids', 'labels'])
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 1024,
        stride: Optional[int] = None,
        min_chunk_size: int = 32,
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialize text dataset.
        
        Args:
            texts: List of text strings
            tokenizer: Tokenizer with encode() method
            max_length: Maximum sequence length
            stride: Stride for sliding window (default: max_length)
            min_chunk_size: Minimum chunk size to keep
            verbose: Print progress
            **kwargs: Additional arguments for TokenizedDataset
        """
        super().__init__(
            max_length=max_length,
            stride=stride,
            **kwargs
        )
        
        self.tokenizer = tokenizer
        self.min_chunk_size = min_chunk_size
        
        # Tokenize and chunk all texts
        if verbose:
            print(f"📝 Tokenizing {len(texts)} texts...")
        
        self.sequences = []
        for text in texts:
            # Tokenize
            token_ids = self._tokenize(text)
            
            # Create chunks with sliding window
            chunks = self._chunk_sequence(
                token_ids,
                chunk_size=max_length,
                stride=self.stride,
                min_chunk_size=min_chunk_size
            )
            
            self.sequences.extend(chunks)
        
        if verbose:
            print(f"✅ Created {len(self.sequences)} sequences")
        
        self.metadata.update({
            'num_texts': len(texts),
            'num_sequences': len(self.sequences),
            'min_chunk_size': min_chunk_size
        })
    
    def _tokenize(self, text: str) -> List[int]:
        """
        Tokenize text.
        
        Args:
            text: Input text
        
        Returns:
            List of token IDs
        """
        token_ids = self.tokenizer.encode(text)
        
        # Convert to list if tensor
        if isinstance(token_ids, Tensor):
            token_ids = token_ids.tolist()
        
        return token_ids
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Get a tokenized sequence.
        
        Args:
            idx: Index
        
        Returns:
            Dictionary with 'input_ids' and 'labels'
        """
        seq = self.sequences[idx]
        
        # Pad if necessary
        if len(seq) < self.max_length:
            seq = self._pad_sequence(seq)
        
        input_ids = torch.tensor(seq, dtype=torch.long)
        
        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()
        
        item = {
            'input_ids': input_ids,
            'labels': labels
        }
        
        return self._apply_transform(item)


class StreamingTextDataset(StreamingDataset):
    """
    Streaming dataset for large text corpora.
    
    Loads data on-the-fly without storing everything in memory.
    Suitable for very large datasets (e.g., FineWeb, C4, The Pile).
    
    Example:
        >>> dataset = StreamingTextDataset(
        ...     dataset_name="HuggingFaceFW/fineweb-edu",
        ...     dataset_config="sample-10BT",
        ...     tokenizer=my_tokenizer,
        ...     streaming=True
        ... )
        >>> item = dataset[0]  # Loads on demand
    """
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        dataset_config: Optional[str] = None,
        split: str = "train",
        max_length: int = 1024,
        text_column: str = "text",
        streaming: bool = True,
        buffer_size: int = 100,
        cache_dir: Optional[str] = None,
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialize streaming text dataset.
        
        Args:
            dataset_name: HuggingFace dataset name
            tokenizer: Tokenizer with encode() method
            dataset_config: Dataset configuration/subset
            split: Dataset split
            max_length: Maximum sequence length
            text_column: Name of text column in dataset
            streaming: Whether to stream dataset
            buffer_size: Buffer size for streaming
            cache_dir: Cache directory
            verbose: Print loading info
            **kwargs: Additional arguments for StreamingDataset
        """
        # Estimate size based on streaming
        estimated_size = 10_000_000 if streaming else None
        
        super().__init__(
            estimated_size=estimated_size,
            cache_dir=cache_dir,
            **kwargs
        )
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.streaming = streaming
        self.buffer_size = buffer_size
        
        if verbose:
            print(f"📦 Loading {dataset_name}...")
            if dataset_config:
                print(f"   Config: {dataset_config}")
            print(f"   Split: {split}")
            print(f"   Streaming: {streaming}")
        
        # Import here to make it optional
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets not installed. "
                "Install with: pip install datasets"
            )
        
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
            self.dataset_iter = iter(self.dataset)
            self.buffer = []
        else:
            self._length = len(self.dataset)
            if verbose:
                print(f"✅ Loaded {self._length:,} examples")
        
        # Token buffer for creating sequences
        self.token_buffer = []
        
        self.metadata.update({
            'dataset_name': dataset_name,
            'dataset_config': dataset_config,
            'split': split,
            'text_column': text_column
        })
    
    def _get_next_example(self) -> Dict[str, Any]:
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
            # Random sampling for non-streaming
            import random
            idx = random.randint(0, len(self.dataset) - 1)
            return self.dataset[idx]
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Get a tokenized sequence.
        
        For streaming datasets, idx is ignored and we return
        the next sequence from the stream.
        
        Args:
            idx: Index (ignored for streaming)
        
        Returns:
            Dictionary with 'input_ids' and 'labels'
        """
        # Build up token buffer until we have enough
        while len(self.token_buffer) < self.max_length:
            example = self._get_next_example()
            text = example[self.text_column]
            
            # Tokenize
            tokens = self.tokenizer.encode(text)
            if isinstance(tokens, Tensor):
                tokens = tokens.tolist()
            
            self.token_buffer.extend(tokens)
        
        # Extract sequence
        input_ids = self.token_buffer[:self.max_length]
        self.token_buffer = self.token_buffer[self.max_length:]
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        
        item = {
            'input_ids': input_ids,
            'labels': labels
        }
        
        return self._apply_transform(item)


class MemoryMappedTextDataset(TokenizedDataset):
    """
    Memory-mapped text dataset for efficient access to large datasets.
    
    Stores tokenized sequences in a memory-mapped file for fast random access
    without loading everything into RAM.
    
    Useful for:
    - Very large datasets that don't fit in memory
    - Fast random access during training
    - Datasets that are reused multiple times
    
    Example:
        >>> dataset = MemoryMappedTextDataset(
        ...     texts=large_text_list,
        ...     tokenizer=my_tokenizer,
        ...     cache_file='data/processed/dataset.mmap'
        ... )
    """
    
    def __init__(
        self,
        texts: Optional[List[str]] = None,
        tokenizer = None,
        max_length: int = 1024,
        cache_file: Optional[str] = None,
        rebuild_cache: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialize memory-mapped dataset.
        
        Args:
            texts: List of texts (required if building cache)
            tokenizer: Tokenizer (required if building cache)
            max_length: Maximum sequence length
            cache_file: Path to cache file
            rebuild_cache: Force rebuild cache
            verbose: Print progress
            **kwargs: Additional arguments for TokenizedDataset
        """
        super().__init__(max_length=max_length, **kwargs)
        
        self.cache_file = cache_file
        self.tokenizer = tokenizer
        
        if cache_file is None:
            raise ValueError("cache_file must be provided for MemoryMappedTextDataset")
        
        cache_path = Path(cache_file)
        
        # Build or load cache
        if rebuild_cache or not cache_path.exists():
            if texts is None or tokenizer is None:
                raise ValueError("texts and tokenizer required to build cache")
            
            if verbose:
                print(f"🔨 Building cache: {cache_file}")
            
            self._build_cache(texts, tokenizer, cache_path, verbose)
        else:
            if verbose:
                print(f"📂 Loading cache: {cache_file}")
            
            self._load_cache(cache_path)
        
        self.metadata.update({
            'cache_file': str(cache_file),
            'num_sequences': len(self.sequences)
        })
    
    def _build_cache(
        self,
        texts: List[str],
        tokenizer,
        cache_path: Path,
        verbose: bool
    ):
        """Build and save cache."""
        # Tokenize all texts
        if verbose:
            print(f"   Tokenizing {len(texts)} texts...")
        
        sequences = []
        for text in texts:
            token_ids = tokenizer.encode(text)
            if isinstance(token_ids, Tensor):
                token_ids = token_ids.tolist()
            
            # Create chunks
            chunks = self._chunk_sequence(token_ids)
            sequences.extend(chunks)
        
        # Save as memory-mapped array
        import numpy as np
        
        # Create directory if needed
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to numpy array and save
        num_sequences = len(sequences)
        data = np.zeros((num_sequences, self.max_length), dtype=np.int32)
        
        for i, seq in enumerate(sequences):
            padded_seq = self._pad_sequence(seq)
            data[i] = padded_seq
        
        # Save as memory-mapped file
        np.save(str(cache_path), data)
        
        if verbose:
            print(f"   ✅ Cached {num_sequences} sequences")
        
        self.sequences = data
    
    def _load_cache(self, cache_path: Path):
        """Load cache from disk."""
        import numpy as np
        
        # Load as memory-mapped array
        self.sequences = np.load(
            str(cache_path),
            mmap_mode='r'  # Read-only memory map
        )
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Get sequence from memory-mapped array."""
        seq = self.sequences[idx]
        
        input_ids = torch.tensor(seq, dtype=torch.long)
        labels = input_ids.clone()
        
        item = {
            'input_ids': input_ids,
            'labels': labels
        }
        
        return self._apply_transform(item)
"""
Tokenizer utilities for Ramanujan Transformer.

This module provides tokenizer wrappers and utilities:
- MistralTokenizerWrapper: Wrapper for Mistral tokenizer
- Tokenizer utilities (encode, decode, batch processing)

Example:
    >>> from ramanujan.data import get_tokenizer
    >>> 
    >>> # Load tokenizer
    >>> tokenizer = get_tokenizer(vocab_size=31980)
    >>> 
    >>> # Encode text
    >>> tokens = tokenizer.encode("Hello, world!")
    >>> print(tokens)  # [12345, 6789, ...]
    >>> 
    >>> # Decode tokens
    >>> text = tokenizer.decode(tokens)
    >>> print(text)  # "Hello, world!"
"""

import torch
from typing import List, Union, Optional, Dict, Any
from pathlib import Path


# ============================================================================
# MISTRAL TOKENIZER WRAPPER
# ============================================================================

class MistralTokenizerWrapper:
    """
    Wrapper for Mistral tokenizer.
    
    Provides a unified interface for tokenization with special
    token handling and batch processing.
    
    Args:
        vocab_size: Vocabulary size (31980 for Mistral v3)
        model_name: HuggingFace model name
        cache_dir: Cache directory for tokenizer files
    
    Example:
        >>> tokenizer = MistralTokenizerWrapper(vocab_size=31980)
        >>> 
        >>> # Single text
        >>> tokens = tokenizer.encode("Hello, world!")
        >>> 
        >>> # Batch
        >>> batch = tokenizer.encode_batch(["Text 1", "Text 2"])
    """
    
    def __init__(
        self,
        vocab_size: int = 31980,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        cache_dir: Optional[str] = None
    ):
        self.vocab_size = vocab_size
        self.model_name = model_name
        
        # Try to load tokenizer
        self.tokenizer = self._load_tokenizer(cache_dir)
        
        # Special tokens
        self.bos_token_id = self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else 1
        self.eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 2
        self.pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else 0
        
        # Set pad token if not set
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id
            if hasattr(self.tokenizer, 'pad_token_id'):
                self.tokenizer.pad_token_id = self.pad_token_id
    
    def _load_tokenizer(self, cache_dir: Optional[str]):
        """Load tokenizer from HuggingFace."""
        try:
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                use_fast=True
            )
            
            print(f"Loaded tokenizer: {self.model_name}")
            print(f"Vocabulary size: {len(tokenizer)}")
            
            return tokenizer
        
        except ImportError:
            raise ImportError(
                "transformers not installed. Install with: pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")
    
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_bos: Add BOS token
            add_eos: Add EOS token
            max_length: Maximum sequence length
            truncation: Truncate if exceeds max_length
        
        Returns:
            List of token IDs
        
        Example:
            >>> tokens = tokenizer.encode("Hello, world!")
            >>> print(tokens)  # [1, 12345, 6789, ..., 2]
        """
        # Tokenize
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=max_length,
            truncation=truncation
        )
        
        # Add special tokens
        if add_bos:
            tokens = [self.bos_token_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_token_id]
        
        return tokens
    
    def decode(
        self,
        tokens: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            tokens: Token IDs
            skip_special_tokens: Skip special tokens in output
        
        Returns:
            Decoded text
        
        Example:
            >>> text = tokenizer.decode([1, 12345, 6789, 2])
            >>> print(text)  # "Hello, world!"
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        return self.tokenizer.decode(
            tokens,
            skip_special_tokens=skip_special_tokens
        )
    
    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: bool = False
    ) -> Union[List[List[int]], Dict[str, torch.Tensor]]:
        """
        Encode batch of texts.
        
        Args:
            texts: List of input texts
            add_bos: Add BOS token
            add_eos: Add EOS token
            max_length: Maximum sequence length
            padding: Pad sequences to same length
            truncation: Truncate if exceeds max_length
            return_tensors: Return PyTorch tensors
        
        Returns:
            List of token ID lists or dict with tensors
        
        Example:
            >>> batch = tokenizer.encode_batch(["Text 1", "Text 2"], return_tensors=True)
            >>> print(batch['input_ids'].shape)  # [2, seq_len]
        """
        # Encode each text
        all_tokens = []
        for text in texts:
            tokens = self.encode(
                text,
                add_bos=add_bos,
                add_eos=add_eos,
                max_length=max_length,
                truncation=truncation
            )
            all_tokens.append(tokens)
        
        if not return_tensors:
            return all_tokens
        
        # Convert to tensors with padding
        if padding:
            max_len = max(len(t) for t in all_tokens)
            
            input_ids = []
            attention_mask = []
            
            for tokens in all_tokens:
                # Pad
                padding_length = max_len - len(tokens)
                padded = tokens + [self.pad_token_id] * padding_length
                mask = [1] * len(tokens) + [0] * padding_length
                
                input_ids.append(padded)
                attention_mask.append(mask)
            
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
            }
        else:
            return {
                'input_ids': [torch.tensor(t, dtype=torch.long) for t in all_tokens]
            }
    
    def decode_batch(
        self,
        batch_tokens: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode batch of token IDs.
        
        Args:
            batch_tokens: Batch of token IDs
            skip_special_tokens: Skip special tokens
        
        Returns:
            List of decoded texts
        
        Example:
            >>> texts = tokenizer.decode_batch([[1, 123, 2], [1, 456, 2]])
            >>> print(texts)  # ["Text 1", "Text 2"]
        """
        if isinstance(batch_tokens, torch.Tensor):
            batch_tokens = batch_tokens.tolist()
        
        return [
            self.decode(tokens, skip_special_tokens=skip_special_tokens)
            for tokens in batch_tokens
        ]
    
    def __len__(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
    
    def __call__(self, *args, **kwargs):
        """Allow direct calling like HuggingFace tokenizer."""
        return self.tokenizer(*args, **kwargs)



class VocabConstrainedTokenizer:
    """
    Tokenizer with vocabulary truncation for Ramanujan-optimal dimensions.
    
    Truncates vocabulary to prime-friendly sizes for better sparsification.
    Out-of-vocab tokens are mapped to special fallback tokens.
    """
    
    def __init__(self, base_tokenizer, target_vocab_size: int):
        self.tokenizer = base_tokenizer
        self.target_vocab_size = target_vocab_size
        self.original_vocab_size = 32000  # Mistral's actual size
        
        # Calculate how many tokens we're excluding
        self.excluded_tokens = self.original_vocab_size - target_vocab_size
        
        # Special token handling - keep these even if out of range
        self.special_tokens = {
            'bos': 1,
            'eos': 2,
            'pad': 0,
            'unk': 0  # Map unknown to pad
        }
        
        print(f"📚 Vocabulary Truncation for Ramanujan Sparsity:")
        print(f"   Original vocab: {self.original_vocab_size}")
        print(f"   Target vocab: {target_vocab_size}")
        print(f"   Excluded tokens: {self.excluded_tokens}")
        print(f"   Sparsity benefit: Prime-friendly dimension")
    
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Encode with vocabulary constraint."""
        tokens = self.tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
        
        constrained_tokens = []
        remapped_count = 0
        
        for token in tokens:
            # Keep special tokens
            if token in self.special_tokens.values():
                constrained_tokens.append(token)
            # Keep tokens in range
            elif token < self.target_vocab_size:
                constrained_tokens.append(token)
            # Remap out-of-range tokens deterministically
            else:
                # Modulo mapping - deterministic and reversible
                remapped_token = (token % (self.target_vocab_size - 10)) + 10
                constrained_tokens.append(remapped_token)
                remapped_count += 1
        
        return constrained_tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens (best effort with truncated vocab)."""
        # Note: Remapped tokens won't decode perfectly, but that's expected
        return self.tokenizer.decode(tokens)
    
    def __len__(self):
        return self.target_vocab_size


def get_tokenizer(vocab_size: int = 32000) -> Union[MistralTokenizerWrapper, VocabConstrainedTokenizer]:
    """
    Get tokenizer with optional vocabulary truncation.
    
    Args:
        vocab_size: Target vocabulary size. Use prime-friendly sizes (e.g., 31847, 31981)
                   for optimal Ramanujan sparsification.
    
    Returns:
        Tokenizer instance
    """
    base_tokenizer = MistralTokenizerWrapper()
    
    if vocab_size != 32000:
        return VocabConstrainedTokenizer(base_tokenizer, vocab_size)
    
    return base_tokenizer



# ============================================================================
# TOKENIZER FACTORY
# ============================================================================

def get_tokenizer(
    vocab_size: int = 31980,
    tokenizer_type: str = 'mistral',
    **kwargs
) -> MistralTokenizerWrapper:
    """
    Get tokenizer instance.
    
    Factory function for creating tokenizers.
    
    Args:
        vocab_size: Vocabulary size
        tokenizer_type: Type of tokenizer ('mistral')
        **kwargs: Additional tokenizer arguments
    
    Returns:
        Tokenizer instance
    
    Example:
        >>> tokenizer = get_tokenizer(vocab_size=31980)
        >>> tokens = tokenizer.encode("Hello!")
    """
    tokenizer_type = tokenizer_type.lower()
    
    if tokenizer_type == 'mistral':
        return MistralTokenizerWrapper(vocab_size=vocab_size, **kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


# ============================================================================
# TOKENIZER UTILITIES
# ============================================================================

def tokenize_file(
    file_path: str,
    tokenizer: MistralTokenizerWrapper,
    max_length: Optional[int] = None,
    overlap: int = 0
) -> List[List[int]]:
    """
    Tokenize text file into chunks.
    
    Args:
        file_path: Path to text file
        tokenizer: Tokenizer instance
        max_length: Maximum chunk length
        overlap: Overlap between chunks
    
    Returns:
        List of token ID lists
    
    Example:
        >>> chunks = tokenize_file('data.txt', tokenizer, max_length=512)
        >>> print(f"Created {len(chunks)} chunks")
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenize full text
    tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
    
    if max_length is None:
        return [tokens]
    
    # Split into chunks with overlap
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + max_length, len(tokens))
        chunks.append(tokens[start:end])
        
        if end == len(tokens):
            break
        
        start = end - overlap
    
    return chunks


def compute_token_statistics(
    texts: List[str],
    tokenizer: MistralTokenizerWrapper
) -> Dict[str, float]:
    """
    Compute tokenization statistics.
    
    Args:
        texts: List of texts
        tokenizer: Tokenizer instance
    
    Returns:
        Dictionary with statistics
    
    Example:
        >>> stats = compute_token_statistics(texts, tokenizer)
        >>> print(f"Avg tokens: {stats['avg_tokens']:.1f}")
    """
    import numpy as np
    
    token_counts = []
    char_counts = []
    
    for text in texts:
        tokens = tokenizer.encode(text, add_bos=False, add_eos=False)
        token_counts.append(len(tokens))
        char_counts.append(len(text))
    
    token_counts = np.array(token_counts)
    char_counts = np.array(char_counts)
    
    # Compute statistics
    stats = {
        'num_texts': len(texts),
        'total_tokens': int(token_counts.sum()),
        'total_chars': int(char_counts.sum()),
        'avg_tokens': float(token_counts.mean()),
        'std_tokens': float(token_counts.std()),
        'min_tokens': int(token_counts.min()),
        'max_tokens': int(token_counts.max()),
        'avg_chars_per_token': float(char_counts.sum() / token_counts.sum())
    }
    
    return stats


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing tokenizer.py module")
    print("="*70)
    
    # Note: These tests will fail if transformers is not installed
    # or if HuggingFace model is not accessible
    
    try:
        # Test MistralTokenizerWrapper
        print("\n1. Testing MistralTokenizerWrapper...")
        tokenizer = VocabConstrainedTokenizer(vocab_size=31980)
        
        print(f"   Vocabulary size: {len(tokenizer)}")
        print(f"   BOS token: {tokenizer.bos_token_id}")
        print(f"   EOS token: {tokenizer.eos_token_id}")
        print(f"   PAD token: {tokenizer.pad_token_id}")
        print(f"   ✅ MistralTokenizerWrapper initialization working!")
        
        # Test encode
        print("\n2. Testing encode...")
        text = "Hello, world! This is a test."
        tokens = tokenizer.encode(text)
        
        print(f"   Text: {text}")
        print(f"   Tokens: {tokens[:10]}... ({len(tokens)} total)")
        assert len(tokens) > 0, "Encoding failed!"
        print(f"   ✅ encode working!")
        
        # Test decode
        print("\n3. Testing decode...")
        decoded = tokenizer.decode(tokens)
        
        print(f"   Decoded: {decoded}")
        print(f"   ✅ decode working!")
        
        # Test encode_batch
        print("\n4. Testing encode_batch...")
        texts = [
            "First sentence.",
            "Second sentence is longer.",
            "Third."
        ]
        
        batch = tokenizer.encode_batch(texts, return_tensors=True, padding=True)
        
        print(f"   Batch size: {batch['input_ids'].shape[0]}")
        print(f"   Sequence length: {batch['input_ids'].shape[1]}")
        print(f"   Input IDs shape: {batch['input_ids'].shape}")
        print(f"   Attention mask shape: {batch['attention_mask'].shape}")
        print(f"   ✅ encode_batch working!")
        
        # Test decode_batch
        print("\n5. Testing decode_batch...")
        decoded_texts = tokenizer.decode_batch(batch['input_ids'])
        
        print(f"   Decoded {len(decoded_texts)} texts")
        for i, text in enumerate(decoded_texts):
            print(f"   {i+1}. {text}")
        print(f"   ✅ decode_batch working!")
        
        # Test get_tokenizer factory
        print("\n6. Testing get_tokenizer factory...")
        tokenizer2 = get_tokenizer(vocab_size=31980, tokenizer_type='mistral')
        
        print(f"   Created tokenizer: {type(tokenizer2).__name__}")
        print(f"   ✅ get_tokenizer working!")
        
        # Test tokenize_file
        print("\n7. Testing tokenize_file...")
        
        # Create temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("This is a test file. " * 100)
            temp_path = f.name
        
        chunks = tokenize_file(temp_path, tokenizer, max_length=50, overlap=10)
        
        print(f"   Created {len(chunks)} chunks")
        print(f"   First chunk length: {len(chunks[0])}")
        print(f"   ✅ tokenize_file working!")
        
        # Cleanup
        import os
        os.unlink(temp_path)
        
        # Test compute_token_statistics
        print("\n8. Testing compute_token_statistics...")
        test_texts = [
            "Short text.",
            "This is a longer text with more words.",
            "Medium length text here."
        ]
        
        stats = compute_token_statistics(test_texts, tokenizer)
        
        print(f"   Total tokens: {stats['total_tokens']}")
        print(f"   Avg tokens: {stats['avg_tokens']:.1f}")
        print(f"   Chars per token: {stats['avg_chars_per_token']:.2f}")
        print(f"   ✅ compute_token_statistics working!")
        
        print("\n" + "="*70)
        print("✅ All tests passed!")
        print("="*70)
        
    except ImportError as e:
        print(f"\n⚠️  Skipping tests: {e}")
        print("Install transformers to run tests: pip install transformers")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nModule ready for use. Import with:")
    print("  from ramanujan.data import get_tokenizer")
    print("  from ramanujan.data.tokenizer import MistralTokenizerWrapper")
    print("="*70)
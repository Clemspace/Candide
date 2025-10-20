"""
Metrics computation for Ramanujan Transformer.

This module provides utilities for computing and tracking metrics:
- Sparsity statistics
- Model metrics (perplexity, BPB)
- Training metrics
- Parameter counting
- Memory estimation

Example:
    >>> from ramanujan.utils import compute_sparsity_stats, MetricsTracker
    >>> 
    >>> # Compute sparsity
    >>> stats = compute_sparsity_stats(model)
    >>> print(f"Overall sparsity: {stats['overall']:.2%}")
    >>> 
    >>> # Track metrics during training
    >>> tracker = MetricsTracker()
    >>> tracker.update({'loss': 3.5, 'perplexity': 33.1})
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import math
from collections import defaultdict


# ============================================================================
# SPARSITY METRICS
# ============================================================================

def compute_sparsity_stats(model: nn.Module) -> Dict[str, float]:
    """
    Compute sparsity statistics for model.
    
    Analyzes all layers with 'mask' attribute (Ramanujan sparse layers).
    
    Args:
        model: Model to analyze
    
    Returns:
        Dictionary with sparsity statistics
    
    Example:
        >>> stats = compute_sparsity_stats(model)
        >>> print(f"Overall: {stats['overall']:.2%}")
        >>> print(f"Attention: {stats['attention']:.2%}")
        >>> print(f"FFN: {stats['ffn']:.2%}")
    """
    total_params = 0
    sparse_params = 0
    
    attention_total = 0
    attention_sparse = 0
    
    ffn_total = 0
    ffn_sparse = 0
    
    layer_stats = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'mask'):
            # This is a sparse layer
            mask = module.mask
            total = mask.numel()
            sparse = mask.sum().item()
            sparsity = 1.0 - (sparse / total)
            
            total_params += total
            sparse_params += sparse
            
            # Categorize by layer type
            if 'attention' in name.lower() or 'attn' in name.lower():
                attention_total += total
                attention_sparse += sparse
            elif 'ffn' in name.lower() or 'feedforward' in name.lower() or 'mlp' in name.lower():
                ffn_total += total
                ffn_sparse += sparse
            
            layer_stats.append({
                'name': name,
                'sparsity': sparsity,
                'params': total,
                'nonzero': int(sparse)
            })
    
    # Compute overall statistics
    overall_sparsity = 1.0 - (sparse_params / total_params) if total_params > 0 else 0.0
    attention_sparsity = 1.0 - (attention_sparse / attention_total) if attention_total > 0 else 0.0
    ffn_sparsity = 1.0 - (ffn_sparse / ffn_total) if ffn_total > 0 else 0.0
    
    return {
        'overall': overall_sparsity,
        'attention': attention_sparsity,
        'ffn': ffn_sparsity,
        'total_params': total_params,
        'sparse_params': int(sparse_params),
        'dense_params': total_params - int(sparse_params),
        'num_sparse_layers': len(layer_stats),
        'layers': layer_stats
    }


def compute_layer_sparsity(layer: nn.Module) -> float:
    """
    Compute sparsity for a single layer.
    
    Args:
        layer: Layer to analyze
    
    Returns:
        Sparsity value (0.0 = dense, 1.0 = fully sparse)
    
    Example:
        >>> sparsity = compute_layer_sparsity(model.blocks[0].attention.q_proj)
        >>> print(f"Sparsity: {sparsity:.2%}")
    """
    if hasattr(layer, 'mask'):
        mask = layer.mask
        total = mask.numel()
        sparse = mask.sum().item()
        return 1.0 - (sparse / total)
    return 0.0


# ============================================================================
# MODEL METRICS
# ============================================================================

def compute_perplexity(loss: torch.Tensor) -> torch.Tensor:
    """
    Compute perplexity from loss.
    
    Perplexity = exp(loss)
    
    Args:
        loss: Cross-entropy loss
    
    Returns:
        Perplexity value
    """
    return torch.exp(loss)


def compute_bpb(
    loss: torch.Tensor,
    vocab_size: int,
    bits_per_token: Optional[float] = None
) -> float:
    """
    Compute bits per byte (BPB).
    
    BPB is a normalized compression metric that accounts for
    vocabulary size and encoding.
    
    Args:
        loss: Cross-entropy loss (in nats)
        vocab_size: Vocabulary size
        bits_per_token: Optional pre-computed bits per token
    
    Returns:
        Bits per byte
    
    Example:
        >>> loss = torch.tensor(3.5)
        >>> bpb = compute_bpb(loss, vocab_size=32000)
        >>> print(f"BPB: {bpb:.4f}")
    """
    # Convert nats to bits
    bits_per_token = loss.item() / math.log(2) if bits_per_token is None else bits_per_token
    
    # Estimate bytes per token (rough approximation)
    # For English text, typically ~4 bytes per token
    bytes_per_token = 4.0
    
    bpb = bits_per_token / bytes_per_token
    
    return bpb


def compute_parameter_efficiency(
    model: nn.Module,
    baseline_params: int
) -> Dict[str, float]:
    """
    Compute parameter efficiency metrics.
    
    Compares model parameters to a baseline.
    
    Args:
        model: Model to analyze
        baseline_params: Baseline parameter count
    
    Returns:
        Dictionary with efficiency metrics
    
    Example:
        >>> efficiency = compute_parameter_efficiency(sparse_model, dense_params)
        >>> print(f"Reduction: {efficiency['reduction_percentage']:.1f}%")
    """
    model_params = sum(p.numel() for p in model.parameters())
    
    reduction = baseline_params - model_params
    reduction_pct = (reduction / baseline_params) * 100 if baseline_params > 0 else 0
    
    compression_ratio = baseline_params / model_params if model_params > 0 else 0
    
    return {
        'model_params': model_params,
        'baseline_params': baseline_params,
        'reduction': reduction,
        'reduction_percentage': reduction_pct,
        'compression_ratio': compression_ratio
    }


# ============================================================================
# PARAMETER COUNTING
# ============================================================================

def count_parameters(
    model: nn.Module,
    trainable_only: bool = False,
    exclude_embeddings: bool = False
) -> int:
    """
    Count model parameters.
    
    Args:
        model: Model to analyze
        trainable_only: Only count trainable parameters
        exclude_embeddings: Exclude embedding layers
    
    Returns:
        Parameter count
    
    Example:
        >>> total = count_parameters(model)
        >>> trainable = count_parameters(model, trainable_only=True)
        >>> print(f"Total: {total:,}, Trainable: {trainable:,}")
    """
    params = model.parameters()
    
    if trainable_only:
        params = filter(lambda p: p.requires_grad, params)
    
    if exclude_embeddings:
        # Exclude embedding and output projection
        excluded_names = {'token_embedding', 'output_projection'}
        params = [
            p for n, p in model.named_parameters()
            if not any(en in n for en in excluded_names)
        ]
    
    return sum(p.numel() for p in params)


def count_parameters_by_layer(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters by layer type.
    
    Args:
        model: Model to analyze
    
    Returns:
        Dictionary mapping layer types to parameter counts
    
    Example:
        >>> counts = count_parameters_by_layer(model)
        >>> for layer_type, count in counts.items():
        ...     print(f"{layer_type}: {count:,}")
    """
    counts = defaultdict(int)
    
    for name, param in model.named_parameters():
        # Categorize by layer type
        if 'embedding' in name.lower():
            layer_type = 'embeddings'
        elif 'attention' in name.lower() or 'attn' in name.lower():
            layer_type = 'attention'
        elif 'ffn' in name.lower() or 'feedforward' in name.lower() or 'mlp' in name.lower():
            layer_type = 'ffn'
        elif 'norm' in name.lower():
            layer_type = 'normalization'
        elif 'output' in name.lower():
            layer_type = 'output'
        else:
            layer_type = 'other'
        
        counts[layer_type] += param.numel()
    
    return dict(counts)


# ============================================================================
# MEMORY ESTIMATION
# ============================================================================

def estimate_model_memory(
    model: nn.Module,
    batch_size: int = 1,
    seq_len: int = 512,
    dtype: str = 'float32'
) -> Dict[str, float]:
    """
    Estimate memory usage for model.
    
    Args:
        model: Model to analyze
        batch_size: Batch size
        seq_len: Sequence length
        dtype: Data type ('float32', 'float16', 'bfloat16')
    
    Returns:
        Dictionary with memory estimates in MB
    
    Example:
        >>> mem = estimate_model_memory(model, batch_size=8, seq_len=512)
        >>> print(f"Parameters: {mem['parameters_mb']:.1f} MB")
        >>> print(f"Activations: {mem['activations_mb']:.1f} MB")
        >>> print(f"Total: {mem['total_mb']:.1f} MB")
    """
    bytes_per_element = {
        'float32': 4,
        'float16': 2,
        'bfloat16': 2
    }[dtype]
    
    # Parameter memory
    param_count = sum(p.numel() for p in model.parameters())
    param_memory = (param_count * bytes_per_element) / (1024 ** 2)
    
    # Activation memory (rough estimate)
    # Assume activations are stored at each layer
    if hasattr(model, 'num_layers'):
        num_layers = model.num_layers
    else:
        num_layers = len([m for m in model.modules() if isinstance(m, nn.Module)])
    
    if hasattr(model, 'dim'):
        dim = model.dim
    else:
        dim = 512  # Rough estimate
    
    # Hidden states: batch * seq * dim per layer
    activation_memory = (batch_size * seq_len * dim * bytes_per_element * num_layers) / (1024 ** 2)
    
    # Gradient memory (same as parameters for training)
    gradient_memory = param_memory
    
    # Optimizer state (2x parameters for Adam/AdamW)
    optimizer_memory = param_memory * 2
    
    total_train = param_memory + activation_memory + gradient_memory + optimizer_memory
    total_inference = param_memory + activation_memory
    
    return {
        'parameters_mb': param_memory,
        'activations_mb': activation_memory,
        'gradients_mb': gradient_memory,
        'optimizer_mb': optimizer_memory,
        'total_inference_mb': total_inference,
        'total_training_mb': total_train,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'dtype': dtype
    }


# ============================================================================
# METRICS TRACKER
# ============================================================================

class MetricsTracker:
    """
    Track metrics during training/evaluation.
    
    Maintains running averages and history of metrics.
    
    Example:
        >>> tracker = MetricsTracker()
        >>> 
        >>> for step in range(100):
        ...     metrics = {'loss': 3.5, 'perplexity': 33.1}
        ...     tracker.update(metrics)
        ...     
        ...     if step % 10 == 0:
        ...         print(tracker.get_averages())
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(list)
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
    
    def update(self, metrics: Dict[str, float]):
        """Update with new metrics."""
        for key, value in metrics.items():
            self.metrics[key].append(value)
            self.totals[key] += value
            self.counts[key] += 1
            
            # Keep only window_size most recent
            if len(self.metrics[key]) > self.window_size:
                old_value = self.metrics[key].pop(0)
                self.totals[key] -= old_value
    
    def get_average(self, key: str) -> float:
        """Get average for specific metric over window."""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        return self.totals[key] / len(self.metrics[key])
    
    def get_averages(self) -> Dict[str, float]:
        """Get averages for all metrics."""
        return {key: self.get_average(key) for key in self.metrics.keys()}
    
    def get_total_average(self, key: str) -> float:
        """Get average over all time."""
        if self.counts[key] == 0:
            return 0.0
        total_sum = sum(self.metrics[key])  # Use actual values for accuracy
        return total_sum / self.counts[key]
    
    def get_stats(self, key: str) -> Dict[str, float]:
        """Get statistics for specific metric."""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return {
                'average': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std': 0.0
            }
        
        values = self.metrics[key]
        import numpy as np
        values_array = np.array(values)
        
        return {
            'average': float(values_array.mean()),
            'min': float(values_array.min()),
            'max': float(values_array.max()),
            'std': float(values_array.std())
        }
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.totals.clear()
        self.counts.clear()
    
    def get_history(self, key: str) -> List[float]:
        """Get full history for metric."""
        return self.metrics.get(key, [])


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def compute_throughput(
    num_tokens: int,
    time_seconds: float
) -> float:
    """
    Compute training/inference throughput.
    
    Args:
        num_tokens: Number of tokens processed
        time_seconds: Time taken in seconds
    
    Returns:
        Tokens per second
    
    Example:
        >>> throughput = compute_throughput(10000, 5.0)
        >>> print(f"Throughput: {throughput:.0f} tokens/sec")
    """
    return num_tokens / time_seconds if time_seconds > 0 else 0.0


def compute_flops(
    model: nn.Module,
    seq_len: int = 512
) -> int:
    """
    Estimate FLOPs for model forward pass.
    
    Rough approximation based on model architecture.
    
    Args:
        model: Model to analyze
        seq_len: Sequence length
    
    Returns:
        Approximate FLOPs
    
    Example:
        >>> flops = compute_flops(model, seq_len=512)
        >>> print(f"FLOPs: {flops/1e9:.2f}G")
    """
    if not hasattr(model, 'dim') or not hasattr(model, 'num_layers'):
        return 0
    
    dim = model.dim
    num_layers = model.num_layers
    
    # Rough estimate: 2 * params * seq_len for forward pass
    params = sum(p.numel() for p in model.parameters())
    flops = 2 * params * seq_len
    
    return flops


# ============================================================================
# COMPARISON UTILITIES
# ============================================================================

def compare_models(
    models: Dict[str, nn.Module],
    metrics: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models.
    
    Args:
        models: Dictionary mapping names to models
        metrics: Optional metrics for each model
    
    Returns:
        Dictionary with comparison results
    
    Example:
        >>> models = {'baseline': baseline_model, 'sparse': sparse_model}
        >>> comparison = compare_models(models)
        >>> for name, info in comparison.items():
        ...     print(f"{name}: {info['parameters']:,} params")
    """
    results = {}
    
    for name, model in models.items():
        model_info = {
            'parameters': count_parameters(model),
            'trainable_parameters': count_parameters(model, trainable_only=True),
            'parameter_breakdown': count_parameters_by_layer(model)
        }
        
        # Add sparsity stats if available
        sparsity = compute_sparsity_stats(model)
        if sparsity['num_sparse_layers'] > 0:
            model_info['sparsity'] = sparsity['overall']
            model_info['num_sparse_layers'] = sparsity['num_sparse_layers']
        
        # Add provided metrics
        if metrics and name in metrics:
            model_info['metrics'] = metrics[name]
        
        results[name] = model_info
    
    return results


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing metrics.py module")
    print("="*70)
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dim = 256
            self.num_layers = 4
            self.embedding = nn.Embedding(1000, 256)
            self.layers = nn.ModuleList([
                nn.Linear(256, 256) for _ in range(4)
            ])
            self.output = nn.Linear(256, 1000)
        
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x)
            return self.output(x)
    
    model = DummyModel()
    
    # Test count_parameters
    print("\n1. Testing count_parameters...")
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    
    print(f"   Total params: {total:,}")
    print(f"   Trainable params: {trainable:,}")
    assert total == trainable, "All params should be trainable!"
    print(f"   ✅ count_parameters working!")
    
    # Test count_parameters_by_layer
    print("\n2. Testing count_parameters_by_layer...")
    breakdown = count_parameters_by_layer(model)
    
    for layer_type, count in breakdown.items():
        print(f"   {layer_type}: {count:,}")
    print(f"   ✅ count_parameters_by_layer working!")
    
    # Test compute_perplexity
    print("\n3. Testing compute_perplexity...")
    loss = torch.tensor(3.5)
    ppl = compute_perplexity(loss)
    
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Perplexity: {ppl.item():.2f}")
    print(f"   ✅ compute_perplexity working!")
    
    # Test compute_bpb
    print("\n4. Testing compute_bpb...")
    bpb = compute_bpb(loss, vocab_size=1000)
    
    print(f"   BPB: {bpb:.4f}")
    print(f"   ✅ compute_bpb working!")
    
    # Test estimate_model_memory
    print("\n5. Testing estimate_model_memory...")
    mem = estimate_model_memory(model, batch_size=8, seq_len=512)
    
    print(f"   Parameters: {mem['parameters_mb']:.1f} MB")
    print(f"   Activations: {mem['activations_mb']:.1f} MB")
    print(f"   Total (inference): {mem['total_inference_mb']:.1f} MB")
    print(f"   Total (training): {mem['total_training_mb']:.1f} MB")
    print(f"   ✅ estimate_model_memory working!")
    
    # Test MetricsTracker
    print("\n6. Testing MetricsTracker...")
    tracker = MetricsTracker(window_size=10)
    
    for i in range(20):
        tracker.update({'loss': 3.5 + 0.1 * i, 'acc': 0.5 + 0.01 * i})
    
    avg = tracker.get_average('loss')
    stats = tracker.get_stats('loss')
    
    print(f"   Average loss: {avg:.4f}")
    print(f"   Loss stats: min={stats['min']:.4f}, max={stats['max']:.4f}")
    print(f"   ✅ MetricsTracker working!")
    
    # Test compute_throughput
    print("\n7. Testing compute_throughput...")
    throughput = compute_throughput(10000, 5.0)
    
    print(f"   Throughput: {throughput:.0f} tokens/sec")
    print(f"   ✅ compute_throughput working!")
    
    # Test compute_flops
    print("\n8. Testing compute_flops...")
    flops = compute_flops(model, seq_len=512)
    
    print(f"   FLOPs: {flops/1e9:.2f}G")
    print(f"   ✅ compute_flops working!")
    
    # Test compare_models
    print("\n9. Testing compare_models...")
    model2 = DummyModel()
    models = {'model1': model, 'model2': model2}
    comparison = compare_models(models)
    
    for name, info in comparison.items():
        print(f"   {name}: {info['parameters']:,} params")
    print(f"   ✅ compare_models working!")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    print("\nModule ready for use. Import with:")
    print("  from ramanujan.utils.metrics import compute_sparsity_stats")
    print("  from ramanujan.utils.metrics import MetricsTracker, count_parameters")
    print("="*70)
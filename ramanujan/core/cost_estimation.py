"""
Cost estimation system for Ramanujan computation graphs.

Estimates computational cost (FLOPs, parameters, memory) for graphs
before instantiation. Critical for architecture search.

Features:
- FLOPs estimation
- Parameter counting
- Memory estimation
- Per-component breakdown

Usage:
    from ramanujan.core.cost_estimation import estimate_cost
    
    cost = estimate_cost(graph, {'input': (8, 512)})
    print(f"FLOPs: {cost.flops:,}, Params: {cost.params:,}")
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ComputeCost:
    """
    Computational cost estimate.
    
    Attributes:
        flops: Floating point operations
        params: Number of parameters
        memory_mb: Peak memory usage (MB)
        latency_ms: Estimated latency (optional)
        breakdown: Per-node cost breakdown
    """
    flops: int = 0
    params: int = 0
    memory_mb: float = 0.0
    latency_ms: Optional[float] = None
    breakdown: Dict[str, 'ComputeCost'] = field(default_factory=dict)
    
    def __add__(self, other: 'ComputeCost') -> 'ComputeCost':
        """Add two costs together."""
        return ComputeCost(
            flops=self.flops + other.flops,
            params=self.params + other.params,
            memory_mb=self.memory_mb + other.memory_mb,
            latency_ms=(
                (self.latency_ms or 0) + (other.latency_ms or 0)
                if (self.latency_ms is not None or other.latency_ms is not None)
                else None
            ),
            breakdown={**self.breakdown, **other.breakdown}
        )
    
    def __str__(self) -> str:
        """Human-readable format."""
        lines = [
            f"FLOPs: {self.flops:,}",
            f"Params: {self.params:,}",
            f"Memory: {self.memory_mb:.1f} MB"
        ]
        if self.latency_ms is not None:
            lines.append(f"Latency: {self.latency_ms:.2f} ms")
        return " | ".join(lines)
    
    def summary(self) -> str:
        """Detailed summary with breakdown."""
        lines = [str(self)]
        if self.breakdown:
            lines.append("\nPer-node breakdown:")
            for node_id, cost in self.breakdown.items():
                lines.append(f"  {node_id}: {cost}")
        return '\n'.join(lines)


class CostEstimator:
    """
    Estimates computational cost for graphs.
    
    Example:
        >>> estimator = CostEstimator(graph)
        >>> cost = estimator.estimate({'input': (8, 512)})
        >>> print(cost.summary())
    """
    
    def __init__(self, graph):
        self.graph = graph
        self.graph.validate()
    
    def estimate(
        self,
        input_shapes: Dict[str, Tuple[int, ...]]
    ) -> ComputeCost:
        """
        Estimate total cost for the graph.
        
        Args:
            input_shapes: Input node shapes
        
        Returns:
            Total ComputeCost with per-node breakdown
        """
        # First infer shapes
        from ramanujan.core.shape_inference import infer_shapes
        
        try:
            shapes = infer_shapes(self.graph, input_shapes, strict=False)
        except:
            # Can't infer shapes - return zero cost
            return ComputeCost()
        
        # Estimate cost for each node
        total = ComputeCost()
        
        for node in self.graph.topological_sort():
            try:
                node_cost = self._estimate_node(node, shapes)
                total = total + node_cost
                total.breakdown[node.id] = node_cost
            except Exception:
                # Can't estimate this node - skip
                continue
        
        return total
    
    def _estimate_node(self, node, shapes) -> ComputeCost:
        """Estimate cost for a single node."""
        from ramanujan.core.registry import ComponentRegistry, get_component
        
        # Get component class
        component_cls = None
        for category in ComponentRegistry.list_categories():
            if ComponentRegistry.has(category, node.component_type):
                component_cls = get_component(category, node.component_type)
                break
        
        if component_cls is None:
            return ComputeCost()
        
        # Check if component has custom cost estimation
        if hasattr(component_cls, 'estimate_cost'):
            try:
                return component_cls.estimate_cost(
                    shapes.get(node.id, {}),
                    node.config
                )
            except:
                pass
        
        # Use default estimators based on component type
        if node.id not in shapes:
            return ComputeCost()
        
        output_shapes = shapes[node.id]
        return self._default_estimate(node, output_shapes)
    
    def _default_estimate(self, node, output_shapes) -> ComputeCost:
        """
        Default cost estimation based on component type.
        
        These are rough estimates - components should implement
        their own estimate_cost() for accuracy.
        """
        config = node.config
        
        # Get primary output shape
        if 'x' in output_shapes:
            output_shape = output_shapes['x']
        elif output_shapes:
            output_shape = next(iter(output_shapes.values()))
        else:
            return ComputeCost()
        
        # Skip if shape is symbolic
        if not all(isinstance(d, int) for d in output_shape):
            return ComputeCost()
        
        # Estimate based on component type
        comp_type = node.component_type
        
        if 'embed' in comp_type.lower():
            return self._estimate_embedding(output_shape, config)
        elif 'attention' in comp_type.lower() or 'transformer' in comp_type.lower():
            return self._estimate_attention(output_shape, config)
        elif 'ffn' in comp_type.lower() or 'mlp' in comp_type.lower():
            return self._estimate_ffn(output_shape, config)
        elif 'norm' in comp_type.lower():
            return self._estimate_norm(output_shape, config)
        else:
            # Generic estimate
            return self._estimate_generic(output_shape, config)
    
    def _estimate_embedding(self, output_shape, config) -> ComputeCost:
        """Estimate embedding layer cost."""
        vocab_size = config.get('vocab_size', 50000)
        dim = config.get('dim', output_shape[-1] if output_shape else 768)
        
        params = vocab_size * dim
        flops = 0  # Embedding is just lookup
        memory_mb = (params * 4) / (1024 ** 2)  # 4 bytes per float32
        
        return ComputeCost(flops=flops, params=params, memory_mb=memory_mb)
    
    def _estimate_attention(self, output_shape, config) -> ComputeCost:
        """Estimate attention mechanism cost."""
        if len(output_shape) < 3:
            return ComputeCost()
        
        batch, seq_len, dim = output_shape[0], output_shape[1], output_shape[-1]
        num_heads = config.get('num_heads', 12)
        
        # QKV projections: 3 * (dim * dim)
        params_proj = 3 * dim * dim
        
        # Output projection
        params_out = dim * dim
        
        # Total params
        params = params_proj + params_out
        
        # FLOPs calculation:
        # QKV proj: 3 * (batch * seq * dim * dim)
        # Attention scores: batch * heads * seq * seq * (dim / heads)
        # Attention output: batch * heads * seq * seq * (dim / heads)
        # Output proj: batch * seq * dim * dim
        flops_qkv = 3 * batch * seq_len * dim * dim
        flops_attn = 2 * batch * num_heads * seq_len * seq_len * (dim // num_heads)
        flops_out = batch * seq_len * dim * dim
        
        flops = flops_qkv + flops_attn + flops_out
        
        # Memory: activations + parameters
        memory_mb = (params * 4 + batch * seq_len * dim * 4) / (1024 ** 2)
        
        return ComputeCost(flops=flops, params=params, memory_mb=memory_mb)
    
    def _estimate_ffn(self, output_shape, config) -> ComputeCost:
        """Estimate feed-forward network cost."""
        if len(output_shape) < 2:
            return ComputeCost()
        
        batch_seq = output_shape[0] * output_shape[1] if len(output_shape) > 2 else output_shape[0]
        dim = output_shape[-1]
        hidden_dim = config.get('hidden_dim', dim * 4)
        
        # Two linear layers: dim -> hidden, hidden -> dim
        params = dim * hidden_dim + hidden_dim * dim
        
        # FLOPs: 2 * (batch * seq * dim * hidden)
        flops = 2 * batch_seq * dim * hidden_dim
        
        memory_mb = (params * 4 + batch_seq * hidden_dim * 4) / (1024 ** 2)
        
        return ComputeCost(flops=flops, params=params, memory_mb=memory_mb)
    
    def _estimate_norm(self, output_shape, config) -> ComputeCost:
        """Estimate normalization layer cost."""
        dim = output_shape[-1] if output_shape else 768
        
        # LayerNorm/RMSNorm: scale and shift
        params = 2 * dim if config.get('affine', True) else 0
        
        # FLOPs: mean, var, normalize
        batch_seq = 1
        for d in output_shape[:-1]:
            batch_seq *= d
        flops = 5 * batch_seq * dim  # Rough estimate
        
        memory_mb = (params * 4) / (1024 ** 2)
        
        return ComputeCost(flops=flops, params=params, memory_mb=memory_mb)
    
    def _estimate_generic(self, output_shape, config) -> ComputeCost:
        """Generic estimate for unknown components."""
        # Very rough: assume linear transformation
        num_elements = 1
        for d in output_shape:
            num_elements *= d
        
        dim = output_shape[-1] if output_shape else 768
        params = dim * dim  # Assume square matrix
        flops = num_elements * dim
        memory_mb = (params * 4) / (1024 ** 2)
        
        return ComputeCost(flops=flops, params=params, memory_mb=memory_mb)


def estimate_cost(
    graph,
    input_shapes: Dict[str, Tuple[int, ...]],
    detailed: bool = False
) -> ComputeCost:
    """
    Estimate computational cost for a graph.
    
    Args:
        graph: ComputationGraph to analyze
        input_shapes: Input node shapes
        detailed: Include per-node breakdown
    
    Returns:
        ComputeCost estimate
    
    Example:
        >>> cost = estimate_cost(graph, {'input': (8, 512)})
        >>> print(f"FLOPs: {cost.flops:,}")
        >>> print(f"Params: {cost.params:,}")
        >>> 
        >>> if detailed:
        ...     print(cost.summary())
    """
    estimator = CostEstimator(graph)
    cost = estimator.estimate(input_shapes)
    
    if not detailed:
        # Clear breakdown to save memory
        cost.breakdown = {}
    
    return cost
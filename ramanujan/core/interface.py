"""
Core interfaces for Ramanujan components.

This module defines the fundamental protocols and data structures that all
Ramanujan components must implement. The design is protocol-based (duck typing)
rather than inheritance-based for maximum flexibility.

Key Design Principles:
- Protocols are optional - implement only what you need
- Zero runtime overhead (protocols are type hints only)
- External code can be wrapped without modification
- Self-documenting through type hints and dataclasses

Example:
    >>> @register_component('block', 'my_block')
    >>> class MyBlock(nn.Module):
    ...     @property
    ...     def component_type(self) -> str:
    ...         return 'block'
    ...     
    ...     # ... implement other protocol methods
"""

from typing import Protocol, Dict, Any, Optional, Tuple, Callable, Union, List, runtime_checkable
from dataclasses import dataclass, field
from enum import Enum
import torch
from torch import Tensor


# ============================================================================
# ENUMS
# ============================================================================


class Modality(Enum):
    """
    Tensor modality types.
    
    Used to specify what kind of data a tensor represents.
    This enables multi-modal models and helps with validation.
    
    Attributes:
        TEXT: Discrete tokens (language)
        IMAGE: Spatial data (vision)
        AUDIO: Temporal waveforms (speech, music)
        VIDEO: Spatiotemporal data (vision + time)
        LATENT: Continuous embeddings (hidden representations)
        GRAPH: Graph-structured data (nodes + edges)
        GENERIC: Unknown or flexible type
    """
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    LATENT = "latent"
    GRAPH = "graph"
    GENERIC = "generic"


class ForwardMode(Enum):
    """
    Forward pass execution mode.
    
    Components can behave differently depending on the mode:
    - TRAIN: Training mode (dropout enabled, deep supervision, etc.)
    - EVAL: Evaluation mode (deterministic, no dropout)
    - GENERATE: Generation mode (autoregressive, beam search, etc.)
    - DIFFUSION_STEP: Single diffusion timestep
    - INFERENCE: General inference (may differ from eval in some components)
    
    Example:
        >>> # Training with multiple supervision steps
        >>> output = trm(x, mode=ForwardMode.TRAIN, max_steps=16)
        >>> 
        >>> # Fast inference with fewer steps
        >>> output = trm(x, mode=ForwardMode.INFERENCE, max_steps=4)
        >>> 
        >>> # High-quality generation
        >>> output = trm(x, mode=ForwardMode.GENERATE, max_steps=16)
    
    Usage:
        Components implementing MultiModeComponent protocol can access mode:
        
        >>> def forward(self, x, mode: ForwardMode = ForwardMode.TRAIN):
        ...     if mode == ForwardMode.TRAIN:
        ...         return self._train_forward(x)
        ...     elif mode == ForwardMode.GENERATE:
        ...         return self._generate_forward(x)
        ...     else:
        ...         return self._default_forward(x)
    """
    TRAIN = "train"
    EVAL = "eval"
    GENERATE = "generate"
    DIFFUSION_STEP = "diffusion_step"
    INFERENCE = "inference"
    
    def is_training(self) -> bool:
        """Check if this mode represents training."""
        return self == ForwardMode.TRAIN
    
    def is_inference(self) -> bool:
        """Check if this mode represents any kind of inference."""
        return self in {ForwardMode.EVAL, ForwardMode.GENERATE, 
                       ForwardMode.INFERENCE, ForwardMode.DIFFUSION_STEP}


# ============================================================================
# DATACLASSES
# ============================================================================


@dataclass(frozen=True)
class TensorSpec:
    """
    Specification for input/output tensors.
    
    Uses symbolic dimensions that are resolved at runtime.
    This enables:
    - Shape validation
    - Automatic shape inference
    - Self-documenting code
    - Multi-modal support
    
    Attributes:
        shape: Symbolic shape tuple (e.g., ('batch', 'seq', 'dim'))
        dtype: PyTorch data type
        modality: Type of data this tensor represents
        optional: Whether this tensor is required or optional
        description: Human-readable description
    
    Example:
        >>> # Define spec for text embeddings
        >>> spec = TensorSpec(
        ...     shape=('batch', 'seq', 'dim'),
        ...     dtype=torch.float32,
        ...     modality=Modality.TEXT,
        ...     description="Token embeddings after embedding layer"
        ... )
        >>> 
        >>> # Check if tensor matches
        >>> x = torch.randn(8, 512, 768)
        >>> assert spec.matches(x, {'batch': 8, 'seq': 512, 'dim': 768})
        >>> 
        >>> # Resolve to concrete shape
        >>> concrete_shape = spec.resolve_shape({'batch': 8, 'seq': 512, 'dim': 768})
        >>> assert concrete_shape == (8, 512, 768)
    """
    shape: Tuple[str, ...]
    dtype: torch.dtype = torch.float32
    modality: Modality = Modality.GENERIC
    optional: bool = False
    description: str = ""
    
    def matches(
        self, 
        tensor: Tensor, 
        dim_values: Optional[Dict[str, int]] = None,
        strict: bool = False
    ) -> bool:
        """
        Check if tensor matches this specification.
        
        Args:
            tensor: Tensor to check
            dim_values: Optional mapping of symbolic dims to values
            strict: If True, enforce exact dim_values match
        
        Returns:
            True if tensor matches spec
        
        Example:
            >>> spec = TensorSpec(shape=('batch', 'seq', 'dim'))
            >>> x = torch.randn(8, 512, 768)
            >>> 
            >>> # Flexible matching (just check ndim)
            >>> assert spec.matches(x)
            >>> 
            >>> # Strict matching
            >>> assert spec.matches(x, {'batch': 8, 'seq': 512, 'dim': 768}, strict=True)
            >>> assert not spec.matches(x, {'batch': 4, 'seq': 512, 'dim': 768}, strict=True)
        """
        # Check dtype
        if tensor.dtype != self.dtype:
            return False
        
        # Check shape dimensionality
        if len(tensor.shape) != len(self.shape):
            return False
        
        # Check specific dimensions if provided
        if dim_values and strict:
            for symbolic, actual in zip(self.shape, tensor.shape):
                if symbolic in dim_values:
                    if actual != dim_values[symbolic]:
                        return False
        
        return True
    
    def resolve_shape(self, dim_values: Dict[str, int]) -> Tuple[int, ...]:
        """
        Resolve symbolic shape to concrete shape.
        
        Args:
            dim_values: Mapping of symbolic dimension names to values
        
        Returns:
            Concrete shape tuple
        
        Raises:
            KeyError: If a symbolic dimension is not in dim_values
        
        Example:
            >>> spec = TensorSpec(shape=('batch', 'seq', 'dim'))
            >>> concrete = spec.resolve_shape({'batch': 8, 'seq': 512, 'dim': 768})
            >>> assert concrete == (8, 512, 768)
        """
        return tuple(dim_values[dim] for dim in self.shape)
    
    def __repr__(self) -> str:
        """Pretty string representation."""
        shape_str = f"[{', '.join(self.shape)}]"
        parts = [f"shape={shape_str}", f"dtype={self.dtype}"]
        if self.modality != Modality.GENERIC:
            parts.append(f"modality={self.modality.value}")
        if self.optional:
            parts.append("optional=True")
        if self.description:
            parts.append(f'"{self.description}"')
        return f"TensorSpec({', '.join(parts)})"


@dataclass(frozen=True)
class DynamicTensorSpec(TensorSpec):
    """
    TensorSpec with dynamic shape computation.
    
    For components where output shape depends on input shape
    in non-trivial ways (pooling, pruning, packing, etc.).
    
    Attributes:
        shape_fn: Function that computes output shape from input shapes
    
    Example:
        >>> # Global average pooling: [B, C, H, W] → [B, C]
        >>> def pool_shape_fn(input_shapes):
        ...     x_shape = input_shapes['x']
        ...     return (x_shape[0], x_shape[1])  # Keep batch and channels
        >>> 
        >>> spec = DynamicTensorSpec(
        ...     shape=('batch', 'channels'),
        ...     shape_fn=pool_shape_fn
        ... )
        >>> 
        >>> output_shape = spec.resolve_output_shape({'x': (8, 256, 32, 32)})
        >>> assert output_shape == (8, 256)
    """
    shape_fn: Optional[Callable[[Dict[str, Tuple[int, ...]]], Tuple[int, ...]]] = None
    
    def resolve_output_shape(self, input_shapes: Dict[str, Tuple[int, ...]]) -> Tuple[int, ...]:
        """
        Compute output shape from input shapes.
        
        Args:
            input_shapes: Dict mapping input names to their shapes
        
        Returns:
            Output shape
        
        Example:
            >>> spec = DynamicTensorSpec.from_reduction('x', reduced_dims=[2, 3])
            >>> output_shape = spec.resolve_output_shape({'x': (8, 256, 32, 32)})
            >>> assert output_shape == (8, 256)  # Spatial dims reduced
        """
        if self.shape_fn:
            return self.shape_fn(input_shapes)
        # Fallback to static shape if no shape_fn provided
        return self.shape
    
    @staticmethod
    def from_reduction(
        input_name: str,
        reduced_dims: List[int],
        description: str = ""
    ) -> 'DynamicTensorSpec':
        """
        Create spec for dimension reduction (pooling, etc.).
        
        Args:
            input_name: Name of input to reduce
            reduced_dims: List of dimension indices to remove
            description: Optional description
        
        Returns:
            DynamicTensorSpec for reduction operation
        
        Example:
            >>> # Reduce spatial dimensions [B, C, H, W] → [B, C]
            >>> spec = DynamicTensorSpec.from_reduction('x', reduced_dims=[2, 3])
            >>> 
            >>> # Reduce sequence dimension [B, L, D] → [B, D]
            >>> spec = DynamicTensorSpec.from_reduction('x', reduced_dims=[1])
        """
        def shape_fn(inputs: Dict[str, Tuple[int, ...]]) -> Tuple[int, ...]:
            input_shape = inputs[input_name]
            return tuple(s for i, s in enumerate(input_shape) if i not in reduced_dims)
        
        return DynamicTensorSpec(
            shape=('batch', '...'),  # Symbolic, actual computed by shape_fn
            shape_fn=shape_fn,
            description=description or f"Reduction of {input_name} dims {reduced_dims}"
        )
    
    @staticmethod
    def from_transformation(
        input_name: str,
        transform: Callable[[Tuple[int, ...]], Tuple[int, ...]],
        description: str = ""
    ) -> 'DynamicTensorSpec':
        """
        Create spec with arbitrary shape transformation.
        
        Args:
            input_name: Name of input to transform
            transform: Function that transforms shape tuple
            description: Optional description
        
        Returns:
            DynamicTensorSpec for transformation
        
        Example:
            >>> # Double sequence length
            >>> spec = DynamicTensorSpec.from_transformation(
            ...     'x',
            ...     lambda shape: (shape[0], shape[1] * 2, shape[2]),
            ...     description="Sequence length doubling"
            ... )
            >>> 
            >>> # Flatten spatial dimensions [B, C, H, W] → [B, C*H*W]
            >>> spec = DynamicTensorSpec.from_transformation(
            ...     'x',
            ...     lambda shape: (shape[0], shape[1] * shape[2] * shape[3])
            ... )
        """
        return DynamicTensorSpec(
            shape=('batch', '...'),
            shape_fn=lambda inputs: transform(inputs[input_name]),
            description=description
        )


@dataclass
class ForwardMetadata:
    """
    Metadata about forward pass execution.
    
    Components can optionally return this alongside outputs to provide
    information about their execution. Useful for:
    - Mixture of Experts (which experts fired)
    - Adaptive computation (when did it stop)
    - Routing decisions (which path was taken)
    - Profiling (compute used)
    - Debugging (execution trace)
    
    All fields are optional - populate only what's relevant.
    
    Attributes:
        executed_components: Names of sub-components that executed
        flops: Floating point operations (estimated or measured)
        memory_mb: Memory used in megabytes
        routing_decisions: Any routing decisions made
        exit_layer: For early exit, which layer exited at
        confidence: Confidence score for early exit decision
        num_steps: Number of iteration steps (for recursive models)
        custom: Any custom metrics specific to component
    
    Example:
        >>> # Mixture of Experts
        >>> metadata = ForwardMetadata(
        ...     executed_components=['expert_0', 'expert_3', 'expert_7'],
        ...     routing_decisions={'top_k_indices': [0, 3, 7]},
        ...     custom={'router_entropy': 2.1}
        ... )
        >>> 
        >>> # TRM reasoning
        >>> metadata = ForwardMetadata(
        ...     num_steps=8,
        ...     exit_layer=None,  # Didn't exit early
        ...     custom={'final_loss': 0.23, 'halted_early': False}
        ... )
        >>> 
        >>> # Adaptive depth
        >>> metadata = ForwardMetadata(
        ...     exit_layer=5,
        ...     confidence=0.97,
        ...     executed_components=[f'layer_{i}' for i in range(6)]
        ... )
    """
    # Execution trace
    executed_components: List[str] = field(default_factory=list)
    
    # Compute metrics
    flops: Optional[int] = None
    memory_mb: Optional[float] = None
    latency_ms: Optional[float] = None
    
    # Routing & decisions
    routing_decisions: Dict[str, Any] = field(default_factory=dict)
    
    # Early exit / adaptive computation
    exit_layer: Optional[int] = None
    confidence: Optional[float] = None
    
    # Iterative / recursive models
    num_steps: Optional[int] = None
    converged: Optional[bool] = None
    
    # Loss breakdown (for multi-loss components)
    loss_components: Dict[str, float] = field(default_factory=dict)
    
    # Custom metrics (component-specific)
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def add_component(self, name: str):
        """Add component to execution trace."""
        self.executed_components.append(name)
    
    def set_routing(self, decision_name: str, value: Any):
        """Record a routing decision."""
        self.routing_decisions[decision_name] = value
    
    def set_custom(self, key: str, value: Any):
        """Set custom metric."""
        self.custom[key] = value
    
    def merge(self, other: 'ForwardMetadata') -> 'ForwardMetadata':
        """
        Merge two metadata objects.
        
        Useful when composing multiple components.
        
        Args:
            other: Another ForwardMetadata to merge
        
        Returns:
            New merged ForwardMetadata
        
        Example:
            >>> meta1 = ForwardMetadata(executed_components=['layer_0'])
            >>> meta2 = ForwardMetadata(executed_components=['layer_1'])
            >>> merged = meta1.merge(meta2)
            >>> assert merged.executed_components == ['layer_0', 'layer_1']
        """
        return ForwardMetadata(
            executed_components=self.executed_components + other.executed_components,
            flops=(self.flops or 0) + (other.flops or 0) if (self.flops or other.flops) else None,
            memory_mb=(self.memory_mb or 0) + (other.memory_mb or 0) if (self.memory_mb or other.memory_mb) else None,
            routing_decisions={**self.routing_decisions, **other.routing_decisions},
            exit_layer=other.exit_layer if other.exit_layer is not None else self.exit_layer,
            confidence=other.confidence if other.confidence is not None else self.confidence,
            num_steps=(self.num_steps or 0) + (other.num_steps or 0) if (self.num_steps or other.num_steps) else None,
            loss_components={**self.loss_components, **other.loss_components},
            custom={**self.custom, **other.custom}
        )
    
    def summary(self) -> str:
        """
        Get human-readable summary.
        
        Returns:
            Formatted string with key metrics
        
        Example:
            >>> metadata = ForwardMetadata(
            ...     executed_components=['expert_0', 'expert_3'],
            ...     num_steps=8,
            ...     confidence=0.95
            ... )
            >>> print(metadata.summary())
            Executed: expert_0, expert_3 (2 components)
            Steps: 8
            Confidence: 0.95
        """
        lines = []
        
        if self.executed_components:
            comp_str = ', '.join(self.executed_components[:5])
            if len(self.executed_components) > 5:
                comp_str += f', ... ({len(self.executed_components)} total)'
            lines.append(f"Executed: {comp_str}")
        
        if self.num_steps is not None:
            lines.append(f"Steps: {self.num_steps}")
        
        if self.exit_layer is not None:
            lines.append(f"Exited at layer: {self.exit_layer}")
        
        if self.confidence is not None:
            lines.append(f"Confidence: {self.confidence:.3f}")
        
        if self.flops:
            lines.append(f"FLOPs: {self.flops:,}")
        
        if self.memory_mb:
            lines.append(f"Memory: {self.memory_mb:.1f} MB")
        
        if self.routing_decisions:
            lines.append(f"Routing: {self.routing_decisions}")
        
        if self.loss_components:
            loss_str = ', '.join(f"{k}={v:.3f}" for k, v in self.loss_components.items())
            lines.append(f"Loss components: {loss_str}")
        
        if self.custom:
            lines.append(f"Custom metrics: {self.custom}")
        
        return '\n'.join(lines) if lines else "No metadata recorded"
    
    def __repr__(self) -> str:
        """Concise representation."""
        parts = []
        if self.executed_components:
            parts.append(f"{len(self.executed_components)} components")
        if self.num_steps:
            parts.append(f"{self.num_steps} steps")
        if self.exit_layer is not None:
            parts.append(f"exit@{self.exit_layer}")
        return f"ForwardMetadata({', '.join(parts)})" if parts else "ForwardMetadata(empty)"


# ============================================================================
# PROTOCOLS
# ============================================================================


@runtime_checkable
class Component(Protocol):
    """
    Base protocol for all Ramanujan components.
    
    This is the minimal interface every component must implement.
    No inheritance required - just implement these methods.
    
    Design rationale:
    - Protocol (not ABC) allows external code to work without modification
    - Minimal interface keeps simple components simple
    - Additional capabilities via optional protocols (MultiModeComponent, etc.)
    
    All components must implement:
    - component_type: Category string ('block', 'attention', etc.)
    - input_spec: Specification of expected inputs
    - output_spec: Specification of outputs
    - forward: The actual computation
    - get_config: Configuration for serialization
    
    Example:
        >>> @register_component('block', 'simple_block')
        >>> class SimpleBlock(nn.Module):
        ...     def __init__(self, dim: int):
        ...         super().__init__()
        ...         self.dim = dim
        ...         self.linear = nn.Linear(dim, dim)
        ...     
        ...     @property
        ...     def component_type(self) -> str:
        ...         return 'block'
        ...     
        ...     @property
        ...     def input_spec(self) -> Dict[str, TensorSpec]:
        ...         return {
        ...             'x': TensorSpec(
        ...                 shape=('batch', 'seq', 'dim'),
        ...                 dtype=torch.float32
        ...             )
        ...         }
        ...     
        ...     @property
        ...     def output_spec(self) -> Dict[str, TensorSpec]:
        ...         return {
        ...             'x': TensorSpec(
        ...                 shape=('batch', 'seq', 'dim'),
        ...                 dtype=torch.float32
        ...             )
        ...         }
        ...     
        ...     def forward(self, x: Tensor) -> Tensor:
        ...         return self.linear(x)
        ...     
        ...     def get_config(self) -> Dict[str, Any]:
        ...         return {'dim': self.dim}
    """
    
    @property
    def component_type(self) -> str:
        """
        Component category.
        
        Standard types:
        - 'embedding': Token/position embeddings
        - 'encoder': Encoding blocks (transformer, mamba, etc.)
        - 'decoder': Decoding blocks
        - 'attention': Attention mechanisms
        - 'ffn': Feed-forward networks
        - 'norm': Normalization layers
        - 'block': Complete transformer-style blocks
        - 'head': Output heads (LM head, classifier, etc.)
        - 'reasoning': Reasoning modules (TRM, etc.)
        - 'vision': Vision components (ResNet, ViT, etc.)
        - 'audio': Audio components
        - 'diffusion': Diffusion model components
        
        Custom types are allowed - this is primarily for organization.
        """
        ...
    
    @property
    def input_spec(self) -> Dict[str, TensorSpec]:
        """
        Specification of expected inputs.
        
        Returns:
            Dictionary mapping input names to their specifications
        
        Example:
            >>> {
            ...     'x': TensorSpec(shape=('batch', 'seq', 'dim')),
            ...     'mask': TensorSpec(shape=('batch', 'seq'), dtype=torch.bool, optional=True)
            ... }
        """
        ...
    
    @property
    def output_spec(self) -> Dict[str, TensorSpec]:
        """
        Specification of outputs.
        
        Returns:
            Dictionary mapping output names to their specifications
        
        Example:
            >>> {
            ...     'x': TensorSpec(shape=('batch', 'seq', 'dim')),
            ...     'attention_weights': TensorSpec(shape=('batch', 'heads', 'seq', 'seq'))
            ... }
        """
        ...
    
    def forward(self, **inputs: Tensor) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Forward pass through component.
        
        Args:
            **inputs: Named tensor inputs matching input_spec
        
        Returns:
            Either:
            - Single tensor (if component has one output named 'x')
            - Dict of named tensors (if multiple outputs)
        
        Note:
            The forward signature should match input_spec keys.
            For example, if input_spec has 'x' and 'mask', forward should be:
            def forward(self, x, mask=None)
        """
        ...
    
    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration dictionary.
        
        Should contain all hyperparameters needed to reconstruct this component.
        Used for:
        - Serialization
        - Logging
        - Reproducibility
        - Model cards
        
        Returns:
            Configuration dictionary
        
        Example:
            >>> {
            ...     'dim': 768,
            ...     'num_heads': 12,
            ...     'dropout': 0.1,
            ...     'component_class': 'TransformerBlock',
            ...     'component_type': 'block'
            ... }
        """
        ...


@runtime_checkable
class StatefulComponent(Component, Protocol):
    """
    Component with trainable parameters.
    
    Most neural network components are stateful (have parameters).
    This protocol extends Component with standard PyTorch methods.
    
    Note: If your component inherits from nn.Module, you automatically
    implement this protocol (nn.Module has these methods).
    """
    
    def parameters(self) -> Any:
        """Return trainable parameters (nn.Module.parameters())."""
        ...
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary (nn.Module.state_dict())."""
        ...
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary (nn.Module.load_state_dict())."""
        ...


@runtime_checkable
class ComposableComponent(Component, Protocol):
    """
    Component that exposes its sub-components.
    
    Optional protocol for components that want to provide fine-grained
    access to their building blocks. Useful for:
    - Layer-wise learning rates
    - Component-specific optimizers
    - Freezing sub-components
    - Introspection
    
    Example:
        >>> class TransformerBlock(nn.Module):
        ...     def __init__(self, ...):
        ...         self.attention = GQA(...)
        ...         self.ffn = SwiGLU(...)
        ...         self.norm1 = RMSNorm(...)
        ...         self.norm2 = RMSNorm(...)
        ...     
        ...     def get_subcomponents(self) -> Dict[str, Component]:
        ...         return {
        ...             'attention': self.attention,
        ...             'ffn': self.ffn,
        ...             'norm1': self.norm1,
        ...             'norm2': self.norm2
        ...         }
    """
    
    def get_subcomponents(self) -> Dict[str, Component]:
        """
        Return sub-components.
        
        Returns:
            Dictionary mapping names to sub-components
        """
        ...


@runtime_checkable
class MultiModeComponent(Component, Protocol):
    """
    Component with multiple forward modes.
    
    For components that behave differently depending on task:
    - TRAIN: Training with full supervision
    - EVAL: Deterministic evaluation
    - GENERATE: Autoregressive generation
    - DIFFUSION_STEP: Single diffusion timestep
    - INFERENCE: General inference
    
    Example:
        >>> @register_component('reasoning', 'trm')
        >>> class TRM(nn.Module):
        ...     def forward(
        ...         self,
        ...         x,
        ...         mode: ForwardMode = ForwardMode.TRAIN,
        ...         max_steps: int | None = None,
        ...         **kwargs
        ...     ):
        ...         if mode == ForwardMode.TRAIN:
        ...             return self._train_forward(x, max_steps=16)
        ...         elif mode == ForwardMode.INFERENCE:
        ...             return self._inference_forward(x, max_steps=max_steps or 8)
        ...         elif mode == ForwardMode.GENERATE:
        ...             return self._generate_forward(x, max_steps=max_steps or 16)
        ...         else:
        ...             raise ValueError(f"Unsupported mode: {mode}")
        ...     
        ...     def get_supported_modes(self) -> List[ForwardMode]:
        ...         return [ForwardMode.TRAIN, ForwardMode.INFERENCE, ForwardMode.GENERATE]
    """
    
    def forward(
        self,
        mode: ForwardMode = ForwardMode.TRAIN,
        **inputs
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Forward pass with explicit mode.
        
        Args:
            mode: Execution mode
            **inputs: Named tensor inputs
        
        Returns:
            Output tensors
        """
        ...
    
    def get_supported_modes(self) -> List[ForwardMode]:
        """
        Return list of supported modes.
        
        Returns:
            List of ForwardMode enums this component supports
        """
        ...


@runtime_checkable
class MetadataComponent(Component, Protocol):
    """
    Component that returns execution metadata.
    
    For components that want to expose information about their execution:
    - Which sub-components ran (MoE, routing)
    - How many steps (adaptive depth, TRM)
    - Compute metrics (FLOPs, memory)
    - Confidence scores
    
    Example:
        >>> @register_component('block', 'moe')
        >>> class MoE(nn.Module):
        ...     def forward(self, x, return_metadata=False):
        ...         # Route to experts
        ...         top_k = self.router(x)
        ...         output = sum(self.experts[i](x) for i in top_k)
        ...         
        ...         if return_metadata:
        ...             metadata = ForwardMetadata(
        ...                 executed_components=[f'expert_{i}' for i in top_k],
        ...                 routing_decisions={'top_k': top_k.tolist()}
        ...             )
        ...             return output, metadata
        ...         return output
    """
    
    def forward(
        self,
        **inputs
    ) -> Tuple[Union[Tensor, Dict[str, Tensor]], ForwardMetadata]:
        """
        Forward pass with metadata.
        
        Returns:
            (outputs, metadata) tuple
        """
        ...


@runtime_checkable
class MutatingComponent(Component, Protocol):
    """
    Component that mutates inputs or internal state.
    
    Explicitly marking mutations helps with:
    - Debugging (know which ops modify data)
    - Graph optimization (can't reorder mutating ops)
    - Distributed training (special handling needed)
    
    Example:
        >>> @register_component('norm', 'inplace_rmsnorm')
        >>> class InPlaceRMSNorm(nn.Module):
        ...     @property
        ...     def mutates_inputs(self) -> bool:
        ...         return True  # Modifies x in-place
        ...     
        ...     @property
        ...     def mutates_self(self) -> bool:
        ...         return False
        ...     
        ...     def forward(self, x):
        ...         x.div_(x.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)
        ...         x.mul_(self.weight)
        ...         return x
    """
    
    @property
    def mutates_inputs(self) -> bool:
        """Returns True if forward() modifies input tensors in-place."""
        ...
    
    @property
    def mutates_self(self) -> bool:
        """Returns True if forward() modifies component state."""
        ...
    
    def clone_for_mutation(self) -> 'MutatingComponent':
        """
        Create a copy that can be safely mutated.
        
        Returns:
            New instance with same config but independent state
        """
        ...


@runtime_checkable
class SerializableComponent(Component, Protocol):
    """
    Component with custom serialization logic.
    
    For components that need special handling when saving/loading:
    - External API clients
    - Compiled code
    - Large cached data
    - Non-tensor state
    
    Example:
        >>> @register_component('encoder', 'openai_embeddings')
        >>> class OpenAIEmbeddings(nn.Module):
        ...     def __init__(self, api_key, model='text-embedding-ada-002'):
        ...         self.api_key = api_key  # Don't serialize
        ...         self.model = model
        ...         self.client = OpenAI(api_key)
        ...     
        ...     def get_serializable_state(self) -> Dict[str, Any]:
        ...         return {'model': self.model}  # Don't save api_key!
        ...     
        ...     def load_serializable_state(self, state: Dict[str, Any]) -> None:
        ...         self.model = state['model']
        ...         # Client must be reconstructed externally
    """
    
    def get_serializable_state(self) -> Dict[str, Any]:
        """
        Return state that can be serialized.
        
        Exclude:
        - API clients
        - File handles
        - Compiled code
        
        Include:
        - Configuration
        - Learnable parameters (via state_dict)
        - Essential buffers
        """
        ...
    
    def load_serializable_state(self, state: Dict[str, Any]) -> None:
        """
        Restore from serialized state.
        
        Reconstruct:
        - API connections
        - Compiled code
        - Cached data
        """
        ...


@dataclass
class ForwardContext:
    """
    Metadata passed through forward pass.
    
    This enables components to access information beyond their direct inputs:
    - Attention masks (padding, causal)
    - Position indices (for RoPE, sinusoidal)
    - KV cache (for autoregressive generation)
    - Mode-specific behavior (train vs eval vs generate)
    - Custom fields for specialized components
    
    Example:
        >>> ctx = ForwardContext(
        ...     attention_mask=causal_mask,
        ...     positions=torch.arange(seq_len),
        ...     use_cache=True,
        ...     mode=ForwardMode.GENERATE
        ... )
        >>> output = model(input_ids, ctx)
    """
    
    # Attention-related
    attention_mask: Optional[Tensor] = None  # [batch, seq, seq] or [batch, 1, seq, seq]
    positions: Optional[Tensor] = None  # [batch, seq] or [seq]
    past_key_values: Optional[List[Any]] = None  # List of (key, value) tuples per layer
    use_cache: bool = False  # Whether to return KV cache for generation
    
    # Mode
    mode: ForwardMode = ForwardMode.TRAIN
    
    # Custom fields (for specialized components)
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def clone(self) -> 'ForwardContext':
        """Create a copy of this context."""
        return ForwardContext(
            attention_mask=self.attention_mask,
            positions=self.positions,
            past_key_values=self.past_key_values,
            use_cache=self.use_cache,
            mode=self.mode,
            custom=self.custom.copy()
        )
    
    def with_updates(self, **kwargs) -> 'ForwardContext':
        """Create a new context with updated fields."""
        ctx = self.clone()
        for key, value in kwargs.items():
            if hasattr(ctx, key):
                setattr(ctx, key, value)
            else:
                ctx.custom[key] = value
        return ctx


class ComponentOutput:
    """
    Structured output from a component.
    
    Components can return:
    1. Just a tensor (backward compatible)
    2. ComponentOutput with primary tensor + auxiliary outputs
    
    This enables:
    - Attention returning weights alongside output
    - Components providing KV cache for generation
    - Accessing intermediate activations for losses
    
    Example:
        >>> # Component returns multiple outputs
        >>> result = attention(x, ctx)
        >>> output = result.primary  # Main output tensor
        >>> attn_weights = result.auxiliary['attention_weights']
        >>> kv_cache = result.auxiliary['present_key_values']
        >>> 
        >>> # Or unpack (backward compatible)
        >>> output, aux = result
    """
    
    def __init__(self, primary: Tensor, **auxiliary):
        """
        Args:
            primary: Main output tensor (required)
            **auxiliary: Additional outputs as keyword arguments
        """
        self.primary = primary
        self.auxiliary = auxiliary
    
    def __getitem__(self, key: Union[int, str]):
        """
        Allow both indexing and key access.
        
        result[0] → primary
        result[1] → auxiliary dict
        result['attention_weights'] → specific auxiliary output
        """
        if isinstance(key, int):
            if key == 0:
                return self.primary
            elif key == 1:
                return self.auxiliary
            else:
                raise IndexError(f"ComponentOutput only supports indices 0, 1, got {key}")
        else:
            return self.auxiliary[key]
    
    def __iter__(self):
        """Allow unpacking: output, aux = component(x)"""
        return iter([self.primary, self.auxiliary])
    
    def get(self, key: str, default=None):
        """Get auxiliary output with default."""
        return self.auxiliary.get(key, default)
    
    def keys(self):
        """Get auxiliary output keys."""
        return self.auxiliary.keys()
    
    def __repr__(self):
        aux_keys = list(self.auxiliary.keys())
        return f"ComponentOutput(primary={self.primary.shape}, auxiliary={aux_keys})"


# ============================================================================
# TYPE ALIASES
# ============================================================================

# Common component type combinations
# Since StatefulComponent already inherits from Component, we can just use type aliases
StandardComponent = StatefulComponent  # Already includes Component

@runtime_checkable
class AdvancedComponent(StatefulComponent, ComposableComponent, MetadataComponent, Protocol):
    """A fully-featured component implementing all core protocols."""
    pass

# Type aliases for clarity
ComponentType = str
ComponentName = str
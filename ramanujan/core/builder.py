"""
Model builder - constructs executable models from computation graphs.

Takes a ComputationGraph and instantiates all components, creating
an executable nn.Module that runs the graph.

Key Features:
- Automatic component instantiation from registry
- Optimized sequential execution
- General DAG execution
- Shape inference
- Metadata tracking

Example:
    >>> from ramanujan.core import ComputationGraph, ModelBuilder
    >>> 
    >>> # Create graph
    >>> graph = ComputationGraph.from_sequential([
    ...     {'type': 'embedding', 'config': {'vocab_size': 32000, 'dim': 768}},
    ...     {'type': 'transformer_block', 'config': {'dim': 768, 'num_heads': 12}}
    ... ])
    >>> 
    >>> # Build executable model
    >>> model = ModelBuilder.build(graph)
    >>> 
    >>> # Use like any nn.Module
    >>> output = model(input_ids)
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from collections import OrderedDict

from .graph import ComputationGraph, Node
from .registry import ComponentRegistry, get_component, create_component
from .interface import Component, ForwardMetadata


"""
Updated GraphExecutor for ramanujan/core/builder.py

This replaces your existing GraphExecutor class.
Adds support for:
- ForwardContext passing
- ComponentOutput handling
- Intermediate activation access
- KV cache collection
- Backward compatible with existing components
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
from ramanujan.core.graph import ComputationGraph
from ramanujan.core.registry import ComponentRegistry
from ramanujan.core.interface import ForwardContext, ComponentOutput, ForwardMetadata


class GraphExecutor(nn.Module):
    """
    Executes a computation graph with context support.
    
    Enhanced from original to support:
    - ForwardContext for metadata (masks, positions, cache, mode)
    - ComponentOutput for multi-value returns (attention weights, cache)
    - Intermediate activation access (for losses like semantic entropy)
    - Both sequential and DAG execution modes
    
    Two execution modes:
    1. Sequential (optimized): For linear graphs, uses nn.ModuleList
    2. DAG (general): For arbitrary graphs, uses topological execution
    
    Attributes:
        graph: The computation graph being executed
        components: Dictionary of instantiated components
        execution_order: Pre-computed execution order
        is_sequential: Whether graph is purely sequential
    """
    
    def __init__(self, graph: ComputationGraph):
        """
        Initialize executor.
        
        Args:
            graph: Validated ComputationGraph to execute
        
        Raises:
            ValueError: If graph is invalid
        """
        super().__init__()
        
        # Validate graph
        graph.validate()
        
        self.graph = graph
        self.is_sequential = graph.is_sequential()
        
        # Pre-compute execution order
        self.execution_order = graph.topological_sort()
        
        # Instantiate all components
        self.components = self._instantiate_components()
        
        # Optimize for sequential case
        if self.is_sequential:
            self._setup_sequential_execution()
        else:
            self._setup_dag_execution()
    
    def _instantiate_components(self) -> nn.ModuleDict:
        """
        Instantiate all components from graph nodes.
        
        Returns:
            ModuleDict of instantiated components
        """
        components = nn.ModuleDict()
        
        for node in self.execution_order:
            try:
                # Get component class from registry
                component_cls = ComponentRegistry.get(
                    self._get_category(node.component_type),
                    node.component_type
                )
                
                # Instantiate component
                component = component_cls(**node.config)
                components[node.id] = component
                
            except Exception as e:
                raise ValueError(
                    f"Failed to instantiate component '{node.id}' "
                    f"of type '{node.component_type}': {e}"
                ) from e
        
        return components
    
    def _get_category(self, component_type: str) -> str:
        """
        Infer category from component type.
        
        Tries to find the component in registry by checking all categories.
        """
        # Try all categories
        for category in ComponentRegistry.list_categories():
            if ComponentRegistry.has(category, component_type):
                return category
        
        # If not found, raise helpful error
        available = []
        for cat in ComponentRegistry.list_categories():
            available.extend(
                f"{cat}/{name}" 
                for name in ComponentRegistry.list_components(cat)
            )
        
        raise ValueError(
            f"Component type '{component_type}' not found in registry.\n"
            f"Available components:\n" + '\n'.join(f"  - {a}" for a in available[:20])
        )
    
    def _setup_sequential_execution(self):
        """Setup optimized sequential execution."""
        # Store as ModuleList for fast iteration
        self.sequential_components = nn.ModuleList([
            self.components[node.id] for node in self.execution_order
        ])
    
    def _setup_dag_execution(self):
        """Setup general DAG execution."""
        # Pre-compute execution plan
        self.execution_plan = []
        
        for i, node in enumerate(self.execution_order):
            component = self.components[node.id]
            
            # Map input node IDs to their execution order indices
            input_indices = []
            for input_id in node.inputs:
                for j, n in enumerate(self.execution_order):
                    if n.id == input_id:
                        input_indices.append(j)
                        break
            
            self.execution_plan.append({
                'component': component,
                'node': node,
                'node_id': node.id,
                'input_indices': input_indices,
                'output_index': i
            })
    
    def forward(
        self,
        *args,
        ctx: Optional[ForwardContext] = None,
        return_metadata: bool = False,
        return_intermediates: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], Tuple]:
        """
        Execute graph with optional context and intermediate outputs.
        
        Args:
            *args: Positional arguments for input nodes
            ctx: Optional ForwardContext (masks, positions, cache, mode)
            return_metadata: If True, include metadata in return
            return_intermediates: If True, return intermediate activations
            **kwargs: Keyword arguments for input nodes
        
        Returns:
            - Just output: tensor or dict of tensors
            - With metadata: (output, metadata)
            - With intermediates: (output, intermediates_dict)
            - With both: (output, metadata, intermediates_dict)
        
        Example:
            >>> # Simple forward
            >>> output = model(input_ids)
            >>> 
            >>> # With context
            >>> ctx = ForwardContext(attention_mask=mask, use_cache=True)
            >>> output = model(input_ids, ctx=ctx)
            >>> 
            >>> # Get intermediates for loss
            >>> output, intermediates = model(
            ...     input_ids, 
            ...     ctx=ctx,
            ...     return_intermediates=True
            ... )
            >>> hidden_states = intermediates['block_5']  # For semantic entropy
        """
        if self.is_sequential:
            return self._forward_sequential(
                *args,
                ctx=ctx,
                return_metadata=return_metadata,
                return_intermediates=return_intermediates,
                **kwargs
            )
        else:
            return self._forward_dag(
                *args,
                ctx=ctx,
                return_metadata=return_metadata,
                return_intermediates=return_intermediates,
                **kwargs
            )
    
    def _forward_sequential(
        self,
        x: torch.Tensor,
        ctx: Optional[ForwardContext] = None,
        return_metadata: bool = False,
        return_intermediates: bool = False
    ) -> Union[torch.Tensor, Tuple]:
        """
        Optimized sequential forward pass with context support.
        """
        metadata = ForwardMetadata() if return_metadata else None
        intermediates = {} if return_intermediates else None
        auxiliary_outputs = {}
        
        # Simple loop through components
        for i, (component, node) in enumerate(zip(
            self.sequential_components, 
            self.execution_order
        )):
            if return_metadata:
                metadata.add_component(component.__class__.__name__)
            
            # Prepare component inputs
            component_inputs = self._prepare_component_inputs(
                component, node, x, ctx
            )
            
            # Execute component
            result = component(**component_inputs)
            
            # Handle ComponentOutput
            if isinstance(result, ComponentOutput):
                x = result.primary
                auxiliary_outputs[node.id] = result.auxiliary
            elif isinstance(result, dict):
                x = result.get('x', result)
            else:
                x = result
            
            # Store intermediate if requested
            if return_intermediates:
                intermediates[node.id] = x
        
        # Build return value
        return self._build_return_value(
            x, metadata, intermediates, 
            return_metadata, return_intermediates
        )
    
    def _forward_dag(
        self,
        *args,
        ctx: Optional[ForwardContext] = None,
        return_metadata: bool = False,
        return_intermediates: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], Tuple]:
        """
        General DAG forward pass with context support.
        """
        # Cache for intermediate outputs
        cache = [None] * len(self.execution_order)
        metadata = ForwardMetadata() if return_metadata else None
        intermediates = {} if return_intermediates else None
        auxiliary_outputs = {}
        
        # Handle inputs - first node gets the input
        if args:
            cache[0] = args[0]
        elif kwargs:
            cache[0] = kwargs
        
        # Execute graph
        for step in self.execution_plan:
            component = step['component']
            node = step['node']
            
            # Gather inputs from cache
            if step['input_indices']:
                inputs = [cache[i] for i in step['input_indices']]
                if len(inputs) == 1:
                    inp = inputs[0]
                else:
                    inp = inputs
            else:
                # Input node - use initial input
                inp = cache[0]
            
            # Prepare component inputs with context
            component_inputs = self._prepare_component_inputs(
                component, node, inp, ctx
            )
            
            if return_metadata:
                metadata.add_component(step['node_id'])
            
            # Execute component
            result = component(**component_inputs)
            
            # Handle ComponentOutput
            if isinstance(result, ComponentOutput):
                output = result.primary
                auxiliary_outputs[node.id] = result.auxiliary
            else:
                output = result
            
            # Cache output
            cache[step['output_index']] = output
            
            # Store intermediate if requested
            if return_intermediates:
                intermediates[node.id] = output
        
        # Return output from last node
        final_output = cache[-1]
        
        return self._build_return_value(
            final_output, metadata, intermediates,
            return_metadata, return_intermediates
        )
    
    def _prepare_component_inputs(
        self,
        component: nn.Module,
        node: Any,
        data_input: Any,
        ctx: Optional[ForwardContext]
    ) -> Dict[str, Any]:
        """
        Prepare inputs for component forward call.
        
        Handles:
        - Named vs positional inputs
        - Context injection
        - Input spec matching
        
        Args:
            component: Component to call
            node: Graph node
            data_input: Data input(s) from previous component
            ctx: Optional forward context
        
        Returns:
            Dictionary of keyword arguments for component
        """
        inputs = {}
        
        # Check if component has input_spec
        if hasattr(component, 'input_spec'):
            input_spec = component.input_spec
            
            # Iterate through expected inputs
            for input_name in input_spec.keys():
                if input_name == 'ctx':
                    # Special case: inject context
                    if ctx is not None:
                        inputs['ctx'] = ctx
                elif input_name == 'x':
                    # Primary input
                    if isinstance(data_input, dict):
                        inputs['x'] = data_input.get('x', data_input)
                    else:
                        inputs['x'] = data_input
                else:
                    # Other inputs from node config or data_input
                    if isinstance(data_input, dict) and input_name in data_input:
                        inputs[input_name] = data_input[input_name]
                    elif not input_spec[input_name].optional:
                        # Required input not found
                        pass  # Component will handle the error
        else:
            # Component doesn't have input_spec
            # Try to pass data as-is
            if isinstance(data_input, dict):
                inputs.update(data_input)
            else:
                inputs['x'] = data_input
            
            # Try to add context if component accepts it
            # (check signature or just try)
            if ctx is not None:
                inputs['ctx'] = ctx
        
        return inputs
    
    def _build_return_value(
        self,
        output: Any,
        metadata: Optional[ForwardMetadata],
        intermediates: Optional[Dict],
        return_metadata: bool,
        return_intermediates: bool
    ) -> Union[Any, Tuple]:
        """
        Build return value based on flags.
        
        Args:
            output: Final output
            metadata: Optional metadata
            intermediates: Optional intermediate activations
            return_metadata: Whether to return metadata
            return_intermediates: Whether to return intermediates
        
        Returns:
            Output possibly with metadata and/or intermediates
        """
        returns = [output]
        
        if return_metadata and metadata is not None:
            returns.append(metadata)
        
        if return_intermediates and intermediates is not None:
            returns.append(intermediates)
        
        if len(returns) == 1:
            return returns[0]
        else:
            return tuple(returns)
    
    def get_component(self, node_id: str) -> nn.Module:
        """
        Get instantiated component by node ID.
        
        Args:
            node_id: Node ID in graph
        
        Returns:
            Component module
        
        Example:
            >>> # Access specific component for fine-tuning
            >>> trm = model.get_component('reasoning')
            >>> for param in trm.parameters():
            ...     param.requires_grad = False
        """
        if node_id not in self.components:
            raise ValueError(f"Component '{node_id}' not found")
        return self.components[node_id]
    
    def get_all_components(self) -> Dict[str, nn.Module]:
        """Get all instantiated components."""
        return dict(self.components)


class ModelBuilder:
    """
    Factory for building executable models from graphs.
    
    Main entry point for model construction.
    """
    
    @staticmethod
    def build(graph: ComputationGraph, **kwargs) -> nn.Module:
        """
        Build executable model from computation graph.
        
        Args:
            graph: Validated ComputationGraph
            **kwargs: Additional options (e.g., compile=True for torch.compile)
        
        Returns:
            Executable nn.Module
        
        Example:
            >>> graph = ComputationGraph.from_sequential([...])
            >>> model = ModelBuilder.build(graph)
            >>> 
            >>> # With optimization
            >>> model = ModelBuilder.build(graph, compile=True)
        """
        executor = GraphExecutor(graph)
        
        # Optional: Compile with torch.compile (PyTorch 2.0+)
        if kwargs.get('compile', False):
            try:
                executor = torch.compile(executor)
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}")
        
        return executor
    
    @staticmethod
    def build_from_config(config: Dict[str, Any], **kwargs) -> nn.Module:
        """
        Build model from config dictionary.
        
        Args:
            config: Configuration dict with graph specification
            **kwargs: Additional options
        
        Returns:
            Executable nn.Module
        
        Example:
            >>> config = {
            ...     'nodes': [
            ...         {'id': 'input', 'type': 'embedding', 'config': {...}},
            ...         {'id': 'layer_0', 'type': 'transformer_block', 'config': {...}}
            ...     ],
            ...     'inputs': ['input'],
            ...     'outputs': ['layer_0']
            ... }
            >>> model = ModelBuilder.build_from_config(config)
        """
        graph = ComputationGraph.from_dict(config)
        return ModelBuilder.build(graph, **kwargs)


# Convenience function
def build_model(graph: ComputationGraph, **kwargs) -> nn.Module:
    """
    Build model from graph (convenience function).
    
    Example:
        >>> from ramanujan.core import build_model, ComputationGraph
        >>> 
        >>> graph = ComputationGraph.from_sequential([...])
        >>> model = build_model(graph)
    """
    return ModelBuilder.build(graph, **kwargs)
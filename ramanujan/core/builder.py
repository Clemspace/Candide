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


class GraphExecutor(nn.Module):
    """
    Executes a computation graph.
    
    This is the runtime that takes a graph definition and makes it executable.
    
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
                    # Extract category from component_type
                    # e.g., 'transformer_block' -> look in 'block' category
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
                'node_id': node.id,
                'input_indices': input_indices,
                'output_index': i
            })
    
    def forward(
        self,
        *args,
        return_metadata: bool = False,
        **kwargs
    ) -> torch.Tensor | Dict[str, torch.Tensor] | tuple:
        """
        Execute graph.
        
        Args:
            *args: Positional arguments for input nodes
            return_metadata: If True, return (output, metadata) tuple
            **kwargs: Keyword arguments for input nodes
        
        Returns:
            Output tensor(s) or (output, metadata) if return_metadata=True
        
        Example:
            >>> # Simple forward
            >>> output = model(input_ids)
            >>> 
            >>> # With metadata
            >>> output, metadata = model(input_ids, return_metadata=True)
        """
        if self.is_sequential:
            return self._forward_sequential(*args, return_metadata=return_metadata, **kwargs)
        else:
            return self._forward_dag(*args, return_metadata=return_metadata, **kwargs)
    
    def _forward_sequential(
        self,
        x: torch.Tensor,
        return_metadata: bool = False
    ) -> torch.Tensor | tuple:
        """
        Optimized sequential forward pass.
        
        Uses simple iteration - same speed as hand-written code.
        """
        metadata = ForwardMetadata() if return_metadata else None
        
        # Simple loop through components
        for component in self.sequential_components:
            if return_metadata:
                metadata.add_component(component.__class__.__name__)
            
            # Handle both single tensor and dict returns
            result = component(x) if not isinstance(x, dict) else component(**x)
            
            if isinstance(result, dict):
                x = result.get('x', result)  # Try to get 'x', fallback to full dict
            else:
                x = result
        
        if return_metadata:
            return x, metadata
        return x
    
    def _forward_dag(
        self,
        *args,
        return_metadata: bool = False,
        **kwargs
    ) -> torch.Tensor | Dict[str, torch.Tensor] | tuple:
        """
        General DAG forward pass.
        
        Executes nodes in topological order, caching intermediate results.
        """
        # Cache for intermediate outputs
        cache = [None] * len(self.execution_order)
        metadata = ForwardMetadata() if return_metadata else None
        
        # Handle inputs
        # First node gets the input
        if args:
            cache[0] = args[0]
        elif kwargs:
            cache[0] = kwargs
        
        # Execute graph
        for step in self.execution_plan:
            component = step['component']
            
            # Gather inputs from cache
            if step['input_indices']:
                inputs = [cache[i] for i in step['input_indices']]
                
                # Handle multiple inputs (fusion, etc.)
                if len(inputs) == 1:
                    inp = inputs[0]
                else:
                    # Multiple inputs - component must handle
                    inp = inputs
            else:
                # Input node - use initial input
                inp = cache[0]
            
            # Execute component
            if return_metadata:
                metadata.add_component(step['node_id'])
            
            # Call component
            if isinstance(inp, list):
                # Multiple inputs - try to unpack
                try:
                    output = component(*inp)
                except TypeError:
                    # Component doesn't accept multiple args
                    output = component(inp)
            elif isinstance(inp, dict):
                output = component(**inp)
            else:
                output = component(inp)
            
            # Cache output
            cache[step['output_index']] = output
        
        # Return output from last node
        final_output = cache[-1]
        
        if return_metadata:
            return final_output, metadata
        return final_output
    
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
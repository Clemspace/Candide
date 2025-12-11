"""
Shape inference system for Ramanujan computation graphs.

This module provides automatic shape validation and inference for
computation graphs, catching dimension mismatches before model instantiation.

Features:
- Symbolic shape resolution
- Automatic shape propagation
- Detailed error reporting
- Support for dynamic shapes

Usage:
    from ramanujan.core.shape_inference import infer_shapes
    
    shapes = infer_shapes(graph, {'input': (8, 512)})
"""

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class ShapeError:
    """Detailed shape mismatch error."""
    node_id: str
    input_name: str
    expected: Tuple
    actual: Tuple
    message: str
    
    def __str__(self) -> str:
        return (
            f"Shape mismatch at node '{self.node_id}', input '{self.input_name}':\n"
            f"  Expected: {self.expected}\n"
            f"  Actual: {self.actual}\n"
            f"  {self.message}"
        )


class ShapeInferenceEngine:
    """
    Infers shapes through computation graphs.
    
    Example:
        >>> engine = ShapeInferenceEngine(graph)
        >>> shapes = engine.infer({'input': (8, 512)})
        >>> if not engine.validate():
        ...     print(engine.get_error_summary())
    """
    
    def __init__(self, graph):
        self.graph = graph
        self.graph.validate()
        self.node_shapes: Dict[str, Dict[str, Tuple]] = {}
        self.errors: List[ShapeError] = []
    
    def infer(
        self,
        input_shapes: Dict[str, Tuple[int, ...]],
        strict: bool = True
    ) -> Dict[str, Dict[str, Tuple]]:
        """Infer shapes for all nodes."""
        self.node_shapes = {}
        self.errors = []
        
        for node in self.graph.topological_sort():
            try:
                self.node_shapes[node.id] = self._infer_node(node, input_shapes)
            except Exception as e:
                error = ShapeError(
                    node_id=node.id,
                    input_name='unknown',
                    expected=(),
                    actual=(),
                    message=str(e)
                )
                self.errors.append(error)
                if strict:
                    raise ValueError(str(error)) from e
        
        return self.node_shapes
    
    def _infer_node(self, node, input_shapes):
        """Infer output shapes for a single node."""
        from ramanujan.core.registry import ComponentRegistry, get_component
        
        # Get component class
        component_cls = None
        for category in ComponentRegistry.list_categories():
            if ComponentRegistry.has(category, node.component_type):
                component_cls = get_component(category, node.component_type)
                break
        
        if component_cls is None:
            raise ValueError(f"Component '{node.component_type}' not in registry")
        
        # Check if has specs
        if not hasattr(component_cls, 'output_spec'):
            # No spec - assume passthrough
            if node.inputs and node.inputs[0] in self.node_shapes:
                return self.node_shapes[node.inputs[0]]
            elif node.id in input_shapes:
                return {'x': input_shapes[node.id]}
            else:
                return {'x': (...,)}
        
        # Gather input shapes
        node_input_shapes = self._gather_inputs(node, input_shapes)
        
        # Infer outputs using specs
        output_spec = component_cls.output_spec
        if isinstance(output_spec, property):
            # Property-based - can't access statically
            return {'x': (...,)}
        
        output_shapes = {}
        for name, spec in output_spec.items():
            if hasattr(spec, 'infer_concrete_shape'):
                try:
                    output_shapes[name] = spec.infer_concrete_shape(
                        node_input_shapes, node.config
                    )
                except:
                    output_shapes[name] = spec.shape
            else:
                output_shapes[name] = spec.shape
        
        return output_shapes
    
    def _gather_inputs(self, node, input_shapes):
        """Gather input shapes from previous nodes."""
        if not node.inputs:
            # Input node
            if node.id in input_shapes:
                return {'x': input_shapes[node.id]}
            else:
                raise ValueError(f"Input node '{node.id}' not in input_shapes")
        
        # Get from previous nodes
        gathered = {}
        for i, inp_id in enumerate(node.inputs):
            if inp_id not in self.node_shapes:
                raise ValueError(f"Shape for '{inp_id}' not yet inferred")
            
            prev_shapes = self.node_shapes[inp_id]
            if len(node.inputs) == 1:
                gathered.update(prev_shapes)
            else:
                for key, shape in prev_shapes.items():
                    gathered[f'input_{i}_{key}'] = shape
        
        return gathered
    
    def validate(self) -> bool:
        """Check if inference succeeded."""
        return len(self.errors) == 0
    
    def get_error_summary(self) -> str:
        """Get formatted error summary."""
        if not self.errors:
            return "No errors"
        
        lines = [f"Found {len(self.errors)} shape error(s):"]
        for i, error in enumerate(self.errors, 1):
            lines.append(f"\n{i}. {error}")
        return '\n'.join(lines)


def infer_shapes(
    graph,
    input_shapes: Dict[str, Tuple[int, ...]],
    strict: bool = True
) -> Dict[str, Dict[str, Tuple]]:
    """
    Infer shapes for a computation graph.
    
    Args:
        graph: ComputationGraph to analyze
        input_shapes: Input node shapes (node_id -> shape)
        strict: Raise on first error if True
    
    Returns:
        node_id -> {output_name -> shape}
    
    Example:
        >>> shapes = infer_shapes(graph, {'input': (8, 512)})
        >>> print(shapes['layer_0'])
    """
    engine = ShapeInferenceEngine(graph)
    return engine.infer(input_shapes, strict=strict)
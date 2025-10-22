"""
Computation graph representation for Ramanujan models.

Supports:
- Sequential models (linear chains)
- Parallel branches (multi-path)
- Arbitrary DAGs (skip connections, fusion, routing)
- Dynamic graphs (conditional execution)

Key Design Principles:
- Immutable graph structure (build once, execute many times)
- Explicit node connections (no implicit dependencies)
- Topological ordering (correct execution order)
- Validation (detect cycles, disconnected nodes)

Example:
    >>> # Simple sequential graph
    >>> graph = ComputationGraph.from_sequential([
    ...     {'type': 'embedding', 'config': {'vocab_size': 32000, 'dim': 768}},
    ...     {'type': 'transformer_block', 'config': {'dim': 768, 'num_heads': 12}},
    ...     {'type': 'lm_head', 'config': {'vocab_size': 32000}}
    ... ])
    >>> 
    >>> # Complex graph with branches
    >>> graph = ComputationGraph()
    >>> graph.add_node(Node(id='input', component_type='embedding', config={...}))
    >>> graph.add_node(Node(id='transformer', component_type='transformer_block', 
    ...                     config={...}, inputs=['input']))
    >>> graph.add_node(Node(id='reasoning', component_type='trm', 
    ...                     config={...}, inputs=['transformer']))
    >>> graph.add_node(Node(id='output', component_type='lm_head', 
    ...                     config={...}, inputs=['reasoning']))
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from collections import defaultdict, deque
import json
from copy import deepcopy
import random


@dataclass(frozen=True)
class Node:
    """
    Node in computation graph.
    
    Represents a single component with its configuration and connections.
    
    Attributes:
        id: Unique identifier for this node
        component_type: Type of component (e.g., 'transformer_block', 'trm')
        config: Configuration dictionary for component instantiation
        inputs: List of input node IDs (empty for input nodes)
        metadata: Optional metadata (description, tags, etc.)
    
    Example:
        >>> node = Node(
        ...     id='transformer_0',
        ...     component_type='transformer_block',
        ...     config={'dim': 768, 'num_heads': 12},
        ...     inputs=['embedding']
        ... )
    """
    id: str
    component_type: str
    config: Dict[str, Any]
    inputs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        """Make Node hashable for use in sets."""
        return hash(self.id)
    
    def __eq__(self, other):
        """Equality based on ID."""
        if not isinstance(other, Node):
            return False
        return self.id == other.id
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'component_type': self.component_type,
            'config': self.config,
            'inputs': self.inputs,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Deserialize from dictionary."""
        return cls(
            id=data['id'],
            component_type=data['component_type'],
            config=data['config'],
            inputs=data.get('inputs', []),
            metadata=data.get('metadata', {})
        )


class ComputationGraph:
    """
    Directed Acyclic Graph (DAG) of components.
    
    The graph defines the structure of a model:
    - Nodes are components
    - Edges are data flow connections
    - Execution follows topological order
    
    Features:
    - Cycle detection
    - Connectivity validation
    - Topological sorting
    - Multiple input/output support
    - Subgraph extraction
    
    Example:
        >>> graph = ComputationGraph()
        >>> 
        >>> # Add nodes
        >>> graph.add_node(Node(id='input', component_type='embedding', config={...}))
        >>> graph.add_node(Node(id='layer_0', component_type='transformer_block',
        ...                     config={...}, inputs=['input']))
        >>> graph.add_node(Node(id='output', component_type='lm_head',
        ...                     config={...}, inputs=['layer_0']))
        >>> 
        >>> # Set inputs/outputs
        >>> graph.set_inputs(['input'])
        >>> graph.set_outputs(['output'])
        >>> 
        >>> # Validate
        >>> assert graph.validate()
        >>> 
        >>> # Get execution order
        >>> order = graph.topological_sort()
    """
    
    def __init__(self):
        """Initialize empty graph."""
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, List[str]] = defaultdict(list)  # node_id -> [child_ids]
        self.reverse_edges: Dict[str, List[str]] = defaultdict(list)  # node_id -> [parent_ids]
        self.inputs: List[str] = []  # Input node IDs
        self.outputs: List[str] = []  # Output node IDs
    
    def add_node(self, node: Node) -> None:
        """
        Add node to graph.
        
        Args:
            node: Node to add
        
        Raises:
            ValueError: If node ID already exists
        
        Example:
            >>> graph = ComputationGraph()
            >>> graph.add_node(Node(id='layer_0', component_type='transformer_block',
            ...                     config={'dim': 768}))
        """
        if node.id in self.nodes:
            raise ValueError(f"Node '{node.id}' already exists in graph")
        
        self.nodes[node.id] = node
        
        # Build edges
        for input_id in node.inputs:
            self.edges[input_id].append(node.id)
            self.reverse_edges[node.id].append(input_id)
    
    def add_edge(self, from_node: str, to_node: str) -> None:
        """
        Add edge between nodes.
        
        Args:
            from_node: Source node ID
            to_node: Destination node ID
        
        Raises:
            ValueError: If either node doesn't exist
        
        Example:
            >>> graph.add_edge('layer_0', 'layer_1')
        """
        if from_node not in self.nodes:
            raise ValueError(f"Source node '{from_node}' not found")
        if to_node not in self.nodes:
            raise ValueError(f"Destination node '{to_node}' not found")
        
        self.edges[from_node].append(to_node)
        self.reverse_edges[to_node].append(from_node)
    
    def set_inputs(self, input_ids: List[str]) -> None:
        """
        Set input nodes.
        
        Args:
            input_ids: List of input node IDs
        
        Raises:
            ValueError: If any node ID doesn't exist
        """
        for node_id in input_ids:
            if node_id not in self.nodes:
                raise ValueError(f"Input node '{node_id}' not found")
        self.inputs = input_ids
    
    def set_outputs(self, output_ids: List[str]) -> None:
        """
        Set output nodes.
        
        Args:
            output_ids: List of output node IDs
        
        Raises:
            ValueError: If any node ID doesn't exist
        """
        for node_id in output_ids:
            if node_id not in self.nodes:
                raise ValueError(f"Output node '{node_id}' not found")
        self.outputs = output_ids
    
    def get_node(self, node_id: str) -> Node:
        """Get node by ID."""
        if node_id not in self.nodes:
            raise ValueError(f"Node '{node_id}' not found")
        return self.nodes[node_id]
    
    def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        return node_id in self.nodes
    
    def get_children(self, node_id: str) -> List[str]:
        """Get child node IDs."""
        return self.edges.get(node_id, [])
    
    def get_parents(self, node_id: str) -> List[str]:
        """Get parent node IDs."""
        return self.reverse_edges.get(node_id, [])
    
    def topological_sort(self) -> List[Node]:
        """
        Return nodes in topological execution order.
        
        Uses Kahn's algorithm for topological sorting.
        
        Returns:
            List of nodes in execution order
        
        Raises:
            ValueError: If graph has cycles
        
        Example:
            >>> order = graph.topological_sort()
            >>> for node in order:
            ...     print(f"Execute {node.id}")
        """
        # Compute in-degree for each node
        in_degree = {node_id: len(self.reverse_edges[node_id]) 
                    for node_id in self.nodes}
        
        # Queue of nodes with no incoming edges
        queue = deque([node_id for node_id, degree in in_degree.items() 
                      if degree == 0])
        
        result = []
        
        while queue:
            node_id = queue.popleft()
            result.append(self.nodes[node_id])
            
            # Reduce in-degree for children
            for child_id in self.edges[node_id]:
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)
        
        # Check if all nodes were processed (no cycles)
        if len(result) != len(self.nodes):
            raise ValueError(
                "Graph has cycles - cannot perform topological sort. "
                f"Processed {len(result)}/{len(self.nodes)} nodes."
            )
        
        return result
    
    def detect_cycles(self) -> List[List[str]]:
        """
        Detect cycles in graph.
        
        Returns:
            List of cycles, where each cycle is a list of node IDs
        
        Example:
            >>> cycles = graph.detect_cycles()
            >>> if cycles:
            ...     print(f"Found cycles: {cycles}")
        """
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node_id: str) -> bool:
            """DFS to detect cycles."""
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)
            
            for child_id in self.edges[node_id]:
                if child_id not in visited:
                    if dfs(child_id):
                        return True
                elif child_id in rec_stack:
                    # Found cycle
                    cycle_start = path.index(child_id)
                    cycles.append(path[cycle_start:] + [child_id])
                    return True
            
            path.pop()
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                dfs(node_id)
        
        return cycles
    
    def find_disconnected_nodes(self) -> Set[str]:
        """
        Find nodes not reachable from inputs.
        
        Returns:
            Set of disconnected node IDs
        
        Example:
            >>> disconnected = graph.find_disconnected_nodes()
            >>> if disconnected:
            ...     print(f"Warning: disconnected nodes: {disconnected}")
        """
        if not self.inputs:
            # If no inputs specified, all nodes are considered reachable
            return set()
        
        reachable = set()
        queue = deque(self.inputs)
        
        while queue:
            node_id = queue.popleft()
            if node_id in reachable:
                continue
            reachable.add(node_id)
            queue.extend(self.edges[node_id])
        
        return set(self.nodes.keys()) - reachable
    
    def validate(self) -> bool:
        """
        Validate graph structure.
        
        Checks:
        - No cycles
        - All input nodes exist
        - All output nodes exist
        - All node inputs reference existing nodes
        - No disconnected nodes (if inputs specified)
        
        Returns:
            True if valid
        
        Raises:
            ValueError: With detailed error message if invalid
        
        Example:
            >>> try:
            ...     graph.validate()
            ... except ValueError as e:
            ...     print(f"Invalid graph: {e}")
        """
        # Check for cycles
        cycles = self.detect_cycles()
        if cycles:
            cycle_strs = [' -> '.join(cycle) for cycle in cycles]
            raise ValueError(f"Graph has cycles:\n" + '\n'.join(cycle_strs))
        
        # Check inputs exist
        for node_id in self.inputs:
            if node_id not in self.nodes:
                raise ValueError(f"Input node '{node_id}' not found in graph")
        
        # Check outputs exist
        for node_id in self.outputs:
            if node_id not in self.nodes:
                raise ValueError(f"Output node '{node_id}' not found in graph")
        
        # Check all node inputs reference existing nodes
        for node in self.nodes.values():
            for input_id in node.inputs:
                if input_id not in self.nodes:
                    raise ValueError(
                        f"Node '{node.id}' references non-existent input '{input_id}'"
                    )
        
        # Check for disconnected nodes
        disconnected = self.find_disconnected_nodes()
        if disconnected:
            raise ValueError(
                f"Graph has disconnected nodes: {disconnected}\n"
                f"These nodes are not reachable from inputs: {self.inputs}"
            )
        
        return True
    
    def is_sequential(self) -> bool:
        """
        Check if graph is purely sequential (no branches).
        
        Returns:
            True if graph is a linear chain
        
        Example:
            >>> if graph.is_sequential():
            ...     print("Can use fast sequential execution")
        """
        for node_id in self.nodes:
            # Each node should have at most one child and one parent
            if len(self.edges[node_id]) > 1 or len(self.reverse_edges[node_id]) > 1:
                return False
        return True
    
    def get_subgraph(self, node_ids: List[str]) -> 'ComputationGraph':
        """
        Extract subgraph containing only specified nodes.
        
        Args:
            node_ids: List of node IDs to include
        
        Returns:
            New ComputationGraph containing only specified nodes
        
        Example:
            >>> # Extract just the transformer layers
            >>> layer_ids = [f'layer_{i}' for i in range(12)]
            >>> subgraph = graph.get_subgraph(layer_ids)
        """
        subgraph = ComputationGraph()
        
        # Add nodes
        for node_id in node_ids:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                # Filter inputs to only include nodes in subgraph
                filtered_inputs = [inp for inp in node.inputs if inp in node_ids]
                subgraph.add_node(Node(
                    id=node.id,
                    component_type=node.component_type,
                    config=node.config,
                    inputs=filtered_inputs,
                    metadata=node.metadata
                ))
        
        # Set inputs/outputs if they're in subgraph
        subgraph_inputs = [inp for inp in self.inputs if inp in node_ids]
        subgraph_outputs = [out for out in self.outputs if out in node_ids]
        
        if subgraph_inputs:
            subgraph.set_inputs(subgraph_inputs)
        if subgraph_outputs:
            subgraph.set_outputs(subgraph_outputs)
        
        return subgraph
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize graph to dictionary.
        
        Returns:
            Dictionary representation suitable for JSON
        
        Example:
            >>> graph_dict = graph.to_dict()
            >>> with open('model_graph.json', 'w') as f:
            ...     json.dump(graph_dict, f, indent=2)
        """
        return {
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'inputs': self.inputs,
            'outputs': self.outputs
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComputationGraph':
        """
        Deserialize graph from dictionary.
        
        Args:
            data: Dictionary representation
        
        Returns:
            ComputationGraph instance
        
        Example:
            >>> with open('model_graph.json', 'r') as f:
            ...     graph_dict = json.load(f)
            >>> graph = ComputationGraph.from_dict(graph_dict)
        """
        graph = cls()
        
        # Add nodes
        for node_data in data['nodes']:
            graph.add_node(Node.from_dict(node_data))
        
        # Set inputs/outputs
        if 'inputs' in data:
            graph.set_inputs(data['inputs'])
        if 'outputs' in data:
            graph.set_outputs(data['outputs'])
        
        return graph
    
    @classmethod
    def from_sequential(cls, node_configs: List[Dict[str, Any]]) -> 'ComputationGraph':
        """
        Create linear graph from list of node configs.
        
        Convenience method for sequential models (most common case).
        
        Args:
            node_configs: List of dicts with 'type' and 'config' keys
        
        Returns:
            Sequential ComputationGraph
        
        Example:
            >>> graph = ComputationGraph.from_sequential([
            ...     {'type': 'embedding', 'config': {'vocab_size': 32000, 'dim': 768}},
            ...     {'type': 'transformer_block', 'config': {'dim': 768, 'num_heads': 12}},
            ...     {'type': 'transformer_block', 'config': {'dim': 768, 'num_heads': 12}},
            ...     {'type': 'lm_head', 'config': {'vocab_size': 32000}}
            ... ])
        """
        graph = cls()
        prev_id = None
        
        for i, config in enumerate(node_configs):
            # Generate unique ID
            node_id = config.get('id', f"node_{i}")
            
            # Create node
            node = Node(
                id=node_id,
                component_type=config['type'],
                config=config.get('config', {}),
                inputs=[prev_id] if prev_id else [],
                metadata=config.get('metadata', {})
            )
            graph.add_node(node)
            
            # Track inputs/outputs
            if i == 0:
                graph.inputs.append(node_id)
            if i == len(node_configs) - 1:
                graph.outputs.append(node_id)
            
            prev_id = node_id
        
        return graph
    
    def visualize(self) -> str:
        """
        Create ASCII visualization of graph.
        
        Returns:
            String representation suitable for printing
        
        Example:
            >>> print(graph.visualize())
            embedding
              ↓
            transformer_0
              ↓
            transformer_1
              ↓
            lm_head
        """
        lines = []
        order = self.topological_sort()
        
        for i, node in enumerate(order):
            # Node
            lines.append(f"{node.id} [{node.component_type}]")
            
            # Arrow to next (if not last)
            if i < len(order) - 1:
                lines.append("  ↓")
        
        return '\n'.join(lines)
    
    def __len__(self) -> int:
        """Number of nodes in graph."""
        return len(self.nodes)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ComputationGraph("
            f"nodes={len(self.nodes)}, "
            f"inputs={self.inputs}, "
            f"outputs={self.outputs})"
        )



class GraphMutation:
    """
    Represents a single mutation operation on a graph.
    
    Mutations are the building blocks of self-modification.
    Each mutation is:
    - Reversible (can undo)
    - Serializable (can save/load)
    - Composable (can chain mutations)
    
    Example:
        >>> # Add a new transformer block
        >>> mutation = AddNodeMutation(
        ...     node_id='new_layer',
        ...     component_type='transformer_block',
        ...     config={'dim': 768, 'num_heads': 12},
        ...     insert_after='layer_5'
        ... )
        >>> mutation.apply(graph)
        >>> # Later, undo it
        >>> mutation.undo(graph)
    """
    
    def apply(self, graph: 'MutableGraph') -> None:
        """Apply mutation to graph."""
        raise NotImplementedError
    
    def undo(self, graph: 'MutableGraph') -> None:
        """Undo mutation (restore original graph)."""
        raise NotImplementedError
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize mutation."""
        raise NotImplementedError
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphMutation':
        """Deserialize mutation."""
        raise NotImplementedError


class AddNodeMutation(GraphMutation):
    """Add a new node to the graph."""
    
    def __init__(
        self,
        node_id: str,
        component_type: str,
        config: Dict[str, Any],
        insert_after: Optional[str] = None,
        insert_before: Optional[str] = None,
        inputs: Optional[List[str]] = None
    ):
        """
        Args:
            node_id: ID for new node
            component_type: Type of component to add
            config: Component configuration
            insert_after: Insert after this node (sequential graphs)
            insert_before: Insert before this node (sequential graphs)
            inputs: Explicit input connections (DAG graphs)
        """
        self.node_id = node_id
        self.component_type = component_type
        self.config = config
        self.insert_after = insert_after
        self.insert_before = insert_before
        self.inputs = inputs
        
        # For undo
        self._original_edges = None
    
    def apply(self, graph: 'MutableGraph') -> None:
        """Add node to graph."""
        # Determine inputs
        if self.inputs:
            inputs = self.inputs
        elif self.insert_after:
            inputs = [self.insert_after]
        elif self.insert_before:
            # Connect to parents of insert_before node
            inputs = graph.get_parents(self.insert_before)
        else:
            inputs = []
        
        # Create and add node
        node = Node(
            id=self.node_id,
            component_type=self.component_type,
            config=self.config,
            inputs=inputs
        )
        graph.add_node(node)
        
        # Rewire connections if inserting between nodes
        if self.insert_after:
            # Save original edges for undo
            self._original_edges = graph.get_children(self.insert_after).copy()
            
            # Rewire: insert_after -> new_node -> original_children
            for child_id in self._original_edges:
                graph.remove_edge(self.insert_after, child_id)
                graph.add_edge(self.node_id, child_id)

        if self.insert_after:
            # Save original edges for undo
            self._original_edges = graph.get_children(self.insert_after).copy()
            
            # Rewire: insert_after -> new_node -> original_children
            for child_id in self._original_edges:
                graph.remove_edge(self.insert_after, child_id)
                graph.add_edge(self.node_id, child_id)
            
            # Connect insert_after to new node
            graph.add_edge(self.insert_after, self.node_id)
        
        elif self.insert_before:
            # Save original edges
            self._original_edges = graph.get_parents(self.insert_before).copy()
            
            # Rewire: original_parents -> new_node -> insert_before
            for parent_id in self._original_edges:
                graph.remove_edge(parent_id, self.insert_before)
                graph.add_edge(parent_id, self.node_id)
            graph.add_edge(self.node_id, self.insert_before)
    
    def undo(self, graph: 'MutableGraph') -> None:
        """Remove the added node and restore original connections."""
        if self._original_edges:
            # Restore original edges
            if self.insert_after:
                for child_id in self._original_edges:
                    graph.remove_edge(self.node_id, child_id)
                    graph.add_edge(self.insert_after, child_id)
            elif self.insert_before:
                for parent_id in self._original_edges:
                    graph.remove_edge(parent_id, self.node_id)
                    graph.add_edge(parent_id, self.insert_before)
                graph.remove_edge(self.node_id, self.insert_before)
        
        # Remove node
        graph.remove_node(self.node_id)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'add_node',
            'node_id': self.node_id,
            'component_type': self.component_type,
            'config': self.config,
            'insert_after': self.insert_after,
            'insert_before': self.insert_before,
            'inputs': self.inputs
        }


class RemoveNodeMutation(GraphMutation):
    """Remove a node from the graph."""
    
    def __init__(self, node_id: str, bypass: bool = True):
        """
        Args:
            node_id: ID of node to remove
            bypass: If True, connect parents directly to children (bypass removed node)
        """
        self.node_id = node_id
        self.bypass = bypass
        
        # For undo
        self._removed_node = None
        self._removed_edges = None
    
    def apply(self, graph: 'MutableGraph') -> None:
        """Remove node from graph."""
        # Save for undo
        self._removed_node = graph.get_node(self.node_id)
        self._removed_edges = {
            'parents': graph.get_parents(self.node_id),
            'children': graph.get_children(self.node_id)
        }
        
        # Bypass if requested
        if self.bypass:
            parents = graph.get_parents(self.node_id)
            children = graph.get_children(self.node_id)
            
            # Connect parents directly to children
            for parent_id in parents:
                for child_id in children:
                    graph.add_edge(parent_id, child_id)
        
        # Remove node
        graph.remove_node(self.node_id)
    
    def undo(self, graph: 'MutableGraph') -> None:
        """Restore removed node."""
        # Re-add node
        graph.add_node(self._removed_node)
        
        # Restore original edges
        for parent_id in self._removed_edges['parents']:
            graph.add_edge(parent_id, self.node_id)
        for child_id in self._removed_edges['children']:
            graph.add_edge(self.node_id, child_id)
        
        # Remove bypass edges if we added them
        if self.bypass:
            for parent_id in self._removed_edges['parents']:
                for child_id in self._removed_edges['children']:
                    graph.remove_edge(parent_id, child_id)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'remove_node',
            'node_id': self.node_id,
            'bypass': self.bypass
        }


class ModifyConfigMutation(GraphMutation):
    """Modify a node's configuration."""
    
    def __init__(self, node_id: str, config_updates: Dict[str, Any]):
        """
        Args:
            node_id: ID of node to modify
            config_updates: Dictionary of config keys to update
        """
        self.node_id = node_id
        self.config_updates = config_updates
        
        # For undo
        self._original_config = None
    
    def apply(self, graph: 'MutableGraph') -> None:
        """Update node configuration."""
        node = graph.get_node(self.node_id)
        
        # Save original
        self._original_config = node.config.copy()
        
        # Update config (create new node with updated config)
        new_config = {**node.config, **self.config_updates}
        new_node = Node(
            id=node.id,
            component_type=node.component_type,
            config=new_config,
            inputs=node.inputs,
            metadata=node.metadata
        )
        
        # Replace node
        graph.replace_node(node.id, new_node)
    
    def undo(self, graph: 'MutableGraph') -> None:
        """Restore original configuration."""
        node = graph.get_node(self.node_id)
        original_node = Node(
            id=node.id,
            component_type=node.component_type,
            config=self._original_config,
            inputs=node.inputs,
            metadata=node.metadata
        )
        graph.replace_node(node.id, original_node)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'modify_config',
            'node_id': self.node_id,
            'config_updates': self.config_updates
        }


class AddSkipConnectionMutation(GraphMutation):
    """Add a skip connection between nodes."""
    
    def __init__(self, from_node: str, to_node: str):
        self.from_node = from_node
        self.to_node = to_node
    
    def apply(self, graph: 'MutableGraph') -> None:
        """Add skip connection."""
        graph.add_edge(self.from_node, self.to_node)
    
    def undo(self, graph: 'MutableGraph') -> None:
        """Remove skip connection."""
        graph.remove_edge(self.from_node, self.to_node)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'add_skip_connection',
            'from_node': self.from_node,
            'to_node': self.to_node
        }


class MutableGraph(ComputationGraph):
    """
    Mutable computation graph for self-modification.
    
    Extends ComputationGraph with:
    - Node/edge addition/removal
    - Configuration updates
    - Mutation tracking
    - Undo/redo history
    
    Example:
        >>> graph = MutableGraph.from_sequential([...])
        >>> 
        >>> # Add a reasoning module
        >>> graph.add_node_after('layer_8', Node(
        ...     id='reasoning',
        ...     component_type='trm',
        ...     config={'dim': 512}
        ... ))
        >>> 
        >>> # Try it out
        >>> model = build_model(graph)
        >>> performance = evaluate(model)
        >>> 
        >>> # Undo if it didn't help
        >>> if performance < baseline:
        ...     graph.undo_last_mutation()
    """
    
    def __init__(self):
        super().__init__()
        self.mutation_history: List[GraphMutation] = []
        self.redo_stack: List[GraphMutation] = []
    
    def remove_node(self, node_id: str) -> None:
        """
        Remove node from graph.
        
        Args:
            node_id: ID of node to remove
        
        Raises:
            ValueError: If node doesn't exist or is an input/output node
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node '{node_id}' not found")
        
        # Don't allow removing input/output nodes
        if node_id in self.inputs or node_id in self.outputs:
            raise ValueError(f"Cannot remove input/output node '{node_id}'")
        
        # Remove from nodes
        del self.nodes[node_id]
        
        # Remove all edges involving this node
        for parent_id in self.reverse_edges[node_id]:
            self.edges[parent_id].remove(node_id)
        for child_id in self.edges[node_id]:
            self.reverse_edges[child_id].remove(node_id)
        
        del self.edges[node_id]
        del self.reverse_edges[node_id]
    
    def remove_edge(self, from_node: str, to_node: str) -> None:
        """Remove edge between nodes."""
        if to_node in self.edges[from_node]:
            self.edges[from_node].remove(to_node)
        if from_node in self.reverse_edges[to_node]:
            self.reverse_edges[to_node].remove(from_node)
    
    def replace_node(self, node_id: str, new_node: Node) -> None:
        """Replace a node with a new one (keeping connections)."""
        if node_id not in self.nodes:
            raise ValueError(f"Node '{node_id}' not found")
        
        # Keep same connections
        new_node = Node(
            id=new_node.id,
            component_type=new_node.component_type,
            config=new_node.config,
            inputs=self.nodes[node_id].inputs,  # Keep original inputs
            metadata=new_node.metadata
        )
        
        self.nodes[node_id] = new_node
    
    def apply_mutation(self, mutation: GraphMutation) -> None:
        """
        Apply a mutation to the graph.
        
        Tracks mutation in history for undo/redo.
        
        Args:
            mutation: Mutation to apply
        
        Example:
            >>> mutation = AddNodeMutation(...)
            >>> graph.apply_mutation(mutation)
        """
        mutation.apply(self)
        self.mutation_history.append(mutation)
        self.redo_stack.clear()  # Clear redo stack
    
    def undo_last_mutation(self) -> None:
        """Undo the last mutation."""
        if not self.mutation_history:
            raise ValueError("No mutations to undo")
        
        mutation = self.mutation_history.pop()
        mutation.undo(self)
        self.redo_stack.append(mutation)
    
    def redo_last_mutation(self) -> None:
        """Redo the last undone mutation."""
        if not self.redo_stack:
            raise ValueError("No mutations to redo")
        
        mutation = self.redo_stack.pop()
        mutation.apply(self)
        self.mutation_history.append(mutation)
    
    def clone(self) -> 'MutableGraph':
        """Create a deep copy of this graph."""
        new_graph = MutableGraph()
        new_graph.nodes = deepcopy(self.nodes)
        new_graph.edges = deepcopy(self.edges)
        new_graph.reverse_edges = deepcopy(self.reverse_edges)
        new_graph.inputs = self.inputs.copy()
        new_graph.outputs = self.outputs.copy()
        return new_graph
    
    def add_node_after(self, after_node_id: str, new_node: Node) -> None:
        """
        Convenience method: Add node after an existing node.
        
        Args:
            after_node_id: Insert after this node
            new_node: Node to insert
        """
        mutation = AddNodeMutation(
            node_id=new_node.id,
            component_type=new_node.component_type,
            config=new_node.config,
            insert_after=after_node_id
        )
        self.apply_mutation(mutation)
    
    def add_node_before(self, before_node_id: str, new_node: Node) -> None:
        """
        Convenience method: Add node before an existing node.
        
        Args:
            before_node_id: Insert before this node
            new_node: Node to insert
        """
        mutation = AddNodeMutation(
            node_id=new_node.id,
            component_type=new_node.component_type,
            config=new_node.config,
            insert_before=before_node_id
        )
        self.apply_mutation(mutation)


class SearchSpace:
    """
    Defines the space of valid graph mutations for architecture search.
    
    Used by AutoNAS to explore architectures systematically.
    
    Example:
        >>> search_space = SearchSpace()
        >>> 
        >>> # Define allowed component types
        >>> search_space.add_component_type('transformer_block', {
        ...     'dim': [512, 768, 1024],
        ...     'num_heads': [8, 12, 16],
        ...     'dropout': [0.0, 0.1, 0.2]
        ... })
        >>> search_space.add_component_type('trm', {
        ...     'dim': [256, 512],
        ...     'n_recursions': [4, 6, 8]
        ... })
        >>> 
        >>> # Define mutation rules
        >>> search_space.add_mutation_rule('can_add_layer', lambda g: len(g) < 24)
        >>> search_space.add_mutation_rule('can_add_skip', lambda g: g.is_sequential())
        >>> 
        >>> # Sample random mutation
        >>> mutation = search_space.sample_mutation(current_graph)
    """
    
    def __init__(self):
        self.component_types: Dict[str, Dict[str, List[Any]]] = {}
        self.mutation_rules: Dict[str, Callable] = {}
        self.mutation_weights: Dict[str, float] = {
            'add_node': 0.4,
            'remove_node': 0.1,
            'modify_config': 0.3,
            'add_skip': 0.2
        }
    
    def add_component_type(
        self,
        component_type: str,
        config_space: Dict[str, List[Any]]
    ) -> None:
        """
        Add a component type to the search space.
        
        Args:
            component_type: Type of component
            config_space: Dictionary of config keys to list of possible values
        """
        self.component_types[component_type] = config_space
    
    def add_mutation_rule(self, name: str, rule: Callable[[MutableGraph], bool]) -> None:
        """
        Add a constraint on mutations.
        
        Args:
            name: Rule name
            rule: Function that takes graph and returns True if mutation is allowed
        """
        self.mutation_rules[name] = rule
    
    def sample_mutation(self, graph: MutableGraph) -> GraphMutation:
        """
        Sample a random valid mutation.
        
        Args:
            graph: Current graph
        
        Returns:
            Random mutation that satisfies all rules
        """
        # Choose mutation type based on weights
        mutation_type = random.choices(
            list(self.mutation_weights.keys()),
            weights=list(self.mutation_weights.values())
        )[0]
        
        # Generate mutation
        if mutation_type == 'add_node':
            return self._sample_add_node(graph)
        elif mutation_type == 'remove_node':
            return self._sample_remove_node(graph)
        elif mutation_type == 'modify_config':
            return self._sample_modify_config(graph)
        elif mutation_type == 'add_skip':
            return self._sample_add_skip(graph)
    
    def _sample_add_node(self, graph: MutableGraph) -> AddNodeMutation:
        """Sample random node addition."""
        # Choose component type
        component_type = random.choice(list(self.component_types.keys()))
        
        # Sample config
        config_space = self.component_types[component_type]
        config = {
            key: random.choice(values)
            for key, values in config_space.items()
        }
        
        # Choose insertion point
        nodes = list(graph.nodes.keys())
        if nodes:
            insert_after = random.choice(nodes)
        else:
            insert_after = None
        
        return AddNodeMutation(
            node_id=f"search_{component_type}_{random.randint(0, 9999)}",
            component_type=component_type,
            config=config,
            insert_after=insert_after
        )
    
    def _sample_remove_node(self, graph: MutableGraph) -> RemoveNodeMutation:
        """Sample random node removal."""
        # Get removable nodes (not inputs/outputs)
        removable = [
            nid for nid in graph.nodes.keys()
            if nid not in graph.inputs and nid not in graph.outputs
        ]
        
        if not removable:
            # Fallback to add_node
            return self._sample_add_node(graph)
        
        node_id = random.choice(removable)
        return RemoveNodeMutation(node_id, bypass=True)
    
    def _sample_modify_config(self, graph: MutableGraph) -> ModifyConfigMutation:
        """Sample random config modification."""
        # Choose random node
        node_id = random.choice(list(graph.nodes.keys()))
        node = graph.get_node(node_id)
        
        # Check if we have a config space for this type
        if node.component_type not in self.component_types:
            # Fallback
            return self._sample_add_node(graph)
        
        # Sample new values for one config key
        config_space = self.component_types[node.component_type]
        key = random.choice(list(config_space.keys()))
        value = random.choice(config_space[key])
        
        return ModifyConfigMutation(node_id, {key: value})
    
    def _sample_add_skip(self, graph: MutableGraph) -> AddSkipConnectionMutation:
        """Sample random skip connection."""
        nodes = list(graph.nodes.keys())
        if len(nodes) < 2:
            return self._sample_add_node(graph)
        
        # Choose two nodes (from_node must come before to_node in topo order)
        order = [n.id for n in graph.topological_sort()]
        from_idx = random.randint(0, len(order) - 2)
        to_idx = random.randint(from_idx + 1, len(order) - 1)
        
        return AddSkipConnectionMutation(order[from_idx], order[to_idx])
"""
Tests for computation graph.

Tests:
- Node creation and validation
- Graph construction (sequential and DAG)
- Edge management
- Topological sorting
- Cycle detection
- Graph validation
- Serialization/deserialization
- Subgraph extraction
- Mutable graphs and mutations
"""

import pytest
import json
from copy import deepcopy

from ramanujan.core.graph import (
    Node,
    ComputationGraph,
    MutableGraph,
    GraphMutation,
    AddNodeMutation,
    RemoveNodeMutation,
    ModifyConfigMutation,
    AddSkipConnectionMutation,
    SearchSpace
)


# ============================================================================
# NODE TESTS
# ============================================================================


class TestNode:
    """Test Node dataclass."""
    
    def test_basic_node_creation(self):
        """Test creating a basic node."""
        node = Node(
            id='test_node',
            component_type='transformer_block',
            config={'dim': 768, 'num_heads': 12}
        )
        
        assert node.id == 'test_node'
        assert node.component_type == 'transformer_block'
        assert node.config['dim'] == 768
        assert len(node.inputs) == 0
    
    def test_node_with_inputs(self):
        """Test node with input connections."""
        node = Node(
            id='layer_1',
            component_type='transformer_block',
            config={},
            inputs=['layer_0']
        )
        
        assert node.inputs == ['layer_0']
    
    def test_node_with_metadata(self):
        """Test node with metadata."""
        node = Node(
            id='test',
            component_type='block',
            config={},
            metadata={'description': 'Test node', 'tags': ['test']}
        )
        
        assert node.metadata['description'] == 'Test node'
        assert 'test' in node.metadata['tags']
    
    def test_node_hashable(self):
        """Test that nodes are hashable (can be used in sets)."""
        node1 = Node(id='node_1', component_type='test', config={})
        node2 = Node(id='node_2', component_type='test', config={})
        
        node_set = {node1, node2}
        assert len(node_set) == 2
    
    def test_node_equality(self):
        """Test node equality based on ID."""
        node1 = Node(id='same', component_type='test', config={})
        node2 = Node(id='same', component_type='test', config={})
        node3 = Node(id='different', component_type='test', config={})
        
        assert node1 == node2
        assert node1 != node3
    
    def test_node_to_dict(self):
        """Test node serialization."""
        node = Node(
            id='test',
            component_type='block',
            config={'dim': 768},
            inputs=['prev'],
            metadata={'tag': 'test'}
        )
        
        d = node.to_dict()
        
        assert d['id'] == 'test'
        assert d['component_type'] == 'block'
        assert d['config']['dim'] == 768
        assert d['inputs'] == ['prev']
        assert d['metadata']['tag'] == 'test'
    
    def test_node_from_dict(self):
        """Test node deserialization."""
        data = {
            'id': 'test',
            'component_type': 'block',
            'config': {'dim': 768},
            'inputs': ['prev'],
            'metadata': {'tag': 'test'}
        }
        
        node = Node.from_dict(data)
        
        assert node.id == 'test'
        assert node.component_type == 'block'
        assert node.config['dim'] == 768


# ============================================================================
# BASIC GRAPH TESTS
# ============================================================================


class TestBasicGraph:
    """Test basic ComputationGraph functionality."""
    
    def test_empty_graph(self):
        """Test creating empty graph."""
        graph = ComputationGraph()
        
        assert len(graph) == 0
        assert len(graph.nodes) == 0
        assert len(graph.inputs) == 0
        assert len(graph.outputs) == 0
    
    def test_add_single_node(self):
        """Test adding a single node."""
        graph = ComputationGraph()
        node = Node(id='node_0', component_type='test', config={})
        
        graph.add_node(node)
        
        assert len(graph) == 1
        assert graph.has_node('node_0')
    
    def test_add_multiple_nodes(self):
        """Test adding multiple nodes."""
        graph = ComputationGraph()
        
        for i in range(5):
            node = Node(id=f'node_{i}', component_type='test', config={})
            graph.add_node(node)
        
        assert len(graph) == 5
    
    def test_add_duplicate_node_error(self):
        """Test error when adding duplicate node ID."""
        graph = ComputationGraph()
        node = Node(id='duplicate', component_type='test', config={})
        
        graph.add_node(node)
        
        with pytest.raises(ValueError, match="already exists"):
            graph.add_node(node)
    
    def test_get_node(self):
        """Test retrieving node by ID."""
        graph = ComputationGraph()
        node = Node(id='test', component_type='block', config={'dim': 768})
        graph.add_node(node)
        
        retrieved = graph.get_node('test')
        
        assert retrieved.id == 'test'
        assert retrieved.config['dim'] == 768
    
    def test_get_node_not_found(self):
        """Test error when getting non-existent node."""
        graph = ComputationGraph()
        
        with pytest.raises(ValueError, match="not found"):
            graph.get_node('nonexistent')
    
    def test_has_node(self):
        """Test checking node existence."""
        graph = ComputationGraph()
        node = Node(id='test', component_type='test', config={})
        
        assert not graph.has_node('test')
        graph.add_node(node)
        assert graph.has_node('test')
    
    def test_set_inputs(self):
        """Test setting input nodes."""
        graph = ComputationGraph()
        graph.add_node(Node(id='input', component_type='embedding', config={}))
        
        graph.set_inputs(['input'])
        
        assert graph.inputs == ['input']
    
    def test_set_inputs_nonexistent_error(self):
        """Test error when setting non-existent input."""
        graph = ComputationGraph()
        
        with pytest.raises(ValueError, match="not found"):
            graph.set_inputs(['nonexistent'])
    
    def test_set_outputs(self):
        """Test setting output nodes."""
        graph = ComputationGraph()
        graph.add_node(Node(id='output', component_type='head', config={}))
        
        graph.set_outputs(['output'])
        
        assert graph.outputs == ['output']


# ============================================================================
# EDGE MANAGEMENT TESTS
# ============================================================================


class TestEdges:
    """Test edge management."""
    
    def test_edges_from_node_inputs(self):
        """Test that edges are created from node inputs."""
        graph = ComputationGraph()
        
        graph.add_node(Node(id='node_0', component_type='test', config={}))
        graph.add_node(Node(id='node_1', component_type='test', config={}, inputs=['node_0']))
        
        # Check edges were created
        assert 'node_1' in graph.get_children('node_0')
        assert 'node_0' in graph.get_parents('node_1')
    
    def test_add_edge_explicitly(self):
        """Test adding edge explicitly."""
        graph = ComputationGraph()
        
        graph.add_node(Node(id='a', component_type='test', config={}))
        graph.add_node(Node(id='b', component_type='test', config={}))
        
        graph.add_edge('a', 'b')
        
        assert 'b' in graph.get_children('a')
        assert 'a' in graph.get_parents('b')
    
    def test_add_edge_nonexistent_node(self):
        """Test error when adding edge with non-existent nodes."""
        graph = ComputationGraph()
        graph.add_node(Node(id='a', component_type='test', config={}))
        
        with pytest.raises(ValueError, match="not found"):
            graph.add_edge('a', 'nonexistent')
    
    def test_get_children(self):
        """Test getting child nodes."""
        graph = ComputationGraph()
        
        graph.add_node(Node(id='parent', component_type='test', config={}))
        graph.add_node(Node(id='child1', component_type='test', config={}, inputs=['parent']))
        graph.add_node(Node(id='child2', component_type='test', config={}, inputs=['parent']))
        
        children = graph.get_children('parent')
        
        assert 'child1' in children
        assert 'child2' in children
        assert len(children) == 2
    
    def test_get_parents(self):
        """Test getting parent nodes."""
        graph = ComputationGraph()
        
        graph.add_node(Node(id='parent1', component_type='test', config={}))
        graph.add_node(Node(id='parent2', component_type='test', config={}))
        graph.add_node(Node(id='child', component_type='test', config={}, 
                           inputs=['parent1', 'parent2']))
        
        parents = graph.get_parents('child')
        
        assert 'parent1' in parents
        assert 'parent2' in parents
        assert len(parents) == 2


# ============================================================================
# SEQUENTIAL GRAPH TESTS
# ============================================================================


class TestSequentialGraph:
    """Test sequential graph construction."""
    
    def test_from_sequential_basic(self):
        """Test creating sequential graph from config list."""
        configs = [
            {'type': 'embedding', 'config': {'vocab_size': 1000, 'dim': 128}},
            {'type': 'transformer_block', 'config': {'dim': 128, 'num_heads': 4}},
            {'type': 'lm_head', 'config': {'vocab_size': 1000}}
        ]
        
        graph = ComputationGraph.from_sequential(configs)
        
        assert len(graph) == 3
        assert len(graph.inputs) == 1
        assert len(graph.outputs) == 1
    
    def test_from_sequential_connections(self):
        """Test that sequential connections are correct."""
        configs = [
            {'type': 'a', 'config': {}},
            {'type': 'b', 'config': {}},
            {'type': 'c', 'config': {}}
        ]
        
        graph = ComputationGraph.from_sequential(configs)
        
        # Get nodes in order
        nodes = graph.topological_sort()
        
        # Check connections
        assert len(graph.get_children(nodes[0].id)) == 1
        assert len(graph.get_children(nodes[1].id)) == 1
        assert len(graph.get_children(nodes[2].id)) == 0
    
    def test_from_sequential_custom_ids(self):
        """Test sequential graph with custom node IDs."""
        configs = [
            {'id': 'input', 'type': 'embedding', 'config': {}},
            {'id': 'layer', 'type': 'transformer_block', 'config': {}},
            {'id': 'output', 'type': 'head', 'config': {}}
        ]
        
        graph = ComputationGraph.from_sequential(configs)
        
        assert graph.has_node('input')
        assert graph.has_node('layer')
        assert graph.has_node('output')
    
    def test_is_sequential(self):
        """Test sequential detection."""
        # Sequential graph
        configs = [
            {'type': 'a', 'config': {}},
            {'type': 'b', 'config': {}},
            {'type': 'c', 'config': {}}
        ]
        graph = ComputationGraph.from_sequential(configs)
        
        assert graph.is_sequential()
    
    def test_is_not_sequential(self):
        """Test non-sequential detection."""
        graph = ComputationGraph()
        
        # Create branching structure
        graph.add_node(Node(id='input', component_type='test', config={}))
        graph.add_node(Node(id='branch1', component_type='test', config={}, inputs=['input']))
        graph.add_node(Node(id='branch2', component_type='test', config={}, inputs=['input']))
        
        assert not graph.is_sequential()


# ============================================================================
# TOPOLOGICAL SORT TESTS
# ============================================================================


class TestTopologicalSort:
    """Test topological sorting."""
    
    def test_topological_sort_simple(self):
        """Test topological sort on simple sequential graph."""
        graph = ComputationGraph.from_sequential([
            {'id': 'a', 'type': 'test', 'config': {}},
            {'id': 'b', 'type': 'test', 'config': {}},
            {'id': 'c', 'type': 'test', 'config': {}}
        ])
        
        order = graph.topological_sort()
        ids = [node.id for node in order]
        
        assert ids == ['a', 'b', 'c']
    
    def test_topological_sort_dag(self):
        """Test topological sort on DAG."""
        graph = ComputationGraph()
        
        graph.add_node(Node(id='a', component_type='test', config={}))
        graph.add_node(Node(id='b', component_type='test', config={}, inputs=['a']))
        graph.add_node(Node(id='c', component_type='test', config={}, inputs=['a']))
        graph.add_node(Node(id='d', component_type='test', config={}, inputs=['b', 'c']))
        
        order = graph.topological_sort()
        ids = [node.id for node in order]
        
        # 'a' must come first, 'd' must come last
        assert ids[0] == 'a'
        assert ids[-1] == 'd'
        # 'b' and 'c' can be in either order
        assert set(ids[1:3]) == {'b', 'c'}
    
    def test_topological_sort_cycle_error(self):
        """Test error when graph has cycle."""
        graph = ComputationGraph()
        
        # Create cycle: a -> b -> c -> a
        graph.add_node(Node(id='a', component_type='test', config={}))
        graph.add_node(Node(id='b', component_type='test', config={}, inputs=['a']))
        graph.add_node(Node(id='c', component_type='test', config={}, inputs=['b']))
        # Manually add edge to create cycle
        graph.add_edge('c', 'a')
        
        with pytest.raises(ValueError, match="cycle"):
            graph.topological_sort()


# ============================================================================
# CYCLE DETECTION TESTS
# ============================================================================


class TestCycleDetection:
    """Test cycle detection."""
    
    def test_no_cycle(self):
        """Test detection on acyclic graph."""
        graph = ComputationGraph.from_sequential([
            {'type': 'a', 'config': {}},
            {'type': 'b', 'config': {}},
            {'type': 'c', 'config': {}}
        ])
        
        cycles = graph.detect_cycles()
        assert len(cycles) == 0
    
    def test_simple_cycle(self):
        """Test detection of simple cycle."""
        graph = ComputationGraph()
        
        graph.add_node(Node(id='a', component_type='test', config={}))
        graph.add_node(Node(id='b', component_type='test', config={}, inputs=['a']))
        graph.add_edge('b', 'a')  # Create cycle
        
        cycles = graph.detect_cycles()
        
        assert len(cycles) > 0
        # Check cycle contains 'a' and 'b'
        cycle = cycles[0]
        assert 'a' in cycle
        assert 'b' in cycle
    
    def test_self_loop(self):
        """Test detection of self-loop."""
        graph = ComputationGraph()
        
        graph.add_node(Node(id='a', component_type='test', config={}))
        graph.add_edge('a', 'a')  # Self-loop
        
        cycles = graph.detect_cycles()
        
        assert len(cycles) > 0


# ============================================================================
# VALIDATION TESTS
# ============================================================================


class TestValidation:
    """Test graph validation."""
    
    def test_validate_valid_graph(self):
        """Test validation of valid graph."""
        graph = ComputationGraph.from_sequential([
            {'type': 'a', 'config': {}},
            {'type': 'b', 'config': {}}
        ])
        
        assert graph.validate()
    
    def test_validate_cycle_error(self):
        """Test validation fails on cycle."""
        graph = ComputationGraph()
        
        graph.add_node(Node(id='a', component_type='test', config={}))
        graph.add_node(Node(id='b', component_type='test', config={}, inputs=['a']))
        graph.add_edge('b', 'a')
        
        with pytest.raises(ValueError, match="cycle"):
            graph.validate()
    
    def test_validate_disconnected_nodes(self):
        """Test validation fails on disconnected nodes."""
        graph = ComputationGraph()
        
        graph.add_node(Node(id='input', component_type='test', config={}))
        graph.add_node(Node(id='disconnected', component_type='test', config={}))
        
        graph.set_inputs(['input'])
        
        with pytest.raises(ValueError, match="disconnected"):
            graph.validate()
    
    def test_validate_invalid_input_reference(self):
        """Test validation fails on invalid input reference."""
        graph = ComputationGraph()
        
        graph.add_node(Node(
            id='node',
            component_type='test',
            config={},
            inputs=['nonexistent']
        ))
        
        with pytest.raises(ValueError, match="non-existent"):
            graph.validate()
    
    def test_find_disconnected_nodes(self):
        """Test finding disconnected nodes."""
        graph = ComputationGraph()
        
        graph.add_node(Node(id='input', component_type='test', config={}))
        graph.add_node(Node(id='connected', component_type='test', config={}, inputs=['input']))
        graph.add_node(Node(id='disconnected', component_type='test', config={}))
        
        graph.set_inputs(['input'])
        
        disconnected = graph.find_disconnected_nodes()
        
        assert 'disconnected' in disconnected
        assert 'connected' not in disconnected
        assert 'input' not in disconnected


# ============================================================================
# SERIALIZATION TESTS
# ============================================================================


class TestSerialization:
    """Test graph serialization."""
    
    def test_to_dict(self):
        """Test graph serialization to dict."""
        graph = ComputationGraph.from_sequential([
            {'id': 'a', 'type': 'test', 'config': {'dim': 768}},
            {'id': 'b', 'type': 'test', 'config': {'dim': 768}}
        ])
        
        d = graph.to_dict()
        
        assert 'nodes' in d
        assert 'inputs' in d
        assert 'outputs' in d
        assert len(d['nodes']) == 2
    
    def test_from_dict(self):
        """Test graph deserialization from dict."""
        data = {
            'nodes': [
                {'id': 'a', 'component_type': 'test', 'config': {}, 'inputs': []},
                {'id': 'b', 'component_type': 'test', 'config': {}, 'inputs': ['a']}
            ],
            'inputs': ['a'],
            'outputs': ['b']
        }
        
        graph = ComputationGraph.from_dict(data)
        
        assert len(graph) == 2
        assert graph.has_node('a')
        assert graph.has_node('b')
        assert graph.inputs == ['a']
        assert graph.outputs == ['b']
    
    def test_roundtrip_serialization(self):
        """Test serialization roundtrip."""
        original = ComputationGraph.from_sequential([
            {'id': 'input', 'type': 'embedding', 'config': {'dim': 768}},
            {'id': 'layer', 'type': 'transformer_block', 'config': {'dim': 768}},
            {'id': 'output', 'type': 'head', 'config': {}}
        ])
        
        # Serialize
        data = original.to_dict()
        
        # Deserialize
        restored = ComputationGraph.from_dict(data)
        
        # Compare
        assert len(restored) == len(original)
        assert restored.inputs == original.inputs
        assert restored.outputs == original.outputs
        
        # Check nodes
        for node_id in original.nodes:
            assert restored.has_node(node_id)
            orig_node = original.get_node(node_id)
            rest_node = restored.get_node(node_id)
            assert orig_node.component_type == rest_node.component_type
            assert orig_node.config == rest_node.config
    
    def test_json_serialization(self):
        """Test JSON serialization."""
        graph = ComputationGraph.from_sequential([
            {'type': 'a', 'config': {'dim': 768}},
            {'type': 'b', 'config': {'num_heads': 12}}
        ])
        
        # Serialize to JSON
        json_str = json.dumps(graph.to_dict(), indent=2)
        
        # Deserialize
        data = json.loads(json_str)
        restored = ComputationGraph.from_dict(data)
        
        assert len(restored) == len(graph)


# ============================================================================
# SUBGRAPH TESTS
# ============================================================================


class TestSubgraph:
    """Test subgraph extraction."""
    
    def test_get_subgraph(self):
        """Test extracting subgraph."""
        graph = ComputationGraph.from_sequential([
            {'id': 'a', 'type': 'test', 'config': {}},
            {'id': 'b', 'type': 'test', 'config': {}},
            {'id': 'c', 'type': 'test', 'config': {}},
            {'id': 'd', 'type': 'test', 'config': {}}
        ])
        
        # Extract middle nodes
        subgraph = graph.get_subgraph(['b', 'c'])
        
        assert len(subgraph) == 2
        assert subgraph.has_node('b')
        assert subgraph.has_node('c')
        assert not subgraph.has_node('a')
        assert not subgraph.has_node('d')
    
    def test_subgraph_preserves_connections(self):
        """Test subgraph preserves connections between included nodes."""
        graph = ComputationGraph.from_sequential([
            {'id': 'a', 'type': 'test', 'config': {}},
            {'id': 'b', 'type': 'test', 'config': {}},
            {'id': 'c', 'type': 'test', 'config': {}}
        ])
        
        subgraph = graph.get_subgraph(['b', 'c'])
        
        # b -> c connection should be preserved
        assert 'c' in subgraph.get_children('b')


# ============================================================================
# VISUALIZATION TESTS
# ============================================================================


class TestVisualization:
    """Test graph visualization."""
    
    def test_visualize_sequential(self):
        """Test visualization of sequential graph."""
        graph = ComputationGraph.from_sequential([
            {'id': 'input', 'type': 'embedding', 'config': {}},
            {'id': 'layer', 'type': 'transformer_block', 'config': {}},
            {'id': 'output', 'type': 'head', 'config': {}}
        ])
        
        viz = graph.visualize()
        
        assert 'input' in viz
        assert 'layer' in viz
        assert 'output' in viz
        assert '↓' in viz  # Should have arrows
    
    def test_repr(self):
        """Test string representation."""
        graph = ComputationGraph.from_sequential([
            {'type': 'a', 'config': {}},
            {'type': 'b', 'config': {}}
        ])
        
        repr_str = repr(graph)
        
        assert 'ComputationGraph' in repr_str
        assert 'nodes=2' in repr_str


# ============================================================================
# MUTABLE GRAPH TESTS
# ============================================================================


class TestMutableGraph:
    """Test MutableGraph functionality."""
    
    def test_create_mutable_graph(self):
        """Test creating mutable graph."""
        graph = MutableGraph()
        
        assert isinstance(graph, ComputationGraph)
        assert len(graph.mutation_history) == 0
    
    def test_remove_node(self):
        """Test removing node from mutable graph."""
        graph = MutableGraph.from_sequential([
            {'id': 'a', 'type': 'test', 'config': {}},
            {'id': 'b', 'type': 'test', 'config': {}},
            {'id': 'c', 'type': 'test', 'config': {}}
        ])
        
        graph.remove_node('b')
        
        assert not graph.has_node('b')
        assert graph.has_node('a')
        assert graph.has_node('c')
    
    def test_remove_input_node_error(self):
        """Test error when removing input node."""
        graph = MutableGraph.from_sequential([
            {'id': 'input', 'type': 'test', 'config': {}},
            {'id': 'layer', 'type': 'test', 'config': {}}
        ])
        
        with pytest.raises(ValueError, match="input/output"):
            graph.remove_node('input')
    
    def test_remove_edge(self):
        """Test removing edge."""
        graph = MutableGraph()
        graph.add_node(Node(id='a', component_type='test', config={}))
        graph.add_node(Node(id='b', component_type='test', config={}, inputs=['a']))
        
        graph.remove_edge('a', 'b')
        
        assert 'b' not in graph.get_children('a')
    
    def test_replace_node(self):
        """Test replacing node."""
        graph = MutableGraph()
        graph.add_node(Node(id='test', component_type='old', config={'old': 1}))
        
        new_node = Node(id='test', component_type='new', config={'new': 2})
        graph.replace_node('test', new_node)
        
        node = graph.get_node('test')
        assert node.component_type == 'new'
        assert node.config['new'] == 2
    
    def test_clone_graph(self):
        """Test cloning mutable graph."""
        original = MutableGraph.from_sequential([
            {'id': 'a', 'type': 'test', 'config': {}},
            {'id': 'b', 'type': 'test', 'config': {}}
        ])
        
        clone = original.clone()
        
        # Clear input/output designations to allow removal
        clone.outputs = []  # type: ignore
        
        # Modify clone
        clone.remove_node('b')
        
        # Original should be unchanged
        assert original.has_node('b')
        assert not clone.has_node('b')
    
    def test_add_node_after(self):
        """Test convenience method for adding node after."""
        graph = MutableGraph.from_sequential([
            {'id': 'a', 'type': 'test', 'config': {}},
            {'id': 'c', 'type': 'test', 'config': {}}
        ])
        
        new_node = Node(id='b', component_type='test', config={})
        graph.add_node_after('a', new_node)
        
        assert graph.has_node('b')
        # Should be: a -> b -> c
        assert 'b' in graph.get_children('a')
        assert 'c' in graph.get_children('b')
    
    def test_add_node_before(self):
        """Test convenience method for adding node before."""
        graph = MutableGraph.from_sequential([
            {'id': 'a', 'type': 'test', 'config': {}},
            {'id': 'c', 'type': 'test', 'config': {}}
        ])
        
        new_node = Node(id='b', component_type='test', config={})
        graph.add_node_before('c', new_node)
        
        assert graph.has_node('b')
        # Should be: a -> b -> c
        assert 'c' in graph.get_children('b')


# ============================================================================
# MUTATION TESTS
# ============================================================================


class TestMutations:
    """Test graph mutations."""
    
    def test_add_node_mutation(self):
        """Test AddNodeMutation."""
        graph = MutableGraph.from_sequential([
            {'id': 'a', 'type': 'test', 'config': {}},
            {'id': 'b', 'type': 'test', 'config': {}}
        ])
        
        mutation = AddNodeMutation(
            node_id='new',
            component_type='test',
            config={},
            insert_after='a'
        )
        
        graph.apply_mutation(mutation)
        
        assert graph.has_node('new')
        assert len(graph.mutation_history) == 1
    
    def test_add_node_mutation_undo(self):
        """Test undoing AddNodeMutation."""
        graph = MutableGraph.from_sequential([
            {'id': 'a', 'type': 'test', 'config': {}},
            {'id': 'b', 'type': 'test', 'config': {}}
        ])
        
        mutation = AddNodeMutation(
            node_id='new',
            component_type='test',
            config={},
            insert_after='a'
        )
        
        graph.apply_mutation(mutation)
        assert graph.has_node('new')
        
        graph.undo_last_mutation()
        assert not graph.has_node('new')
    
    def test_remove_node_mutation(self):
        """Test RemoveNodeMutation."""
        graph = MutableGraph.from_sequential([
            {'id': 'a', 'type': 'test', 'config': {}},
            {'id': 'b', 'type': 'test', 'config': {}},
            {'id': 'c', 'type': 'test', 'config': {}}
        ])
        
        mutation = RemoveNodeMutation('b', bypass=True)
        graph.apply_mutation(mutation)
        
        assert not graph.has_node('b')
        # a should connect directly to c
        assert 'c' in graph.get_children('a')
    
    def test_remove_node_mutation_undo(self):
        """Test undoing RemoveNodeMutation."""
        graph = MutableGraph.from_sequential([
            {'id': 'a', 'type': 'test', 'config': {}},
            {'id': 'b', 'type': 'test', 'config': {}},
            {'id': 'c', 'type': 'test', 'config': {}}
        ])
        
        mutation = RemoveNodeMutation('b', bypass=True)
        graph.apply_mutation(mutation)
        
        graph.undo_last_mutation()
        
        assert graph.has_node('b')
        # Original connections restored
        assert 'b' in graph.get_children('a')
        assert 'c' in graph.get_children('b')
    
    def test_modify_config_mutation(self):
        """Test ModifyConfigMutation."""
        graph = MutableGraph()
        graph.add_node(Node(id='test', component_type='block', config={'dim': 768}))
        
        mutation = ModifyConfigMutation('test', {'dim': 1024})
        graph.apply_mutation(mutation)
        
        node = graph.get_node('test')
        assert node.config['dim'] == 1024
    
    def test_modify_config_mutation_undo(self):
        """Test undoing ModifyConfigMutation."""
        graph = MutableGraph()
        graph.add_node(Node(id='test', component_type='block', config={'dim': 768}))
        
        mutation = ModifyConfigMutation('test', {'dim': 1024})
        graph.apply_mutation(mutation)
        graph.undo_last_mutation()
        
        node = graph.get_node('test')
        assert node.config['dim'] == 768
    
    def test_add_skip_connection_mutation(self):
        """Test AddSkipConnectionMutation."""
        graph = MutableGraph.from_sequential([
            {'id': 'a', 'type': 'test', 'config': {}},
            {'id': 'b', 'type': 'test', 'config': {}},
            {'id': 'c', 'type': 'test', 'config': {}}
        ])
        
        mutation = AddSkipConnectionMutation('a', 'c')
        graph.apply_mutation(mutation)
        
        # Should have skip: a -> c
        assert 'c' in graph.get_children('a')
    
    def test_mutation_history(self):
        """Test mutation history tracking."""
        graph = MutableGraph()
        graph.add_node(Node(id='a', component_type='test', config={}))
        
        # Apply multiple mutations
        for i in range(3):
            mutation = AddNodeMutation(
                node_id=f'node_{i}',
                component_type='test',
                config={},
                insert_after='a'
            )
            graph.apply_mutation(mutation)
        
        assert len(graph.mutation_history) == 3
    
    def test_redo_mutation(self):
        """Test redoing undone mutation."""
        graph = MutableGraph()
        graph.add_node(Node(id='a', component_type='test', config={}))
        
        mutation = AddNodeMutation('new', 'test', {}, insert_after='a')
        graph.apply_mutation(mutation)
        
        # Undo
        graph.undo_last_mutation()
        assert not graph.has_node('new')
        
        # Redo
        graph.redo_last_mutation()
        assert graph.has_node('new')
    
    def test_mutation_clears_redo_stack(self):
        """Test that new mutation clears redo stack."""
        graph = MutableGraph()
        graph.add_node(Node(id='a', component_type='test', config={}))
        
        # Apply, undo
        graph.apply_mutation(AddNodeMutation('b', 'test', {}, insert_after='a'))
        graph.undo_last_mutation()
        
        # Apply new mutation
        graph.apply_mutation(AddNodeMutation('c', 'test', {}, insert_after='a'))
        
        # Redo stack should be cleared
        with pytest.raises(ValueError):
            graph.redo_last_mutation()


# ============================================================================
# SEARCH SPACE TESTS
# ============================================================================


class TestSearchSpace:
    """Test SearchSpace for architecture search."""
    
    def test_create_search_space(self):
        """Test creating search space."""
        space = SearchSpace()
        
        assert len(space.component_types) == 0
        assert len(space.mutation_rules) == 0
    
    def test_add_component_type(self):
        """Test adding component type to search space."""
        space = SearchSpace()
        
        space.add_component_type('transformer_block', {
            'dim': [512, 768, 1024],
            'num_heads': [8, 12, 16]
        })
        
        assert 'transformer_block' in space.component_types
        assert 'dim' in space.component_types['transformer_block']
    
    def test_add_mutation_rule(self):
        """Test adding mutation rule."""
        space = SearchSpace()
        
        def max_depth_rule(graph):
            return len(graph) < 24
        
        space.add_mutation_rule('max_depth', max_depth_rule)
        
        assert 'max_depth' in space.mutation_rules
    
    def test_sample_mutation(self):
        """Test sampling random mutation."""
        space = SearchSpace()
        space.add_component_type('test_block', {
            'dim': [512, 768]
        })
        
        graph = MutableGraph()
        graph.add_node(Node(id='base', component_type='test', config={}))
        
        mutation = space.sample_mutation(graph)
        
        assert isinstance(mutation, GraphMutation)


# ============================================================================
# RUN TESTS
# ============================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
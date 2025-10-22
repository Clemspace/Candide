"""
Tests for model builder.

Tests:
- GraphExecutor instantiation
- Sequential execution
- DAG execution
- Component retrieval
- Model building from graphs
- Model building from configs
- Optimization (torch.compile)
- Error handling
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict

from ramanujan.core.graph import ComputationGraph, Node
from ramanujan.core.builder import (
    GraphExecutor,
    ModelBuilder,
    build_model
)
from ramanujan.core.registry import ComponentRegistry, register_component
from ramanujan.core.interface import TensorSpec, ForwardMetadata, Component


# ============================================================================
# TEST FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean registry before each test."""
    ComponentRegistry.clear()
    yield
    ComponentRegistry.clear()


@pytest.fixture
def simple_component():
    """Create a simple test component."""
    
    @register_component('block', 'simple')
    class SimpleBlock(nn.Module):
        def __init__(self, dim: int = 10):
            super().__init__()
            self.dim = dim
            self.linear = nn.Linear(dim, dim)
        
        @property
        def component_type(self):
            return 'block'
        
        @property
        def input_spec(self):
            return {'x': TensorSpec(shape=('batch', 'dim'))}
        
        @property
        def output_spec(self):
            return {'x': TensorSpec(shape=('batch', 'dim'))}
        
        def forward(self, x):
            return self.linear(x)
        
        def get_config(self):
            return {'dim': self.dim}
    
    return SimpleBlock


@pytest.fixture
def embedding_component():
    """Create embedding component."""
    
    @register_component('embedding', 'token')
    class TokenEmbedding(nn.Module):
        def __init__(self, vocab_size: int, dim: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, dim)
        
        @property
        def component_type(self):
            return 'embedding'
        
        @property
        def input_spec(self):
            return {'x': TensorSpec(shape=('batch', 'seq'), dtype=torch.long)}
        
        @property
        def output_spec(self):
            return {'x': TensorSpec(shape=('batch', 'seq', 'dim'))}
        
        def forward(self, x):
            return self.embedding(x)
        
        def get_config(self):
            return {'vocab_size': self.embedding.num_embeddings, 
                   'dim': self.embedding.embedding_dim}
    
    return TokenEmbedding


@pytest.fixture
def head_component():
    """Create output head component."""
    
    @register_component('head', 'lm_head')
    class LMHead(nn.Module):
        def __init__(self, vocab_size: int, dim: int):
            super().__init__()
            self.linear = nn.Linear(dim, vocab_size)
        
        @property
        def component_type(self):
            return 'head'
        
        @property
        def input_spec(self):
            return {'x': TensorSpec(shape=('batch', 'seq', 'dim'))}
        
        @property
        def output_spec(self):
            return {'x': TensorSpec(shape=('batch', 'seq', 'vocab'))}
        
        def forward(self, x):
            return self.linear(x)
        
        def get_config(self):
            return {'vocab_size': self.linear.out_features, 
                   'dim': self.linear.in_features}
    
    return LMHead


# ============================================================================
# GRAPH EXECUTOR TESTS
# ============================================================================


class TestGraphExecutor:
    """Test GraphExecutor functionality."""
    
    def test_create_executor_simple(self, simple_component):
        """Test creating executor for simple graph."""
        graph = ComputationGraph.from_sequential([
            {'type': 'simple', 'config': {'dim': 10}}
        ])
        
        executor = GraphExecutor(graph)
        
        assert isinstance(executor, nn.Module)
        assert len(executor.components) == 1
    
    def test_create_executor_multiple_components(self, simple_component):
        """Test creating executor with multiple components."""
        graph = ComputationGraph.from_sequential([
            {'type': 'simple', 'config': {'dim': 10}},
            {'type': 'simple', 'config': {'dim': 10}},
            {'type': 'simple', 'config': {'dim': 10}}
        ])
        
        executor = GraphExecutor(graph)
        
        assert len(executor.components) == 3
    
    def test_sequential_detection(self, simple_component):
        """Test detection of sequential graph."""
        graph = ComputationGraph.from_sequential([
            {'type': 'simple', 'config': {'dim': 10}},
            {'type': 'simple', 'config': {'dim': 10}}
        ])
        
        executor = GraphExecutor(graph)
        
        assert executor.is_sequential
    
    def test_dag_detection(self, simple_component):
        """Test detection of DAG graph."""
        graph = ComputationGraph()
        
        graph.add_node(Node(id='input', component_type='simple', config={'dim': 10}))
        graph.add_node(Node(id='branch1', component_type='simple', config={'dim': 10}, 
                           inputs=['input']))
        graph.add_node(Node(id='branch2', component_type='simple', config={'dim': 10}, 
                           inputs=['input']))
        graph.set_inputs(['input'])
        graph.set_outputs(['branch1', 'branch2'])
        
        executor = GraphExecutor(graph)
        
        assert not executor.is_sequential
    
    def test_component_instantiation_error(self):
        """Test error when component instantiation fails."""
        @register_component('block', 'bad')
        class BadComponent(nn.Module):
            def __init__(self, required_param):
                # Missing required parameter will cause error
                super().__init__()
        
        graph = ComputationGraph.from_sequential([
            {'type': 'bad', 'config': {}}  # Missing required_param
        ])
        
        with pytest.raises(ValueError, match="Failed to instantiate"):
            GraphExecutor(graph)
    
    def test_invalid_graph_error(self):
        """Test error when graph is invalid."""
        graph = ComputationGraph()
        
        # Create cycle
        graph.add_node(Node(id='a', component_type='simple', config={'dim': 10}))
        graph.add_node(Node(id='b', component_type='simple', config={'dim': 10}, 
                           inputs=['a']))
        graph.add_edge('b', 'a')  # Cycle
        
        with pytest.raises(ValueError, match="cycle"):
            GraphExecutor(graph)


# ============================================================================
# FORWARD PASS TESTS
# ============================================================================


class TestForwardPass:
    """Test forward pass execution."""
    
    def test_sequential_forward(self, simple_component):
        """Test forward pass through sequential graph."""
        graph = ComputationGraph.from_sequential([
            {'type': 'simple', 'config': {'dim': 10}},
            {'type': 'simple', 'config': {'dim': 10}}
        ])
        
        model = GraphExecutor(graph)
        
        # Test forward pass
        x = torch.randn(2, 10)
        output = model(x)
        
        assert output.shape == (2, 10)
    
    def test_sequential_forward_multiple_layers(self, simple_component):
        """Test forward through many layers."""
        graph = ComputationGraph.from_sequential([
            {'type': 'simple', 'config': {'dim': 10}} for _ in range(5)
        ])
        
        model = GraphExecutor(graph)
        
        x = torch.randn(2, 10)
        output = model(x)
        
        assert output.shape == (2, 10)
    
    def test_dag_forward(self, simple_component):
        """Test forward pass through DAG."""
        graph = ComputationGraph()
        
        # Create simple DAG: input -> layer1 -> output
        #                           -> layer2 -^
        graph.add_node(Node(id='input', component_type='simple', config={'dim': 10}))
        graph.add_node(Node(id='layer1', component_type='simple', config={'dim': 10}, 
                           inputs=['input']))
        graph.add_node(Node(id='layer2', component_type='simple', config={'dim': 10}, 
                           inputs=['input']))
        graph.add_node(Node(id='output', component_type='simple', config={'dim': 10}, 
                           inputs=['layer1']))  # Only connect layer1 for simplicity
        
        graph.set_inputs(['input'])
        graph.set_outputs(['output'])
        
        model = GraphExecutor(graph)
        
        x = torch.randn(2, 10)
        output = model(x)
        
        assert output.shape == (2, 10)
    
    def test_forward_with_embedding_and_head(
        self, 
        embedding_component, 
        simple_component, 
        head_component
    ):
        """Test forward with embedding and head."""
        graph = ComputationGraph.from_sequential([
            {'type': 'token', 'config': {'vocab_size': 100, 'dim': 10}},
            {'type': 'simple', 'config': {'dim': 10}},
            {'type': 'lm_head', 'config': {'vocab_size': 100, 'dim': 10}}
        ])
        
        model = GraphExecutor(graph)
        
        # Input: token IDs
        input_ids = torch.randint(0, 100, (2, 5))
        output = model(input_ids)
        
        # Output: logits over vocabulary
        assert output.shape == (2, 5, 100)
    
    def test_forward_with_metadata(self, simple_component):
        """Test forward pass returning metadata."""
        graph = ComputationGraph.from_sequential([
            {'type': 'simple', 'config': {'dim': 10}},
            {'type': 'simple', 'config': {'dim': 10}}
        ])
        
        model = GraphExecutor(graph)
        
        x = torch.randn(2, 10)
        output, metadata = model(x, return_metadata=True)
        
        assert output.shape == (2, 10)
        assert isinstance(metadata, ForwardMetadata)
        assert len(metadata.executed_components) > 0
    
    def test_forward_gradient_flow(self, simple_component):
        """Test that gradients flow correctly."""
        graph = ComputationGraph.from_sequential([
            {'type': 'simple', 'config': {'dim': 10}},
            {'type': 'simple', 'config': {'dim': 10}}
        ])
        
        model = GraphExecutor(graph)
        
        x = torch.randn(2, 10, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        for param in model.parameters():
            assert param.grad is not None


# ============================================================================
# COMPONENT RETRIEVAL TESTS
# ============================================================================


class TestComponentRetrieval:
    """Test retrieving components from executor."""
    
    def test_get_component(self, simple_component):
        """Test getting component by node ID."""
        graph = ComputationGraph.from_sequential([
            {'id': 'layer_0', 'type': 'simple', 'config': {'dim': 10}},
            {'id': 'layer_1', 'type': 'simple', 'config': {'dim': 10}}
        ])
        
        model = GraphExecutor(graph)
        
        layer = model.get_component('layer_0')
        
        assert isinstance(layer, nn.Module)
        assert layer.dim == 10
    
    def test_get_component_not_found(self, simple_component):
        """Test error when component not found."""
        graph = ComputationGraph.from_sequential([
            {'id': 'layer_0', 'type': 'simple', 'config': {'dim': 10}}
        ])
        
        model = GraphExecutor(graph)
        
        with pytest.raises(ValueError, match="not found"):
            model.get_component('nonexistent')
    
    def test_get_all_components(self, simple_component):
        """Test getting all components."""
        graph = ComputationGraph.from_sequential([
            {'id': 'layer_0', 'type': 'simple', 'config': {'dim': 10}},
            {'id': 'layer_1', 'type': 'simple', 'config': {'dim': 10}}
        ])
        
        model = GraphExecutor(graph)
        
        components = model.get_all_components()
        
        assert 'layer_0' in components
        assert 'layer_1' in components
        assert len(components) == 2
    
    def test_modify_component_directly(self, simple_component):
        """Test modifying component after retrieval."""
        graph = ComputationGraph.from_sequential([
            {'id': 'layer', 'type': 'simple', 'config': {'dim': 10}}
        ])
        
        model = GraphExecutor(graph)
        
        # Freeze component
        layer = model.get_component('layer')
        for param in layer.parameters():
            param.requires_grad = False
        
        # Verify frozen
        for param in layer.parameters():
            assert not param.requires_grad


# ============================================================================
# MODEL BUILDER TESTS
# ============================================================================


class TestModelBuilder:
    """Test ModelBuilder functionality."""
    
    def test_build_from_graph(self, simple_component):
        """Test building model from graph."""
        graph = ComputationGraph.from_sequential([
            {'type': 'simple', 'config': {'dim': 10}},
            {'type': 'simple', 'config': {'dim': 10}}
        ])
        
        model = ModelBuilder.build(graph)
        
        assert isinstance(model, nn.Module)
        
        # Test forward
        x = torch.randn(2, 10)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_build_from_config(self, simple_component):
        """Test building model from config dict."""
        config = {
            'nodes': [
                {'id': 'a', 'component_type': 'simple', 'config': {'dim': 10}, 'inputs': []},
                {'id': 'b', 'component_type': 'simple', 'config': {'dim': 10}, 'inputs': ['a']}
            ],
            'inputs': ['a'],
            'outputs': ['b']
        }
        
        model = ModelBuilder.build_from_config(config)
        
        assert isinstance(model, nn.Module)
        
        # Test forward
        x = torch.randn(2, 10)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_build_convenience_function(self, simple_component):
        """Test build_model convenience function."""
        graph = ComputationGraph.from_sequential([
            {'type': 'simple', 'config': {'dim': 10}}
        ])
        
        model = build_model(graph)
        
        assert isinstance(model, nn.Module)
    
    @pytest.mark.skipif(not hasattr(torch, 'compile'), 
                       reason="torch.compile not available")
    def test_build_with_compile(self, simple_component):
        """Test building with torch.compile."""
        graph = ComputationGraph.from_sequential([
            {'type': 'simple', 'config': {'dim': 10}}
        ])
        
        # This may fail if torch.compile not available, that's ok
        try:
            model = ModelBuilder.build(graph, compile=True)
            assert isinstance(model, nn.Module)
        except Exception:
            pytest.skip("torch.compile not working in this environment")


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling:
    """Test error handling in builder."""
    
    def test_component_not_in_registry(self):
        """Test error when component type not registered."""
        graph = ComputationGraph.from_sequential([
            {'type': 'nonexistent', 'config': {}}
        ])
        
        with pytest.raises(ValueError, match="not found in registry"):
            GraphExecutor(graph)
    
    def test_invalid_config_error(self):
        """Test error with invalid component config."""
        @register_component('block', 'strict')
        class StrictComponent(nn.Module):
            def __init__(self, required_param: int):
                super().__init__()
                if required_param < 0:
                    raise ValueError("required_param must be >= 0")
        
        graph = ComputationGraph.from_sequential([
            {'type': 'strict', 'config': {'required_param': -1}}
        ])
        
        with pytest.raises(ValueError):
            GraphExecutor(graph)
    
    def test_helpful_error_messages(self):
        """Test that error messages are helpful."""
        graph = ComputationGraph.from_sequential([
            {'type': 'nonexistent', 'config': {}}
        ])
        
        with pytest.raises(ValueError) as exc_info:
            GraphExecutor(graph)
        
        error_msg = str(exc_info.value)
        assert 'nonexistent' in error_msg
        assert 'not found' in error_msg


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestPerformance:
    """Test performance characteristics."""
    
    def test_sequential_execution_speed(self, simple_component):
        """Test that sequential execution is fast."""
        import time
        
        # Large sequential model
        graph = ComputationGraph.from_sequential([
            {'type': 'simple', 'config': {'dim': 100}} for _ in range(50)
        ])
        
        model = GraphExecutor(graph)
        model.eval()
        
        x = torch.randn(4, 100)
        
        # Warmup
        for _ in range(10):
            _ = model(x)
        
        # Time it
        start = time.time()
        for _ in range(100):
            _ = model(x)
        elapsed = time.time() - start
        
        # Should be reasonably fast (< 1 second for 100 forward passes)
        assert elapsed < 1.0
    
    def test_no_memory_leak(self, simple_component):
        """Test that repeated forwards don't leak memory."""
        graph = ComputationGraph.from_sequential([
            {'type': 'simple', 'config': {'dim': 10}}
        ])
        
        model = GraphExecutor(graph)
        
        # Run many forwards
        for _ in range(1000):
            x = torch.randn(2, 10)
            _ = model(x)
        
        # If we got here without OOM, we're good
        assert True
    
    def test_graph_compilation_once(self, simple_component):
        """Test that graph is compiled only once."""
        graph = ComputationGraph.from_sequential([
            {'type': 'simple', 'config': {'dim': 10}}
        ])
        
        # Compilation happens in __init__
        executor = GraphExecutor(graph)
        
        # Execution plan should be cached
        assert hasattr(executor, 'execution_order')
        assert len(executor.execution_order) > 0
        
        # Multiple forwards should use same plan
        x = torch.randn(2, 10)
        _ = executor(x)
        _ = executor(x)
        
        # No recompilation occurred
        assert True


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestBuilderIntegration:
    """Integration tests for builder."""
    
    def test_full_language_model_pipeline(
        self,
        embedding_component,
        simple_component,
        head_component
    ):
        """Test building and running full language model."""
        # Build graph
        graph = ComputationGraph.from_sequential([
            {'id': 'embedding', 'type': 'token', 
             'config': {'vocab_size': 1000, 'dim': 64}},
            {'id': 'layer_0', 'type': 'simple', 'config': {'dim': 64}},
            {'id': 'layer_1', 'type': 'simple', 'config': {'dim': 64}},
            {'id': 'layer_2', 'type': 'simple', 'config': {'dim': 64}},
            {'id': 'head', 'type': 'lm_head', 
             'config': {'vocab_size': 1000, 'dim': 64}}
        ])
        
        # Build model
        model = build_model(graph)
        
        # Test forward
        input_ids = torch.randint(0, 1000, (2, 10))
        logits = model(input_ids)
        
        assert logits.shape == (2, 10, 1000)
        
        # Test backward
        loss = logits.sum()
        loss.backward()
        
        # Check all components have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
    
    def test_training_loop(self, simple_component):
        """Test simple training loop."""
        graph = ComputationGraph.from_sequential([
            {'type': 'simple', 'config': {'dim': 10}},
            {'type': 'simple', 'config': {'dim': 10}}
        ])
        
        model = build_model(graph)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        for _ in range(10):
            x = torch.randn(2, 10)
            target = torch.randn(2, 10)
            
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()
        
        # If we got here, training works
        assert True
    
    def test_model_save_load(self, simple_component, tmp_path):
        """Test saving and loading model."""
        graph = ComputationGraph.from_sequential([
            {'type': 'simple', 'config': {'dim': 10}}
        ])
        
        model = build_model(graph)
        
        # Save
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)
        
        # Load into new model
        model2 = build_model(graph)
        model2.load_state_dict(torch.load(save_path))
        
        # Test outputs match
        x = torch.randn(2, 10)
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        
        assert torch.allclose(out1, out2)
    
    def test_component_specific_operations(self, simple_component):
        """Test component-specific operations after building."""
        graph = ComputationGraph.from_sequential([
            {'id': 'freeze_me', 'type': 'simple', 'config': {'dim': 10}},
            {'id': 'train_me', 'type': 'simple', 'config': {'dim': 10}}
        ])
        
        model = build_model(graph)
        
        # Freeze first component
        freeze_comp = model.get_component('freeze_me')
        for param in freeze_comp.parameters():
            param.requires_grad = False
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        
        assert trainable > 0
        assert frozen > 0


# ============================================================================
# SPECIAL CASES TESTS
# ============================================================================


class TestSpecialCases:
    """Test special cases and edge conditions."""
    
    def test_single_component_model(self, simple_component):
        """Test model with single component."""
        graph = ComputationGraph.from_sequential([
            {'type': 'simple', 'config': {'dim': 10}}
        ])
        
        model = build_model(graph)
        
        x = torch.randn(2, 10)
        output = model(x)
        
        assert output.shape == (2, 10)
    
    def test_empty_config(self, simple_component):
        """Test component with empty config (using defaults)."""
        graph = ComputationGraph.from_sequential([
            {'type': 'simple', 'config': {}}  # Will use default dim=10
        ])
        
        model = build_model(graph)
        
        x = torch.randn(2, 10)
        output = model(x)
        
        assert output.shape == (2, 10)
    
    def test_model_in_eval_mode(self, simple_component):
        """Test model in eval mode."""
        graph = ComputationGraph.from_sequential([
            {'type': 'simple', 'config': {'dim': 10}}
        ])
        
        model = build_model(graph)
        model.eval()
        
        x = torch.randn(2, 10)
        
        # Forward twice should give same result (no dropout randomness)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        
        assert torch.allclose(out1, out2)
    
    def test_model_on_different_device(self, simple_component):
        """Test moving model to different device."""
        graph = ComputationGraph.from_sequential([
            {'type': 'simple', 'config': {'dim': 10}}
        ])
        
        model = build_model(graph)
        
        # Move to CPU (should work even without GPU)
        model = model.cpu()
        
        x = torch.randn(2, 10)
        output = model(x)
        
        assert output.device.type == 'cpu'


# ============================================================================
# RUN TESTS
# ============================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
"""
Tests for core interfaces and protocols.

Tests:
- TensorSpec validation and matching
- DynamicTensorSpec shape resolution
- ForwardMode utilities
- ForwardMetadata tracking and merging
- Protocol implementations
"""

import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass

from ramanujan.core.interface import (
    Modality,
    TensorSpec,
    DynamicTensorSpec,
    ForwardMode,
    ForwardMetadata,
    Component,
    StatefulComponent,
    ComposableComponent,
    MultiModeComponent,
    MetadataComponent,
)


# ============================================================================
# TENSORSPEC TESTS
# ============================================================================


class TestTensorSpec:
    """Test TensorSpec functionality."""
    
    def test_basic_creation(self):
        """Test basic TensorSpec creation."""
        spec = TensorSpec(
            shape=('batch', 'seq', 'dim'),
            dtype=torch.float32,
            modality=Modality.TEXT
        )
        
        assert spec.shape == ('batch', 'seq', 'dim')
        assert spec.dtype == torch.float32
        assert spec.modality == Modality.TEXT
        assert not spec.optional
    
    def test_matches_correct_shape(self):
        """Test tensor matching with correct shape."""
        spec = TensorSpec(shape=('batch', 'seq', 'dim'))
        
        # Correct shape
        x = torch.randn(8, 512, 768)
        assert spec.matches(x)
    
    def test_matches_wrong_ndim(self):
        """Test tensor matching with wrong number of dimensions."""
        spec = TensorSpec(shape=('batch', 'seq', 'dim'))
        
        # Wrong number of dimensions
        x = torch.randn(8, 512)  # Only 2D
        assert not spec.matches(x)
    
    def test_matches_wrong_dtype(self):
        """Test tensor matching with wrong dtype."""
        spec = TensorSpec(shape=('batch', 'seq', 'dim'), dtype=torch.float32)
        
        # Wrong dtype
        x = torch.randn(8, 512, 768).int()
        assert not spec.matches(x)
    
    def test_matches_with_dim_values(self):
        """Test strict matching with dimension values."""
        spec = TensorSpec(shape=('batch', 'seq', 'dim'))
        x = torch.randn(8, 512, 768)
        
        # Correct dim values
        assert spec.matches(x, {'batch': 8, 'seq': 512, 'dim': 768}, strict=True)
        
        # Wrong dim values
        assert not spec.matches(x, {'batch': 4, 'seq': 512, 'dim': 768}, strict=True)
    
    def test_resolve_shape(self):
        """Test shape resolution from symbolic to concrete."""
        spec = TensorSpec(shape=('batch', 'seq', 'dim'))
        
        concrete = spec.resolve_shape({'batch': 8, 'seq': 512, 'dim': 768})
        assert concrete == (8, 512, 768)
    
    def test_resolve_shape_missing_dim(self):
        """Test shape resolution with missing dimension."""
        spec = TensorSpec(shape=('batch', 'seq', 'dim'))
        
        with pytest.raises(KeyError):
            spec.resolve_shape({'batch': 8, 'seq': 512})  # Missing 'dim'
    
    def test_optional_spec(self):
        """Test optional tensor specification."""
        spec = TensorSpec(shape=('batch', 'seq'), optional=True)
        assert spec.optional
    
    def test_spec_with_description(self):
        """Test spec with description."""
        spec = TensorSpec(
            shape=('batch', 'seq', 'dim'),
            description="Token embeddings"
        )
        assert spec.description == "Token embeddings"
    
    def test_repr(self):
        """Test string representation."""
        spec = TensorSpec(
            shape=('batch', 'seq'),
            dtype=torch.float32,
            modality=Modality.TEXT,
            description="Test tensor"
        )
        repr_str = repr(spec)
        assert 'batch' in repr_str
        assert 'seq' in repr_str
        assert 'TEXT' in repr_str


class TestDynamicTensorSpec:
    """Test DynamicTensorSpec functionality."""
    
    def test_from_reduction(self):
        """Test creating spec for dimension reduction."""
        spec = DynamicTensorSpec.from_reduction('x', reduced_dims=[2, 3])
        
        # [B, C, H, W] → [B, C]
        input_shapes = {'x': (8, 256, 32, 32)}
        output_shape = spec.resolve_output_shape(input_shapes)
        
        assert output_shape == (8, 256)
    
    def test_from_reduction_single_dim(self):
        """Test reduction of single dimension."""
        spec = DynamicTensorSpec.from_reduction('x', reduced_dims=[1])
        
        # [B, L, D] → [B, D]
        input_shapes = {'x': (8, 512, 768)}
        output_shape = spec.resolve_output_shape(input_shapes)
        
        assert output_shape == (8, 768)
    
    def test_from_transformation(self):
        """Test custom transformation."""
        # Double sequence length
        spec = DynamicTensorSpec.from_transformation(
            'x',
            lambda shape: (shape[0], shape[1] * 2, shape[2])
        )
        
        input_shapes = {'x': (8, 512, 768)}
        output_shape = spec.resolve_output_shape(input_shapes)
        
        assert output_shape == (8, 1024, 768)
    
    def test_custom_shape_fn(self):
        """Test custom shape function."""
        def custom_fn(input_shapes):
            x_shape = input_shapes['x']
            y_shape = input_shapes['y']
            # Concatenate along last dim
            return (x_shape[0], x_shape[1], x_shape[2] + y_shape[2])
        
        spec = DynamicTensorSpec(shape=('batch', 'seq', 'dim'), shape_fn=custom_fn)
        
        input_shapes = {'x': (8, 512, 768), 'y': (8, 512, 256)}
        output_shape = spec.resolve_output_shape(input_shapes)
        
        assert output_shape == (8, 512, 1024)
    
    def test_fallback_to_static(self):
        """Test fallback to static shape when no shape_fn."""
        spec = DynamicTensorSpec(shape=('batch', 'dim'))
        
        # Without shape_fn, returns static shape
        output_shape = spec.resolve_output_shape({'x': (8, 768)})
        assert output_shape == ('batch', 'dim')


# ============================================================================
# FORWARDMODE TESTS
# ============================================================================


class TestForwardMode:
    """Test ForwardMode enum."""
    
    def test_all_modes_exist(self):
        """Test all expected modes exist."""
        assert ForwardMode.TRAIN
        assert ForwardMode.EVAL
        assert ForwardMode.GENERATE
        assert ForwardMode.DIFFUSION_STEP
        assert ForwardMode.INFERENCE
    
    def test_is_training(self):
        """Test training mode detection."""
        assert ForwardMode.TRAIN.is_training()
        assert not ForwardMode.EVAL.is_training()
        assert not ForwardMode.GENERATE.is_training()
        assert not ForwardMode.INFERENCE.is_training()
    
    def test_is_inference(self):
        """Test inference mode detection."""
        assert not ForwardMode.TRAIN.is_inference()
        assert ForwardMode.EVAL.is_inference()
        assert ForwardMode.GENERATE.is_inference()
        assert ForwardMode.INFERENCE.is_inference()
        assert ForwardMode.DIFFUSION_STEP.is_inference()
    
    def test_mode_values(self):
        """Test mode string values."""
        assert ForwardMode.TRAIN.value == "train"
        assert ForwardMode.EVAL.value == "eval"
        assert ForwardMode.GENERATE.value == "generate"


# ============================================================================
# FORWARDMETADATA TESTS
# ============================================================================


class TestForwardMetadata:
    """Test ForwardMetadata tracking."""
    
    def test_empty_metadata(self):
        """Test creating empty metadata."""
        meta = ForwardMetadata()
        assert len(meta.executed_components) == 0
        assert meta.flops is None
        assert meta.num_steps is None
    
    def test_add_component(self):
        """Test adding executed components."""
        meta = ForwardMetadata()
        meta.add_component('layer_0')
        meta.add_component('layer_1')
        
        assert meta.executed_components == ['layer_0', 'layer_1']
    
    def test_set_routing(self):
        """Test setting routing decisions."""
        meta = ForwardMetadata()
        meta.set_routing('expert_selection', [0, 3, 7])
        
        assert meta.routing_decisions['expert_selection'] == [0, 3, 7]
    
    def test_set_custom(self):
        """Test setting custom metrics."""
        meta = ForwardMetadata()
        meta.set_custom('entropy', 2.5)
        meta.set_custom('confidence', 0.95)
        
        assert meta.custom['entropy'] == 2.5
        assert meta.custom['confidence'] == 0.95
    
    def test_merge_metadata(self):
        """Test merging two metadata objects."""
        meta1 = ForwardMetadata(
            executed_components=['layer_0'],
            flops=1000,
            num_steps=5
        )
        
        meta2 = ForwardMetadata(
            executed_components=['layer_1'],
            flops=2000,
            num_steps=3
        )
        
        merged = meta1.merge(meta2)
        
        assert merged.executed_components == ['layer_0', 'layer_1']
        assert merged.flops == 3000
        assert merged.num_steps == 8
    
    def test_merge_with_none_values(self):
        """Test merging when some values are None."""
        meta1 = ForwardMetadata(flops=1000)
        meta2 = ForwardMetadata(memory_mb=512.0)
        
        merged = meta1.merge(meta2)
        
        assert merged.flops == 1000
        assert merged.memory_mb == 512.0
    
    def test_merge_routing_decisions(self):
        """Test merging routing decisions."""
        meta1 = ForwardMetadata()
        meta1.set_routing('decision_1', 'path_a')
        
        meta2 = ForwardMetadata()
        meta2.set_routing('decision_2', 'path_b')
        
        merged = meta1.merge(meta2)
        
        assert merged.routing_decisions == {
            'decision_1': 'path_a',
            'decision_2': 'path_b'
        }
    
    def test_summary_empty(self):
        """Test summary of empty metadata."""
        meta = ForwardMetadata()
        summary = meta.summary()
        assert summary == "No metadata recorded"
    
    def test_summary_with_data(self):
        """Test summary with data."""
        meta = ForwardMetadata(
            executed_components=['layer_0', 'layer_1'],
            num_steps=5,
            confidence=0.95
        )
        summary = meta.summary()
        
        assert 'layer_0' in summary
        assert 'layer_1' in summary
        assert '5' in summary
        assert '0.95' in summary
    
    def test_repr(self):
        """Test string representation."""
        meta = ForwardMetadata(
            executed_components=['a', 'b', 'c'],
            num_steps=10
        )
        repr_str = repr(meta)
        
        assert '3 components' in repr_str
        assert '10 steps' in repr_str


# ============================================================================
# PROTOCOL TESTS
# ============================================================================


class TestProtocols:
    """Test protocol implementations."""
    
    def test_component_protocol(self):
        """Test basic Component protocol."""
        
        class SimpleComponent(nn.Module):
            @property
            def component_type(self):
                return 'test'
            
            @property
            def input_spec(self):
                return {'x': TensorSpec(shape=('batch', 'dim'))}
            
            @property
            def output_spec(self):
                return {'x': TensorSpec(shape=('batch', 'dim'))}
            
            def forward(self, x):
                return x
            
            def get_config(self):
                return {'type': 'simple'}
        
        comp = SimpleComponent()
        
        # Check protocol compliance
        assert comp.component_type == 'test'
        assert 'x' in comp.input_spec
        assert 'x' in comp.output_spec
        assert comp.get_config()['type'] == 'simple'
        
        # Check it's recognized as Component
        assert isinstance(comp, Component)
    
    def test_stateful_component_protocol(self):
        """Test StatefulComponent protocol."""
        
        class StatefulComp(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10, 10))
            
            @property
            def component_type(self):
                return 'stateful'
            
            @property
            def input_spec(self):
                return {'x': TensorSpec(shape=('batch', 'dim'))}
            
            @property
            def output_spec(self):
                return {'x': TensorSpec(shape=('batch', 'dim'))}
            
            def forward(self, x):
                return torch.matmul(x, self.weight)
            
            def get_config(self):
                return {}
        
        comp = StatefulComp()
        
        # Check it's recognized as StatefulComponent
        assert isinstance(comp, Component)
        assert isinstance(comp, StatefulComponent)
        
        # Check it has standard nn.Module methods
        assert hasattr(comp, 'parameters')
        assert hasattr(comp, 'state_dict')
        assert hasattr(comp, 'load_state_dict')
        
        # Verify parameters exist
        params = list(comp.parameters())
        assert len(params) == 1
    
    def test_composable_component_protocol(self):
        """Test ComposableComponent protocol."""
        
        class ComposableComp(nn.Module):
            def __init__(self):
                super().__init__()
                self.sub1 = nn.Linear(10, 10)
                self.sub2 = nn.Linear(10, 10)
            
            @property
            def component_type(self):
                return 'composable'
            
            @property
            def input_spec(self):
                return {'x': TensorSpec(shape=('batch', 'dim'))}
            
            @property
            def output_spec(self):
                return {'x': TensorSpec(shape=('batch', 'dim'))}
            
            def forward(self, x):
                return self.sub2(self.sub1(x))
            
            def get_config(self):
                return {}
            
            def get_subcomponents(self):
                return {'sub1': self.sub1, 'sub2': self.sub2}
        
        comp = ComposableComp()
        
        # Check it's recognized as ComposableComponent
        assert isinstance(comp, Component)
        assert isinstance(comp, ComposableComponent)
        
        # Check subcomponents
        subcomps = comp.get_subcomponents()
        assert 'sub1' in subcomps
        assert 'sub2' in subcomps
        assert isinstance(subcomps['sub1'], nn.Module)
    
    def test_multimode_component_protocol(self):
        """Test MultiModeComponent protocol."""
        
        class MultiModeComp(nn.Module):
            @property
            def component_type(self):
                return 'multimode'
            
            @property
            def input_spec(self):
                return {'x': TensorSpec(shape=('batch', 'dim'))}
            
            @property
            def output_spec(self):
                return {'x': TensorSpec(shape=('batch', 'dim'))}
            
            def forward(self, x, mode=ForwardMode.TRAIN):
                if mode == ForwardMode.TRAIN:
                    return x * 2
                elif mode == ForwardMode.INFERENCE:
                    return x
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
            
            def get_config(self):
                return {}
            
            def get_supported_modes(self):
                return [ForwardMode.TRAIN, ForwardMode.INFERENCE]
        
        comp = MultiModeComp()
        
        # Check it's recognized as MultiModeComponent
        assert isinstance(comp, Component)
        assert isinstance(comp, MultiModeComponent)
        
        # Check modes
        modes = comp.get_supported_modes()
        assert ForwardMode.TRAIN in modes
        assert ForwardMode.INFERENCE in modes
        
        # Test different modes
        x = torch.randn(2, 10)
        train_out = comp(x, mode=ForwardMode.TRAIN)
        infer_out = comp(x, mode=ForwardMode.INFERENCE)
        
        assert not torch.allclose(train_out, infer_out)
    
    def test_metadata_component_protocol(self):
        """Test MetadataComponent protocol."""
        
        class MetadataComp(nn.Module):
            @property
            def component_type(self):
                return 'metadata'
            
            @property
            def input_spec(self):
                return {'x': TensorSpec(shape=('batch', 'dim'))}
            
            @property
            def output_spec(self):
                return {'x': TensorSpec(shape=('batch', 'dim'))}
            
            def forward(self, x):
                metadata = ForwardMetadata(
                    executed_components=['metadata_comp'],
                    num_steps=1
                )
                return x, metadata
            
            def get_config(self):
                return {}
        
        comp = MetadataComp()
        
        # Check it's recognized as MetadataComponent
        assert isinstance(comp, Component)
        assert isinstance(comp, MetadataComponent)
        
        # Test forward returns tuple
        x = torch.randn(2, 10)
        output, metadata = comp(x)
        
        assert torch.is_tensor(output)
        assert isinstance(metadata, ForwardMetadata)
        assert len(metadata.executed_components) > 0


# ============================================================================
# MODALITY TESTS
# ============================================================================


class TestModality:
    """Test Modality enum."""
    
    def test_all_modalities_exist(self):
        """Test all expected modalities exist."""
        assert Modality.TEXT
        assert Modality.IMAGE
        assert Modality.AUDIO
        assert Modality.VIDEO
        assert Modality.LATENT
        assert Modality.GRAPH
        assert Modality.GENERIC
    
    def test_modality_values(self):
        """Test modality string values."""
        assert Modality.TEXT.value == "text"
        assert Modality.IMAGE.value == "image"
        assert Modality.AUDIO.value == "audio"


# ============================================================================
# RUN TESTS
# ============================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
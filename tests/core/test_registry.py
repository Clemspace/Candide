"""
Tests for component registry.

Tests:
- Component registration
- Component retrieval
- Registry validation
- Error handling
- Metadata tracking
"""

import pytest
import torch.nn as nn

from ramanujan.core.registry import (
    ComponentRegistry,
    register_component,
    get_component,
    create_component
)
from ramanujan.core.interface import TensorSpec


# ============================================================================
# TEST FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean registry before each test."""
    # Clear all categories
    ComponentRegistry.clear()
    yield
    # Clean up after test
    ComponentRegistry.clear()


@pytest.fixture
def sample_component():
    """Create a sample component class."""
    
    class SampleComponent(nn.Module):
        def __init__(self, dim: int = 10):
            super().__init__()
            self.dim = dim
            self.linear = nn.Linear(dim, dim)
        
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
            return self.linear(x)
        
        def get_config(self):
            return {'dim': self.dim}
    
    return SampleComponent


# ============================================================================
# REGISTRATION TESTS
# ============================================================================


class TestRegistration:
    """Test component registration."""
    
    def test_register_component(self, sample_component):
        """Test registering a component."""
        # Register
        register_component('test', 'sample')(sample_component)
        
        # Verify registration
        assert ComponentRegistry.has('test', 'sample')
    
    def test_register_via_decorator(self):
        """Test registration via decorator syntax."""
        
        @register_component('test', 'decorated')
        class DecoratedComponent(nn.Module):
            pass
        
        assert ComponentRegistry.has('test', 'decorated')
    
    def test_register_duplicate_error(self, sample_component):
        """Test error when registering duplicate component."""
        # Register once
        register_component('test', 'sample')(sample_component)
        
        # Try to register again - should fail
        with pytest.raises(ValueError, match="already registered"):
            register_component('test', 'sample')(sample_component)
    
    def test_register_duplicate_with_override(self, sample_component):
        """Test overriding existing component."""
        # Register once
        register_component('test', 'sample')(sample_component)
        
        # Override
        class NewComponent(nn.Module):
            pass
        
        register_component('test', 'sample', override=True)(NewComponent)
        
        # Verify new component is registered
        retrieved = ComponentRegistry.get('test', 'sample')
        assert retrieved is NewComponent
    
    def test_register_multiple_categories(self, sample_component):
        """Test registering components in different categories."""
        register_component('category1', 'comp1')(sample_component)
        register_component('category2', 'comp2')(sample_component)
        
        assert ComponentRegistry.has('category1', 'comp1')
        assert ComponentRegistry.has('category2', 'comp2')
    
    def test_register_auto_create_category(self, sample_component):
        """Test auto-creation of new category."""
        # Register in non-existent category
        register_component('new_category', 'comp')(sample_component)
        
        # Verify category was created
        assert 'new_category' in ComponentRegistry.list_categories()
        assert ComponentRegistry.has('new_category', 'comp')


# ============================================================================
# RETRIEVAL TESTS
# ============================================================================


class TestRetrieval:
    """Test component retrieval."""
    
    def test_get_component(self, sample_component):
        """Test retrieving registered component."""
        register_component('test', 'sample')(sample_component)
        
        retrieved = ComponentRegistry.get('test', 'sample')
        assert retrieved is sample_component
    
    def test_get_component_not_found(self):
        """Test error when component not found."""
        with pytest.raises(ValueError, match="not found"):
            ComponentRegistry.get('test', 'nonexistent')
    
    def test_get_component_category_not_found(self):
        """Test error when category not found."""
        with pytest.raises(ValueError, match="Category.*not found"):
            ComponentRegistry.get('nonexistent_category', 'comp')
    
    def test_get_component_convenience_function(self, sample_component):
        """Test get_component convenience function."""
        register_component('test', 'sample')(sample_component)
        
        retrieved = get_component('test', 'sample')
        assert retrieved is sample_component
    
    def test_has_component(self, sample_component):
        """Test checking if component exists."""
        assert not ComponentRegistry.has('test', 'sample')
        
        register_component('test', 'sample')(sample_component)
        
        assert ComponentRegistry.has('test', 'sample')
    
    def test_list_categories(self, sample_component):
        """Test listing all categories."""
        register_component('cat1', 'comp1')(sample_component)
        register_component('cat2', 'comp2')(sample_component)
        
        categories = ComponentRegistry.list_categories()
        
        assert 'cat1' in categories
        assert 'cat2' in categories
    
    def test_list_components(self, sample_component):
        """Test listing components in a category."""
        register_component('test', 'comp1')(sample_component)
        register_component('test', 'comp2')(sample_component)
        
        components = ComponentRegistry.list_components('test')
        
        assert 'comp1' in components
        assert 'comp2' in components
    
    def test_list_components_with_metadata(self, sample_component):
        """Test listing components with metadata."""
        sample_component.__doc__ = "Test component documentation"
        register_component('test', 'sample')(sample_component)
        
        components = ComponentRegistry.list_components('test', include_metadata=True)
        
        assert 'sample' in components
        assert 'module' in components['sample']
        assert 'class' in components['sample']
        assert 'doc' in components['sample']
    
    def test_list_components_empty_category(self):
        """Test listing components from empty category."""
        components = ComponentRegistry.list_components('nonexistent')
        assert components == []


# ============================================================================
# CREATE COMPONENT TESTS
# ============================================================================


class TestCreateComponent:
    """Test component instantiation."""
    
    def test_create_component_basic(self, sample_component):
        """Test creating component instance."""
        register_component('test', 'sample')(sample_component)
        
        instance = create_component('test', 'sample', {'dim': 20})
        
        assert isinstance(instance, sample_component)
        assert instance.dim == 20
    
    def test_create_component_default_config(self, sample_component):
        """Test creating with default config."""
        register_component('test', 'sample')(sample_component)
        
        instance = create_component('test', 'sample', {})
        
        assert isinstance(instance, sample_component)
        assert instance.dim == 10  # Default value
    
    def test_create_component_invalid_config(self, sample_component):
        """Test error with invalid config."""
        register_component('test', 'sample')(sample_component)
        
        # Pass invalid parameter
        with pytest.raises(TypeError):
            create_component('test', 'sample', {'invalid_param': 123})
    
    def test_create_component_not_found(self):
        """Test error when creating non-existent component."""
        with pytest.raises(ValueError):
            create_component('test', 'nonexistent', {})


# ============================================================================
# REGISTRY MANAGEMENT TESTS
# ============================================================================


class TestRegistryManagement:
    """Test registry management operations."""
    
    def test_clear_all(self, sample_component):
        """Test clearing entire registry."""
        register_component('cat1', 'comp1')(sample_component)
        register_component('cat2', 'comp2')(sample_component)
        
        assert ComponentRegistry.has('cat1', 'comp1')
        assert ComponentRegistry.has('cat2', 'comp2')
        
        ComponentRegistry.clear()
        
        # All should be gone
        assert not ComponentRegistry.has('cat1', 'comp1')
        assert not ComponentRegistry.has('cat2', 'comp2')
    
    def test_clear_category(self, sample_component):
        """Test clearing specific category."""
        register_component('cat1', 'comp1')(sample_component)
        register_component('cat2', 'comp2')(sample_component)
        
        ComponentRegistry.clear('cat1')
        
        # Only cat1 should be cleared
        assert not ComponentRegistry.has('cat1', 'comp1')
        assert ComponentRegistry.has('cat2', 'comp2')
    
    def test_registry_isolation(self, sample_component):
        """Test that registry changes don't affect other tests."""
        # Register component
        register_component('test', 'isolated')(sample_component)
        
        # Component should exist
        assert ComponentRegistry.has('test', 'isolated')
        
        # After fixture cleanup, it should be gone
        # (verified by autouse=True clean_registry fixture)


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_helpful_error_message_component_not_found(self):
        """Test that error messages are helpful."""
        with pytest.raises(ValueError) as exc_info:
            ComponentRegistry.get('test', 'nonexistent')
        
        error_msg = str(exc_info.value)
        assert 'nonexistent' in error_msg
        assert 'not found' in error_msg
        assert 'Available' in error_msg or 'available' in error_msg
    
    def test_helpful_error_message_category_not_found(self):
        """Test helpful error for missing category."""
        with pytest.raises(ValueError) as exc_info:
            ComponentRegistry.get('nonexistent_category', 'comp')
        
        error_msg = str(exc_info.value)
        assert 'Category' in error_msg
        assert 'not found' in error_msg
    
    def test_helpful_error_message_duplicate(self, sample_component):
        """Test helpful error for duplicate registration."""
        register_component('test', 'sample')(sample_component)
        
        with pytest.raises(ValueError) as exc_info:
            register_component('test', 'sample')(sample_component)
        
        error_msg = str(exc_info.value)
        assert 'already registered' in error_msg
        assert 'override=True' in error_msg


# ============================================================================
# METADATA TESTS
# ============================================================================


class TestMetadata:
    """Test metadata tracking."""
    
    def test_metadata_stored(self, sample_component):
        """Test that metadata is stored on registration."""
        sample_component.__doc__ = "Test documentation"
        register_component('test', 'sample')(sample_component)
        
        metadata = ComponentRegistry.list_components('test', include_metadata=True)
        
        assert 'sample' in metadata
        assert metadata['sample']['class'] == 'SampleComponent'
        assert 'Test documentation' in metadata['sample']['doc']
    
    def test_metadata_module_tracking(self, sample_component):
        """Test that module path is tracked."""
        register_component('test', 'sample')(sample_component)
        
        metadata = ComponentRegistry.list_components('test', include_metadata=True)
        
        assert 'module' in metadata['sample']
        # Should contain test module path
        assert 'test' in metadata['sample']['module']


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for common workflows."""
    
    def test_full_workflow(self, sample_component):
        """Test complete register -> retrieve -> instantiate workflow."""
        # 1. Register
        register_component('test', 'sample')(sample_component)
        
        # 2. List to verify
        components = ComponentRegistry.list_components('test')
        assert 'sample' in components
        
        # 3. Retrieve class
        comp_class = get_component('test', 'sample')
        assert comp_class is sample_component
        
        # 4. Create instance
        instance = create_component('test', 'sample', {'dim': 15})
        assert isinstance(instance, sample_component)
        assert instance.dim == 15
        
        # 5. Use instance
        import torch
        x = torch.randn(2, 15)
        output = instance(x)
        assert output.shape == (2, 15)
    
    def test_multiple_component_workflow(self, sample_component):
        """Test workflow with multiple components."""
        # Register multiple components
        for i in range(5):
            @register_component('test', f'comp_{i}')
            class TestComp(nn.Module):
                pass
        
        # List all
        components = ComponentRegistry.list_components('test')
        assert len(components) == 5
        
        # Retrieve each
        for i in range(5):
            comp = get_component('test', f'comp_{i}')
            assert comp is not None


# ============================================================================
# RUN TESTS
# ============================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
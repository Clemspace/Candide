"""Pytest configuration for core tests."""
import pytest
from ramanujan.core.registry import ComponentRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    """
    Reset registry before and after each test.
    
    This ensures:
    1. Tests start with a clean registry
    2. No test pollution between tests
    3. Categories exist when tests try to clear them
    """
    # Clear all existing registrations before test
    for category in list(ComponentRegistry._registry.keys()):
        ComponentRegistry._registry[category].clear()
        ComponentRegistry._metadata[category].clear()
    
    yield
    
    # Clean up after test
    for category in list(ComponentRegistry._registry.keys()):
        ComponentRegistry._registry[category].clear()
        ComponentRegistry._metadata[category].clear()


@pytest.fixture
def sample_component():
    """Create a sample component class for testing."""
    import torch.nn as nn
    
    class TestComponent(nn.Module):
        """Simple test component."""
        
        def __init__(self, dim: int = 768, **kwargs):
            super().__init__()
            self.dim = dim
            self.linear = nn.Linear(dim, dim)
        
        @property
        def component_type(self) -> str:
            return "test"
        
        @property
        def component_name(self) -> str:
            return "test_component"
        
        def forward(self, x):
            return self.linear(x)
    
    return TestComponent
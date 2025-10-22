"""
Global component registry with zero magic.

Components register themselves via decorator.
Lookup is explicit - no auto-discovery, no side effects.
"""

from typing import Dict, Type, Optional, List, Callable, Any
import inspect
from .interface import Component, ComponentType, ComponentName


class ComponentRegistry:
    """
    Global registry for all components.
    
    Design principles:
    - Explicit registration (decorator)
    - Zero side effects (no auto-import)
    - Fast lookup (O(1) dict access)
    - Clear errors (helpful messages)
    
    Example:
        >>> from ramanujan.core import ComponentRegistry, register_component
        >>> 
        >>> @register_component('block', 'my_block')
        >>> class MyBlock(nn.Module):
        ...     pass
        >>> 
        >>> # Later, get the class
        >>> block_cls = ComponentRegistry.get('block', 'my_block')
        >>> block = block_cls(dim=768)
    """
    
    # Category -> Name -> Class
    _registry: Dict[ComponentType, Dict[ComponentName, Type]] = {
        'embedding': {},
        'encoder': {},
        'decoder': {},
        'attention': {},
        'ffn': {},
        'norm': {},
        'block': {},
        'head': {},
        'reasoning': {},
        'vision': {},
        'audio': {},
        'diffusion': {},
        # Infinitely extensible - add categories as needed
    }
    
    # Track registration metadata for debugging
    _metadata: Dict[ComponentType, Dict[ComponentName, Dict]] = {
        cat: {} for cat in _registry.keys()
}    
    @classmethod
    def register(
        cls,
        category: ComponentType,
        name: ComponentName,
        override: bool = False
    ) -> Callable:
        """
        Register a component class.
        
        Args:
            category: Component category ('block', 'attention', etc.)
            name: Unique name within category
            override: Allow overriding existing component (default: False)
        
        Returns:
            Decorator function
        
        Raises:
            ValueError: If component already registered and override=False
            ValueError: If category doesn't exist
        
        Example:
            >>> @register_component('block', 'transformer')
            >>> class TransformerBlock(nn.Module):
            ...     pass
        """
        def decorator(component_cls: Type) -> Type:
            # Validate category exists
            if category not in cls._registry:
                # Auto-create category (flexible)
                cls._registry[category] = {}
                cls._metadata[category] = {}
            
            # Check for conflicts
            if name in cls._registry[category] and not override:
                existing = cls._registry[category][name]
                raise ValueError(
                    f"Component '{name}' already registered in category '{category}'. "
                    f"Existing: {existing.__module__}.{existing.__name__}. "
                    f"Use override=True to replace."
                )
            
            # Register component
            cls._registry[category][name] = component_cls
            
            # Store metadata for debugging
            cls._metadata[category][name] = {
                'module': component_cls.__module__,
                'class': component_cls.__name__,
                'doc': inspect.getdoc(component_cls),
            }
            
            return component_cls
        
        return decorator
    
    @classmethod
    def get(cls, category: ComponentType, name: ComponentName) -> Type:
        """
        Get registered component class.
        
        Args:
            category: Component category
            name: Component name
        
        Returns:
            Component class
        
        Raises:
            ValueError: If category or component not found
        
        Example:
            >>> TransformerBlock = ComponentRegistry.get('block', 'transformer')
            >>> block = TransformerBlock(dim=768, num_heads=12)
        """
        if category not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Category '{category}' not found. "
                f"Available categories: {available}"
            )
        
        if name not in cls._registry[category]:
            available = list(cls._registry[category].keys())
            raise ValueError(
                f"Component '{name}' not found in category '{category}'. "
                f"Available components: {available}"
            )
        
        return cls._registry[category][name]
    
    @classmethod
    def has(cls, category: ComponentType, name: ComponentName) -> bool:
        """Check if component is registered."""
        return (
            category in cls._registry and
            name in cls._registry[category]
        )
    
    @classmethod
    def list_categories(cls) -> List[ComponentType]:
        """List all registered categories."""
        return sorted(cls._registry.keys())
    
    @classmethod
    def list_components(
        cls,
        category: ComponentType,
        include_metadata: bool = False
    ) -> List[ComponentName] | Dict[ComponentName, Dict]:
        """
        List all components in a category.
        
        Args:
            category: Category to list
            include_metadata: If True, return dict with metadata
        
        Returns:
            List of component names, or dict with metadata
        
        Example:
            >>> ComponentRegistry.list_components('block')
            ['transformer', 'mamba', 'my_custom_block']
            >>> 
            >>> ComponentRegistry.list_components('block', include_metadata=True)
            {
                'transformer': {
                    'module': 'ramanujan.components.blocks',
                    'class': 'TransformerBlock',
                    'doc': 'Standard transformer block...'
                },
                ...
            }
        """
        if category not in cls._registry:
            return [] if not include_metadata else {}
        
        if include_metadata:
            return cls._metadata[category].copy()
        else:
            return sorted(cls._registry[category].keys())
    
    @classmethod
    def clear(cls, category: Optional[ComponentType] = None):
        """
        Clear registry (mainly for testing).
        
        Args:
            category: If provided, clear only this category.
                     If None, clear everything.
        """
        if category:
            if category in cls._registry:
                cls._registry[category].clear()
                cls._metadata[category].clear()
        else:
            for cat in cls._registry:
                cls._registry[cat].clear()
                cls._metadata[cat].clear()


# Convenience function (more readable than ComponentRegistry.register)
def register_component(
    category: ComponentType,
    name: ComponentName,
    override: bool = False
) -> Callable:
    """
    Register a component (convenience wrapper).
    
    This is the primary way users should register components.
    
    Example:
        >>> from ramanujan.core import register_component
        >>> 
        >>> @register_component('block', 'my_block')
        >>> class MyBlock(nn.Module):
        ...     def __init__(self, dim):
        ...         super().__init__()
        ...         self.dim = dim
        ...     
        ...     @property
        ...     def component_type(self):
        ...         return 'block'
        ...     
        ...     # ... implement Component protocol
    """
    return ComponentRegistry.register(category, name, override)


def get_component(category: ComponentType, name: ComponentName) -> Type:
    """
    Get component class (convenience wrapper).
    
    Example:
        >>> from ramanujan.core import get_component
        >>> 
        >>> MyBlock = get_component('block', 'my_block')
        >>> block = MyBlock(dim=768)
    """
    return ComponentRegistry.get(category, name)


def create_component(
    category: ComponentType,
    name: ComponentName,
    config: Dict[str, Any]
) -> Component:
    """
    Create component instance from config.
    
    Convenience function that gets class and instantiates it.
    
    Args:
        category: Component category
        name: Component name
        config: Configuration dictionary (passed as **kwargs)
    
    Returns:
        Instantiated component
    
    Example:
        >>> from ramanujan.core import create_component
        >>> 
        >>> block = create_component('block', 'transformer', {
        ...     'dim': 768,
        ...     'num_heads': 12,
        ...     'dropout': 0.1
        ... })
    """
    component_cls = ComponentRegistry.get(category, name)
    return component_cls(**config)
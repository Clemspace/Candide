"""
State management system for stateful Ramanujan components.

Add this file to: ramanujan/core/state_manager.py

Manages persistent state for components like Titans memory, RNNs,
and other stateful architectures that need to maintain state across
forward passes.

Features:
- Automatic state save/load
- State checkpointing
- State reset
- Memory-efficient state storage

Usage:
    from ramanujan.core.state_manager import StateManager
    
    manager = StateManager(components)
    manager.save_states()
    # ... forward pass ...
    manager.load_states()
"""

from typing import Dict, Any, Optional, List
from collections import OrderedDict
import copy


class StateManager:
    """
    Manages state for stateful components.
    
    Automatically detects which components are stateful (implement
    get_state/set_state) and manages their state across forward passes.
    
    Example:
        >>> components = {'memory': TitansMemory(), 'rnn': GRUCell()}
        >>> manager = StateManager(components)
        >>> 
        >>> # Before forward
        >>> manager.load_states()
        >>> output = model.forward(x)
        >>> 
        >>> # After forward
        >>> manager.save_states()
    """
    
    def __init__(self, components: Dict[str, Any]):
        """
        Initialize state manager.
        
        Args:
            components: Dictionary of component_id -> component instance
        """
        self.components = components
        
        # Identify stateful components
        self.stateful_components: Dict[str, Any] = {}
        for comp_id, component in components.items():
            if self._is_stateful(component):
                self.stateful_components[comp_id] = component
        
        # Storage for states
        self.states: Dict[str, Dict[str, Any]] = {}
        
        # Checkpoint history
        self.checkpoints: List[Dict[str, Dict[str, Any]]] = []
        self.max_checkpoints = 10
    
    def _is_stateful(self, component: Any) -> bool:
        """Check if component is stateful."""
        return (
            hasattr(component, 'get_state') and
            hasattr(component, 'set_state') and
            hasattr(component, 'reset_state')
        )
    
    def save_states(self) -> None:
        """
        Save current state from all stateful components.
        
        Call this after forward pass to persist state changes.
        
        Example:
            >>> manager.save_states()
        """
        for comp_id, component in self.stateful_components.items():
            try:
                state = component.get_state()
                if state is not None:
                    # Deep copy to prevent modification
                    self.states[comp_id] = copy.deepcopy(state)
            except Exception as e:
                print(f"Warning: Failed to save state for '{comp_id}': {e}")
    
    def load_states(self) -> None:
        """
        Load saved state into all stateful components.
        
        Call this before forward pass to restore state.
        
        Example:
            >>> manager.load_states()
        """
        for comp_id, component in self.stateful_components.items():
            if comp_id in self.states:
                try:
                    # Deep copy to prevent accidental modification
                    state = copy.deepcopy(self.states[comp_id])
                    component.set_state(state)
                except Exception as e:
                    print(f"Warning: Failed to load state for '{comp_id}': {e}")
    
    def reset_states(self) -> None:
        """
        Reset all stateful components to initial state.
        
        Useful for starting a new sequence or episode.
        
        Example:
            >>> manager.reset_states()
        """
        self.states.clear()
        for comp_id, component in self.stateful_components.items():
            try:
                component.reset_state()
            except Exception as e:
                print(f"Warning: Failed to reset state for '{comp_id}': {e}")
    
    def checkpoint(self) -> int:
        """
        Create a checkpoint of current states.
        
        Returns:
            Checkpoint index
        
        Example:
            >>> checkpoint_id = manager.checkpoint()
            >>> # ... do some forward passes ...
            >>> manager.restore_checkpoint(checkpoint_id)
        """
        # Save current states
        self.save_states()
        
        # Create checkpoint
        checkpoint = copy.deepcopy(self.states)
        self.checkpoints.append(checkpoint)
        
        # Limit checkpoint history
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints.pop(0)
        
        return len(self.checkpoints) - 1
    
    def restore_checkpoint(self, checkpoint_id: int) -> bool:
        """
        Restore from a checkpoint.
        
        Args:
            checkpoint_id: Index of checkpoint to restore
        
        Returns:
            True if successful, False otherwise
        
        Example:
            >>> if manager.restore_checkpoint(0):
            ...     print("Restored to earliest checkpoint")
        """
        if 0 <= checkpoint_id < len(self.checkpoints):
            # Restore states
            self.states = copy.deepcopy(self.checkpoints[checkpoint_id])
            self.load_states()
            return True
        return False
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of current state.
        
        Returns:
            Dictionary with state information
        
        Example:
            >>> summary = manager.get_state_summary()
            >>> print(f"Stateful components: {summary['num_stateful']}")
        """
        summary = {
            'num_total': len(self.components),
            'num_stateful': len(self.stateful_components),
            'stateful_components': list(self.stateful_components.keys()),
            'has_state': list(self.states.keys()),
            'num_checkpoints': len(self.checkpoints)
        }
        
        # Add state sizes
        state_sizes = {}
        for comp_id, state in self.states.items():
            if isinstance(state, dict):
                # Count elements in state dict
                size = sum(
                    v.numel() if hasattr(v, 'numel') else 1
                    for v in state.values()
                )
                state_sizes[comp_id] = size
        
        summary['state_sizes'] = state_sizes
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize state manager to dictionary.
        
        Returns:
            Dictionary that can be saved/loaded
        
        Example:
            >>> state_dict = manager.to_dict()
            >>> # Save to file
            >>> torch.save(state_dict, 'state_manager.pt')
        """
        return {
            'states': self.states,
            'checkpoints': self.checkpoints,
            'stateful_components': list(self.stateful_components.keys())
        }
    
    def from_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state manager from dictionary.
        
        Args:
            state_dict: Dictionary from to_dict()
        
        Example:
            >>> state_dict = torch.load('state_manager.pt')
            >>> manager.from_dict(state_dict)
            >>> manager.load_states()  # Apply to components
        """
        self.states = state_dict.get('states', {})
        self.checkpoints = state_dict.get('checkpoints', [])
        
        # Validate that stateful components match
        saved_comps = set(state_dict.get('stateful_components', []))
        current_comps = set(self.stateful_components.keys())
        
        if saved_comps != current_comps:
            print(
                f"Warning: Stateful components mismatch.\n"
                f"  Saved: {saved_comps}\n"
                f"  Current: {current_comps}"
            )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StateManager("
            f"stateful={len(self.stateful_components)}, "
            f"checkpoints={len(self.checkpoints)})"
        )


class StatefulWrapper:
    """
    Wrapper that adds state management to a GraphExecutor.
    
    This is a convenience class that automatically manages state
    for all stateful components in a model.
    
    Example:
        >>> from ramanujan.core import build_model
        >>> from ramanujan.core.state_manager import StatefulWrapper
        >>> 
        >>> model = build_model(graph)
        >>> stateful_model = StatefulWrapper(model)
        >>> 
        >>> # State is automatically managed
        >>> output = stateful_model(input_ids)
        >>> 
        >>> # Reset between sequences
        >>> stateful_model.reset_state()
    """
    
    def __init__(self, executor):
        """
        Wrap a GraphExecutor with state management.
        
        Args:
            executor: GraphExecutor instance
        """
        self.executor = executor
        
        # Create state manager for executor's components
        if hasattr(executor, 'components'):
            self.state_manager = StateManager(dict(executor.components))
        else:
            self.state_manager = StateManager({})
    
    def forward(self, *args, **kwargs):
        """Forward pass with automatic state management."""
        # Load states before forward
        self.state_manager.load_states()
        
        # Run forward
        output = self.executor.forward(*args, **kwargs)
        
        # Save states after forward
        self.state_manager.save_states()
        
        return output
    
    def __call__(self, *args, **kwargs):
        """Make wrapper callable."""
        return self.forward(*args, **kwargs)
    
    def reset_state(self):
        """Reset all stateful components."""
        self.state_manager.reset_states()
    
    def checkpoint(self) -> int:
        """Create state checkpoint."""
        return self.state_manager.checkpoint()
    
    def restore_checkpoint(self, checkpoint_id: int) -> bool:
        """Restore from checkpoint."""
        return self.state_manager.restore_checkpoint(checkpoint_id)
    
    def state_dict(self):
        """Get state dictionary for saving."""
        return {
            'executor': self.executor.state_dict() if hasattr(self.executor, 'state_dict') else {},
            'state_manager': self.state_manager.to_dict()
        }
    
    def load_state_dict(self, state_dict):
        """Load state dictionary."""
        if 'executor' in state_dict and hasattr(self.executor, 'load_state_dict'):
            self.executor.load_state_dict(state_dict['executor'])
        
        if 'state_manager' in state_dict:
            self.state_manager.from_dict(state_dict['state_manager'])
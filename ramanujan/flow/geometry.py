"""
Geometric analysis of reasoning flows in representation space.

Based on "The Geometry of Reasoning: Flowing Logics in Representation Space"
(https://arxiv.org/abs/2510.09782)

This module provides tools to:
1. Compute reasoning trajectories (progressive prefix extension)
2. Calculate geometric properties (position, velocity, curvature)
3. Measure similarity between flows
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, Union


class FlowTrajectoryComputer:
    """
    Extract reasoning flows from models via progressive prefix extension.
    
    Algorithm 1 from the paper: Context-cumulative flow extraction.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
    
    @torch.no_grad()
    def compute_trajectory(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        num_steps: int = 8,
        layer_idx: int = -1,
        method: str = 'uniform'
    ) -> torch.Tensor:
        """
        Compute reasoning trajectory through progressive prefix extension.
        
        Args:
            model: Model with forward pass that returns hidden states
            input_ids: Input token IDs [batch_size, seq_len]
            num_steps: Number of trajectory points to extract
            layer_idx: Which transformer layer to extract from (-1 = last)
            method: Sampling method ('uniform', 'exponential', 'fibonacci')
            
        Returns:
            trajectory: [num_steps, batch_size, hidden_dim]
            
        Example:
            >>> computer = FlowTrajectoryComputer()
            >>> trajectory = computer.compute_trajectory(model, tokens, num_steps=10)
            >>> print(trajectory.shape)  # [10, 1, 768]
        """
        model.eval()
        input_ids = input_ids.to(self.device)
        
        batch_size, seq_len = input_ids.shape
        
        # Determine step positions
        step_positions = self._get_step_positions(seq_len, num_steps, method)
        
        trajectories = []
        
        for pos in step_positions:
            # Progressive prefix: tokens from 0 to pos
            prefix = input_ids[:, :pos]
            
            # Forward pass with hidden states
            hidden_states = self._extract_hidden_states(model, prefix, layer_idx)
            
            # Take last token representation (context-cumulative)
            h_t = hidden_states[:, -1, :]  # [batch_size, hidden_dim]
            
            trajectories.append(h_t)
        
        # Stack into trajectory tensor
        trajectory = torch.stack(trajectories, dim=0)  # [num_steps, batch_size, hidden_dim]
        
        return trajectory
    
    def _get_step_positions(
        self, 
        seq_len: int, 
        num_steps: int, 
        method: str
    ) -> list:
        """Generate positions for trajectory sampling."""
        if method == 'uniform':
            # Evenly spaced steps
            step_size = max(1, seq_len // num_steps)
            positions = [min((i + 1) * step_size, seq_len) for i in range(num_steps)]
            
        elif method == 'exponential':
            # More steps early in sequence (when learning is faster)
            positions = [int(seq_len * (1 - np.exp(-3 * i / num_steps))) 
                        for i in range(1, num_steps + 1)]
            positions = [max(1, p) for p in positions]
            
        elif method == 'fibonacci':
            # Fibonacci-spaced (golden ratio sampling)
            phi = (1 + np.sqrt(5)) / 2
            positions = [int(seq_len * (i / num_steps) ** phi) 
                        for i in range(1, num_steps + 1)]
            positions = [max(1, min(p, seq_len)) for p in positions]
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return positions
    
    def _extract_hidden_states(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Extract hidden states from model.
        
        Tries multiple methods to be compatible with different architectures.
        """
        # Method 1: Model has built-in support
        if hasattr(model, 'forward') and 'return_hidden_states' in model.forward.__code__.co_varnames:
            outputs = model(input_ids, return_hidden_states=True)
            if isinstance(outputs, tuple):
                hidden_states = outputs[1] if len(outputs) > 1 else outputs[0]
            else:
                hidden_states = outputs
            
            # Handle list of hidden states (one per layer)
            if isinstance(hidden_states, (list, tuple)):
                return hidden_states[layer_idx]
            return hidden_states
        
        # Method 2: Use hooks
        hidden_state = None
        
        def hook_fn(module, input, output):
            nonlocal hidden_state
            hidden_state = output[0] if isinstance(output, tuple) else output
        
        # Try to find the right layer
        target_layer = self._find_target_layer(model, layer_idx)
        if target_layer is not None:
            handle = target_layer.register_forward_hook(hook_fn)
            _ = model(input_ids)
            handle.remove()
            
            if hidden_state is not None:
                return hidden_state
        
        # Method 3: Just run forward and hope for the best
        output = model(input_ids)
        if isinstance(output, tuple):
            return output[0]  # Assume first element is hidden states
        return output
    
    def _find_target_layer(self, model: nn.Module, layer_idx: int) -> Optional[nn.Module]:
        """Find the transformer layer at layer_idx."""
        # Common patterns
        for attr in ['layers', 'blocks', 'h', 'transformer', 'encoder']:
            if hasattr(model, attr):
                layers = getattr(model, attr)
                if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                    idx = layer_idx if layer_idx >= 0 else len(layers) + layer_idx
                    if 0 <= idx < len(layers):
                        return layers[idx]
        
        return None


class GeometricMetrics:
    """
    Compute geometric properties of reasoning flows.
    
    Key metrics:
    - Position similarity: Semantic alignment (0th order)
    - Velocity similarity: Reasoning dynamics (1st order)
    - Curvature: Logical structure (2nd order)
    
    From paper: "Logic acts as a controller of flow velocity"
    """
    
    @staticmethod
    def menger_curvature(
        trajectory: torch.Tensor,
        epsilon: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute Menger curvature for each point in trajectory.
        
        Proposition C.8 from paper:
        For three consecutive points (y_{t-1}, y_t, y_{t+1}), curvature is:
        
            κ_t = 2 * sqrt(1 - cos²(θ)) / ||y_{t+1} - y_{t-1}||
        
        where θ is angle between vectors u = y_t - y_{t-1} and v = y_{t+1} - y_t
        
        Args:
            trajectory: [num_steps, batch_size, hidden_dim] where num_steps >= 3
            epsilon: Small constant for numerical stability
            
        Returns:
            curvatures: [num_steps-2, batch_size] curvature at each interior point
            
        Example:
            >>> traj = torch.randn(10, 2, 768)
            >>> curv = GeometricMetrics.menger_curvature(traj)
            >>> print(curv.shape)  # [8, 2]
        """
        if trajectory.size(0) < 3:
            raise ValueError(f"Need at least 3 points, got {trajectory.size(0)}")
        
        # Get consecutive triples
        y_prev = trajectory[:-2]  # [T-2, B, D]
        y_curr = trajectory[1:-1]  # [T-2, B, D]
        y_next = trajectory[2:]    # [T-2, B, D]
        
        # Compute difference vectors
        u = y_curr - y_prev  # [T-2, B, D]
        v = y_next - y_curr  # [T-2, B, D]
        
        # Normalize to unit vectors
        u_norm = F.normalize(u, p=2, dim=-1)
        v_norm = F.normalize(v, p=2, dim=-1)
        
        # Cosine similarity between consecutive vectors
        cos_sim = (u_norm * v_norm).sum(dim=-1)  # [T-2, B]
        
        # Clamp to valid range (numerical stability)
        cos_sim = torch.clamp(cos_sim, -1.0 + epsilon, 1.0 - epsilon)
        
        # Sine from cosine: sin²(θ) = 1 - cos²(θ)
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_sim**2, min=epsilon))
        
        # Chord length (distance between endpoints)
        chord = torch.norm(y_next - y_prev, p=2, dim=-1)  # [T-2, B]
        
        # Menger curvature: κ = 2*sin(θ) / chord
        curvature = 2.0 * sin_theta / (chord + epsilon)
        
        return curvature
    
    @staticmethod
    def position_similarity(
        traj1: torch.Tensor,
        traj2: torch.Tensor,
        method: str = 'cosine'
    ) -> float:
        """
        Measure similarity at position level (0th order).
        
        This captures semantic alignment between flows.
        From paper: "Position similarity dominated by surface semantics"
        
        Args:
            traj1, traj2: [num_steps, batch_size, hidden_dim]
            method: 'cosine', 'euclidean', or 'correlation'
            
        Returns:
            similarity: Scalar in [0, 1] for cosine, [-∞, 0] for euclidean
        """
        # Flatten trajectories
        flat1 = traj1.reshape(-1, traj1.size(-1))  # [T*B, D]
        flat2 = traj2.reshape(-1, traj2.size(-1))  # [T*B, D]
        
        if method == 'cosine':
            # Cosine similarity
            similarity = F.cosine_similarity(flat1, flat2, dim=-1).mean()
            return similarity.item()
            
        elif method == 'euclidean':
            # Negative euclidean distance
            distance = torch.norm(flat1 - flat2, p=2, dim=-1).mean()
            return -distance.item()
            
        elif method == 'correlation':
            # Pearson correlation
            flat1_centered = flat1 - flat1.mean(dim=0, keepdim=True)
            flat2_centered = flat2 - flat2.mean(dim=0, keepdim=True)
            
            numerator = (flat1_centered * flat2_centered).sum()
            denominator = torch.sqrt(
                (flat1_centered**2).sum() * (flat2_centered**2).sum()
            )
            
            correlation = numerator / (denominator + 1e-8)
            return correlation.item()
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def velocity_similarity(
        traj1: torch.Tensor,
        traj2: torch.Tensor
    ) -> float:
        """
        Measure similarity of flow velocities (1st order).
        
        This captures reasoning dynamics.
        From paper: "Velocity similarity reveals logical structure"
        
        Args:
            traj1, traj2: [num_steps, batch_size, hidden_dim]
            
        Returns:
            similarity: Cosine similarity in [-1, 1]
        """
        # Compute velocities (first differences)
        vel1 = traj1[1:] - traj1[:-1]  # [T-1, B, D]
        vel2 = traj2[1:] - traj2[:-1]  # [T-1, B, D]
        
        # Flatten
        vel1_flat = vel1.reshape(-1, vel1.size(-1))
        vel2_flat = vel2.reshape(-1, vel2.size(-1))
        
        # Cosine similarity
        similarity = F.cosine_similarity(vel1_flat, vel2_flat, dim=-1).mean()
        
        return similarity.item()
    
    @staticmethod
    def curvature_correlation(
        traj1: torch.Tensor,
        traj2: torch.Tensor
    ) -> float:
        """
        Pearson correlation between flow curvatures (2nd order).
        
        **KEY METRIC FROM PAPER**:
        "Curvature similarity remains highly consistent between flows 
        sharing the same logical skeleton, even across unrelated topics"
        
        This is modality-invariant and captures abstract reasoning!
        
        Args:
            traj1, traj2: [num_steps, batch_size, hidden_dim]
            
        Returns:
            correlation: Pearson r in [-1, 1]
            
        Example:
            >>> # Same logic, different topics should have high correlation
            >>> math_traj = compute_trajectory(model, math_problem)
            >>> code_traj = compute_trajectory(model, code_problem)
            >>> r = GeometricMetrics.curvature_correlation(math_traj, code_traj)
            >>> print(f"Logical similarity: {r:.3f}")  # Should be high!
        """
        # Compute curvatures
        curv1 = GeometricMetrics.menger_curvature(traj1)  # [T-2, B]
        curv2 = GeometricMetrics.menger_curvature(traj2)  # [T-2, B]
        
        # Flatten
        c1 = curv1.flatten()
        c2 = curv2.flatten()
        
        # Center
        c1_centered = c1 - c1.mean()
        c2_centered = c2 - c2.mean()
        
        # Pearson correlation
        numerator = (c1_centered * c2_centered).sum()
        denominator = torch.sqrt(
            (c1_centered**2).sum() * (c2_centered**2).sum()
        )
        
        correlation = numerator / (denominator + 1e-8)
        
        return correlation.item()
    
    @staticmethod
    def compute_all_metrics(
        traj1: torch.Tensor,
        traj2: torch.Tensor
    ) -> Dict[str, float]:
        """
        Convenience function to compute all similarity metrics.
        
        Returns dict with:
        - position_similarity: 0th order (semantics)
        - velocity_similarity: 1st order (dynamics)
        - curvature_correlation: 2nd order (logic)
        """
        return {
            'position_similarity': GeometricMetrics.position_similarity(traj1, traj2),
            'velocity_similarity': GeometricMetrics.velocity_similarity(traj1, traj2),
            'curvature_correlation': GeometricMetrics.curvature_correlation(traj1, traj2),
        }


class FlowAnalyzer:
    """
    High-level interface for flow analysis.
    
    Combines trajectory computation and geometric metrics.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.computer = FlowTrajectoryComputer(device=device)
        self.metrics = GeometricMetrics()
        self.device = device
    
    def analyze_model(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_steps: int = 8,
        layer_idx: int = -1
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """
        Complete flow analysis for a single model.
        
        Args:
            model: Model to analyze
            inputs: Dict with 'input_ids' and optionally other fields
            num_steps: Number of trajectory points
            layer_idx: Which layer to extract
            
        Returns:
            Dictionary with trajectory and statistics
        """
        # Compute trajectory
        trajectory = self.computer.compute_trajectory(
            model,
            inputs['input_ids'],
            num_steps=num_steps,
            layer_idx=layer_idx
        )
        
        # Compute curvatures
        curvatures = self.metrics.menger_curvature(trajectory)
        
        return {
            'trajectory': trajectory,
            'curvatures': curvatures,
            'mean_curvature': curvatures.mean().item(),
            'std_curvature': curvatures.std().item(),
            'max_curvature': curvatures.max().item(),
            'num_steps': num_steps,
        }
    
    def compare_models(
        self,
        model1: nn.Module,
        model2: nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_steps: int = 8,
        layer_idx: int = -1
    ) -> Dict[str, float]:
        """
        Compare reasoning flows between two models.
        
        Use case: Compare expert models to measure task similarity.
        
        Returns:
            All geometric similarity metrics
        """
        # Compute trajectories
        traj1 = self.computer.compute_trajectory(
            model1, inputs['input_ids'], num_steps, layer_idx
        )
        traj2 = self.computer.compute_trajectory(
            model2, inputs['input_ids'], num_steps, layer_idx
        )
        
        # Compute all metrics
        metrics = self.metrics.compute_all_metrics(traj1, traj2)
        
        return metrics
    
    def batch_analyze(
        self,
        model: nn.Module,
        dataloader,
        num_samples: int = 100,
        num_steps: int = 8,
        layer_idx: int = -1
    ) -> Dict[str, np.ndarray]:
        """
        Analyze flow properties over a dataset.
        
        Useful for characterizing task-specific flow patterns.
        """
        all_curvatures = []
        
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            trajectory = self.computer.compute_trajectory(
                model,
                batch['input_ids'].to(self.device),
                num_steps=num_steps,
                layer_idx=layer_idx
            )
            
            curvatures = self.metrics.menger_curvature(trajectory)
            all_curvatures.append(curvatures.cpu().numpy())
        
        # Concatenate
        all_curvatures = np.concatenate(all_curvatures, axis=0)
        
        return {
            'curvatures': all_curvatures,
            'mean': all_curvatures.mean(),
            'std': all_curvatures.std(),
            'percentiles': np.percentile(all_curvatures, [25, 50, 75, 95]),
        }


# Convenience functions for quick usage
def quick_curvature(model, input_ids, num_steps=8):
    """Quick curvature computation."""
    analyzer = FlowAnalyzer()
    result = analyzer.analyze_model(model, {'input_ids': input_ids}, num_steps)
    return result['curvatures']


def quick_compare(model1, model2, input_ids, num_steps=8):
    """Quick model comparison."""
    analyzer = FlowAnalyzer()
    return analyzer.compare_models(model1, model2, {'input_ids': input_ids}, num_steps)



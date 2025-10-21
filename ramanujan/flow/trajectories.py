import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowTrajectoryComputer:
    """
    Implements Algorithm 1 from reasoning flow paper
    Context-cumulative flow extraction
    """
    
    @staticmethod
    def compute_trajectory(
        model: nn.Module,
        input_ids: torch.Tensor,
        num_steps: int = 8,
        layer_idx: int = -1,
        return_all_layers: bool = False
    ) -> torch.Tensor:
        """
        Progressive prefix extension to get reasoning flow
        
        Args:
            model: Any encoder (TextCandide, VisionCandide, etc.)
            input_ids: Input sequence [batch, seq_len]
            num_steps: How many trajectory points to extract
            layer_idx: Which layer to extract (-1 = last)
            
        Returns:
            trajectory: [num_steps, batch, hidden_dim]
        """
        model.eval()
        trajectories = []
        
        # Compute step size
        seq_len = input_ids.size(1)
        step_size = seq_len // num_steps
        
        with torch.no_grad():
            for t in range(1, num_steps + 1):
                # Progressive prefix
                end_idx = min(t * step_size, seq_len)
                prefix = input_ids[:, :end_idx]
                
                # Forward pass with hidden state extraction
                if hasattr(model, 'forward_with_hidden_states'):
                    _, hidden_states = model.forward_with_hidden_states(prefix)
                    h_t = hidden_states[layer_idx][:, -1, :]  # Last token
                else:
                    # Fallback: add hook
                    h_t = extract_hidden_with_hook(model, prefix, layer_idx)
                
                trajectories.append(h_t)
        
        return torch.stack(trajectories)  # [T, B, D]


class GeometricMetrics:
    """
    All geometric computations from the paper
    """
    
    @staticmethod
    def menger_curvature(trajectory: torch.Tensor) -> torch.Tensor:
        """
        Proposition C.8 from paper
        
        Args:
            trajectory: [T, B, D] where T >= 3
            
        Returns:
            curvatures: [T-2, B] curvature at each point
        """
        # Get consecutive triples
        y_prev = trajectory[:-2]  # [T-2, B, D]
        y_curr = trajectory[1:-1]
        y_next = trajectory[2:]
        
        # Compute vectors
        u = y_curr - y_prev  # [T-2, B, D]
        v = y_next - y_curr
        
        # Cosine similarity
        u_norm = F.normalize(u, dim=-1)
        v_norm = F.normalize(v, dim=-1)
        cos_sim = (u_norm * v_norm).sum(dim=-1)  # [T-2, B]
        
        # Chord length
        chord = torch.norm(y_next - y_prev, dim=-1)  # [T-2, B]
        
        # Menger curvature formula
        numerator = 2 * torch.sqrt(torch.clamp(1 - cos_sim**2, min=1e-8))
        curvature = numerator / (chord + 1e-8)
        
        return curvature
    
    @staticmethod
    def velocity_similarity(traj1: torch.Tensor, traj2: torch.Tensor) -> float:
        """
        Cosine similarity of velocity vectors
        """
        # Compute velocities (first-order differences)
        vel1 = traj1[1:] - traj1[:-1]
        vel2 = traj2[1:] - traj2[:-1]
        
        # Flatten and compute cosine similarity
        vel1_flat = vel1.reshape(-1, vel1.size(-1))
        vel2_flat = vel2.reshape(-1, vel2.size(-1))
        
        cos_sim = F.cosine_similarity(vel1_flat, vel2_flat, dim=-1)
        return cos_sim.mean().item()
    
    @staticmethod
    def curvature_correlation(traj1: torch.Tensor, traj2: torch.Tensor) -> float:
        """
        Pearson correlation of curvatures (key metric from paper!)
        """
        curv1 = GeometricMetrics.menger_curvature(traj1)
        curv2 = GeometricMetrics.menger_curvature(traj2)
        
        # Flatten
        c1 = curv1.flatten()
        c2 = curv2.flatten()
        
        # Pearson correlation
        c1_centered = c1 - c1.mean()
        c2_centered = c2 - c2.mean()
        
        correlation = (c1_centered * c2_centered).sum() / (
            torch.sqrt((c1_centered**2).sum() * (c2_centered**2).sum()) + 1e-8
        )
        
        return correlation.item()
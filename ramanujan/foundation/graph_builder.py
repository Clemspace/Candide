import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .math_core import RamanujanMath

class RamanujanGraphBuilder:
    """Graph builder with proper tensor handling"""
    
    def __init__(self, math_foundation: RamanujanMath):
        self.math = math_foundation
    
    def build_lps_adjacency(self, p: int, q: int) -> torch.Tensor:  # FIXED: Return torch.Tensor
        """Build LPS adjacency matrix"""
        generators = self.math.four_square_cache[p]
        graph_size = q * (q * q - 1) // 2
        
        adjacency = np.zeros((graph_size, graph_size), dtype=np.float32)
        
        for i in range(graph_size):
            connections = 0
            for a0, a1, a2, a3 in generators:
                if connections >= len(generators):
                    break
                
                j = (i * a0 + a1 * 17 + a2 * 19 + a3 * 23) % graph_size
                
                if i != j and adjacency[i, j] == 0:
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0
                    connections += 1
        
        # FIXED: Convert to torch tensor properly
        return torch.from_numpy(adjacency).float()
    
    def build_biregular_biadjacency(self, q: int, l: int) -> torch.Tensor:  # FIXED: Return torch.Tensor
        """FIXED: Build bi-regular bi-adjacency matrix"""
        
        # Create cyclic shift permutation matrix P
        P = np.zeros((q, q), dtype=np.float32)
        for i in range(q):
            j = (i - 1) % q
            P[i, j] = 1.0
        
        # Build bi-adjacency matrix
        q_squared = q * q
        lq = l * q
        B = np.zeros((q_squared, lq), dtype=np.float32)
        I_q = np.eye(q, dtype=np.float32)
        
        for row_block in range(q):
            for col_block in range(l):
                # Power of P: P^(row_block * col_block)
                power = row_block * col_block
                P_power = self._matrix_power_cyclic(P, power)
                
                row_start = row_block * q
                row_end = (row_block + 1) * q
                col_start = col_block * q
                col_end = (col_block + 1) * q
                
                if col_block == 0:
                    B[row_start:row_end, col_start:col_end] = I_q
                else:
                    B[row_start:row_end, col_start:col_end] = P_power
        
        # FIXED: Convert to torch tensor properly
        return torch.from_numpy(B).float()
    
    def _matrix_power_cyclic(self, P: np.ndarray, power: int) -> np.ndarray:
        """Compute P^power for cyclic shift matrix"""
        q = P.shape[0]
        if power == 0:
            return np.eye(q, dtype=np.float32)
        
        result = np.zeros((q, q), dtype=np.float32)
        for i in range(q):
            j = (i - power) % q
            result[i, j] = 1.0
        return result
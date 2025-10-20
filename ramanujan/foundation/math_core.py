import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

class RamanujanMath:
    """Mathematical foundations with larger bi-regular configs"""
    
    def __init__(self, max_prime: int = 1000):
        self.max_prime = max_prime
        self.primes_1_mod_4 = self._sieve_primes_1_mod_4(max_prime)
        self.legendre_table = self._compute_legendre_symbols()
        self.four_square_cache = self._compute_four_square_representations()
        self.lps_pairs = self._compute_lps_pairs()
        self.biregular_configs = self._compute_biregular_configs()  # FIXED
    
    def _sieve_primes_1_mod_4(self, limit: int) -> List[int]:
        """Find primes p ≡ 1 (mod 4)"""
        if limit < 5:
            return []
        
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        
        for i in range(2, int(math.sqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        
        return [p for p in range(5, limit + 1) if is_prime[p] and p % 4 == 1]
    
    def _compute_legendre_symbols(self) -> Dict[Tuple[int, int], int]:
        """Compute Legendre symbols (q/p)"""
        table = {}
        for p in self.primes_1_mod_4:
            for q in self.primes_1_mod_4:
                if p != q:
                    table[(q, p)] = pow(q, (p - 1) // 2, p)
                    if table[(q, p)] == p - 1:
                        table[(q, p)] = -1
        return table
    
    def _compute_four_square_representations(self) -> Dict[int, List[Tuple[int, int, int, int]]]:
        """Jacobi's four-square theorem implementation"""
        cache = {}
        for p in self.primes_1_mod_4:
            solutions = []
            max_search = int(math.sqrt(p)) + 1
            
            for a0 in range(1, max_search, 2):  # a₀ odd, positive
                if a0 * a0 > p:
                    break
                for a1 in range(0, max_search, 2):  # a₁ even
                    if a0*a0 + a1*a1 > p:
                        break
                    for a2 in range(0, max_search, 2):  # a₂ even
                        remainder = p - a0*a0 - a1*a1 - a2*a2
                        if remainder < 0:
                            break
                        a3 = int(math.sqrt(remainder))
                        if a3 % 2 == 0 and a3*a3 == remainder:
                            solutions.append((a0, a1, a2, a3))
                            if len(solutions) >= p + 1:
                                break
                    if len(solutions) >= p + 1:
                        break
                if len(solutions) >= p + 1:
                    break
            
            while len(solutions) < p + 1:
                solutions.append(solutions[0] if solutions else (1, 0, 0, 0))
            
            cache[p] = solutions[:p + 1]
        return cache
    
    def _compute_lps_pairs(self) -> List[Dict]:
        """Compute valid LPS pairs"""
        pairs = []
        for q in self.primes_1_mod_4:
            for p in self.primes_1_mod_4:
                if p == q:
                    continue
                
                if self.legendre_table.get((q, p), 0) != -1:
                    continue
                
                graph_size = q * (q * q - 1) // 2
                degree = p + 1
                sparsity = 1.0 - degree / graph_size
                
                if 0.1 <= sparsity <= 0.99:
                    pairs.append({
                        'p': p, 'q': q,
                        'graph_size': graph_size,
                        'degree': degree,
                        'sparsity': sparsity,
                        'type': 'lps'
                    })
        
        return sorted(pairs, key=lambda x: x['sparsity'])
    
    def _compute_biregular_configs(self) -> List[Dict]:
        """Compute bi-regular configurations with larger graphs"""
        configs = []
        
        # FIXED: Use larger primes and more l values for bigger graphs
        biregular_primes = [q for q in self.primes_1_mod_4 if q <= 200]  # Increased from 50
        
        for q in biregular_primes:
            # FIXED: Allow larger l values for bigger graphs
            max_l = min(100, 2000 // q)  # Much larger l values
            
            for l in range(2, max_l):
                graph_out = q * q
                graph_in = l * q
                
                # Sparsity calculation
                total_edges = graph_out * l
                total_possible = graph_out * graph_in
                sparsity = 1.0 - total_edges / total_possible
                
                if 0.3 <= sparsity <= 0.98:
                    configs.append({
                        'q': q, 'l': l,
                        'graph_out': graph_out,
                        'graph_in': graph_in,
                        'sparsity': sparsity,
                        'type': 'biregular'
                    })
        
        return sorted(configs, key=lambda x: x['sparsity'])

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
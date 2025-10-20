import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple
from .math_core import RamanujanMath
from .graph_builder import RamanujanGraphBuilder

class RamanujanLinearLayer(nn.Module):
    """Linear layer with proper method selection and tensor handling"""
    
    def __init__(self, in_features: int, out_features: int, target_sparsity: float,
                 bias: bool = True, math_foundation: RamanujanMath = None,
                 force_method: str = None):  # ADDED: force_method parameter
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.target_sparsity = target_sparsity
        self.force_method = force_method  # ADDED
        
        if math_foundation is None:
            math_foundation = RamanujanMath()
        
        self.math = math_foundation
        self.builder = RamanujanGraphBuilder(math_foundation)  # FIXED: Add builder
        
        # Parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Generate mask
        mask, info = self._generate_ramanujan_mask()  # FIXED
        self.register_buffer('mask', mask)
        self.construction_info = info
        
        self.reset_parameters()
    
    def _generate_ramanujan_mask(self) -> Tuple[torch.Tensor, Dict]:
        """Generate mask with proper method selection"""
        
        config = self._find_configuration()  # FIXED
        
        if config is None:
            raise ValueError(f"No Ramanujan configuration found for {self.out_features}×{self.in_features} @ {self.target_sparsity}")
        
        # Build graph based on type
        if config['type'] == 'lps':
            adjacency = self.builder.build_lps_adjacency(config['p'], config['q'])
            mask = self._adapt_adjacency_to_layer(adjacency)
            method = 'lps'
        elif config['type'] == 'biregular':
            biadjacency = self.builder.build_biregular_biadjacency(config['q'], config['l'])
            mask = self._adapt_biadjacency_to_layer(biadjacency)
            method = 'biregular'
        else:
            raise ValueError(f"Unknown configuration type: {config['type']}")
        
        actual_sparsity = 1.0 - mask.mean().item()
        
        info = {
            'method': method,
            'config': config,
            'target_sparsity': self.target_sparsity,
            'actual_sparsity': actual_sparsity,
            'sparsity_error': abs(actual_sparsity - self.target_sparsity)
        }
        
        return mask, info
    
    def _find_configuration(self) -> Optional[Dict]:
        """Method selection with debug info and force option"""
        
        aspect_ratio = max(self.out_features, self.in_features) / min(self.out_features, self.in_features)
        
        print(f"   🔍 Method selection for {self.out_features}×{self.in_features}")
        print(f"      Aspect ratio: {aspect_ratio:.1f}:1, Target sparsity: {self.target_sparsity:.2f}")
        
        # ADDED: Force method if specified
        if self.force_method == "biregular":
            biregular_config = self._find_best_biregular()
            if biregular_config:
                print(f"      🔴 FORCED bi-regular method")
                return biregular_config
            else:
                print(f"      ❌ FORCED bi-regular failed, falling back to LPS")
        elif self.force_method == "lps":
            lps_config = self._find_best_lps()
            if lps_config:
                print(f"      🔵 FORCED LPS method")
                return lps_config
        
        # Find candidates
        best_lps = self._find_best_lps()
        best_biregular = self._find_best_biregular() if aspect_ratio >= 2.0 else None
        
        print(f"      📊 LPS candidate: {best_lps is not None}")
        if best_lps:
            print(f"         LPS: p={best_lps['p']}, q={best_lps['q']}, sparsity={best_lps['sparsity']:.3f}")
        
        print(f"      📊 Bi-regular candidate: {best_biregular is not None}")
        if best_biregular:
            print(f"         Bi-regular: q={best_biregular['q']}, l={best_biregular['l']}, sparsity={best_biregular['sparsity']:.3f}")
        
        # Select best
        candidates = [c for c in [best_lps, best_biregular] if c is not None]
        if not candidates:
            return None
        
        selected = min(candidates, key=lambda c: abs(c['sparsity'] - self.target_sparsity))
        print(f"      ✅ Selected: {selected['type']} method")
        
        return selected
    
    def _find_best_lps(self) -> Optional[Dict]:
        """Find best LPS configuration"""
        best = None
        min_error = float('inf')
        
        for config in self.math.lps_pairs:
            if config['graph_size'] < max(self.out_features, self.in_features):
                continue
            
            error = abs(config['sparsity'] - self.target_sparsity)
            if error < min_error:
                min_error = error
                best = config
        
        return best
    
    def _find_best_biregular(self) -> Optional[Dict]:
        """Find best bi-regular configuration with relaxed constraints"""
        best = None
        min_error = float('inf')
        
        for config in self.math.biregular_configs:
            # FIXED: More flexible size constraints - allow tiling
            graph_out, graph_in = config['graph_out'], config['graph_in']
            
            # Allow if graph can tile to cover the layer (more flexible)
            if (graph_out * 4 >= self.out_features and graph_in * 4 >= self.in_features):
                error = abs(config['sparsity'] - self.target_sparsity)
                if error < min_error:
                    min_error = error
                    best = config
        
        return best
    
    def _adapt_adjacency_to_layer(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Adapt square adjacency matrix to layer dimensions"""
        graph_size = adjacency.shape[0]
        mask = torch.zeros(self.out_features, self.in_features, dtype=torch.float32)
        
        for i in range(self.out_features):
            for j in range(self.in_features):
                src_i = i % graph_size
                src_j = j % graph_size
                mask[i, j] = adjacency[src_i, src_j]
        
        return mask
    
    def _adapt_biadjacency_to_layer(self, biadjacency: torch.Tensor) -> torch.Tensor:
        """Adapt bi-adjacency matrix to layer dimensions"""
        graph_out, graph_in = biadjacency.shape
        mask = torch.zeros(self.out_features, self.in_features, dtype=torch.float32)
        
        for i in range(self.out_features):
            for j in range(self.in_features):
                src_i = i % graph_out
                src_j = j % graph_in
                mask[i, j] = biadjacency[src_i, src_j]
        
        return mask
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Ramanujan sparsity"""
        masked_weight = self.weight * self.mask
        return nn.functional.linear(x, masked_weight, self.bias)
    
    def get_info(self) -> Dict:
        """Get construction information"""
        return self.construction_info.copy()

class RamanujanFoundation:
    """Foundation with proper builder and larger configs"""
    
    def __init__(self, max_prime: int = 1000):
        self.math = RamanujanMath(max_prime)
        self.builder = RamanujanGraphBuilder(self.math)  # FIXED: Add builder
    
    def create_layer(self, in_features: int, out_features: int, 
                    target_sparsity: float, bias: bool = True,
                    force_method: str = None) -> RamanujanLinearLayer:  # ADDED: force_method
        """Create Ramanujan layer with method forcing option"""
        return RamanujanLinearLayer(
            in_features, out_features, target_sparsity, bias, 
            self.math, force_method=force_method
        )
    
    def get_available_configurations(self) -> Dict:
        """Get available configurations"""
        return {
            'lps_pairs': len(self.math.lps_pairs),
            'biregular_configs': len(self.math.biregular_configs),
            'max_prime': self.math.max_prime,
            'primes_available': len(self.math.primes_1_mod_4)
        }
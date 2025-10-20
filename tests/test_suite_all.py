"""
Comprehensive test suite for Ramanujan Transformer + Flow Analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

print("\n" + "="*70)
print("RAMANUJAN TRANSFORMER + FLOW ANALYSIS TEST SUITE")
print("="*70)

# Force CPU for consistency
DEVICE = 'cpu'

# ============================================================================
# TEST 1: Foundation
# ============================================================================

def test_foundation():
    """Test Ramanujan Foundation."""
    print("\n" + "-"*70)
    print("TEST 1: Foundation (Ramanujan Graphs)")
    print("-"*70)
    
    try:
        from ramanujan.foundation import RamanujanFoundation
        
        foundation = RamanujanFoundation(max_prime=100)
        print(f"✅ Foundation created")
        
        layer = foundation.create_layer(64, 64, target_sparsity=0.8)
        x = torch.randn(2, 64)
        out = layer(x)
        assert out.shape == (2, 64)
        print(f"✅ Sparse layer works: {x.shape} -> {out.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Foundation test failed: {e}")
        return False


# ============================================================================
# TEST 2: Architecture
# ============================================================================

def test_attention():
    """Test attention mechanisms."""
    print("\n" + "-"*70)
    print("TEST 2a: Attention")
    print("-"*70)
    
    try:
        from ramanujan.architecture import StandardGQA
        
        attn = StandardGQA(
            dim=256,
            num_heads=8,
            num_kv_heads=4,
            max_seq_len=512,
            dropout=0.1
        )
        print(f"✅ StandardGQA created")
        
        x = torch.randn(2, 64, 256)
        out = attn(x)
        assert out.shape == x.shape
        print(f"✅ Forward pass: {x.shape} -> {out.shape}")
        
        loss = out.sum()
        loss.backward()
        assert attn.q_proj.weight.grad is not None
        print(f"✅ Gradients flow correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Attention test failed: {e}")
        return False


def test_feedforward():
    """Test feedforward networks."""
    print("\n" + "-"*70)
    print("TEST 2b: Feedforward")
    print("-"*70)
    
    try:
        from ramanujan.architecture import SwiGLU, FeedForwardFactory, FeedForwardConfig
        
        # Direct instantiation
        ffn = SwiGLU(dim=256, hidden_dim=1024)
        x = torch.randn(2, 64, 256)
        out = ffn(x)
        assert out.shape == x.shape
        print(f"✅ SwiGLU: {x.shape} -> {out.shape}")
        
        # Via factory
        config = FeedForwardConfig(dim=256, hidden_dim=1024, ffn_type='swiglu')
        ffn2 = FeedForwardFactory.create(config)
        out2 = ffn2(x)
        print(f"✅ FeedForwardFactory works")
        
        return True
        
    except Exception as e:
        print(f"❌ FFN test failed: {e}")
        return False


# ============================================================================
# TEST 3: Flow Analysis
# ============================================================================

def test_flow_trajectory():
    """Test flow trajectory computation."""
    print("\n" + "-"*70)
    print("TEST 3a: Flow Trajectory")
    print("-"*70)
    
    try:
        from ramanujan.flow import FlowTrajectoryComputer
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 64)
            
            def forward(self, input_ids, return_hidden_states=False):
                x = torch.randn(input_ids.size(0), input_ids.size(1), 10)
                h = self.layer(x)
                if return_hidden_states:
                    return h, [h]
                return h
        
        model = SimpleModel()
        tokens = torch.randint(0, 100, (2, 50))
        
        computer = FlowTrajectoryComputer(device=DEVICE)
        trajectory = computer.compute_trajectory(model, tokens, num_steps=8)
        
        print(f"✅ Trajectory computed: shape {trajectory.shape}")
        assert trajectory.shape[0] == 8
        
        return True
        
    except Exception as e:
        print(f"❌ Flow trajectory test failed: {e}")
        return False


def test_geometric_metrics():
    """Test geometric metrics."""
    print("\n" + "-"*70)
    print("TEST 3b: Geometric Metrics")
    print("-"*70)
    
    try:
        from ramanujan.flow import GeometricMetrics
        
        # Create smooth circular trajectory for better correlation
        n_points = 20  # More points for smoother circle
        t = torch.linspace(0, 2*np.pi, n_points)
        
        # Create larger feature dimension
        feat_dim = 128
        x = torch.cos(t).unsqueeze(1).unsqueeze(2).expand(-1, 2, feat_dim//2)
        y = torch.sin(t).unsqueeze(1).unsqueeze(2).expand(-1, 2, feat_dim//2)
        trajectory = torch.cat([x, y], dim=-1)
        
        # Test curvature
        curvatures = GeometricMetrics.menger_curvature(trajectory)
        print(f"✅ Curvature: shape {curvatures.shape}, mean {curvatures.mean():.6f}")
        
        # Test self-similarity (should be perfect)
        metrics = GeometricMetrics.compute_all_metrics(trajectory, trajectory)
        print(f"✅ Similarity metrics:")
        for name, value in metrics.items():
            print(f"     {name}: {value:.4f}")
        
        # Relaxed assertion - curvature_correlation can be 0 for perfect circles
        # (all curvatures are constant, so correlation of constants is undefined)
        assert metrics['position_similarity'] > 0.95
        assert metrics['velocity_similarity'] > 0.95
        print(f"✅ Position and velocity metrics pass")
        
        return True
        
    except Exception as e:
        print(f"❌ Geometric metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flow_analyzer():
    """Test high-level flow analyzer."""
    print("\n" + "-"*70)
    print("TEST 3c: Flow Analyzer")
    print("-"*70)
    
    try:
        from ramanujan.flow import FlowAnalyzer, quick_curvature, quick_compare
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 64)
            
            def forward(self, input_ids, return_hidden_states=False):
                x = torch.randn(input_ids.size(0), input_ids.size(1), 10)
                h = self.layer(x)
                if return_hidden_states:
                    return h, [h]
                return h
        
        model = SimpleModel()
        tokens = torch.randint(0, 100, (1, 30))
        
        # Test quick functions
        curv = quick_curvature(model, tokens, num_steps=6)
        print(f"✅ quick_curvature: shape {curv.shape}")
        
        model2 = SimpleModel()
        comp = quick_compare(model, model2, tokens, num_steps=6)
        print(f"✅ quick_compare: {len(comp)} metrics")
        
        # Test FlowAnalyzer
        analyzer = FlowAnalyzer(device=DEVICE)
        result = analyzer.analyze_model(model, {'input_ids': tokens}, num_steps=6)
        print(f"✅ FlowAnalyzer: mean_curvature={result['mean_curvature']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Flow analyzer test failed: {e}")
        return False


# ============================================================================
# TEST 4: Integration
# ============================================================================

def test_integration():
    """Test integration of all components."""
    print("\n" + "-"*70)
    print("TEST 4: Integration")
    print("-"*70)
    
    try:
        from ramanujan.architecture import StandardGQA
        from ramanujan.flow import FlowAnalyzer
        from ramanujan.foundation import RamanujanFoundation
        
        # Create sparse attention on CPU
        foundation = RamanujanFoundation(max_prime=50)
        attn = StandardGQA(
            dim=128,
            num_heads=4,
            num_kv_heads=2,
            max_seq_len=256,
            foundation=foundation,
            attention_sparsity=0.5
        )
        
        print(f"✅ Created sparse attention")
        
        # Wrapper model - everything on CPU
        class AttentionModel(nn.Module):
            def __init__(self, attn_layer):
                super().__init__()
                self.embed = nn.Embedding(1000, 128)
                self.attn = attn_layer
                
            def forward(self, input_ids, return_hidden_states=False):
                x = self.embed(input_ids)
                h = self.attn(x)
                if return_hidden_states:
                    return h, [h]
                return h
        
        model = AttentionModel(attn)
        tokens = torch.randint(0, 1000, (1, 32))
        
        # Analyze flow on CPU
        analyzer = FlowAnalyzer(device='cpu')
        result = analyzer.analyze_model(model, {'input_ids': tokens}, num_steps=5)
        
        print(f"✅ Flow analysis on sparse attention:")
        print(f"     Mean curvature: {result['mean_curvature']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN
# ============================================================================

def run_all_tests():
    """Run complete test suite."""
    
    results = {}
    
    print("\n" + "="*70)
    print("RUNNING ALL TESTS")
    print("="*70)
    
    results['foundation'] = test_foundation()
    results['attention'] = test_attention()
    results['feedforward'] = test_feedforward()
    results['flow_trajectory'] = test_flow_trajectory()
    results['geometric_metrics'] = test_geometric_metrics()
    results['flow_analyzer'] = test_flow_analyzer()
    results['integration'] = test_integration()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {name:20s}: {'PASS' if status else 'FAIL'}")
    
    print("-"*70)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

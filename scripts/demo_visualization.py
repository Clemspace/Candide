"""
Demo: Visualize reasoning flows.

Run: python scripts/demo_visualization.py
"""

import sys
sys.path.insert(0, '/home/ccastellon/candide_cracked/candide1.0')

import torch
import torch.nn as nn
from ramanujan import StandardGQA, FlowAnalyzer, FlowVisualizer

print("="*70)
print("Flow Visualization Demo")
print("="*70)

# Create complete models (with embeddings)
class SimpleTransformer(nn.Module):
    """Simple transformer for demo."""
    def __init__(self, vocab_size=1000, dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.attn = StandardGQA(
            dim=dim,
            num_heads=8,
            num_kv_heads=4,
            max_seq_len=512
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, input_ids, return_hidden_states=False):
        x = self.embed(input_ids)
        x = self.attn(x)
        x = self.norm(x)
        
        if return_hidden_states:
            return x, [x]
        return x

print("\n1. Creating models...")
model1 = SimpleTransformer()
model2 = SimpleTransformer()
print("✅ Models created")

# Create sample inputs
tokens = torch.randint(0, 1000, (1, 64))

# Compute trajectories
print("\n2. Computing flow trajectories...")
analyzer = FlowAnalyzer(device='cpu')

result1 = analyzer.analyze_model(model1, {'input_ids': tokens}, num_steps=10)
result2 = analyzer.analyze_model(model2, {'input_ids': tokens}, num_steps=10)
print("✅ Trajectories computed")
print(f"   Model 1 mean curvature: {result1['mean_curvature']:.6f}")
print(f"   Model 2 mean curvature: {result2['mean_curvature']:.6f}")

# Visualize
print("\n3. Creating visualizations...")
viz = FlowVisualizer()

# 3D trajectory
print("   - 3D Trajectory plot...")
viz.plot_trajectory_3d(
    {
        'Model 1': result1['trajectory'],
        'Model 2': result2['trajectory']
    },
    title='Reasoning Flow Comparison',
    save_path='outputs/trajectory_3d.png'
)

# Curvature evolution
print("   - Curvature evolution plot...")
viz.plot_curvature_evolution(
    {
        'Model 1': result1['curvatures'],
        'Model 2': result2['curvatures']
    },
    save_path='outputs/curvature_evolution.png'
)

# Curvature distribution
print("   - Curvature distribution...")
viz.plot_curvature_distribution(
    {
        'Model 1': result1['curvatures'],
        'Model 2': result2['curvatures']
    },
    save_path='outputs/curvature_distribution.png'
)

# Similarity matrix
print("   - Similarity matrix...")
sim_matrix = viz.plot_similarity_matrix(
    {
        'Model 1': result1['trajectory'],
        'Model 2': result2['trajectory']
    },
    metric='curvature',
    save_path='outputs/similarity_matrix.png'
)

print(f"\n   Similarity between models: {sim_matrix[0, 1]:.3f}")

# Model comparison
print("   - Model comparison bar chart...")
viz.plot_model_comparison(
    {
        'Model 1': {
            'Mean Curvature': result1['mean_curvature'],
            'Max Curvature': result1['max_curvature'],
            'Std Curvature': result1['std_curvature']
        },
        'Model 2': {
            'Mean Curvature': result2['mean_curvature'],
            'Max Curvature': result2['max_curvature'],
            'Std Curvature': result2['std_curvature']
        }
    },
    title='Geometric Flow Comparison',
    save_path='outputs/model_comparison.png'
)

print("\n" + "="*70)
print("✅ Demo complete! Saved 5 plots to outputs/")
print("="*70)
print("\nGenerated plots:")
print("  1. outputs/trajectory_3d.png - 3D flow trajectories")
print("  2. outputs/curvature_evolution.png - Curvature over time")
print("  3. outputs/curvature_distribution.png - Distribution histogram")
print("  4. outputs/similarity_matrix.png - Task similarity")
print("  5. outputs/model_comparison.png - Model metrics comparison")
print("\nOpen these images to see your reasoning flow analysis!")
print("="*70)

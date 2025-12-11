# Ramanujan Core Architecture

**Enhanced composition-based framework for building self-modifying neural networks**

[![Tests](https://img.shields.io/badge/tests-208%20passing-brightgreen)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 Design Philosophy

The Ramanujan core is built on four principles:

1. **Protocol-based composition** - Components use duck typing, not inheritance
2. **Explicit registration** - Zero magic, clear component lifecycle
3. **Config-driven architecture** - Build any model from JSON/YAML
4. **Graph-native** - First-class support for arbitrary topologies

**NEW:** Enhanced with automatic shape validation, cost estimation, and state management for self-modifying AI systems.

---

## 📦 Core Modules

```
ramanujan/core/
├── interface.py          # Protocol definitions & data structures
├── registry.py           # Component registration & lookup
├── graph.py              # Computation graph representation
├── builder.py            # Model instantiation from graphs
├── shape_inference.py    # NEW: Automatic shape validation
├── cost_estimation.py    # NEW: Computational cost estimation
└── state_manager.py      # NEW: Stateful component management
```

---

## 🚀 Quick Start

### 1. Install
```bash
pip install -e .
pytest tests/core/  # Verify installation
```

### 2. Define a Component
```python
from ramanujan.core import register_component
from ramanujan.core.interface import TensorSpec
import torch.nn as nn

@register_component('norm', 'rms')
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    @property
    def component_type(self) -> str:
        return 'norm'
    
    @property
    def input_spec(self):
        return {'x': TensorSpec(shape=('batch', 'seq', 'dim'))}
    
    @property
    def output_spec(self):
        return {'x': TensorSpec(shape=('batch', 'seq', 'dim'))}
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
    
    def get_config(self):
        return {'dim': self.weight.shape[0], 'eps': self.eps}
```

### 3. Build a Model with Validation
```python
from ramanujan.core import build_model, ComputationGraph

# Define graph
config = {
    'nodes': [
        {'id': 'emb', 'type': 'embedding', 'config': {'vocab_size': 10000, 'dim': 512}},
        {'id': 'norm', 'type': 'rms', 'config': {'dim': 512}, 'inputs': ['emb']},
        {'id': 'head', 'type': 'lm_head', 'config': {'vocab_size': 10000}, 'inputs': ['norm']}
    ],
    'inputs': ['emb'],
    'outputs': ['head']
}

graph = ComputationGraph.from_dict(config)

# Build with automatic validation
model = build_model(
    graph,
    validate_shapes=True,         # Catch dimension mismatches
    input_shapes={'emb': (8, 512)},
    estimate_cost=True,            # Print FLOPs/params estimate
    manage_state=False             # Enable for stateful components
)

# Use like any PyTorch model
output = model(input_ids)
```

---

## 🔍 New Features

### Shape Inference

Automatically validates tensor shapes flow correctly through your graph:

```python
from ramanujan.core.shape_inference import infer_shapes

# Infer all shapes
shapes = infer_shapes(graph, {'input': (8, 512)})

print(shapes['layer_0'])  # {'x': (8, 512, 768)}

# Or use convenience method
shapes = graph.validate_shapes({'input': (8, 512)})
```

**Why this matters:** Catches dimension mismatches *before* instantiation, essential for self-modifying systems that propose new architectures.

### Cost Estimation

Estimates computational cost without running the model:

```python
from ramanujan.core.cost_estimation import estimate_cost

# Estimate cost
cost = estimate_cost(graph, {'input': (8, 512)}, detailed=True)

print(cost)  # FLOPs: 1,234,567 | Params: 50,000 | Memory: 2.5 MB

# Or use convenience method
cost = graph.estimate_cost({'input': (8, 512)})

# See breakdown
print(cost.summary())
```

**Why this matters:** Architecture search can prune invalid mutations before wasting GPU time.

### State Management

Automatic state management for stateful components (RNNs, Titans memory, etc.):

```python
from ramanujan.core import build_model

# Build with state management
model = build_model(graph, manage_state=True)

# State automatically saved/loaded across forward passes
for batch in dataloader:
    output = model(batch)  # State persists

# Reset state between sequences
model.reset_state()

# Checkpoint and restore
checkpoint_id = model.checkpoint()
# ... do some forward passes ...
model.restore_checkpoint(checkpoint_id)
```

**Why this matters:** Essential for Titans/Hope memory, RNNs, and continual learning.

---

## 🏗️ Architecture Patterns

### Pattern 1: Sequential (GPT-style)
```python
graph = ComputationGraph.from_sequential([
    {'type': 'embedding', 'config': {'vocab_size': 32000, 'dim': 768}},
    {'type': 'transformer_block', 'config': {'dim': 768, 'num_heads': 12}},
    {'type': 'transformer_block', 'config': {'dim': 768, 'num_heads': 12}},
    {'type': 'lm_head', 'config': {'vocab_size': 32000}}
])
```

### Pattern 2: With Reasoning Module (R1-style)
```python
graph = ComputationGraph.from_sequential([
    {'type': 'embedding', 'config': {'vocab_size': 32000, 'dim': 768}},
    {'type': 'transformer_block', 'config': {'dim': 768, 'num_heads': 12}},
    {'type': 'reasoning', 'config': {'dim': 768, 'n_recursions': 8}},  # NEW
    {'type': 'lm_head', 'config': {'vocab_size': 32000}}
])
```

### Pattern 3: With Memory (Titans/Hope)
```python
graph = ComputationGraph.from_sequential([
    {'type': 'embedding', 'config': {'vocab_size': 32000, 'dim': 768}},
    {'type': 'titans_memory', 'config': {'dim': 768, 'memory_size': 1024}},  # NEW
    {'type': 'transformer_block', 'config': {'dim': 768, 'num_heads': 12}},
    {'type': 'lm_head', 'config': {'vocab_size': 32000}}
])

# IMPORTANT: Use state management
model = build_model(graph, manage_state=True)
```

### Pattern 4: Multi-Modal (CLIP-style)
```python
graph = ComputationGraph()

# Vision branch
graph.add_node(Node(id='clip_vision', type='clip_encoder', config={...}))

# Text branch
graph.add_node(Node(id='clip_text', type='clip_encoder', config={...}))

# Fusion
graph.add_node(Node(
    id='fusion',
    type='multimodal_fusion',
    config={...},
    inputs=['clip_vision', 'clip_text']  # Multiple inputs!
))

graph.set_inputs(['clip_vision', 'clip_text'])
graph.set_outputs(['fusion'])
```

### Pattern 5: Self-Modifying (Architecture Search)
```python
from ramanujan.core.graph import MutableGraph, AddNodeMutation

# Start with base graph
graph = MutableGraph.from_sequential([...])

# Propose mutation
mutation = AddNodeMutation(
    node_id='new_reasoning',
    component_type='reasoning',
    config={'dim': 768},
    insert_after='layer_5'
)

# Validate before applying
try:
    # Simulate mutation
    test_graph = graph.clone()
    mutation.apply(test_graph)
    
    # Validate shapes
    shapes = test_graph.validate_shapes({'input': (8, 512)})
    
    # Estimate cost
    cost = test_graph.estimate_cost({'input': (8, 512)})
    
    # Check if feasible
    if cost.memory_mb < MAX_MEMORY:
        # Apply for real
        graph.apply_mutation(mutation)
except ValueError as e:
    print(f"Invalid mutation: {e}")
    # Try different mutation
```

---

## 🧪 Testing

The core has 208+ comprehensive tests:

```bash
# Run all core tests
pytest tests/core/ -v

# Test specific modules
pytest tests/core/test_interface.py -v
pytest tests/core/test_registry.py -v
pytest tests/core/test_graph.py -v
pytest tests/core/test_builder.py -v
pytest tests/core/test_enhancements.py -v  # NEW

# With coverage
pytest tests/core/ --cov=ramanujan.core --cov-report=html
```

**Test Organization:**
- `test_interface.py` - Protocol compliance, data structures
- `test_registry.py` - Registration, lookup, metadata
- `test_graph.py` - Graph construction, validation, serialization
- `test_builder.py` - Model building, execution, integration
- `test_enhancements.py` - **NEW:** Shape inference, cost estimation, state management
- `test_integration.py` - End-to-end workflows

---

## 📊 Performance

**Core Operations:**
- Registration: O(1) component lookup  
- Graph Construction: O(N) where N = number of nodes  
- Topological Sort: O(N + E) where E = number of edges  
- Shape Inference: O(N) with caching
- Cost Estimation: O(N) per node
- Execution: O(N) sequential, O(1) per node  
- Memory: Minimal overhead, graph stored as dicts

**Benchmarks (on CPU):**
- Register component: ~0.1ms
- Build 12-layer GPT graph: ~5ms
- Validate shapes (12 layers): ~10ms
- Estimate cost (12 layers): ~15ms
- Instantiate model: ~100ms (depends on component initialization)
- Forward pass: Same as hand-coded nn.Module

---

## 🎯 Design Decisions

### Why Protocols over Inheritance?
**Flexibility.** Components don't need to inherit from base classes. Any object with the right methods is a Component. This means:
- Wrap external libraries (HuggingFace, timm) without modification
- Mix PyTorch modules with custom components
- Test components in isolation

### Why Explicit Registration?
**Predictability.** No auto-discovery magic that breaks when imports change. Registration happens at module import time, making the available components deterministic.

### Why Immutable Graphs?
**Safety.** Once validated, a graph can't be accidentally broken. For architecture search, use MutableGraph which tracks all changes with undo/redo.

### Why Config-Driven?
**Reproducibility.** Save entire model architecture as JSON. Share configs, not code. Version control architectures easily.

### Why Separate Enhancement Modules?
**Modularity.** Core stays clean and focused. Optional features (shape inference, cost estimation, state management) are truly optional and easy to test in isolation.

---

## 🔮 Why This Matters for AGI

This framework is uniquely positioned for self-modifying AI because:

1. ✅ **Graph-native architecture** - Arbitrary topologies, not just sequential
2. ✅ **Protocol-based** - External code works without modification
3. ✅ **Self-modification by design** - Mutable graphs with undo/redo
4. ✅ **Shape validation** - Catch invalid mutations before execution
5. ✅ **Cost estimation** - Evaluate mutations before wasting compute
6. ✅ **State management** - Support for Titans/Hope memory
7. ✅ **AI-readable/writable** - Components can be introspected and modified programmatically

**The key insight:** Most frameworks are designed for humans to build static models. Candide is designed for AI to build and modify its own architecture.

---

## 📚 Advanced Usage

### Custom Shape Inference

Components can implement custom shape inference:

```python
class GlobalPooling(nn.Module):
    @property
    def output_spec(self):
        return {
            'x': TensorSpec(
                shape=('batch', 'channels'),
                shape_fn=lambda inputs, cfg: (
                    inputs['x'][0],  # batch
                    inputs['x'][1]   # channels
                    # Drop spatial dims [2] and [3]
                )
            )
        }
```

### Custom Cost Estimation

Components can implement accurate cost estimates:

```python
class MyAttention(nn.Module):
    @staticmethod
    def estimate_cost(input_shapes, config):
        batch, seq, dim = input_shapes['x']
        num_heads = config['num_heads']
        
        # QKV projection
        flops_qkv = 3 * batch * seq * dim * dim
        
        # Attention scores
        flops_attn = 2 * batch * num_heads * seq * seq * (dim // num_heads)
        
        # Parameters
        params = 4 * dim * dim  # QKV + output
        
        return ComputeCost(
            flops=flops_qkv + flops_attn,
            params=params,
            memory_mb=(params * 4) / (1024 ** 2)
        )
```

### Stateful Components

Components that need to maintain state:

```python
class TitansMemory(nn.Module):
    def __init__(self, dim, memory_size):
        super().__init__()
        self.memory_net = nn.Sequential(...)
        self.surprise_threshold = 0.1
        self.ema_loss = None
    
    def forward(self, x):
        # ... memory logic ...
        return output
    
    def get_state(self):
        """Return state to persist."""
        return {
            'memory_params': self.memory_net.state_dict(),
            'ema_loss': self.ema_loss
        }
    
    def set_state(self, state):
        """Restore state."""
        self.memory_net.load_state_dict(state['memory_params'])
        self.ema_loss = state['ema_loss']
    
    def reset_state(self):
        """Reset to initial state."""
        self.ema_loss = None
```

---

## 🤝 Contributing

See `../CONTRIBUTING.md` for:
- Component implementation guidelines
- Testing requirements
- Code style (black, ruff, mypy)
- Commit message format

---

## 📄 License

See `../LICENSE`

---

## 🙏 Acknowledgments

Built with insights from:
- Modern neural architecture research
- Production ML systems
- Self-modifying AI requirements
- Community feedback

**Special thanks to the Titans/Hope, R1, and CLIP teams for inspiration.**

---

## 🔗 Related Documentation

- [Architecture Roadmap for AGI](../docs/architecture_roadmap.md)
- [Component Library](../components/README.md)
- [Examples](../examples/README.md)
- [API Reference](../docs/api_reference.md)
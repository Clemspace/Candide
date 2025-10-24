# Ramanujan Core Architecture

**Composition-based framework for building neural networks as directed acyclic graphs (DAGs) of reusable components.**

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

---

## 📦 Core Modules

```
ramanujan/core/
├── interface.py     # Protocol definitions & data structures
├── registry.py      # Component registration & lookup
├── graph.py         # Computation graph representation
└── builder.py       # Model instantiation from graphs
```

### `interface.py` - Protocols & Data Structures

**Purpose:** Define contracts for components without forcing inheritance.

**Key Protocols:**
```python
class Component(Protocol):
    """Minimal component interface."""
    component_type: str
    component_name: str
    def forward(self, x, **kwargs): ...

class StatefulComponent(Component, Protocol):
    """Component with internal state (RNNs, normalizations)."""
    def get_state(self) -> Dict: ...
    def set_state(self, state: Dict): ...
    def reset_state(self): ...

class ComposableComponent(Component, Protocol):
    """Component that can contain sub-components."""
    def get_subcomponents(self) -> Dict[str, Component]: ...

class MultiModeComponent(Component, Protocol):
    """Component with different behaviors per mode."""
    def forward(self, x, mode: ForwardMode, **kwargs): ...
```

**Key Data Structures:**
```python
@dataclass
class TensorSpec:
    """Specifies tensor shape, dtype, and modality."""
    shape: List[Union[int, str]]  # e.g., ["batch", "seq", 768]
    dtype: torch.dtype
    modality: Modality  # TEXT, IMAGE, AUDIO, etc.
    description: Optional[str]

class ForwardMode(Enum):
    """Execution mode for forward pass."""
    TRAIN = "train"
    EVAL = "eval"
    GENERATE = "generate"
    INFERENCE = "inference"

@dataclass
class ForwardMetadata:
    """Optional metadata passed through forward."""
    mode: ForwardMode
    components_used: List[str]
    routing_decisions: Dict[str, Any]
    custom: Dict[str, Any]
```

**Why Protocols?**
- ✅ Any class can be a Component by implementing methods
- ✅ Zero runtime overhead (type hints only)
- ✅ External code (torch.nn.Module) compatible without modification
- ✅ Multiple protocol inheritance for rich behaviors

---

### `registry.py` - Component Registration

**Purpose:** Map string identifiers to component classes for config-driven instantiation.

**Usage:**
```python
from ramanujan.core import register_component, get_component, create_component

# Registration (at module import time)
@register_component('block', 'transformer')
class TransformerBlock(nn.Module):
    @property
    def component_type(self) -> str:
        return 'block'
    
    @property
    def component_name(self) -> str:
        return 'transformer'
    
    # ... rest of implementation

# Lookup (at build time)
BlockClass = get_component('block', 'transformer')
block = BlockClass(dim=768, num_heads=12)

# Or create directly from config
block = create_component('block', 'transformer', {
    'dim': 768,
    'num_heads': 12
})
```

**Registry Structure:**
```python
ComponentRegistry._registry = {
    'embedding': {'token': TokenEmbedding, 'rotary': RoPE, ...},
    'block': {'transformer': TransformerBlock, 'mamba': MambaBlock, ...},
    'attention': {'mha': MultiHeadAttention, 'flash': FlashAttention, ...},
    'ffn': {'mlp': MLP, 'swiglu': SwiGLU, ...},
    'norm': {'layer': LayerNorm, 'rms': RMSNorm, ...},
    'head': {'lm': LMHead, 'classification': ClassificationHead, ...},
    # Infinitely extensible!
}
```

**API:**
```python
ComponentRegistry.get(category, name) -> Type
ComponentRegistry.register(category, name, override=False) -> Decorator
ComponentRegistry.has(category, name) -> bool
ComponentRegistry.list_categories() -> List[str]
ComponentRegistry.list_components(category, include_metadata=False) -> List | Dict
ComponentRegistry.clear(category=None)  # For testing
```

**Why Explicit Registration?**
- ✅ No auto-discovery magic (predictable imports)
- ✅ Clear errors when components missing
- ✅ Easy to swap implementations
- ✅ O(1) lookup performance

---

### `graph.py` - Computation Graph

**Purpose:** Represent models as directed acyclic graphs (DAGs) of components.

**Node Structure:**
```python
@dataclass(frozen=True)
class Node:
    """Node in computation graph."""
    id: str                      # Unique identifier
    component_type: str          # 'block', 'attention', etc.
    config: Dict[str, Any]       # Component configuration
    inputs: List[str]            # IDs of input nodes
    metadata: Dict[str, Any]     # Optional metadata
```

**Graph Types:**

#### 1. `ComputationGraph` - Immutable DAG
```python
graph = ComputationGraph()

# Add nodes
graph.add_node(Node(id='emb', component_type='embedding', config={...}))
graph.add_node(Node(id='block_0', component_type='block', config={...},
                    inputs=['emb']))

# Set boundaries
graph.set_inputs(['emb'])
graph.set_outputs(['head'])

# Validate
graph.validate()  # Checks cycles, connectivity

# Get execution order
execution_order = graph.topological_sort()
```

#### 2. `MutableGraph` - For Architecture Search
```python
graph = MutableGraph.from_sequential([...])

# Mutations
graph.add_node_after('block_5', Node(id='reasoning', ...))
graph.remove_node('block_3', bypass=True)
graph.modify_config('block_0', {'num_heads': 16})
graph.add_edge('block_2', 'block_8')  # Skip connection

# Undo/Redo
graph.undo_last_mutation()
graph.redo_last_mutation()
```

**Graph Capabilities:**
- ✅ Sequential chains (GPT-style)
- ✅ Skip connections (ResNet-style)
- ✅ Parallel branches (multi-path)
- ✅ Fusion nodes (combine multiple inputs)
- ✅ Arbitrary DAGs (any topology)

**Validation:**
- Cycle detection
- Connectivity checks
- Input/output verification
- Type compatibility (future)

**Serialization:**
```python
# To dict/JSON
config = graph.to_dict()

# From dict/JSON
graph = ComputationGraph.from_dict(config)

# Save/Load
graph.save('model_graph.json')
graph = ComputationGraph.load('model_graph.json')
```

---

### `builder.py` - Model Instantiation

**Purpose:** Convert graphs + registry → executable nn.Module.

**Usage:**
```python
from ramanujan.core import build_model

# From config dict
config = {
    'graph': {
        'nodes': [
            {'id': 'emb', 'type': 'embedding', 'name': 'token',
             'config': {'vocab_size': 50000, 'dim': 768}},
            {'id': 'block_0', 'type': 'block', 'name': 'transformer',
             'config': {'dim': 768, 'num_heads': 12},
             'inputs': ['emb']},
            {'id': 'head', 'type': 'head', 'name': 'lm',
             'config': {'vocab_size': 50000},
             'inputs': ['block_0']}
        ],
        'inputs': ['emb'],
        'outputs': ['head']
    }
}

model = build_model(config)

# From YAML file
model = build_model('configs/gpt_small.yaml')

# Forward pass
output = model(input_ids)
```

**GraphExecutor:**
```python
# Under the hood, build_model creates a GraphExecutor
executor = GraphExecutor(graph, registry=ComponentRegistry)

# Instantiates components
executor.components = {
    'emb': TokenEmbedding(vocab_size=50000, dim=768),
    'block_0': TransformerBlock(dim=768, num_heads=12),
    'head': LMHead(vocab_size=50000)
}

# Executes in topological order
def forward(self, x):
    outputs = {}
    for node_id in self.execution_order:
        node = self.graph.nodes[node_id]
        component = self.components[node_id]
        
        # Gather inputs
        if node.inputs:
            inputs = [outputs[inp] for inp in node.inputs]
            x = inputs[0] if len(inputs) == 1 else tuple(inputs)
        
        # Execute
        outputs[node_id] = component(x)
    
    return outputs[self.output_node_id]
```

**Detection:**
- Sequential graphs → optimized linear execution
- DAGs → topological execution with caching
- Multi-output → returns dict of outputs

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
    def component_name(self) -> str:
        return 'rms'
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

### 3. Build a Model
```python
from ramanujan.core import build_model

config = {
    'graph': {
        'nodes': [
            {'id': 'emb', 'type': 'embedding', 'name': 'token',
             'config': {'vocab_size': 10000, 'dim': 512}},
            {'id': 'norm', 'type': 'norm', 'name': 'rms',
             'config': {'dim': 512}, 'inputs': ['emb']},
            {'id': 'head', 'type': 'head', 'name': 'lm',
             'config': {'vocab_size': 10000}, 'inputs': ['norm']}
        ],
        'inputs': ['emb'],
        'outputs': ['head']
    }
}

model = build_model(config)
print(f"Built model with {sum(p.numel() for p in model.parameters()):,} parameters")
```

### 4. Train
```python
# Just use it like any PyTorch model!
optimizer = torch.optim.Adam(model.parameters())

for batch in dataloader:
    optimizer.zero_grad()
    output = model(batch['input_ids'])
    loss = F.cross_entropy(output.view(-1, vocab_size), batch['labels'].view(-1))
    loss.backward()
    optimizer.step()
```

---

## 🏗️ Architecture Patterns

### Pattern 1: Sequential (GPT-style)
```python
ComputationGraph.from_sequential([
    {'type': 'embedding', 'name': 'token', 'config': {...}},
    {'type': 'block', 'name': 'transformer', 'config': {...}},
    {'type': 'block', 'name': 'transformer', 'config': {...}},
    {'type': 'head', 'name': 'lm', 'config': {...}}
])
```

### Pattern 2: Skip Connections (ResNet-style)
```python
graph.add_node(Node(id='block_0', ...))
graph.add_node(Node(id='block_1', ..., inputs=['block_0']))
graph.add_node(Node(id='fusion', ..., inputs=['block_0', 'block_1']))  # Skip!
```

### Pattern 3: Multi-Head (Multi-task)
```python
# Single backbone, multiple heads
graph.add_node(Node(id='backbone', ...))
graph.add_node(Node(id='lm_head', ..., inputs=['backbone']))
graph.add_node(Node(id='class_head', ..., inputs=['backbone']))
graph.set_outputs(['lm_head', 'class_head'])  # Both outputs!
```

### Pattern 4: Encoder-Decoder
```python
graph.add_node(Node(id='encoder', ...))
graph.add_node(Node(id='decoder', ..., inputs=['encoder']))
graph.add_node(Node(id='cross_attn', ..., inputs=['decoder', 'encoder']))
```

### Pattern 5: Hybrid (Mamba + Transformer)
```python
ComputationGraph.from_sequential([
    {'type': 'embedding', 'name': 'token', 'config': {...}},
    {'type': 'block', 'name': 'mamba', 'config': {...}},      # Mamba block
    {'type': 'block', 'name': 'transformer', 'config': {...}}, # Transformer
    {'type': 'block', 'name': 'mamba', 'config': {...}},      # Mamba again
    {'type': 'head', 'name': 'lm', 'config': {...}}
])
```

---

## 🧪 Testing

The core has 208 comprehensive tests covering:

```bash
# Run all core tests
pytest tests/core/ -v

# Test specific modules
pytest tests/core/test_interface.py -v
pytest tests/core/test_registry.py -v
pytest tests/core/test_graph.py -v
pytest tests/core/test_builder.py -v

# With coverage
pytest tests/core/ --cov=ramanujan.core --cov-report=html
```

**Test Organization:**
- `test_interface.py` - Protocol compliance, data structures
- `test_registry.py` - Registration, lookup, metadata
- `test_graph.py` - Graph construction, validation, serialization
- `test_builder.py` - Model building, execution, integration
- `test_integration.py` - End-to-end workflows

---

## 📊 Performance

**Registration:** O(1) component lookup  
**Graph Construction:** O(N) where N = number of nodes  
**Topological Sort:** O(N + E) where E = number of edges  
**Execution:** O(N) sequential, O(1) per node  
**Memory:** Minimal overhead, graph stored as dicts

**Benchmarks (on CPU):**
- Register component: ~0.1ms
- Build 12-layer GPT graph: ~5ms
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

---

## 🔮 Future Extensions

Planned features:
- [ ] Type checking (TensorSpec validation)
- [ ] Dynamic graphs (conditional execution)
- [ ] Parallel execution (multi-GPU)
- [ ] Quantization awareness
- [ ] ONNX export
- [ ] Architecture search (AutoML)
- [ ] Visualization (graphviz export)

---

## 📚 References

**Inspiration:**
- [PyTorch nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [Keras Functional API](https://keras.io/guides/functional_api/)
- [ONNX](https://onnx.ai/)
- [NNI (Neural Network Intelligence)](https://github.com/microsoft/nni)

**Related Work:**
- Protocol-based design: [PEP 544](https://peps.python.org/pep-0544/)
- Graph neural architectures: [Neural Architecture Search](https://arxiv.org/abs/1808.05377)
- Composition over inheritance: [Design Patterns](https://en.wikipedia.org/wiki/Composition_over_inheritance)

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

Built with insights from modern neural architecture research and production ML systems.
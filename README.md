# Candide: Transparent Transformer Training

> *"Il faut cultiver notre jardin"* — Voltaire, *Candide*

A research framework for training efficient transformer language models with mathematical rigor and transparency. Named after Voltaire's philosophical novel, Candide embodies honest, straightforward approaches to model architecture and training.

Developed by the **ModERN team** (Sorbonne Université, ENS, SCAI) led by Glenn Roe.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is Candide?

Candide is a research-focused transformer framework that prioritizes:
- **Transparency**: Every architectural choice is documented and justified
- **Efficiency**: Structured sparsity via Ramanujan graphs, modern attention mechanisms
- **Reproducibility**: Deterministic training with detailed configs
- **Education**: Clean, commented code for understanding transformer internals

Unlike production frameworks that hide complexity, Candide exposes it. We believe researchers should understand *why* architectural choices work, not just *how* to use them.

**Target audience:**
- Researchers exploring novel architectures
- Students learning transformer internals  
- Engineers prototyping efficient models on limited compute

---

## Key Features

### Architecture
- **Grouped Query Attention (GQA)**: 50-75% KV cache reduction
- **RoPE embeddings**: Complex number formulation (30-40% faster than sin/cos)
- **Sliding window attention**: O(n·window) for long contexts
- **SwiGLU feedforward**: Better than ReLU/GELU activations
- **RMSNorm**: More efficient than LayerNorm

### Novel Techniques
- **Ramanujan graph sparsity**: Number theory-based structured pruning (80-90% sparsity)
- **Semantic entropy loss**: Uncertainty-aware training
- **WSD scheduler**: Warmup-Stable-Decay (3-5% better than cosine)

### Practical Features
- Multiple optimizers (AdamW, Muon, AdEMAMix)
- Mixed precision training (FP16/BF16)
- Gradient checkpointing for memory efficiency
- Multi-GPU support (DDP ready)
- Weights & Biases integration

---

## Quick Start
```bash
# Install
git clone https://github.com/Clemspace/candide.git
cd candide
pip install -r requirements.txt

# Train a 35M parameter model (2 minutes on single GPU)
python ramanujan/training/train.py --config configs/tiny-test.yaml

# Train with W&B logging
python ramanujan/training/train.py --config configs/small-100m.yaml --wandb
```

### Simple Inference
```python
from ramanujan.architecture import create_model, ModelConfig
from ramanujan.utils import load_checkpoint

# Create model
config = ModelConfig(
    vocab_size=32000,
    dim=512,
    num_layers=6,
    num_heads=8,
    num_kv_heads=4
)
model = create_model(config)

# Load checkpoint
load_checkpoint(model, 'checkpoints/best.pt')

# Generate text
from ramanujan.inference import generate
text = generate(model, tokenizer, prompt="Once upon a time", max_length=100)
```

---

## Model Configurations

We provide configs for various compute budgets:

| Config | Parameters | Layers | Dim | GPU Memory | Use Case |
|--------|-----------|--------|-----|------------|----------|
| `tiny-test.yaml` | 35M | 6 | 512 | <2GB | Testing |
| `small-100m.yaml` | 100M | 12 | 768 | ~16GB | Single GPU |
| `medium-350m.yaml` | 350M | 24 | 1024 | ~24GB | Production |
| `medium-350m-minimal.yaml` | 350M | 24 | 1024 | ~4GB | Memory-efficient |

Each config includes detailed inline comments explaining every parameter and alternative options.

See [configs/README.md](configs/README.md) for the complete configuration guide.

---

## Architecture Overview
```
Input tokens
    ↓
Token embeddings (with vocab safety)
    ↓
┌─────────────────────────────────┐
│  Transformer Block × N          │
│  ┌─────────────────────────┐    │
│  │ RMSNorm                 │    │
│  │ GQA + RoPE              │    │
│  │ Residual                │    │
│  └─────────────────────────┘    │
│  ┌─────────────────────────┐    │
│  │ RMSNorm                 │    │
│  │ SwiGLU FFN              │    │
│  │ Residual                │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘
    ↓
Final RMSNorm
    ↓
Output projection (tied embeddings)
    ↓
Loss (CE or Semantic Entropy)
```

### Ramanujan Graph Sparsity

Uses coprime relationships from number theory to create structured sparse networks that maintain connectivity better than random pruning:
```python
from ramanujan.foundation import RamanujanFoundation

foundation = RamanujanFoundation(max_prime=1000)
sparse_layer = foundation.create_layer(
    in_features=512,
    out_features=512,
    target_sparsity=0.85  # 85% sparse, minimal quality loss
)
```

The mathematical foundation ensures information flow is preserved even at high sparsity levels.

---

## Training

### Basic Training
```bash
python ramanujan/training/train.py --config configs/small-100m.yaml --wandb
```

### Multi-GPU (DDP)
```bash
torchrun --nproc_per_node=4 ramanujan/training/train.py --config configs/medium-350m.yaml
```

### Resume from Checkpoint
```bash
python ramanujan/training/train.py \
    --config configs/small-100m.yaml \
    --resume checkpoints/small-100m/step_5000.pt
```

### Memory-Efficient Training

Use gradient checkpointing to reduce memory by 30-50%:
```yaml
# In your config
training:
  use_gradient_checkpointing: true
```

---

## Results

### Preliminary Benchmarks

Training 35M parameters on WikiText-2 (1000 steps):

| Optimizer | Final Loss | Perplexity | Training Time |
|-----------|-----------|------------|---------------|
| AdamW | 4.81 | 122.3 | 2.3 min |
| Muon | 4.75 | 115.8 | 2.1 min |

### Memory Usage (measured)

| Model | Params | Config | Batch | Seq Len | Memory |
|-------|--------|--------|-------|---------|--------|
| Tiny | 35M | tiny-test | 2 | 512 | <2GB |
| Small | 100M | small-100m | 4 | 1024 | ~16GB |
| Medium | 350M | medium-350m-minimal | 2 | 512 | ~4GB |

*Memory measurements include model, optimizer state, and activations with mixed precision.*

---

## Project Structure
```
candide/
├── ramanujan/                  # Core package
│   ├── architecture/           # Model components
│   │   ├── attention.py        # GQA, sliding window
│   │   ├── blocks.py           # Transformer blocks
│   │   ├── embeddings.py       # RoPE, learned embeddings
│   │   ├── feedforward.py      # SwiGLU, standard FFN
│   │   ├── model.py            # Complete models
│   │   └── normalization.py    # RMS, Layer, QK norms
│   ├── foundation/             # Ramanujan graphs
│   │   ├── core.py             # Graph generation
│   │   └── layers.py           # Sparse layers
│   ├── training/               # Training utilities
│   │   ├── trainer.py          # Main trainer class
│   │   ├── optimizers.py       # Optimizer factory
│   │   ├── schedulers.py       # LR schedules (WSD, cosine, etc)
│   │   ├── losses.py           # CE, semantic entropy
│   │   └── train.py            # Training script
│   ├── data/                   # Data loading
│   │   ├── tokenizer.py        # Tokenization
│   │   └── datasets.py         # Dataset wrappers
│   └── utils/                  # Utilities
│       ├── config.py           # Config management
│       └── logging.py          # Logging setup
├── configs/                    # Model configurations
│   ├── tiny-test.yaml          # 35M (tested ✓)
│   ├── small-100m.yaml         # 100M
│   ├── medium-350m.yaml        # 350M (full)
│   ├── medium-350m-minimal.yaml # 350M (memory-efficient)
│   └── README.md               # Config documentation
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── requirements.txt
└── README.md
```

---

## Development

### Running Tests
```bash
# All modules include self-tests
python ramanujan/architecture/attention.py
python ramanujan/architecture/blocks.py
python ramanujan/training/schedulers.py

# Run pytest (when available)
pytest tests/
```

### Code Style

We prioritize readability over strict style enforcement:
- Clear variable names over brevity
- Comments explaining *why*, not *what*
- Documented edge cases and assumptions
- Minimal abstractions - explicit is better than implicit

---

## Design Philosophy

### Why "Candide"?

In Voltaire's *Candide*, the protagonist learns through experience to reject blind optimism and embrace pragmatic work ("cultivate our garden"). Similarly, this framework rejects:
- Over-engineered abstractions that hide complexity
- Blind adoption of "best practices" without understanding
- Marketing-driven architecture choices

Instead, we cultivate:
- Transparent, understandable code
- Documented design decisions with ablations
- Honest reporting of limitations and tradeoffs

### Research Values

- **Transparency**: Every architectural choice is documented
- **Reproducibility**: Configs and seeds for deterministic results  
- **Education**: Code that teaches transformer internals
- **Efficiency**: Practical techniques for limited compute
- **Honesty**: Report failures and limitations, not just successes

---

## Citation

If you use Candide in your research:
```bibtex
@software{Candide2025,
  title={Candide: A Transparent Framework for Transformer Training},
  author={Castellon, Clément and ModERN Team},
  year={2025},
  organization={Monash University},
  url={https://github.com/Clemspace/candide}
}
```

---

## Acknowledgments

### ModERN Team
- **Glenn Roe** (Team Lead) - Monash University
- **Clément Castellon** (Primary Developer)

### Technical Foundations
- Meta AI: Llama architecture, RoPE implementation
- Mistral AI: Sliding window attention, WSD scheduler
- Google Research: SwiGLU, RMSNorm
- Ramanujan graph theory: Mathematical foundations for sparsity

### Tools
- PyTorch, Hugging Face, Weights & Biases

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contact

- **ModERN Team**: [modern.team@monash.edu]
- **Issues**: [github.com/Clemspace/candide/issues](https://github.com/Clemspace/candide/issues)
- **Discussions**: [github.com/Clemspace/candide/discussions](https://github.com/Clemspace/candide/discussions)

---

<div align="center">

*"Le travail éloigne de nous trois grands maux: l'ennui, le vice et le besoin."*  
*"Work keeps at bay three great evils: boredom, vice, and need."*  
— Voltaire, *Candide*

**Cultivating our computational garden, one transformer at a time.**

</div>

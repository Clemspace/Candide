# demo-llama

Foundation model project built with Candide.

## Quick Start

```bash
# Train the model
candide train configs/base.yaml

# Validate configuration
candide validate configs/base.yaml

# Check system setup
candide doctor
```

## Project Structure

```
demo-llama/
├── configs/        # Model and training configurations
├── data/          # Training data
├── checkpoints/   # Model checkpoints
└── logs/          # Training logs
```

## Configuration

Edit `configs/base.yaml` to customize your model architecture and training setup.

## Documentation

See https://docs.candide.ai for full documentation.

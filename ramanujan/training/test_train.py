#!/usr/bin/env python3
"""Quick test training run."""

import torch
import yaml
import sys
from pathlib import Path

from ramanujan.models.architectures.config import TransformerConfig
from ramanujan.models.architectures.transformer import RamanujanTransformer
from ramanujan.training.optimizers import create_optimizer_from_config
from ramanujan.training.losses import create_loss_from_config
from ramanujan.training.schedulers import create_scheduler_from_config
from ramanujan.training.trainer import Trainer
from ramanujan.training.trainer.base import TrainingConfig
from ramanujan.training.trainer.callbacks import (
    ProgressCallback,
    MetricsCallback,
    CheckpointCallback,
)

print("="*70)
print("🚀 Test Training Run")
print("="*70)

# Load config - use command line arg or default
if len(sys.argv) > 1:
    config_path = sys.argv[1]
else:
    config_path = "configs/test-config.yaml"

print(f"\n📋 Loading config from {config_path}")
with open(config_path) as f:
    config = yaml.safe_load(f)

# Create model
print("\n🔨 Creating model...")
model_config = TransformerConfig.from_dict(config['model'])
model = RamanujanTransformer(model_config)

n_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model created: {n_params:,} parameters")

# Move to device
device = torch.device(config['hardware']['device'])
model = model.to(device)
print(f"✅ Model on {device}")

# Create optimizer
print("\n⚙️  Creating optimizer...")
optimizer_config = {
    'name': config['training']['optimizer'],
    'lr': config['training']['learning_rate'],
    'weight_decay': config['training']['weight_decay'],
}
optimizer = create_optimizer_from_config(optimizer_config, model.parameters())
print(f"✅ Optimizer: {config['training']['optimizer']}")

# Create scheduler
print("\n📈 Creating scheduler...")
scheduler_config = {
    'name': 'cosine',
    'total_steps': config['training']['max_steps'],
}
scheduler = create_scheduler_from_config(scheduler_config, optimizer)
print(f"✅ Scheduler: cosine")

# Create loss
print("\n🎯 Creating loss function...")
loss_config = {
    'name': 'cross_entropy',
    'vocab_size': config['model']['vocab_size'],
}
loss_fn = create_loss_from_config(loss_config)
print("✅ Loss: cross_entropy")

# Create dataloaders
print("\n📚 Loading data...")

# Create tokenizer using your existing module
print("📝 Loading tokenizer...")
from ramanujan.data.tokenizer import get_tokenizer
tokenizer = get_tokenizer(vocab_size=config['model']['vocab_size'])
print(f"✅ Tokenizer loaded: vocab_size={len(tokenizer)}")

from ramanujan.data.loaders.factory import WikiTextLoader

loader = WikiTextLoader(
    dataset_name=config['data']['subset'],
    tokenizer=tokenizer,
    sequence_length=config['data']['sequence_length'],
    verbose=True,
)

train_loader, val_loader = loader.get_dataloaders(
    batch_size=config['training']['batch_size'],
    num_workers=config['data']['num_workers'],
)

print(f"✅ Train batches: {len(train_loader)}")
print(f"✅ Val batches: {len(val_loader)}")

# Create training config
print("\n⚙️  Creating training config...")
training_config = TrainingConfig(
    output_dir=config['training']['output_dir'],
    max_steps=config['training']['max_steps'],
    batch_size=config['training']['batch_size'],
    gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
    learning_rate=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay'],
    max_grad_norm=config['training']['grad_clip'],
    log_every=config['logging']['log_every'],
    eval_every=config['training']['eval_every'],
    save_every=config['training']['save_every'],
    mixed_precision=config['training']['mixed_precision'],
    device=device,
    seed=config['hardware']['seed'],
    use_wandb=config['logging']['use_wandb'],
)

# Create callbacks
callbacks = [
    ProgressCallback(),
    MetricsCallback(window_size=50, log_every=5),
    CheckpointCallback(save_every=50, keep_last_n=3),
]

# Create trainer
print("\n🏋️  Creating trainer...")
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    config=training_config,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    scheduler=scheduler,
    callbacks=callbacks,
)

print("✅ Trainer ready!")

# Train!
print("\n" + "="*70)
print("🚀 Starting Training!")
print("="*70 + "\n")

try:
    trainer.train()
    
    print("\n" + "="*70)
    print("✅ Training completed successfully!")
    print("="*70)
    print(f"\nFinal step: {trainer.state.global_step}")
    print(f"Output dir: {training_config.output_dir}")
    
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
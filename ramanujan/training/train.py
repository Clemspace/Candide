#!/usr/bin/env python3
"""
Main training script for Ramanujan Transformer.

Usage:
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --config configs/base.yaml --wandb
    python scripts/train.py --config configs/base.yaml --resume checkpoints/latest.pt
"""

import argparse
import os
import sys
import torch
import random
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ramanujan.foundation import RamanujanFoundation
from ramanujan.architecture import create_model
from ramanujan.training import (
    Trainer,
    create_optimizer,
    create_scheduler,
    create_loss
)
from ramanujan.data import WikiTextLoader, get_tokenizer
from ramanujan.utils import (
    load_config,
    setup_logging,
    ExperimentLogger,
    compute_sparsity_stats,
    count_parameters
)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Ramanujan Transformer')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). Overrides config.'
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only run evaluation, no training'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    print("="*70)
    print("Loading configuration...")
    print("="*70)
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.wandb:
        config.training.use_wandb = True
    if args.resume:
        config.training.resume_from_checkpoint = args.resume
    if args.device:
        config.training.device = args.device

    
    
    # Set random seed
    set_seed(config.training.seed)
    print(f"Random seed set to: {config.training.seed}")
    
    # Setup logging
    logger = setup_logging(
        name=config.name,
        log_dir=config.training.log_dir,
        level='INFO'
    )
    
    logger.info("="*70)
    logger.info(f"Experiment: {config.name}")
    logger.info(f"Description: {config.description}")
    logger.info("="*70)
    
    # Print configuration
    logger.info("\nModel Configuration:")
    logger.info(f"  Dimension: {config.model.dim}")
    logger.info(f"  Layers: {config.model.num_layers}")
    logger.info(f"  Heads: {config.model.num_heads} (KV: {config.model.num_kv_heads})")
    logger.info(f"  Vocab size: {config.model.vocab_size}")
    logger.info(f"  Max seq len: {config.model.max_seq_len}")
    
    logger.info("\nSparsity Configuration:")
    logger.info(f"  Attention sparsity: {config.sparsity.attention_sparsity:.1%}")
    logger.info(f"  FFN sparsity: {config.sparsity.ffn_sparsity:.1%}")
    logger.info(f"  Sliding window: {config.sparsity.use_sliding_window}")
    if config.sparsity.use_sliding_window:
        logger.info(f"  Window size: {config.sparsity.window_size}")
    
    logger.info("\nTraining Configuration:")
    logger.info(f"  Max steps: {config.training.max_steps}")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {config.training.batch_size * config.training.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  Optimizer: {config.training.optimizer_type}")
    logger.info(f"  Scheduler: {config.training.scheduler_type}")
    logger.info(f"  Loss type: {config.training.loss_type}")
    logger.info(f"  Device: {config.training.device}")
    logger.info(f"  Mixed precision: {config.training.mixed_precision}")
    
    # Create Ramanujan foundation
    logger.info("\n" + "="*70)
    logger.info("Creating Ramanujan Foundation...")
    logger.info("="*70)
    foundation = RamanujanFoundation(max_prime=config.sparsity.max_prime)
    logger.info(f"Foundation created with max_prime={config.sparsity.max_prime}")
    
    # Create model
    logger.info("\n" + "="*70)
    logger.info("Creating model...")
    logger.info("="*70)
    
    # Add foundation to model config
    from ramanujan.architecture.model import ModelConfig
    model_config = ModelConfig(**config.model.__dict__)
    
    # Import and pass foundation to model
    from ramanujan.architecture import StandardModel, EnhancedPretrainingModel
    
    if config.model.model_type == 'enhanced':
        model = EnhancedPretrainingModel(
            vocab_size=model_config.vocab_size,
            dim=model_config.dim,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
            num_kv_heads=model_config.num_kv_heads,
            hidden_dim=model_config.hidden_dim,
            dropout=model_config.dropout,
            max_seq_len=model_config.max_seq_len,
            pad_token_id=model_config.pad_token_id,
            tie_embeddings=model_config.tie_embeddings,
            foundation=foundation,
            attention_sparsity=config.sparsity.attention_sparsity,
            ffn_sparsity=config.sparsity.ffn_sparsity,
            use_sliding_window=config.sparsity.use_sliding_window,
            window_size=config.sparsity.window_size,
            num_global_tokens=config.sparsity.num_global_tokens
        )
    else:
        model = StandardModel(
            vocab_size=model_config.vocab_size,
            dim=model_config.dim,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
            num_kv_heads=model_config.num_kv_heads,
            hidden_dim=model_config.hidden_dim,
            dropout=model_config.dropout,
            max_seq_len=model_config.max_seq_len,
            pad_token_id=model_config.pad_token_id,
            tie_embeddings=model_config.tie_embeddings,
            foundation=foundation if config.sparsity.attention_sparsity > 0 else None,
            attention_sparsity=config.sparsity.attention_sparsity,
            ffn_sparsity=config.sparsity.ffn_sparsity
        )
    
    # Print model statistics
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    logger.info(f"Model created: {model.__class__.__name__}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Enable gradient checkpointing if requested
    if config.training.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled")
    
    # Compute sparsity statistics
    if config.sparsity.attention_sparsity > 0 or config.sparsity.ffn_sparsity > 0:
        sparsity_stats = compute_sparsity_stats(model)
        logger.info(f"\nSparsity Statistics:")
        logger.info(f"  Overall: {sparsity_stats['overall']:.2%}")
        logger.info(f"  Attention: {sparsity_stats['attention']:.2%}")
        logger.info(f"  FFN: {sparsity_stats['ffn']:.2%}")
        logger.info(f"  Active parameters: {sparsity_stats['sparse_params']:,}")
        logger.info(f"  Savings: {sparsity_stats['dense_params']:,} parameters")
    

    # Load data
    logger.info("\n" + "="*70)
    logger.info("Loading data...")
    logger.info("="*70)
    
    tokenizer = get_tokenizer(vocab_size=config.model.vocab_size)
    
    data_loader = WikiTextLoader(
        dataset_name='wikitext-2-raw-v1',
        vocab_size=config.model.vocab_size,
        sequence_length=config.model.max_seq_len,
        tokenizer=tokenizer
    )
    
    train_loader, eval_loader = data_loader.get_dataloaders(
        batch_size=config.training.batch_size,
        num_workers=4,
        shuffle_train=True
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Eval batches: {len(eval_loader)}")
    
    # Create optimizer
    logger.info("\n" + "="*70)
    logger.info("Creating optimizer...")
    logger.info("="*70)
    
    optimizer = create_optimizer(
        model,
        optimizer_type=config.training.optimizer_type,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    
    # Create scheduler
    logger.info("\n" + "="*70)
    logger.info("Creating scheduler...")
    logger.info("="*70)
    
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=config.training.scheduler_type,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.max_steps,
        min_lr=config.training.min_lr
    )
    logger.info(f"Scheduler: {scheduler.__class__.__name__}")
    
    # Create loss function
    loss_fn = create_loss(
        config.training.loss_type,
        vocab_size=config.model.vocab_size,
        label_smoothing=config.training.label_smoothing,
        alpha=config.training.semantic_entropy_alpha
    )
    loss_fn = loss_fn.to(torch.device(config.training.device))

    logger.info(f"Loss function: {loss_fn.__class__.__name__}")
    
    # Create trainer
    logger.info("\n" + "="*70)
    logger.info("Creating trainer...")
    logger.info("="*70)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        config=config.training
    )
    
    logger.info("Trainer created successfully!")
    
    # Evaluation only mode
    if args.eval_only:
        logger.info("\n" + "="*70)
        logger.info("Running evaluation...")
        logger.info("="*70)
        
        metrics = trainer.evaluate()
        
        logger.info("\nEvaluation Results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return
    
    # Start training
    logger.info("\n" + "="*70)
    logger.info("Starting training...")
    logger.info("="*70)
    
    try:
        trainer.train()
        
        logger.info("\n" + "="*70)
        logger.info("Training completed successfully!")
        logger.info("="*70)
        
        # Final evaluation
        logger.info("\nRunning final evaluation...")
        final_metrics = trainer.evaluate()
        
        logger.info("\nFinal Metrics:")
        for key, value in final_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        logger.info(f"\nBest eval loss: {trainer.best_eval_loss:.4f}")
        logger.info(f"Checkpoints saved to: {config.training.checkpoint_dir}")
        logger.info(f"Logs saved to: {config.training.log_dir}")
        
    except KeyboardInterrupt:
        logger.warning("\n" + "="*70)
        logger.warning("Training interrupted by user!")
        logger.warning("="*70)
        
        # Save checkpoint
        trainer.save_checkpoint('interrupted.pt')
        logger.info("Checkpoint saved: interrupted.pt")
        
    except Exception as e:
        logger.error("\n" + "="*70)
        logger.error(f"Training failed with error: {e}")
        logger.error("="*70)
        
        import traceback
        traceback.print_exc()
        
        # Save checkpoint for debugging
        trainer.save_checkpoint('error.pt')
        logger.info("Checkpoint saved: error.pt")
        
        raise


if __name__ == "__main__":
    main()
"""
Comprehensive testing suite for Ramanujan Transformer training system.

Tests:
1. Configuration loading and validation
2. Model creation
3. Optimizer creation
4. Scheduler creation
5. Loss function creation
6. Data loading
7. Trainer initialization
8. Full training loop (small test)
9. Checkpointing and resuming
10. Callbacks
"""

import torch
import yaml
import tempfile
from pathlib import Path
from typing import Dict, Any
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

def create_test_config() -> Dict[str, Any]:
    """Create a minimal test configuration."""
    return {
        'model': {
            'vocab_size': 1000,  # Small for testing
            'd_model': 128,
            'n_layers': 2,
            'n_heads': 4,
            'd_ff': 512,
            'max_seq_len': 128,
            'dropout': 0.1,
            'norm_first': True,
            'norm_type': 'rms',
            'attention_type': 'standard',
            'ffn_type': 'standard',
            'use_rope': False,
            'bias': False,
            'tie_word_embeddings': True,
        },
        'training': {
            'max_steps': 10,  # Very short for testing
            'batch_size': 2,
            'gradient_accumulation_steps': 1,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'optimizer': 'adamw',
            'lr_scheduler': 'constant',
            'loss': 'cross_entropy',
            'save_every': 5,
            'output_dir': None,  # Will be set to temp dir
        },
        'data': {
            'dataset': 'synthetic',  # Use synthetic data for testing
            'sequence_length': 128,
            'num_workers': 0,
        },
        'logging': {
            'log_every': 2,
            'eval_every': 5,
            'use_wandb': False,
        },
        'hardware': {
            'device': 'cpu',  # Use CPU for testing
            'seed': 42,
        }
    }


# ============================================================================
# SYNTHETIC DATA FOR TESTING
# ============================================================================

class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic dataset for testing."""
    
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random sequence
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {'input_ids': input_ids}


def create_synthetic_dataloaders(config: Dict[str, Any]):
    """Create synthetic dataloaders for testing."""
    vocab_size = config['model']['vocab_size']
    seq_len = config['data']['sequence_length']
    batch_size = config['training']['batch_size']
    
    train_dataset = SyntheticDataset(vocab_size, seq_len, num_samples=20)
    val_dataset = SyntheticDataset(vocab_size, seq_len, num_samples=10)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    return train_loader, val_loader


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_config_validation():
    """Test configuration validation."""
    print("\n" + "="*70)
    print("TEST 1: Configuration Validation")
    print("="*70)
    
    from ramanujan.training.trainer.utils import validate_config
    
    # Test valid config
    config = create_test_config()
    errors = validate_config(config)
    
    if errors:
        print(f"❌ Valid config failed validation: {errors}")
        return False
    
    print("✅ Valid config passed validation")
    
    # Test invalid config (missing required field)
    invalid_config = create_test_config()
    del invalid_config['model']['vocab_size']
    errors = validate_config(invalid_config)
    
    if not errors:
        print("❌ Invalid config passed validation (should have failed)")
        return False
    
    print(f"✅ Invalid config correctly rejected: {errors[0]}")
    
    return True


def test_model_creation():
    """Test model creation."""
    print("\n" + "="*70)
    print("TEST 2: Model Creation")
    print("="*70)
    
    from ramanujan.models.architectures.config import TransformerConfig
    from ramanujan.models.architectures.transformer import RamanujanTransformer
    
    config = create_test_config()
    
    try:
        # Create model config
        model_config = TransformerConfig.from_dict(config['model'])
        
        # Create model
        model = RamanujanTransformer(model_config)
        
        # Check parameter count
        n_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created with {n_params:,} parameters")
        
        # Test forward pass
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, config['model']['vocab_size'], (batch_size, seq_len))
        
        model.eval()
        with torch.no_grad():
            logits = model(input_ids)
        
        expected_shape = (batch_size, seq_len, config['model']['vocab_size'])
        if logits.shape != expected_shape:
            print(f"❌ Output shape mismatch: {logits.shape} vs {expected_shape}")
            return False
        
        print(f"✅ Forward pass successful: {logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimizer_creation():
    """Test optimizer creation."""
    print("\n" + "="*70)
    print("TEST 3: Optimizer Creation")
    print("="*70)
    
    from ramanujan.models.architectures.config import TransformerConfig
    from ramanujan.models.architectures.transformer import RamanujanTransformer
    from ramanujan.training.optimizers import create_optimizer_from_config
    
    config = create_test_config()
    
    try:
        # Create model
        model_config = TransformerConfig.from_dict(config['model'])
        model = RamanujanTransformer(model_config)
        
        # Create optimizer
        optimizer_config = {
            'name': config['training']['optimizer'],
            'lr': config['training']['learning_rate'],
            'weight_decay': config['training']['weight_decay'],
        }
        
        optimizer = create_optimizer_from_config(optimizer_config, model.parameters())
        
        print(f"✅ Optimizer created: {optimizer_config['name']}")
        
        # Test optimizer step
        optimizer.zero_grad()
        
        # Dummy forward and backward
        input_ids = torch.randint(0, config['model']['vocab_size'], (2, 16))
        logits = model(input_ids)
        loss = logits.mean()
        loss.backward()
        
        optimizer.step()
        
        print("✅ Optimizer step successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimizer creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scheduler_creation():
    """Test scheduler creation."""
    print("\n" + "="*70)
    print("TEST 4: Scheduler Creation")
    print("="*70)
    
    from ramanujan.models.architectures.config import TransformerConfig
    from ramanujan.models.architectures.transformer import RamanujanTransformer
    from ramanujan.training.optimizers import create_optimizer_from_config
    from ramanujan.training.schedulers import create_scheduler_from_config
    
    config = create_test_config()
    
    try:
        # Create model and optimizer
        model_config = TransformerConfig.from_dict(config['model'])
        model = RamanujanTransformer(model_config)
        
        optimizer_config = {
            'name': 'adamw',
            'lr': 0.001,
        }
        optimizer = create_optimizer_from_config(optimizer_config, model.parameters())
        
        # Create scheduler
        scheduler_config = {
            'name': config['training']['lr_scheduler'],
        }
        
        scheduler = create_scheduler_from_config(scheduler_config, optimizer)
        
        print(f"✅ Scheduler created: {scheduler_config['name']}")
        
        # Test scheduler step
        initial_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        after_lr = optimizer.param_groups[0]['lr']
        
        print(f"   LR: {initial_lr:.6f} → {after_lr:.6f}")
        print("✅ Scheduler step successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Scheduler creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_creation():
    """Test loss function creation."""
    print("\n" + "="*70)
    print("TEST 5: Loss Function Creation")
    print("="*70)
    
    from ramanujan.training.losses import create_loss_from_config
    
    config = create_test_config()
    
    try:
        loss_config = {
            'name': config['training']['loss'],
            'vocab_size': config['model']['vocab_size'],
        }
        
        loss_fn = create_loss_from_config(loss_config)
        
        print(f"✅ Loss function created: {loss_config['name']}")
        
        # Test loss computation
        batch_size = 2
        seq_len = 16
        vocab_size = config['model']['vocab_size']
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        if hasattr(loss_fn, 'compute'):
            loss, metrics = loss_fn.compute(logits, targets)
        else:
            loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
        
        print(f"✅ Loss computation successful: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading."""
    print("\n" + "="*70)
    print("TEST 6: Data Loading")
    print("="*70)
    
    config = create_test_config()
    
    try:
        # Create synthetic dataloaders
        train_loader, val_loader = create_synthetic_dataloaders(config)
        
        print(f"✅ DataLoaders created")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        
        # Test batch retrieval
        batch = next(iter(train_loader))
        
        if 'input_ids' not in batch:
            print("❌ Batch missing 'input_ids'")
            return False
        
        input_ids = batch['input_ids']
        expected_shape = (config['training']['batch_size'], config['data']['sequence_length'])
        
        if input_ids.shape != expected_shape:
            print(f"❌ Batch shape mismatch: {input_ids.shape} vs {expected_shape}")
            return False
        
        print(f"✅ Batch shape correct: {input_ids.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_initialization():
    """Test trainer initialization."""
    print("\n" + "="*70)
    print("TEST 7: Trainer Initialization")
    print("="*70)
    
    from ramanujan.models.architectures.config import TransformerConfig
    from ramanujan.models.architectures.transformer import RamanujanTransformer
    from ramanujan.training.optimizers import create_optimizer_from_config
    from ramanujan.training.schedulers import create_scheduler_from_config
    from ramanujan.training.losses import create_loss_from_config
    from ramanujan.training.trainer.base import TrainingConfig
    from ramanujan.training.trainer.trainer import Trainer
    
    config = create_test_config()
    
    try:
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            config['training']['output_dir'] = tmp_dir
            
            # Create components
            model_config = TransformerConfig.from_dict(config['model'])
            model = RamanujanTransformer(model_config)
            
            optimizer = create_optimizer_from_config(
                {'name': 'adamw', 'lr': 0.001},
                model.parameters()
            )
            
            scheduler = create_scheduler_from_config(
                {'name': 'constant'},
                optimizer
            )
            
            loss_fn = create_loss_from_config({
                'name': 'cross_entropy',
                'vocab_size': config['model']['vocab_size']
            })
            
            train_loader, val_loader = create_synthetic_dataloaders(config)
            
            # Create training config
            training_config = TrainingConfig(
                output_dir=tmp_dir,
                max_steps=10,
                batch_size=2,
                learning_rate=0.001,
                device='cpu',
                log_every=2,
                eval_every=5,
                save_every=5,
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                config=training_config,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                scheduler=scheduler,
            )
            
            print("✅ Trainer initialized successfully")
            
            return True
            
    except Exception as e:
        print(f"❌ Trainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop():
    """Test full training loop."""
    print("\n" + "="*70)
    print("TEST 8: Training Loop")
    print("="*70)
    
    from ramanujan.models.architectures.config import TransformerConfig
    from ramanujan.models.architectures.transformer import RamanujanTransformer
    from ramanujan.training.optimizers import create_optimizer_from_config
    from ramanujan.training.schedulers import create_scheduler_from_config
    from ramanujan.training.losses import create_loss_from_config
    from ramanujan.training.trainer.base import TrainingConfig
    from ramanujan.training.trainer.trainer import Trainer
    from ramanujan.training.trainer.callbacks import ProgressCallback
    
    config = create_test_config()
    
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config['training']['output_dir'] = tmp_dir
            
            # Create all components
            model_config = TransformerConfig.from_dict(config['model'])
            model = RamanujanTransformer(model_config)
            
            optimizer = create_optimizer_from_config(
                {'name': 'adamw', 'lr': 0.001, 'weight_decay': 0.01},
                model.parameters()
            )
            
            loss_fn = create_loss_from_config({
                'name': 'cross_entropy',
                'vocab_size': config['model']['vocab_size']
            })
            
            train_loader, val_loader = create_synthetic_dataloaders(config)
            
            training_config = TrainingConfig(
                output_dir=tmp_dir,
                max_steps=10,
                batch_size=2,
                learning_rate=0.001,
                device='cpu',
                log_every=2,
                eval_every=5,
                save_every=5,
                use_wandb=False,
            )
            
            # Create trainer with minimal callbacks
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                config=training_config,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                callbacks=[ProgressCallback()],
            )
            
            # Train
            print("\n🏋️  Starting training...")
            trainer.train()
            
            # Check that training completed
            if trainer.state.global_step != 10:
                print(f"❌ Training didn't complete: {trainer.state.global_step}/10 steps")
                return False
            
            print(f"✅ Training completed: {trainer.state.global_step} steps")
            
            return True
            
    except Exception as e:
        print(f"❌ Training loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpointing():
    """Test checkpointing and resuming."""
    print("\n" + "="*70)
    print("TEST 9: Checkpointing and Resuming")
    print("="*70)
    
    from ramanujan.models.architectures.config import TransformerConfig
    from ramanujan.models.architectures.transformer import RamanujanTransformer
    from ramanujan.training.optimizers import create_optimizer_from_config
    from ramanujan.training.losses import create_loss_from_config
    from ramanujan.training.trainer.base import TrainingConfig
    from ramanujan.training.trainer.trainer import Trainer
    
    config = create_test_config()
    
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / 'test_checkpoint.pt'
            
            # Create and train for 5 steps
            model_config = TransformerConfig.from_dict(config['model'])
            model = RamanujanTransformer(model_config)
            
            optimizer = create_optimizer_from_config(
                {'name': 'adamw', 'lr': 0.001},
                model.parameters()
            )
            
            loss_fn = create_loss_from_config({
                'name': 'cross_entropy',
                'vocab_size': config['model']['vocab_size']
            })
            
            train_loader, _ = create_synthetic_dataloaders(config)
            
            training_config = TrainingConfig(
                output_dir=tmp_dir,
                max_steps=5,
                batch_size=2,
                learning_rate=0.001,
                device='cpu',
                log_every=100,
                save_every=100,
            )
            
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                config=training_config,
                train_dataloader=train_loader,
            )
            
            trainer.train()
            
            # Save checkpoint
            trainer.save_checkpoint(str(checkpoint_path))
            
            if not checkpoint_path.exists():
                print("❌ Checkpoint not saved")
                return False
            
            print(f"✅ Checkpoint saved: {checkpoint_path}")
            
            # Create new trainer and load checkpoint
            model2 = RamanujanTransformer(model_config)
            optimizer2 = create_optimizer_from_config(
                {'name': 'adamw', 'lr': 0.001},
                model2.parameters()
            )
            loss_fn2 = create_loss_from_config({
                'name': 'cross_entropy',
                'vocab_size': config['model']['vocab_size']
            })
            
            training_config2 = TrainingConfig(
                output_dir=tmp_dir,
                max_steps=10,
                batch_size=2,
                device='cpu',
                resume_from=str(checkpoint_path),
            )
            
            trainer2 = Trainer(
                model=model2,
                optimizer=optimizer2,
                loss_fn=loss_fn2,
                config=training_config2,
                train_dataloader=train_loader,
            )
            
            # Check that state was restored
            if trainer2.state.global_step != 5:
                print(f"❌ State not restored: step {trainer2.state.global_step} != 5")
                return False
            
            print(f"✅ Checkpoint loaded successfully")
            print(f"   Resumed from step {trainer2.state.global_step}")
            
            return True
            
    except Exception as e:
        print(f"❌ Checkpointing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_callbacks():
    """Test callback system."""
    print("\n" + "="*70)
    print("TEST 10: Callbacks")
    print("="*70)
    
    from ramanujan.training.trainer.callbacks import (
        Callback,
        MetricsCallback,
        ProgressCallback,
    )
    
    try:
        # Test callback creation
        callbacks = [
            ProgressCallback(),
            MetricsCallback(window_size=10),
        ]
        
        print(f"✅ Created {len(callbacks)} callbacks")
        
        # Test callback methods exist
        callback = callbacks[0]
        required_methods = [
            'on_train_begin',
            'on_train_end',
            'on_step_begin',
            'on_step_end',
            'on_validation_begin',
            'on_validation_end',
        ]
        
        for method in required_methods:
            if not hasattr(callback, method):
                print(f"❌ Callback missing method: {method}")
                return False
        
        print("✅ All callback methods present")
        
        return True
        
    except Exception as e:
        print(f"❌ Callback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("🧪 RAMANUJAN TRANSFORMER - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Configuration Validation", test_config_validation),
        ("Model Creation", test_model_creation),
        ("Optimizer Creation", test_optimizer_creation),
        ("Scheduler Creation", test_scheduler_creation),
        ("Loss Function Creation", test_loss_creation),
        ("Data Loading", test_data_loading),
        ("Trainer Initialization", test_trainer_initialization),
        ("Training Loop", test_training_loop),
        ("Checkpointing", test_checkpointing),
        ("Callbacks", test_callbacks),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    # Overall result
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    
    print("\n" + "="*70)
    if passed_count == total:
        print(f"🎉 ALL TESTS PASSED ({passed_count}/{total})")
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed_count}/{total} passed)")
    print("="*70 + "\n")
    
    return passed_count == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
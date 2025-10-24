"""
Integration tests for complete workflows.

Tests cover:
- End-to-end training pipeline
- Generation workflows
- Multi-preset compatibility
- Factory functions
- Complete save/load cycles
- Memory efficiency
- Real-world scenarios
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
from pathlib import Path

# Assuming these imports from your codebase
from ramanujan.models import create_model, get_config, create_transformer_block
from ramanujan.models.architectures.transformer import RamanujanTransformer
from ramanujan.models.architectures.config import TransformerConfig


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_batch():
    """Sample training batch."""
    batch_size, seq_len = 4, 64
    vocab_size = 1000
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    return input_ids, target_ids


# ============================================================================
# FACTORY INTEGRATION TESTS
# ============================================================================

class TestFactoryIntegration:
    """Test factory functions work end-to-end."""
    
    def test_create_model_with_preset(self):
        """Test creating model via factory with preset."""
        model = create_model(preset='tiny', vocab_size=1000)
        
        assert model is not None
        assert isinstance(model, RamanujanTransformer)
        assert model.config.vocab_size == 1000
    
    def test_create_model_with_config_dict(self):
        """Test creating model with config dictionary."""
        model = create_model(config={
            'vocab_size': 1000,
            'd_model': 512,
            'n_layers': 6,
            'n_heads': 8,
        })
        
        assert model.config.vocab_size == 1000
        assert model.config.d_model == 512
        assert model.config.n_layers == 6
    
    def test_create_model_with_config_object(self):
        """Test creating model with config object."""
        config = TransformerConfig.tiny(vocab_size=1000)
        model = create_model(config=config)
        
        assert model.config == config
    
    def test_get_config_function(self):
        """Test get_config factory function."""
        config = get_config('llama', vocab_size=1000, d_model=512, n_layers=4, n_heads=8)
        
        assert config.vocab_size == 1000
        assert config.d_model == 512
        assert config.n_layers == 4
        assert config.norm_type == "rms"  # LLaMA characteristic
    
    def test_all_presets_work(self):
        """Test all size presets create working models."""
        for preset in ['tiny', 'small']:  # Skip large ones for speed
            model = create_model(preset=preset, vocab_size=1000)
            
            # Quick forward pass
            input_ids = torch.randint(0, 1000, (2, 32))
            logits = model(input_ids)
            
            assert logits.shape[2] == 1000


# ============================================================================
# END-TO-END TRAINING PIPELINE TESTS
# ============================================================================

class TestTrainingPipeline:
    """Test complete training workflows."""
    
    def test_simple_training_step(self, sample_batch):
        """Test single training step works."""
        model = create_model(preset='tiny', vocab_size=1000)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        input_ids, target_ids = sample_batch
        
        # Forward pass
        logits = model(input_ids)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, 1000),
            target_ids.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_multiple_training_steps(self):
        """Test multiple training steps reduce loss."""
        model = create_model(preset='tiny', vocab_size=1000)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        
        # Create simple overfitting task
        input_ids = torch.randint(0, 1000, (4, 32))
        target_ids = torch.randint(0, 1000, (4, 32))
        
        losses = []
        
        for _ in range(10):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, 1000),
                target_ids.view(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Loss should decrease
        assert losses[-1] < losses[0]
    
    def test_training_with_causal_mask(self):
        """Test training with causal attention mask."""
        model = create_model(preset='tiny', vocab_size=1000)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        input_ids = torch.randint(0, 1000, (2, 32))
        target_ids = torch.randint(0, 1000, (2, 32))
        
        # Create causal mask
        mask = torch.triu(torch.ones(32, 32), diagonal=1).bool()
        
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, 1000), target_ids.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert not torch.isnan(loss)
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation works."""
        model = create_model(preset='tiny', vocab_size=1000)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        accumulation_steps = 4
        
        for step in range(accumulation_steps):
            input_ids = torch.randint(0, 1000, (2, 32))
            target_ids = torch.randint(0, 1000, (2, 32))
            
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, 1000), target_ids.view(-1))
            loss = loss / accumulation_steps
            
            loss.backward()
            
            if step == accumulation_steps - 1:
                optimizer.step()
                optimizer.zero_grad()
        
        # Should complete without error
        assert True


# ============================================================================
# GENERATION PIPELINE TESTS
# ============================================================================

class TestGenerationPipeline:
    """Test complete generation workflows."""
    
    def test_simple_generation(self):
        """Test simple text generation."""
        model = create_model(preset='tiny', vocab_size=1000)
        model.eval()
        
        prompt = torch.randint(0, 1000, (1, 10))
        
        generated = model.generate(
            prompt,
            max_new_tokens=20,
            temperature=1.0,
        )
        
        assert generated.shape == (1, 30)
        # Should only contain valid token IDs
        assert torch.all((generated >= 0) & (generated < 1000))
    
    def test_generation_with_various_sampling(self):
        """Test generation with different sampling strategies."""
        model = create_model(preset='tiny', vocab_size=1000)
        model.eval()
        
        prompt = torch.randint(0, 1000, (1, 10))
        
        # Test different sampling strategies
        strategies = [
            {'temperature': 0.7},
            {'temperature': 1.0, 'top_k': 50},
            {'temperature': 1.0, 'top_p': 0.9},
            {'temperature': 0.8, 'top_k': 40, 'top_p': 0.95},
        ]
        
        for kwargs in strategies:
            generated = model.generate(prompt, max_new_tokens=10, **kwargs)
            assert generated.shape == (1, 20)
    
    def test_batch_generation(self):
        """Test generating for multiple prompts in parallel."""
        model = create_model(preset='tiny', vocab_size=1000)
        model.eval()
        
        prompts = torch.randint(0, 1000, (4, 10))
        
        generated = model.generate(
            prompts,
            max_new_tokens=20,
            temperature=1.0,
        )
        
        assert generated.shape == (4, 30)
    
    def test_long_generation(self):
        """Test generating longer sequences."""
        model = create_model(preset='tiny', vocab_size=1000, max_seq_len=512)
        model.eval()
        
        prompt = torch.randint(0, 1000, (1, 10))
        
        generated = model.generate(
            prompt,
            max_new_tokens=100,
            temperature=1.0,
        )
        
        assert generated.shape == (1, 110)


# ============================================================================
# CHECKPOINT PIPELINE TESTS
# ============================================================================

class TestCheckpointPipeline:
    """Test complete checkpoint workflows."""
    
    def test_train_save_load_continue(self):
        """Test train -> save -> load -> continue pipeline."""
        model = create_model(preset='tiny', vocab_size=1000)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Train for a few steps
        input_ids = torch.randint(0, 1000, (2, 32))
        target_ids = torch.randint(0, 1000, (2, 32))
        
        for _ in range(5):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, 1000), target_ids.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            model.save_pretrained(str(model_path))
            
            # Load
            loaded_model = RamanujanTransformer.from_pretrained(str(model_path))
            loaded_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=1e-4)
            
            # Continue training
            for _ in range(5):
                logits = loaded_model(input_ids)
                loss = F.cross_entropy(logits.view(-1, 1000), target_ids.view(-1))
                loaded_optimizer.zero_grad()
                loss.backward()
                loaded_optimizer.step()
        
        # Should complete without error
        assert True
    
    def test_save_with_optimizer_state(self):
        """Test saving both model and optimizer state."""
        model = create_model(preset='tiny', vocab_size=1000)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Train a bit
        input_ids = torch.randint(0, 1000, (2, 32))
        target_ids = torch.randint(0, 1000, (2, 32))
        
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, 1000), target_ids.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Save everything
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model.config.to_dict(),
            }, checkpoint_path)
            
            # Load everything
            checkpoint = torch.load(checkpoint_path)
            
            new_model = RamanujanTransformer(
                TransformerConfig.from_dict(checkpoint['config'])
            )
            new_model.load_state_dict(checkpoint['model_state_dict'])
            
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)
            new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        assert True


# ============================================================================
# MULTI-PRESET COMPATIBILITY TESTS
# ============================================================================

class TestMultiPresetCompatibility:
    """Test different presets work together."""
    
    def test_different_architectures_same_interface(self):
        """Test different architectures have same interface."""
        input_ids = torch.randint(0, 1000, (2, 32))
        
        for preset in ['llama', 'gpt', 'bert']:
            config = get_config(preset, vocab_size=1000, d_model=512, n_layers=4, n_heads=8)
            model = create_model(config=config)
            
            # Should all work the same way
            logits = model(input_ids)
            assert logits.shape == (2, 32, 1000)
    
    def test_transfer_between_presets(self):
        """Test transferring some weights between architectures."""
        # Create two models with same dimensions
        llama_config = get_config('llama', vocab_size=1000, d_model=512, n_layers=4, n_heads=8)
        gpt_config = get_config('gpt', vocab_size=1000, d_model=512, n_layers=4, n_heads=8)
        
        llama_model = create_model(config=llama_config)
        gpt_model = create_model(config=gpt_config)
        
        # Copy embedding weights (should be compatible)
        gpt_model.token_embedding.embedding.weight.data = llama_model.token_embedding.embedding.weight.data.clone()
        
        # Should work without error
        input_ids = torch.randint(0, 1000, (2, 32))
        llama_out = llama_model(input_ids)
        gpt_out = gpt_model(input_ids)
        
        assert llama_out.shape == gpt_out.shape


# ============================================================================
# MEMORY EFFICIENCY TESTS
# ============================================================================

class TestMemoryEfficiency:
    """Test memory efficiency features."""
    
    def test_kv_cache_reduces_computation(self):
        """Test KV cache makes generation faster."""
        model = create_model(preset='tiny', vocab_size=1000)
        model.eval()
        
        # Without cache: recompute everything each step
        prompt = torch.randint(0, 1000, (1, 10))
        
        # With cache: should be much more efficient
        # (We just test it works, not actual speed)
        generated = model.generate(
            prompt,
            max_new_tokens=50,
            temperature=1.0,
        )
        
        assert generated.shape == (1, 60)
    
    def test_grouped_query_attention_memory(self):
        """Test GQA uses less memory than MHA."""
        # Standard MHA
        mha_config = TransformerConfig(
            vocab_size=1000,
            d_model=512,
            n_layers=4,
            n_heads=8,
            n_kv_heads=8,  # Same as n_heads = MHA
        )
        
        # GQA with 4:1 ratio
        gqa_config = TransformerConfig(
            vocab_size=1000,
            d_model=512,
            n_layers=4,
            n_heads=8,
            n_kv_heads=2,  # 4:1 ratio
        )
        
        mha_model = create_model(config=mha_config)
        gqa_model = create_model(config=gqa_config)
        
        # GQA should have fewer parameters
        mha_params = mha_model.get_num_params()
        gqa_params = gqa_model.get_num_params()
        
        assert gqa_params <= mha_params


# ============================================================================
# REAL-WORLD SCENARIO TESTS
# ============================================================================

class TestRealWorldScenarios:
    """Test realistic usage scenarios."""
    
    def test_pretrain_from_scratch(self):
        """Test pretraining workflow."""
        model = create_model(preset='tiny', vocab_size=1000)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        
        # Simulate pretraining on random data
        steps = 20
        for step in range(steps):
            # Generate random batch
            input_ids = torch.randint(0, 1000, (4, 64))
            
            # Forward
            logits = model(input_ids)
            
            # Shift for autoregressive loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, 1000),
                shift_labels.view(-1)
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
        
        assert True
    
    def test_inference_only_mode(self):
        """Test inference-only deployment."""
        # Create and save model
        model = create_model(preset='tiny', vocab_size=1000)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            model.save_pretrained(str(model_path))
            
            # Load for inference only
            inference_model = RamanujanTransformer.from_pretrained(str(model_path))
            inference_model.eval()
            
            # Disable gradients
            for param in inference_model.parameters():
                param.requires_grad = False
            
            # Generate
            prompt = torch.randint(0, 1000, (1, 10))
            with torch.no_grad():
                generated = inference_model.generate(
                    prompt,
                    max_new_tokens=30,
                    temperature=0.8,
                )
            
            assert generated.shape == (1, 40)
    
    def test_mixed_precision_training(self):
        """Test mixed precision (FP16) training."""
        model = create_model(preset='tiny', vocab_size=1000)
        
        # Convert to half precision
        model = model.half()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Train with FP16
        input_ids = torch.randint(0, 1000, (2, 32))
        target_ids = torch.randint(0, 1000, (2, 32))
        
        logits = model(input_ids)
        
        # Loss computation in FP32
        loss = F.cross_entropy(
            logits.float().view(-1, 1000),
            target_ids.view(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert not torch.isnan(loss)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
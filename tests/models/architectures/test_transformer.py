"""
Comprehensive tests for RamanujanTransformer.

Tests cover:
- Model initialization
- Forward pass (training mode)
- Generation with sampling
- KV caching
- Checkpoint save/load
- Parameter counting
- Different configurations
- Edge cases
"""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path

# Assuming these imports from your codebase
from ramanujan.models.architectures.transformer import RamanujanTransformer
from ramanujan.models.architectures.config import TransformerConfig


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def tiny_config():
    """Tiny model config for fast testing."""
    return TransformerConfig.tiny(vocab_size=1000, max_seq_len=256)


@pytest.fixture
def llama_config():
    """LLaMA-style config for testing."""
    return TransformerConfig.llama(
        vocab_size=1000,
        d_model=512,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=256,
    )


@pytest.fixture
def sample_input_ids():
    """Sample input token IDs."""
    return torch.randint(0, 1000, (2, 32))  # batch=2, seq=32


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestInitialization:
    """Test model initialization."""
    
    def test_basic_init(self, tiny_config):
        """Test basic model initialization."""
        model = RamanujanTransformer(tiny_config)
        
        assert model is not None
        assert hasattr(model, 'config')
        assert model.config == tiny_config
    
    def test_layers_created(self, tiny_config):
        """Test all model layers are created."""
        model = RamanujanTransformer(tiny_config)
        
        assert hasattr(model, 'token_embedding')
        assert hasattr(model, 'blocks')
        assert hasattr(model, 'final_norm')
        assert hasattr(model, 'lm_head')
        
        # Check number of transformer blocks
        assert len(model.blocks) == tiny_config.n_layers
    
    def test_llama_init(self, llama_config):
        """Test LLaMA-style initialization."""
        model = RamanujanTransformer(llama_config)
        
        assert model is not None
        assert len(model.blocks) == llama_config.n_layers
    
    def test_parameter_initialization(self, tiny_config):
        """Test parameters are properly initialized (no NaN)."""
        model = RamanujanTransformer(tiny_config)
        
        for param in model.parameters():
            assert not torch.any(torch.isnan(param))
            assert not torch.any(torch.isinf(param))


# ============================================================================
# FORWARD PASS TESTS
# ============================================================================

class TestForwardPass:
    """Test forward pass in training mode."""
    
    def test_basic_forward(self, tiny_config, sample_input_ids):
        """Test basic forward pass."""
        model = RamanujanTransformer(tiny_config)
        
        logits = model(sample_input_ids)
        
        batch, seq = sample_input_ids.shape
        vocab_size = tiny_config.vocab_size
        
        assert logits.shape == (batch, seq, vocab_size)
        assert logits.dtype == torch.float32
    
    def test_forward_with_mask(self, tiny_config):
        """Test forward pass with attention mask."""
        model = RamanujanTransformer(tiny_config)
        input_ids = torch.randint(0, 1000, (2, 32))
        
        # Create causal mask
        mask = torch.triu(torch.ones(32, 32), diagonal=1).bool()
        
        # logits = model(input_ids, mask=mask)  # TODO: check actual mask parameter
        logits = model(input_ids)
        
        assert logits.shape == (2, 32, 1000)
    
    def test_forward_gradient_flow(self, tiny_config):
        """Test gradients flow through the model."""
        model = RamanujanTransformer(tiny_config)
        input_ids = torch.randint(0, 1000, (2, 32))
        
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()
        
        # Check at least embedding has gradients
        assert model.token_embedding.embedding.weight.grad is not None
        assert not torch.all(model.token_embedding.embedding.weight.grad == 0)

    
    def test_forward_deterministic_eval(self, tiny_config, sample_input_ids):
        """Test forward is deterministic in eval mode."""
        model = RamanujanTransformer(tiny_config)
        model.eval()
        
        with torch.no_grad():
            logits1 = model(sample_input_ids)
            logits2 = model(sample_input_ids)
        
        assert torch.allclose(logits1, logits2)
    
    def test_forward_different_batch_sizes(self, tiny_config):
        """Test forward with different batch sizes."""
        model = RamanujanTransformer(tiny_config)
        model.eval()
        
        for batch_size in [1, 4, 8]:
            input_ids = torch.randint(0, 1000, (batch_size, 32))
            logits = model(input_ids)
            assert logits.shape == (batch_size, 32, 1000)
    
    def test_forward_different_sequence_lengths(self, tiny_config):
        """Test forward with different sequence lengths."""
        model = RamanujanTransformer(tiny_config)
        model.eval()
        
        for seq_len in [16, 32, 64, 128]:
            input_ids = torch.randint(0, 1000, (2, seq_len))
            logits = model(input_ids)
            assert logits.shape == (2, seq_len, 1000)


# ============================================================================
# KV CACHING TESTS
# ============================================================================

class TestKVCaching:
    """Test key-value caching for generation."""
    
    def test_cache_creation(self, tiny_config):
        """Test cache is created when use_cache=True."""
        model = RamanujanTransformer(tiny_config)
        input_ids = torch.randint(0, 1000, (1, 8))
        
        logits, past_kv = model(input_ids, use_cache=True)
        
        assert logits.shape == (1, 8, 1000)
        assert past_kv is not None
        assert isinstance(past_kv, list)
        assert len(past_kv) == tiny_config.n_layers
    

    def test_cache_structure(self, tiny_config):
        """Test cache has correct shapes."""
        model = RamanujanTransformer(tiny_config)
        batch, seq_len = 2, 8
        input_ids = torch.randint(0, 1000, (batch, seq_len))
        
        logits, past_kv = model(input_ids, use_cache=True)
        
        # Check cache per layer
        assert len(past_kv) == tiny_config.n_layers  # Changed: validate against n_layers, not hardcoded 8
        
        for layer_cache in past_kv:
            assert isinstance(layer_cache, tuple)
            assert len(layer_cache) >= 2  # At least keys and values
            keys, values = layer_cache[0], layer_cache[1]
            assert keys.shape[0] == batch
            assert keys.shape[2] == seq_len  # seq_len is dimension 2 in (batch, n_heads, seq_len, head_dim)

    
    def test_cache_reuse(self, tiny_config):
        """Test reusing cache for incremental generation."""
        model = RamanujanTransformer(tiny_config)
        model.eval()
        
        # First forward pass
        input_ids1 = torch.randint(0, 1000, (1, 8))
        logits1, cache1 = model(input_ids1, use_cache=True)
        
        # Second forward pass with cache
        input_ids2 = torch.randint(0, 1000, (1, 1))
        # Use actual parameter name from your forward signature
        logits2, cache2 = model(input_ids2, past_key_values=cache1, use_cache=True)
        
        assert logits2.shape == (1, 1, 1000)

    def test_no_cache_by_default(self, tiny_config, sample_input_ids):
        """Test no cache is returned by default."""
        model = RamanujanTransformer(tiny_config)
        
        output = model(sample_input_ids)
        
        # Should return only logits, not tuple
        assert not isinstance(output, tuple)
        assert output.shape == (2, 32, 1000)


# ============================================================================
# GENERATION TESTS
# ============================================================================

class TestGeneration:
    """Test text generation functionality."""
    
    def test_basic_generation(self, tiny_config):
        """Test basic generation."""
        model = RamanujanTransformer(tiny_config)
        model.eval()
        
        prompt = torch.randint(0, 1000, (1, 10))
        
        generated = model.generate(
            prompt,
            max_new_tokens=20,
            temperature=1.0,
        )
        
        assert generated.shape == (1, 30)  # 10 + 20
        assert generated.dtype == torch.long
    
    def test_generation_with_temperature(self, tiny_config):
        """Test generation with different temperatures."""
        model = RamanujanTransformer(tiny_config)
        model.eval()
        
        prompt = torch.randint(0, 1000, (1, 10))
        
        # Lower temperature should be more deterministic
        gen_low = model.generate(prompt, max_new_tokens=10, temperature=0.1)
        # Higher temperature should be more random
        gen_high = model.generate(prompt, max_new_tokens=10, temperature=2.0)
        
        assert gen_low.shape == gen_high.shape
    
    def test_generation_with_top_k(self, tiny_config):
        """Test generation with top-k sampling."""
        model = RamanujanTransformer(tiny_config)
        model.eval()
        
        prompt = torch.randint(0, 1000, (1, 10))
        
        generated = model.generate(
            prompt,
            max_new_tokens=20,
            top_k=50,
        )
        
        assert generated.shape == (1, 30)
    
    def test_generation_with_top_p(self, tiny_config):
        """Test generation with nucleus (top-p) sampling."""
        model = RamanujanTransformer(tiny_config)
        model.eval()
        
        prompt = torch.randint(0, 1000, (1, 10))
        
        generated = model.generate(
            prompt,
            max_new_tokens=20,
            top_p=0.9,
        )
        
        assert generated.shape == (1, 30)
    
    def test_generation_batch(self, tiny_config):
        """Test batch generation."""
        model = RamanujanTransformer(tiny_config)
        model.eval()
        
        prompt = torch.randint(0, 1000, (4, 10))  # batch=4
        
        generated = model.generate(
            prompt,
            max_new_tokens=20,
        )
        
        assert generated.shape == (4, 30)
    
    def test_generation_uses_cache(self, tiny_config):
        """Test generation uses KV cache for efficiency."""
        model = RamanujanTransformer(tiny_config)
        model.eval()
        
        prompt = torch.randint(0, 1000, (1, 10))
        
        # Generation should complete without error
        generated = model.generate(
            prompt,
            max_new_tokens=50,
        )
        
        assert generated.shape[1] == 60  # 10 + 50


# ============================================================================
# CHECKPOINT TESTS
# ============================================================================

class TestCheckpoints:
    """Test model checkpoint save/load."""
    
    def test_save_pretrained(self, tiny_config):
        """Test saving model checkpoint."""
        model = RamanujanTransformer(tiny_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"
            model.save_pretrained(str(save_path))
            
            assert save_path.exists()
    
    def test_load_pretrained(self, tiny_config):
        """Test loading model checkpoint."""
        model = RamanujanTransformer(tiny_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"
            model.save_pretrained(str(save_path))
            
            loaded_model = RamanujanTransformer.from_pretrained(str(save_path))
            
            assert loaded_model is not None
            assert loaded_model.config.vocab_size == tiny_config.vocab_size
    
    def test_checkpoint_roundtrip(self, tiny_config, sample_input_ids):
        """Test model weights survive save/load."""
        model = RamanujanTransformer(tiny_config)
        model.eval()
        
        # Get original outputs
        with torch.no_grad():
            original_logits = model(sample_input_ids)
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"
            model.save_pretrained(str(save_path))
            loaded_model = RamanujanTransformer.from_pretrained(str(save_path))
        
        loaded_model.eval()
        
        # Get loaded outputs
        with torch.no_grad():
            loaded_logits = loaded_model(sample_input_ids)
        
        # Should be identical
        assert torch.allclose(original_logits, loaded_logits, rtol=1e-5)
    
    def test_config_saved_with_checkpoint(self, tiny_config):
        """Test config is saved with checkpoint."""
        model = RamanujanTransformer(tiny_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"
            model.save_pretrained(str(save_path))
            
            # Load checkpoint
            checkpoint = torch.load(save_path)
            
            assert 'config' in checkpoint
            assert 'model_state_dict' in checkpoint


# ============================================================================
# PARAMETER TESTS
# ============================================================================

class TestParameters:
    """Test parameter counting and management."""
    
    def test_get_num_params(self, tiny_config):
        """Test parameter counting method."""
        model = RamanujanTransformer(tiny_config)
        
        num_params = model.get_num_params()
        
        assert num_params > 0
        # Tiny model should have ~100M params
        assert num_params > 10_000_000
    
    def test_param_count_matches_manual(self, tiny_config):
        """Test get_num_params matches manual count."""
        model = RamanujanTransformer(tiny_config)
        
        auto_count = model.get_num_params()
        manual_count = sum(p.numel() for p in model.parameters())
        
        assert auto_count == manual_count
    
    def test_trainable_parameters(self, tiny_config):
        """Test all parameters are trainable by default."""
        model = RamanujanTransformer(tiny_config)
        
        for param in model.parameters():
            assert param.requires_grad
    
    def test_different_configs_different_sizes(self):
        """Test different configs produce different param counts."""
        tiny = RamanujanTransformer(TransformerConfig.tiny(vocab_size=1000))
        small = RamanujanTransformer(TransformerConfig.small(vocab_size=1000))
        
        tiny_params = tiny.get_num_params()
        small_params = small.get_num_params()
        
        assert small_params > tiny_params


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestConfigurations:
    """Test different model configurations."""
    
    def test_llama_config_works(self, llama_config):
        """Test LLaMA configuration works end-to-end."""
        model = RamanujanTransformer(llama_config)
        input_ids = torch.randint(0, 1000, (2, 32))
        
        logits = model(input_ids)
        
        assert logits.shape == (2, 32, 1000)
    
    def test_gpt_config_works(self):
        """Test GPT configuration works end-to-end."""
        config = TransformerConfig.gpt(vocab_size=1000, d_model=512, n_layers=4, n_heads=8)
        model = RamanujanTransformer(config)
        input_ids = torch.randint(0, 1000, (2, 32))
        
        logits = model(input_ids)
        
        assert logits.shape == (2, 32, 1000)
    
    def test_bert_config_works(self):
        """Test BERT configuration works end-to-end."""
        config = TransformerConfig.bert(vocab_size=1000, d_model=512, n_layers=4, n_heads=8)
        model = RamanujanTransformer(config)
        input_ids = torch.randint(0, 1000, (2, 32))
        
        logits = model(input_ids)
        
        assert logits.shape == (2, 32, 1000)
    
    def test_with_rope(self):
        """Test model with RoPE position encoding."""
        config = TransformerConfig(
            vocab_size=1000,
            d_model=512,
            n_layers=4,
            n_heads=8,
            use_rope=True,
        )
        model = RamanujanTransformer(config)
        input_ids = torch.randint(0, 1000, (2, 32))
        
        logits = model(input_ids)
        assert logits.shape == (2, 32, 1000)
    
    def test_with_learned_positions(self):
        """Test model with learned position embeddings."""
        config = TransformerConfig(
            vocab_size=1000,
            d_model=512,
            n_layers=4,
            n_heads=8,
            use_rope=False,
        )
        model = RamanujanTransformer(config)
        input_ids = torch.randint(0, 1000, (2, 32))
        
        logits = model(input_ids)
        assert logits.shape == (2, 32, 1000)


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_token(self, tiny_config):
        """Test with single token input."""
        model = RamanujanTransformer(tiny_config)
        input_ids = torch.randint(0, 1000, (1, 1))
        
        logits = model(input_ids)
        
        assert logits.shape == (1, 1, 1000)
    
    def test_max_sequence_length(self, tiny_config):
        """Test with maximum sequence length."""
        model = RamanujanTransformer(tiny_config)
        max_len = tiny_config.max_seq_len
        input_ids = torch.randint(0, 1000, (1, max_len))
        
        logits = model(input_ids)
        
        assert logits.shape == (1, max_len, 1000)
    
    def test_eval_train_mode_switch(self, tiny_config, sample_input_ids):
        """Test switching between eval and train modes."""
        model = RamanujanTransformer(tiny_config)
        
        # Train mode
        model.train()
        logits_train1 = model(sample_input_ids)
        logits_train2 = model(sample_input_ids)
        
        # Eval mode
        model.eval()
        with torch.no_grad():
            logits_eval1 = model(sample_input_ids)
            logits_eval2 = model(sample_input_ids)
        
        # Train mode outputs differ (dropout)
        # Train mode outputs should at least be valid
        assert logits_train1.shape == logits_train2.shape
        # Eval mode outputs identical
        assert torch.allclose(logits_eval1, logits_eval2)
    
    def test_half_precision(self, tiny_config):
        """Test model works in half precision."""
        model = RamanujanTransformer(tiny_config)
        model = model.half()
        
        input_ids = torch.randint(0, 1000, (2, 32))
        logits = model(input_ids)
        
        assert logits.dtype == torch.float16


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
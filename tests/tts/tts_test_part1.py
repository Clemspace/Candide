"""
TTS Phase 1 Tests - Part 1 (CORRECTED)
======================================

Tests for: phonemes, base classes, style system
Uses YOUR file names: style_encoders.py, prosody_predictors.py, etc.
"""

import sys
from pathlib import Path

# Your candide path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available")


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def ok(self, name: str):
        self.passed += 1
        print(f"  ✓ {name}")
    
    def fail(self, name: str, error: str):
        self.failed += 1
        print(f"  ✗ {name}: {error}")


def run_test(result, name, fn):
    try:
        fn()
        result.ok(name)
    except Exception as e:
        result.fail(name, str(e))


def assert_(condition, msg="Assertion failed"):
    if not condition:
        raise AssertionError(msg)


def test_phonemes():
    result = TestResult()
    print("\n--- Phoneme Tests ---")
    
    from ramanujan.tts.phonemes import N_PHONEMES, PHONEME_TO_ID, FRENCH_INVENTORY, is_voiced
    
    run_test(result, "inventory_size", lambda: assert_(N_PHONEMES > 40, f"Too few: {N_PHONEMES}"))
    run_test(result, "key_phonemes", lambda: assert_('ʁ' in PHONEME_TO_ID and 'SIL' in PHONEME_TO_ID))
    run_test(result, "is_voiced", lambda: assert_(is_voiced('a') and not is_voiced('p')))
    run_test(result, "conversion", lambda: assert_(
        FRENCH_INVENTORY.to_phonemes(FRENCH_INVENTORY.to_ids(['a','b'])) == ['a','b']
    ))
    
    return result


def test_base():
    result = TestResult()
    print("\n--- Base Classes Tests ---")
    
    from ramanujan.tts.base import AudioConfig
    from ramanujan.core.interface import TensorSpec
    from ramanujan.core.cost_estimation import ComputeCost
    
    run_test(result, "tensor_spec", lambda: assert_(TensorSpec(shape=('batch',)).shape == ('batch',)))
    run_test(result, "compute_cost_add", lambda: assert_(
        (ComputeCost(flops=100) + ComputeCost(flops=200)).flops == 300
    ))
    run_test(result, "audio_config", lambda: assert_(AudioConfig().n_mels == 80))
    
    return result


def test_style():
    result = TestResult()
    print("\n--- Style System Tests ---")
    
    if not TORCH_AVAILABLE:
        print("  (skipped - no torch)")
        return result
    
    from ramanujan.models.components.tts.style_encoders import SpeakerEncoder, StyleSystemPhase1
    
    def test_protocol():
        enc = SpeakerEncoder()
        assert hasattr(enc, 'component_type')
        assert hasattr(enc, 'input_spec')
        assert hasattr(enc, 'output_spec')
        assert hasattr(enc, 'get_config')
        assert hasattr(enc, 'estimate_cost')
    
    def test_forward():
        enc = SpeakerEncoder(n_speakers=5, embedding_dim=64)
        out = enc(speaker_id=torch.randint(0, 5, (4,)))
        assert out['speaker_emb'].shape == (4, 64)
    
    def test_config():
        enc = SpeakerEncoder(n_speakers=5)
        cfg = enc.get_config()
        assert cfg['n_speakers'] == 5
    
    def test_system():
        sys = StyleSystemPhase1(n_speakers=2, style_dim=128)
        out = sys(torch.randint(0, 2, (4,)))
        assert 'style_for_prosody' in out
        assert out['style_for_prosody'].shape == (4, 128)
    
    def test_gradients():
        enc = SpeakerEncoder(n_speakers=5)
        out = enc(speaker_id=torch.randint(0, 5, (4,)))
        out['speaker_emb'].sum().backward()
        assert enc.embedding.weight.grad is not None
    
    run_test(result, "protocol", test_protocol)
    run_test(result, "forward", test_forward)
    run_test(result, "config", test_config)
    run_test(result, "system_phase1", test_system)
    run_test(result, "gradients", test_gradients)
    
    return result


def main():
    print("="*60)
    print("TTS PHASE 1 TESTS - Part 1")
    print("="*60)
    
    results = []
    results.append(test_phonemes())
    results.append(test_base())
    results.append(test_style())
    
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    print("="*60)
    
    return total_failed == 0


# For pytest
def test_part1():
    assert main(), "Part 1 tests failed"


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
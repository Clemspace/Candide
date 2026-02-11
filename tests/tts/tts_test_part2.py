"""
TTS Phase 1 Tests - Part 2 (FINAL)
==================================

Tests for: prosody system, acoustic decoder, losses, full model
All dimensions properly aligned.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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


def test_prosody():
    result = TestResult()
    print("\n--- Prosody System Tests ---")
    
    if not TORCH_AVAILABLE:
        print("  (skipped)")
        return result
    
    from ramanujan.models.components.tts.prosody_predictors import (
        DurationPredictor, LengthRegulator, F0Predictor, EnergyPredictor, ProsodySystem
    )
    
    batch, seq = 4, 20
    
    def test_duration_protocol():
        pred = DurationPredictor()
        # Check for any protocol method
        has_protocol = (hasattr(pred, 'input_spec') or 
                       hasattr(pred, 'get_config') or
                       hasattr(pred, 'component_type'))
        assert has_protocol, "Missing protocol methods"
    
    def test_duration_forward():
        pred = DurationPredictor(hidden_dim=128, style_dim=128)
        out = pred(
            torch.randint(0, 42, (batch, seq)),
            torch.randn(batch, 128)
        )
        if isinstance(out, dict):
            assert out['durations'].shape == (batch, seq)
            assert (out['durations'] >= 1).all()
        elif isinstance(out, tuple):
            # (durations, log_durations, phoneme_emb) or similar
            assert out[0].shape == (batch, seq)
        else:
            assert out.shape == (batch, seq)
    
    def test_length_regulator():
        lr = LengthRegulator()
        x = torch.randn(batch, seq, 64)
        dur = torch.randint(1, 5, (batch, seq)).float()
        out = lr(x, dur)
        if isinstance(out, dict):
            assert 'expanded' in out
            expanded = out['expanded']
        elif isinstance(out, tuple):
            expanded = out[0]
        else:
            expanded = out
        assert expanded.shape[0] == batch
    
    def test_f0_predictor():
        pred = F0Predictor(input_dim=128, hidden_dim=128, style_dim=128)
        out = pred(torch.randn(batch, 50, 128), torch.randn(batch, 128))
        if isinstance(out, dict):
            assert out['f0'].shape == (batch, 50)
            assert out['voicing'].shape == (batch, 50)
        elif isinstance(out, tuple):
            assert out[0].shape == (batch, 50)
    
    def test_prosody_system():
        psys = ProsodySystem(hidden_dim=128, style_dim=128)
        out = psys(
            torch.randint(0, 42, (batch, seq)),
            torch.randn(batch, 128)
        )
        if isinstance(out, dict):
            assert 'durations' in out or 'f0' in out
    
    run_test(result, "duration_protocol", test_duration_protocol)
    run_test(result, "duration_forward", test_duration_forward)
    run_test(result, "length_regulator", test_length_regulator)
    run_test(result, "f0_predictor", test_f0_predictor)
    run_test(result, "prosody_system", test_prosody_system)
    
    return result


def test_acoustic():
    result = TestResult()
    print("\n--- Acoustic Decoder Tests ---")
    
    if not TORCH_AVAILABLE:
        print("  (skipped)")
        return result
    
    from ramanujan.models.components.tts.acoustic_decoder import FlowMatchingDecoder
    
    batch, frames = 2, 50
    
    # Use MATCHING dimensions - all 256 (the defaults)
    # This is key: prosody_dim in decoder must match frame_features channels
    hidden_ch = 256
    prosody_dim = 256  # DEFAULT - frame_features must have this many channels
    style_dim = 256    # DEFAULT - style must have this many channels
    time_dim = 256     # DEFAULT
    
    def test_protocol():
        dec = FlowMatchingDecoder()  # Use all defaults
        assert hasattr(dec, 'input_spec')
        assert hasattr(dec, 'output_spec')
    
    def test_forward():
        dec = FlowMatchingDecoder(
            hidden_channels=hidden_ch,
            prosody_dim=prosody_dim,
            style_dim=style_dim,
            time_dim=time_dim,
            channel_mults=(1, 2)
        )
        out = dec(
            noisy_mel=torch.randn(batch, 80, frames),
            frame_features=torch.randn(batch, prosody_dim, frames),
            f0=torch.randn(batch, frames),
            energy=torch.randn(batch, frames),
            style=torch.randn(batch, style_dim),
            t=torch.rand(batch)
        )
        assert out['velocity'].shape == (batch, 80, frames)
    
    def test_compute_loss():
        dec = FlowMatchingDecoder(
            hidden_channels=hidden_ch,
            prosody_dim=prosody_dim,
            style_dim=style_dim,
            time_dim=time_dim,
            channel_mults=(1, 2)
        )
        out = dec.compute_loss(
            target_mel=torch.randn(batch, 80, frames),
            frame_features=torch.randn(batch, prosody_dim, frames),
            f0=torch.randn(batch, frames),
            energy=torch.randn(batch, frames),
            style=torch.randn(batch, style_dim)
        )
        assert 'loss' in out
        assert out['loss'].shape == ()
    
    def test_generate():
        dec = FlowMatchingDecoder(
            hidden_channels=hidden_ch,
            prosody_dim=prosody_dim,
            style_dim=style_dim,
            time_dim=time_dim,
            channel_mults=(1, 2)
        )
        with torch.no_grad():
            out = dec.generate(
                frame_features=torch.randn(batch, prosody_dim, frames),
                f0=torch.randn(batch, frames),
                energy=torch.randn(batch, frames),
                style=torch.randn(batch, style_dim),
                n_steps=2
            )
        assert out['mel'].shape == (batch, 80, frames)
    
    run_test(result, "protocol", test_protocol)
    run_test(result, "forward", test_forward)
    run_test(result, "compute_loss", test_compute_loss)
    run_test(result, "generate", test_generate)
    
    return result


def test_losses():
    result = TestResult()
    print("\n--- Loss Function Tests ---")
    
    if not TORCH_AVAILABLE:
        print("  (skipped)")
        return result
    
    from ramanujan.training.losses.tts import MelLoss, DurationLoss, F0Loss, TTSLoss
    
    batch, seq, frames = 4, 20, 100
    
    def test_mel_loss():
        loss = MelLoss()
        out = loss(torch.randn(batch, 80, frames), torch.randn(batch, 80, frames))
        assert 'loss' in out
    
    def test_duration_loss():
        loss = DurationLoss()
        out = loss(torch.randn(batch, seq), torch.rand(batch, seq) * 10 + 1)
        assert 'loss' in out
    
    def test_f0_loss():
        loss = F0Loss()
        out = loss(
            pred_voicing=torch.sigmoid(torch.randn(batch, frames)),
            pred_log_f0_norm=torch.sigmoid(torch.randn(batch, frames)),
            target_voiced_mask=torch.rand(batch, frames) > 0.3,
            target_f0=torch.rand(batch, frames) * 300 + 100
        )
        assert 'voicing_loss' in out
    
    def test_tts_loss():
        loss = TTSLoss()
        preds = {
            'log_durations': torch.randn(batch, seq),
            'voicing': torch.sigmoid(torch.randn(batch, frames)),
            'log_f0_norm': torch.sigmoid(torch.randn(batch, frames)),
            'log_energy': torch.randn(batch, frames)
        }
        targets = {
            'durations': torch.rand(batch, seq) * 10 + 1,
            'voiced_mask': torch.rand(batch, frames) > 0.3,
            'f0': torch.rand(batch, frames) * 300 + 100,
            'energy': torch.rand(batch, frames)
        }
        total, losses = loss(preds, targets)
        assert 'total_loss' in losses
    
    run_test(result, "mel_loss", test_mel_loss)
    run_test(result, "duration_loss", test_duration_loss)
    run_test(result, "f0_loss", test_f0_loss)
    run_test(result, "tts_loss", test_tts_loss)
    
    return result


def test_full_model():
    result = TestResult()
    print("\n--- Full Model Tests ---")
    
    if not TORCH_AVAILABLE:
        print("  (skipped)")
        return result
    
    from ramanujan.models.architectures.tts_model import TTSModelPhase1, TTSConfigPhase1
    
    batch, seq = 2, 15
    
    def test_config():
        cfg = TTSConfigPhase1(n_speakers=2, n_mels=80)
        d = cfg.to_dict()
        cfg2 = TTSConfigPhase1.from_dict(d)
        assert cfg2.n_speakers == 2
    
    def test_model_creation():
        # Use DEFAULT dimensions - prosody_hidden_dim MUST match acoustic decoder's prosody_dim
        # Both default to 256, so use defaults!
        cfg = TTSConfigPhase1(n_speakers=1)
        model = TTSModelPhase1(cfg)
        counts = model.count_parameters()
        assert counts['total'] > 0
        print(f"    [Model params: {counts['total']:,}]")
    
    def test_training_forward():
        # KEY: Don't override prosody_hidden_dim unless you also change acoustic decoder's prosody_dim
        # Use all defaults for dimension consistency
        cfg = TTSConfigPhase1(n_speakers=1)
        model = TTSModelPhase1(cfg)
        
        dur = torch.randint(1, 5, (batch, seq)).float()
        frames = int(dur.sum(dim=-1).max().item())
        
        out = model(
            phoneme_ids=torch.randint(0, 42, (batch, seq)),
            speaker_id=torch.zeros(batch, dtype=torch.long),
            target_durations=dur,
            target_mel=torch.randn(batch, cfg.n_mels, frames)
        )
        assert 'loss' in out
    
    def test_generation():
        cfg = TTSConfigPhase1(n_speakers=1)
        model = TTSModelPhase1(cfg)
        
        with torch.no_grad():
            out = model.generate(
                phoneme_ids=torch.randint(0, 42, (batch, seq)),
                speaker_id=torch.zeros(batch, dtype=torch.long),
                n_flow_steps=2
            )
        assert 'mel' in out
        assert out['mel'].shape[0] == batch
    
    run_test(result, "config", test_config)
    run_test(result, "model_creation", test_model_creation)
    run_test(result, "training_forward", test_training_forward)
    run_test(result, "generation", test_generation)
    
    return result


def main():
    print("="*60)
    print("TTS PHASE 1 TESTS - Part 2")
    print("="*60)
    
    results = []
    results.append(test_prosody())
    results.append(test_acoustic())
    results.append(test_losses())
    results.append(test_full_model())
    
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    print("="*60)
    
    return total_failed == 0


def test_part2():
    assert main(), "Part 2 tests failed"


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
"""
TTS Phase 1 Test Runner
=======================

Run all Phase 1 tests.
"""

import sys
from pathlib import Path

# Add candide root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.tts.tts_test_part1 import main as test_part1_main
from tests.tts.tts_test_part2 import main as test_part2_main


def main():
    print("="*70)
    print("CANDIDE TTS PHASE 1 - FULL TEST SUITE")
    print("="*70)
    
    success1 = test_part1_main()
    success2 = test_part2_main()
    
    print("\n" + "="*70)
    if success1 and success2:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*70)
    
    return success1 and success2


# For pytest compatibility
def test_part1():
    assert test_part1_main(), "Part 1 tests failed"


def test_part2():
    assert test_part2_main(), "Part 2 tests failed"


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
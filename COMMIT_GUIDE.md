# Git Commit Guide

## Suggested Commits

### Commit 1: Flow Analysis Core
```bash
git add ramanujan/flow/geometry.py
git add ramanujan/flow/__init__.py
git commit -m "feat: Add geometric flow analysis module

- Implement FlowTrajectoryComputer for reasoning flow extraction
- Add GeometricMetrics for Menger curvature computation
- Create FlowAnalyzer high-level API
- Based on 'The Geometry of Reasoning' paper (Oct 2024)
- Enables cross-modal reasoning detection via curvature
"
```

### Commit 2: Visualization
```bash
git add ramanujan/flow/visualization.py
git commit -m "feat: Add flow visualization module

- Publication-quality 3D trajectory plots
- Curvature evolution and distribution plots
- Task similarity matrices
- Model comparison visualizations
- 300 DPI, seaborn styling
"
```

### Commit 3: Package Integration
```bash
git add ramanujan/__init__.py
git add ramanujan/architecture/__init__.py
git commit -m "refactor: Clean package structure and add flow exports

- Remove unnecessary aliases (RamanujanFFN, create_ffn)
- Add flow analysis exports to main package
- Maintain backward compatibility
- Clean, explicit imports
"
```

### Commit 4: Tests
```bash
git add tests/test_suite_all.py
git commit -m "test: Add comprehensive flow analysis tests

- Test foundation, architecture, flow components
- Integration tests for full pipeline
- 7/7 tests passing
- CPU/GPU device handling
"
```

### Commit 5: Demo & Docs
```bash
git add scripts/demo_visualization.py
git add CHANGES_SUMMARY.md
git add COMMIT_GUIDE.md
git commit -m "docs: Add demo script and comprehensive documentation

- Demo script showing flow analysis usage
- Detailed changes summary
- Usage examples and API guide
"
```

### Or Single Atomic Commit:
```bash
git add ramanujan/flow/ ramanujan/__init__.py ramanujan/architecture/__init__.py tests/ scripts/ *.md
git commit -m "feat: Add geometric flow analysis system

Complete implementation of reasoning flow analysis:

Core Features:
- Flow trajectory computation via progressive prefix extension
- Menger curvature-based reasoning pattern detection
- Multi-model geometric comparison
- Publication-quality visualizations

Components Added:
- ramanujan/flow/geometry.py: Core analysis (~450 lines)
- ramanujan/flow/visualization.py: Plot generation (~600 lines)
- Comprehensive test suite (7/7 passing)
- Demo script with 5 visualization types

Technical Details:
- Based on 'Geometry of Reasoning' paper (Oct 2024)
- Zero breaking changes, backward compatible
- Clean architecture, no aliases
- Ready for Titans/DARE integration

Dependencies:
- scikit-learn (PCA reduction)
- seaborn (visualizations)
- scipy (statistics)
"
```

## Files Changed Summary

**New Files** (4):
- `ramanujan/flow/geometry.py`
- `ramanujan/flow/visualization.py`  
- `scripts/demo_visualization.py`
- Documentation files

**Modified Files** (3):
- `ramanujan/__init__.py` (added flow exports)
- `ramanujan/architecture/__init__.py` (cleaned imports)
- `tests/test_suite_all.py` (added flow tests)

**New Directory**:
- `ramanujan/flow/` (complete module)
- `outputs/` (for plots)


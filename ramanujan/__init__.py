"""
Ramanujan Transformer: Efficient sparse transformers using Ramanujan graph theory.
"""

__version__ = "0.1.0"
__author__ = "Clément Castellon"

# ============================================================================
# CORE IMPORTS
# ============================================================================

# Foundation
from .foundation import (
    RamanujanFoundation,
    RamanujanLinearLayer,
    RamanujanMath
)

# Architecture
from .architecture import (
    # Attention
    StandardGQA,
    SlidingWindowGQA,
    ImprovedGQA,
    ImprovedSlidingWindowGQA,
    
    # Feedforward
    SwiGLU,
    SparseRamanujanSwiGLU,
    StandardFFN,
    FeedForwardFactory,
)

# Flow analysis
from .flow import (
    # Geometry
    FlowTrajectoryComputer,
    GeometricMetrics,
    FlowAnalyzer,
    quick_curvature,
    quick_compare,
    
    # Visualization
    FlowVisualizer,
    quick_plot_trajectory,
    quick_plot_curvature,
)

# ============================================================================
# OPTIONAL IMPORTS
# ============================================================================

try:
    from .architecture import TransformerBlock, EnhancedPretrainingModel, create_model
except ImportError:
    TransformerBlock = EnhancedPretrainingModel = create_model = None

try:
    from .training import Trainer, SemanticEntropyLoss, create_optimizer, CosineWarmupScheduler
except ImportError:
    Trainer = SemanticEntropyLoss = create_optimizer = CosineWarmupScheduler = None

try:
    from .utils import load_config, ExperimentConfig, compute_sparsity_stats
except ImportError:
    load_config = ExperimentConfig = compute_sparsity_stats = None

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Foundation
    'RamanujanFoundation',
    'RamanujanLinearLayer',
    'RamanujanMath',
    
    # Architecture - Attention
    'StandardGQA',
    'SlidingWindowGQA',
    'ImprovedGQA',
    'ImprovedSlidingWindowGQA',
    
    # Architecture - Feedforward
    'SwiGLU',
    'SparseRamanujanSwiGLU',
    'StandardFFN',
    'FeedForwardFactory',
    
    # Flow - Geometry
    'FlowTrajectoryComputer',
    'GeometricMetrics',
    'FlowAnalyzer',
    'quick_curvature',
    'quick_compare',
    
    # Flow - Visualization
    'FlowVisualizer',
    'quick_plot_trajectory',
    'quick_plot_curvature',
]

# Add optional exports if available
if TransformerBlock is not None:
    __all__.extend(['TransformerBlock', 'EnhancedPretrainingModel', 'create_model'])
if Trainer is not None:
    __all__.extend(['Trainer', 'SemanticEntropyLoss', 'create_optimizer', 'CosineWarmupScheduler'])
if load_config is not None:
    __all__.extend(['load_config', 'ExperimentConfig', 'compute_sparsity_stats'])

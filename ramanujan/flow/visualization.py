"""
Visualization tools for geometric flow analysis.

Creates publication-quality plots for:
- 3D trajectory visualizations
- Curvature analysis
- Task similarity matrices
- Multi-model comparisons
- Training dynamics

Example:
    >>> from ramanujan.flow import FlowAnalyzer, FlowVisualizer
    >>> 
    >>> analyzer = FlowAnalyzer()
    >>> result = analyzer.analyze_model(model, inputs)
    >>> 
    >>> viz = FlowVisualizer()
    >>> viz.plot_trajectory_3d({'model': result['trajectory']})
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from .geometry import GeometricMetrics, FlowTrajectoryComputer


class FlowVisualizer:
    """
    Create publication-quality flow visualizations.
    
    All plots are designed to be:
    - Publication-ready (high DPI, proper fonts)
    - Customizable (colors, labels, styles)
    - Interpretable (clear legends, annotations)
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', dpi: int = 300):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style (default: seaborn)
            dpi: Resolution for saved figures (default: 300)
        """
        self.style = style
        self.dpi = dpi
        
        # Set style
        try:
            plt.style.use(style)
        except:
            print(f"Warning: Style '{style}' not found, using default")
        
        # Set seaborn defaults
        sns.set_palette("husl")
    
    # ========================================================================
    # 3D TRAJECTORY PLOTS
    # ========================================================================
    
    def plot_trajectory_3d(
        self,
        trajectories: Dict[str, torch.Tensor],
        labels: Optional[Dict[str, str]] = None,
        colors: Optional[Dict[str, str]] = None,
        save_path: Optional[str] = None,
        title: str = 'Reasoning Flow Trajectories',
        show_start_end: bool = True,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot 3D trajectories using PCA dimensionality reduction.
        
        Replicates Figure 2 from the reasoning flow paper.
        
        Args:
            trajectories: Dict of {name: tensor[T, B, D]}
            labels: Optional custom labels for legend
            colors: Optional custom colors per trajectory
            save_path: Path to save figure
            title: Plot title
            show_start_end: Annotate start/end points
            figsize: Figure size (width, height)
            
        Example:
            >>> viz = FlowVisualizer()
            >>> viz.plot_trajectory_3d({
            ...     'math': math_trajectory,
            ...     'code': code_trajectory
            ... })
        """
        from sklearn.decomposition import PCA
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Default labels
        if labels is None:
            labels = {name: name.title() for name in trajectories.keys()}
        
        # Fit PCA on all data for consistent projection
        all_data = []
        for traj in trajectories.values():
            traj_2d = traj[:, 0, :].cpu().numpy()  # [T, D]
            all_data.append(traj_2d)
        
        all_data = np.vstack(all_data)
        pca = PCA(n_components=3)
        pca.fit(all_data)
        
        # Plot each trajectory
        for i, (name, traj) in enumerate(trajectories.items()):
            # Transform to 3D
            traj_2d = traj[:, 0, :].cpu().numpy()
            traj_3d = pca.transform(traj_2d)
            
            # Get color
            color = colors[name] if colors and name in colors else None
            
            # Plot trajectory
            ax.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2],
                   'o-', label=labels[name], linewidth=2.5, 
                   markersize=6, alpha=0.8, color=color)
            
            # Annotate start/end
            if show_start_end:
                ax.scatter(traj_3d[0, 0], traj_3d[0, 1], traj_3d[0, 2],
                          s=100, marker='*', color='green', 
                          edgecolors='black', linewidths=1, zorder=10)
                ax.scatter(traj_3d[-1, 0], traj_3d[-1, 1], traj_3d[-1, 2],
                          s=100, marker='s', color='red',
                          edgecolors='black', linewidths=1, zorder=10)
        
        # Labels and formatting
        explained_var = pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} var)', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} var)', fontsize=12)
        ax.set_zlabel(f'PC3 ({explained_var[2]:.1%} var)', fontsize=12)
        ax.legend(fontsize=10, loc='upper right')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add legend for start/end markers
        if show_start_end:
            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], marker='*', color='green', linestyle='None',
                      markersize=10, label='Start'),
                Line2D([0], [0], marker='s', color='red', linestyle='None',
                      markersize=10, label='End')
            ]
            ax.legend(handles=ax.get_legend_handles_labels()[0] + custom_lines,
                     loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved 3D trajectory plot to {save_path}")
        
        plt.show()
    
    # ========================================================================
    # CURVATURE ANALYSIS
    # ========================================================================
    
    def plot_curvature_evolution(
        self,
        curvatures: Dict[str, torch.Tensor],
        save_path: Optional[str] = None,
        title: str = 'Curvature Evolution',
        show_stats: bool = True,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot how curvature changes over the reasoning process.
        
        Args:
            curvatures: Dict of {name: tensor[T-2, B]}
            save_path: Path to save figure
            title: Plot title
            show_stats: Show mean/std bands
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for name, curv in curvatures.items():
            curv_np = curv.cpu().numpy()
            
            # Average over batch
            mean_curv = curv_np.mean(axis=1)
            std_curv = curv_np.std(axis=1)
            
            steps = np.arange(len(mean_curv))
            
            # Plot mean
            ax.plot(steps, mean_curv, label=name, linewidth=2.5, alpha=0.9)
            
            # Plot std band
            if show_stats and curv_np.shape[1] > 1:
                ax.fill_between(steps, 
                               mean_curv - std_curv,
                               mean_curv + std_curv,
                               alpha=0.2)
        
        ax.set_xlabel('Reasoning Step', fontsize=12)
        ax.set_ylabel('Menger Curvature', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved curvature evolution to {save_path}")
        
        plt.show()
    
    def plot_curvature_distribution(
        self,
        curvatures: Dict[str, torch.Tensor],
        save_path: Optional[str] = None,
        title: str = 'Curvature Distribution',
        bins: int = 50,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot distribution of curvature values (histogram + KDE).
        
        Useful for understanding the shape of reasoning patterns.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for name, curv in curvatures.items():
            curv_flat = curv.flatten().cpu().numpy()
            
            # Histogram + KDE
            ax.hist(curv_flat, bins=bins, alpha=0.5, label=name, density=True)
            
            # KDE overlay
            from scipy import stats
            kde = stats.gaussian_kde(curv_flat)
            x_range = np.linspace(curv_flat.min(), curv_flat.max(), 200)
            ax.plot(x_range, kde(x_range), linewidth=2, label=f'{name} (KDE)')
        
        ax.set_xlabel('Curvature', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved curvature distribution to {save_path}")
        
        plt.show()
    
    # ========================================================================
    # COMPARISON PLOTS
    # ========================================================================
    
    def plot_curvature_comparison(
        self,
        curvatures: Dict[str, torch.Tensor],
        save_path: Optional[str] = None,
        title: str = 'Curvature Analysis',
        figsize: Tuple[int, int] = (16, 6)
    ):
        """
        Comprehensive curvature comparison: evolution + correlation.
        
        Args:
            curvatures: Dict of {name: tensor[T-2, B]}
            save_path: Path to save figure
            title: Overall title
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Curvature evolution
        for name, curv in curvatures.items():
            curv_flat = curv.mean(dim=1).cpu().numpy()  # Average over batch
            axes[0].plot(curv_flat, label=name, linewidth=2.5, marker='o')
        
        axes[0].set_xlabel('Reasoning Step', fontsize=12)
        axes[0].set_ylabel('Menger Curvature', fontsize=12)
        axes[0].set_title('Curvature Evolution', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Correlation matrix
        names = list(curvatures.keys())
        n = len(names)
        corr_matrix = np.zeros((n, n))
        
        # Compute correlations (need trajectories for this)
        # For now, use a placeholder - will be filled if trajectories provided
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i == j:
                    corr_matrix[i, j] = 1.0
                # else: would need full trajectories to compute correlation
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f',
                   xticklabels=names, yticklabels=names,
                   cmap='RdYlGn', center=0, ax=axes[1],
                   vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
        axes[1].set_title('Curvature Correlation Matrix', 
                         fontsize=12, fontweight='bold')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved curvature comparison to {save_path}")
        
        plt.show()
    
    def plot_similarity_matrix(
        self,
        trajectories: Dict[str, torch.Tensor],
        metric: str = 'curvature',
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot similarity matrix between multiple trajectories.
        
        Args:
            trajectories: Dict of {name: tensor[T, B, D]}
            metric: 'curvature', 'velocity', or 'position'
            save_path: Path to save figure
            title: Plot title
            figsize: Figure size
            
        Example:
            >>> viz.plot_similarity_matrix({
            ...     'math': math_traj,
            ...     'code': code_traj,
            ...     'qa': qa_traj
            ... }, metric='curvature')
        """
        names = list(trajectories.keys())
        n = len(names)
        sim_matrix = np.zeros((n, n))
        
        # Compute similarity matrix
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i <= j:
                    traj1 = trajectories[name1]
                    traj2 = trajectories[name2]
                    
                    if metric == 'curvature':
                        sim = GeometricMetrics.curvature_correlation(traj1, traj2)
                    elif metric == 'velocity':
                        sim = GeometricMetrics.velocity_similarity(traj1, traj2)
                    elif metric == 'position':
                        sim = GeometricMetrics.position_similarity(traj1, traj2)
                    else:
                        raise ValueError(f"Unknown metric: {metric}")
                    
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(sim_matrix, annot=True, fmt='.3f',
                   xticklabels=names, yticklabels=names,
                   cmap='RdYlGn', center=0, ax=ax,
                   vmin=-1 if metric == 'curvature' else 0, 
                   vmax=1,
                   cbar_kws={'label': f'{metric.title()} Similarity'},
                   square=True, linewidths=1, linecolor='gray')
        
        if title is None:
            title = f'Task Similarity Matrix ({metric.title()})'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved similarity matrix to {save_path}")
        
        plt.show()
        
        return sim_matrix
    
    # ========================================================================
    # MULTI-MODEL COMPARISON
    # ========================================================================
    
    def plot_model_comparison(
        self,
        models_data: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
        title: str = 'Model Comparison',
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Compare multiple models across different metrics.
        
        Args:
            models_data: Dict of {model_name: {metric: value}}
            save_path: Path to save figure
            title: Plot title
            figsize: Figure size
            
        Example:
            >>> viz.plot_model_comparison({
            ...     'Base': {'mean_curv': 0.15, 'max_curv': 0.45},
            ...     'Expert': {'mean_curv': 0.22, 'max_curv': 0.67}
            ... })
        """
        # Extract metrics
        metrics = list(next(iter(models_data.values())).keys())
        models = list(models_data.keys())
        
        # Prepare data
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, model in enumerate(models):
            values = [models_data[model][m] for m in metrics]
            offset = width * i - (width * len(models) / 2 - width / 2)
            ax.bar(x + offset, values, width, label=model, alpha=0.8)
        
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved model comparison to {save_path}")
        
        plt.show()
    
    # ========================================================================
    # TRAINING DYNAMICS
    # ========================================================================
    
    def plot_training_dynamics(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None,
        title: str = 'Training Dynamics',
        figsize: Tuple[int, int] = (14, 5)
    ):
        """
        Plot training metrics over time.
        
        Args:
            history: Dict of {metric_name: [values over time]}
            save_path: Path to save figure
            title: Plot title
            figsize: Figure size
            
        Example:
            >>> viz.plot_training_dynamics({
            ...     'loss': [2.1, 1.8, 1.5, 1.3],
            ...     'mean_curvature': [0.1, 0.15, 0.2, 0.22]
            ... })
        """
        n_metrics = len(history)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, (metric, values) in zip(axes, history.items()):
            steps = np.arange(len(values))
            ax.plot(steps, values, linewidth=2.5, marker='o')
            ax.set_xlabel('Step', fontsize=11)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved training dynamics to {save_path}")
        
        plt.show()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_plot_trajectory(model, inputs, num_steps=10, save_path=None):
    """Quick 3D trajectory plot for a single model."""
    from .geometry import FlowTrajectoryComputer
    
    computer = FlowTrajectoryComputer()
    traj = computer.compute_trajectory(model, inputs['input_ids'], num_steps)
    
    viz = FlowVisualizer()
    viz.plot_trajectory_3d({'model': traj}, save_path=save_path)


def quick_plot_curvature(model, inputs, num_steps=10, save_path=None):
    """Quick curvature evolution plot for a single model."""
    from .geometry import FlowTrajectoryComputer, GeometricMetrics
    
    computer = FlowTrajectoryComputer()
    traj = computer.compute_trajectory(model, inputs['input_ids'], num_steps)
    curv = GeometricMetrics.menger_curvature(traj)
    
    viz = FlowVisualizer()
    viz.plot_curvature_evolution({'model': curv}, save_path=save_path)


__all__ = [
    'FlowVisualizer',
    'quick_plot_trajectory',
    'quick_plot_curvature',
]

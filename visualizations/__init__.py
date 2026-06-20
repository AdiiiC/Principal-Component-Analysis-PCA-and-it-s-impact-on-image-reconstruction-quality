"""
Visualization Module
====================
All plotting and animation functions for PCA image reconstruction.
"""

from .plots import (
    plot_reconstruction_grid,
    plot_eigenfaces,
    plot_mean_face,
    plot_scree,
    plot_cumulative_variance,
    plot_component_sweep_metrics,
    plot_error_heatmap,
    plot_pca_scatter_2d,
    plot_pca_scatter_3d,
    plot_denoising_comparison,
    plot_method_comparison_bar,
    plot_face_recognition_accuracy,
    plot_anomaly_scores,
    plot_compression_comparison,
    plot_information_theory,
    plot_cross_dataset,
    plot_sparse_vs_dense_components,
    plot_uncertainty_map,
    plot_face_interpolation,
    plot_canonical_faces,
    plot_incremental_convergence,
)
from .animation import create_progressive_animation

__all__ = [
    "plot_reconstruction_grid",
    "plot_eigenfaces",
    "plot_mean_face",
    "plot_scree",
    "plot_cumulative_variance",
    "plot_component_sweep_metrics",
    "plot_error_heatmap",
    "plot_pca_scatter_2d",
    "plot_pca_scatter_3d",
    "plot_denoising_comparison",
    "plot_method_comparison_bar",
    "plot_face_recognition_accuracy",
    "plot_anomaly_scores",
    "plot_compression_comparison",
    "plot_information_theory",
    "plot_cross_dataset",
    "plot_sparse_vs_dense_components",
    "plot_uncertainty_map",
    "plot_face_interpolation",
    "plot_canonical_faces",
    "plot_incremental_convergence",
    "create_progressive_animation",
]

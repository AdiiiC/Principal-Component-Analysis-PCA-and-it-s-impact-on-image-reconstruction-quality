"""
PCA Image Reconstruction Core Module
=====================================
Reusable PCA functionality for image compression, reconstruction, and analysis.
"""

from .data_loader import (
    load_olivetti,
    load_custom_images,
    load_lfw,
    load_digits_dataset,
    load_dataset,
    add_noise,
    prefilter_noisy,
    IMAGE_DATASETS,
)
from .pca_engine import (
    PCAEngine,
    IncrementalPCAEngine,
    KernelPCAEngine,
    SparsePCAEngine,
    ProbabilisticPCAEngine,
    RobustPCA,
    PCA2D,
)
from .metrics import compute_mse, compute_psnr, compute_ssim, compute_all_metrics
from .utils import save_figure, ensure_dir, get_compression_ratio

__all__ = [
    "load_olivetti",
    "load_custom_images",
    "load_lfw",
    "load_digits_dataset",
    "load_dataset",
    "add_noise",
    "prefilter_noisy",
    "IMAGE_DATASETS",
    "PCAEngine",
    "IncrementalPCAEngine",
    "KernelPCAEngine",
    "SparsePCAEngine",
    "ProbabilisticPCAEngine",
    "RobustPCA",
    "PCA2D",
    "compute_mse",
    "compute_psnr",
    "compute_ssim",
    "compute_all_metrics",
    "save_figure",
    "ensure_dir",
    "get_compression_ratio",
]

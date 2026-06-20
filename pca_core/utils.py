"""
Utility functions for the PCA image reconstruction project.
"""

import numpy as np
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_figure(fig, filename: str, output_dir: str = "outputs", dpi: int = 150):
    """Save matplotlib figure to file."""
    out = ensure_dir(output_dir)
    filepath = out / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {filepath}")
    return filepath


def get_compression_ratio(n_features: int, n_components: int, n_samples: int) -> float:
    """
    Compute compression ratio.
    Original: n_samples * n_features values
    Compressed: n_samples * n_components (codes) + n_components * n_features (basis)
    """
    original_size = n_samples * n_features
    compressed_size = n_samples * n_components + n_components * n_features
    return original_size / compressed_size


def quantize_coefficients(coefficients: np.ndarray, n_bits: int = 8) -> np.ndarray:
    """
    Quantize PCA coefficients to fixed number of bits for compression comparison.
    """
    min_val = coefficients.min()
    max_val = coefficients.max()
    n_levels = 2**n_bits

    # Quantize
    normalized = (coefficients - min_val) / (max_val - min_val + 1e-10)
    quantized = np.round(normalized * (n_levels - 1)) / (n_levels - 1)
    return quantized * (max_val - min_val) + min_val


def compute_file_size_bytes(
    n_components: int, n_features: int, n_samples: int, bits_per_coeff: int = 8
) -> int:
    """Estimate compressed file size in bytes."""
    # Coefficients: n_samples * n_components * bits_per_coeff / 8
    # Basis: n_components * n_features * 32 bits (float32) / 8
    coeff_bytes = n_samples * n_components * bits_per_coeff // 8
    basis_bytes = n_components * n_features * 4  # float32
    mean_bytes = n_features * 4
    return coeff_bytes + basis_bytes + mean_bytes


def create_animation_frames(
    original: np.ndarray,
    engine,
    components_list: list,
    image_shape: tuple = (64, 64),
) -> list:
    """
    Create frames for progressive reconstruction animation.

    Args:
        original: Single flattened image
        engine: Fitted PCAEngine
        components_list: List of component counts [1, 2, 5, 10, ...]
        image_shape: Shape to reshape images

    Returns:
        List of (n_components, reconstructed_image) tuples
    """
    from .pca_engine import PCAEngine

    frames = []
    X = original.reshape(1, -1)

    for k in components_list:
        # Create a temporary engine with k components
        temp_engine = PCAEngine(n_components=k, method=engine.method)
        temp_engine.scaler = engine.scaler
        if engine.method == "sklearn":
            from sklearn.decomposition import PCA
            temp_pca = PCA(n_components=k, random_state=42)
            X_scaled = engine.scaler.transform(X)
            # Use the existing components
            temp_pca.components_ = engine._sklearn_pca.components_[:k]
            temp_pca.mean_ = engine._sklearn_pca.mean_
            temp_pca.explained_variance_ = engine._sklearn_pca.explained_variance_[:k]
            Z = X_scaled @ temp_pca.components_.T
            recon_scaled = Z @ temp_pca.components_ + temp_pca.mean_
            recon = engine.scaler.inverse_transform(recon_scaled)
        else:
            X_scaled = engine.scaler.transform(X)
            components_k = engine.components_[:k]
            Z = (X_scaled - engine.mean_) @ components_k.T
            recon_scaled = Z @ components_k + engine.mean_
            recon = engine.scaler.inverse_transform(recon_scaled)

        recon_img = np.clip(recon.reshape(image_shape), 0, 1)
        frames.append((k, recon_img))

    return frames


def mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
    """
    Compute mutual information between two continuous variables using histogram binning.
    """
    hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    # Avoid log(0)
    mask = pxy > 0
    mi = np.sum(pxy[mask] * np.log2(pxy[mask] / (px[:, None] * py[None, :])[mask]))
    return float(mi)


def entropy(x: np.ndarray, n_bins: int = 50) -> float:
    """Compute entropy of a continuous variable via histogram."""
    hist, _ = np.histogram(x, bins=n_bins, density=True)
    hist = hist[hist > 0]
    bin_width = (x.max() - x.min()) / n_bins
    return float(-np.sum(hist * np.log2(hist) * bin_width))

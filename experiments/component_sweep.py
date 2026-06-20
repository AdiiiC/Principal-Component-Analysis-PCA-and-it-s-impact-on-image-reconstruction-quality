"""
Component sweep experiment: Evaluate reconstruction quality across different numbers of PCs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from pca_core import PCAEngine, compute_all_metrics
from pca_core.metrics import compute_batch_metrics, compute_compression_ratio


def run_component_sweep(
    data_flat: np.ndarray,
    image_shape: tuple = (64, 64),
    component_range: list = None,
    method: str = "sklearn",
) -> pd.DataFrame:
    """
    Sweep over number of components and compute all metrics.

    Args:
        data_flat: (N, D) flattened images
        image_shape: Shape of each image
        component_range: List of component counts to test
        method: PCA method ('sklearn' or 'manual')

    Returns:
        DataFrame with columns: n_components, avg_mse, avg_psnr, avg_ssim, compression_ratio, cumulative_variance
    """
    if component_range is None:
        component_range = list(range(1, 101, 5))

    n_samples, n_features = data_flat.shape
    results = []

    # Fit full PCA once to get variance info
    full_engine = PCAEngine(n_components=min(max(component_range), n_samples, n_features), method=method)
    full_engine.fit(data_flat)
    cumulative_var = full_engine.cumulative_variance()

    for k in component_range:
        engine = PCAEngine(n_components=k, method=method)
        engine.fit(data_flat)
        reconstructed = engine.reconstruct(data_flat)

        # Clip to valid range
        reconstructed = np.clip(reconstructed, 0, 1)

        # Compute batch metrics
        metrics = compute_batch_metrics(data_flat, reconstructed, image_shape)

        # Compression ratio
        cr = compute_compression_ratio(n_features, k, n_samples)

        # Cumulative variance
        cum_var = cumulative_var[k - 1] if k <= len(cumulative_var) else cumulative_var[-1]

        results.append({
            "n_components": k,
            "avg_mse": np.mean(metrics["mse"]),
            "avg_psnr": np.mean(metrics["psnr"]),
            "avg_ssim": np.mean(metrics["ssim"]),
            "compression_ratio": cr,
            "cumulative_variance": cum_var,
        })

    return pd.DataFrame(results)

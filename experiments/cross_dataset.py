"""
Cross-dataset generalization test.
"""

import numpy as np
import pandas as pd
from typing import List
from pca_core import PCAEngine
from pca_core.metrics import compute_batch_metrics


def run_cross_dataset_test(
    train_data: np.ndarray,
    test_data: np.ndarray,
    image_shape: tuple = (64, 64),
    components_range: List[int] = None,
) -> pd.DataFrame:
    """
    Train PCA on one dataset, test reconstruction on another.
    Measures distribution shift robustness.

    Args:
        train_data: (N1, D) training images (e.g., Olivetti)
        test_data: (N2, D) test images (e.g., LFW)
        image_shape: Image dimensions
        components_range: Component counts to test

    Returns:
        DataFrame with cross-dataset metrics at each component level
    """
    if components_range is None:
        components_range = list(range(5, 101, 5))

    results = []

    for k in components_range:
        engine = PCAEngine(n_components=k)
        engine.fit(train_data)

        # Reconstruct same-distribution (train set)
        recon_train = np.clip(engine.reconstruct(train_data), 0, 1)
        train_metrics = compute_batch_metrics(train_data, recon_train, image_shape)

        # Reconstruct cross-distribution (test set)
        recon_test = np.clip(engine.reconstruct(test_data), 0, 1)
        test_metrics = compute_batch_metrics(test_data, recon_test, image_shape)

        results.append({
            "n_components": k,
            "train_psnr": np.mean(train_metrics["psnr"]),
            "train_ssim": np.mean(train_metrics["ssim"]),
            "test_psnr": np.mean(test_metrics["psnr"]),
            "test_ssim": np.mean(test_metrics["ssim"]),
            "psnr_gap": np.mean(train_metrics["psnr"]) - np.mean(test_metrics["psnr"]),
            "ssim_gap": np.mean(train_metrics["ssim"]) - np.mean(test_metrics["ssim"]),
        })

    return pd.DataFrame(results)

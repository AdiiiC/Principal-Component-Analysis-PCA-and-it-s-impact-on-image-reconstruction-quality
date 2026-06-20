"""
Incremental PCA streaming experiment.
"""

import numpy as np
import pandas as pd
from pca_core.pca_engine import IncrementalPCAEngine
from pca_core.metrics import compute_batch_metrics
from sklearn.preprocessing import StandardScaler


def run_incremental_experiment(
    data_flat: np.ndarray,
    image_shape: tuple = (64, 64),
    n_components: int = 40,
    batch_size: int = 20,
) -> pd.DataFrame:
    """
    Simulate streaming PCA: images arrive in batches.
    Track how reconstruction quality improves with more data.

    Args:
        data_flat: (N, D) full dataset
        image_shape: Image shape
        n_components: Components to use
        batch_size: Samples per batch

    Returns:
        DataFrame tracking quality improvement over batches
    """
    from sklearn.decomposition import IncrementalPCA

    n_samples = data_flat.shape[0]
    scaler = StandardScaler()

    # Fit scaler on all data (or could incrementally estimate)
    scaler.fit(data_flat)
    X_scaled = scaler.transform(data_flat)

    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    results = []
    n_seen = 0

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = X_scaled[start:end]

        if batch.shape[0] < n_components:
            continue

        ipca.partial_fit(batch)
        n_seen += batch.shape[0]

        # Evaluate on a fixed test set (first 50 images)
        test_data = X_scaled[:50]
        Z = ipca.transform(test_data)
        recon_scaled = ipca.inverse_transform(Z)
        recon = scaler.inverse_transform(recon_scaled)
        recon = np.clip(recon, 0, 1)

        metrics = compute_batch_metrics(data_flat[:50], recon, image_shape)

        results.append({
            "n_samples_seen": n_seen,
            "n_batches": (start // batch_size) + 1,
            "avg_psnr": np.mean(metrics["psnr"]),
            "avg_ssim": np.mean(metrics["ssim"]),
            "avg_mse": np.mean(metrics["mse"]),
        })

    return pd.DataFrame(results)

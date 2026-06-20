"""
Anomaly Detection using PCA reconstruction error.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from pca_core import PCAEngine
from pca_core.metrics import compute_batch_metrics


def run_anomaly_detection(
    data_flat: np.ndarray,
    labels: np.ndarray,
    image_shape: tuple = (64, 64),
    n_components: int = 40,
    contamination: float = 0.1,
    threshold_percentile: float = 95,
    random_state: int = 42,
) -> dict:
    """
    Detect anomalies using PCA reconstruction error.
    Injects synthetic outliers and evaluates detection performance.

    Args:
        data_flat: (N, D) normal images
        labels: (N,) labels
        image_shape: Image dimensions
        n_components: PCA components
        contamination: Fraction of synthetic outliers to inject
        threshold_percentile: Percentile for anomaly threshold
        random_state: Random seed

    Returns:
        Dict with anomaly scores, threshold, ROC-AUC, and predictions
    """
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

    rng = np.random.RandomState(random_state)
    n_normal = data_flat.shape[0]
    n_anomalies = int(n_normal * contamination)

    # Create synthetic anomalies
    anomalies = _create_anomalies(data_flat, n_anomalies, image_shape, rng)

    # Combine normal + anomalous
    X_combined = np.vstack([data_flat, anomalies])
    y_true = np.array([0] * n_normal + [1] * n_anomalies)  # 0=normal, 1=anomaly

    # Fit PCA on normal data only
    engine = PCAEngine(n_components=n_components)
    engine.fit(data_flat)

    # Reconstruct all samples
    reconstructed = engine.reconstruct(X_combined)
    reconstructed = np.clip(reconstructed, 0, 1)

    # Compute reconstruction error as anomaly score
    anomaly_scores = np.mean((X_combined - reconstructed) ** 2, axis=1)

    # Set threshold from normal data distribution
    normal_scores = anomaly_scores[:n_normal]
    threshold = np.percentile(normal_scores, threshold_percentile)

    # Predictions
    y_pred = (anomaly_scores > threshold).astype(int)

    # Metrics
    roc_auc = roc_auc_score(y_true, anomaly_scores)
    precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)
    pr_auc = auc(recall, precision)

    return {
        "anomaly_scores": anomaly_scores,
        "y_true": y_true,
        "y_pred": y_pred,
        "threshold": threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "n_normal": n_normal,
        "n_anomalies": n_anomalies,
        "normal_scores": normal_scores,
        "anomaly_only_scores": anomaly_scores[n_normal:],
    }


def _create_anomalies(
    data_flat: np.ndarray, n_anomalies: int, image_shape: tuple, rng: np.random.RandomState
) -> np.ndarray:
    """Create synthetic anomalies: rotated faces, random noise, and partial occlusions."""
    from scipy.ndimage import rotate

    anomalies = []
    n_each = n_anomalies // 3

    # Type 1: Random noise images
    for _ in range(n_each):
        anomalies.append(rng.random(data_flat.shape[1]))

    # Type 2: Heavily rotated faces
    indices = rng.choice(len(data_flat), n_each, replace=True)
    for idx in indices:
        img = data_flat[idx].reshape(image_shape)
        angle = rng.choice([90, 180, 270])
        rotated = rotate(img, angle, reshape=False, mode="constant", cval=0)
        anomalies.append(rotated.flatten())

    # Type 3: Inverted faces
    remaining = n_anomalies - 2 * n_each
    indices = rng.choice(len(data_flat), remaining, replace=True)
    for idx in indices:
        anomalies.append(1.0 - data_flat[idx])

    return np.array(anomalies)

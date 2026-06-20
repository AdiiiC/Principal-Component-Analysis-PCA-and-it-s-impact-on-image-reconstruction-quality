"""
Information-theoretic analysis of PCA components.
"""

import numpy as np
import pandas as pd
from pca_core import PCAEngine
from pca_core.utils import entropy, mutual_information


def run_information_theory_analysis(
    data_flat: np.ndarray,
    labels: np.ndarray,
    n_components: int = 40,
    n_bins_entropy: int = 50,
    n_bins_mi: int = 20,
) -> pd.DataFrame:
    """
    Compute entropy and mutual information for each principal component.

    Args:
        data_flat: (N, D) flattened images
        labels: (N,) subject labels
        n_components: Number of PCA components
        n_bins_entropy: Bins for entropy estimation
        n_bins_mi: Bins for mutual information estimation

    Returns:
        DataFrame with per-component entropy and MI with labels
    """
    engine = PCAEngine(n_components=n_components)
    engine.fit(data_flat)
    Z = engine.transform(data_flat)

    results = []
    for i in range(n_components):
        pc_coeffs = Z[:, i]

        # Entropy of coefficient distribution
        h = entropy(pc_coeffs, n_bins=n_bins_entropy)

        # Mutual information with subject identity
        mi = mutual_information(pc_coeffs, labels.astype(float), n_bins=n_bins_mi)

        results.append({
            "component": i + 1,
            "entropy": h,
            "mutual_information_with_identity": mi,
            "explained_variance_ratio": engine.explained_variance_ratio_[i],
        })

    return pd.DataFrame(results)

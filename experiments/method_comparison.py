"""
Comparison of PCA methods: Standard, Incremental, Kernel, Sparse, Probabilistic, Robust, 2D.
"""

import numpy as np
import pandas as pd
import time
from pca_core import PCAEngine, IncrementalPCAEngine, KernelPCAEngine, SparsePCAEngine, ProbabilisticPCAEngine
from pca_core.pca_engine import RobustPCA, PCA2D
from pca_core.metrics import compute_batch_metrics


def run_method_comparison(
    data_flat: np.ndarray,
    images: np.ndarray,
    image_shape: tuple = (64, 64),
    n_components: int = 40,
) -> pd.DataFrame:
    """
    Compare all PCA methods on the same data.

    Returns:
        DataFrame with method name, avg metrics, and computation time
    """
    results = []

    # Standard PCA (sklearn)
    t0 = time.time()
    engine = PCAEngine(n_components=n_components, method="sklearn")
    engine.fit(data_flat)
    recon = np.clip(engine.reconstruct(data_flat), 0, 1)
    elapsed = time.time() - t0
    metrics = compute_batch_metrics(data_flat, recon, image_shape)
    results.append({
        "method": "Standard PCA (SVD)",
        "avg_mse": np.mean(metrics["mse"]),
        "avg_psnr": np.mean(metrics["psnr"]),
        "avg_ssim": np.mean(metrics["ssim"]),
        "time_seconds": elapsed,
    })

    # Standard PCA (manual eigendecomposition)
    t0 = time.time()
    engine_manual = PCAEngine(n_components=n_components, method="manual")
    engine_manual.fit(data_flat)
    recon = np.clip(engine_manual.reconstruct(data_flat), 0, 1)
    elapsed = time.time() - t0
    metrics = compute_batch_metrics(data_flat, recon, image_shape)
    results.append({
        "method": "Standard PCA (Eigen)",
        "avg_mse": np.mean(metrics["mse"]),
        "avg_psnr": np.mean(metrics["psnr"]),
        "avg_ssim": np.mean(metrics["ssim"]),
        "time_seconds": elapsed,
    })

    # Incremental PCA
    t0 = time.time()
    ipca = IncrementalPCAEngine(n_components=n_components, batch_size=50)
    ipca.fit(data_flat)
    recon = np.clip(ipca.reconstruct(data_flat), 0, 1)
    elapsed = time.time() - t0
    metrics = compute_batch_metrics(data_flat, recon, image_shape)
    results.append({
        "method": "Incremental PCA",
        "avg_mse": np.mean(metrics["mse"]),
        "avg_psnr": np.mean(metrics["psnr"]),
        "avg_ssim": np.mean(metrics["ssim"]),
        "time_seconds": elapsed,
    })

    # Kernel PCA
    t0 = time.time()
    try:
        kpca = KernelPCAEngine(n_components=min(n_components, 30), kernel="rbf")
        kpca.fit(data_flat)
        recon = np.clip(kpca.reconstruct(data_flat), 0, 1)
        elapsed = time.time() - t0
        metrics = compute_batch_metrics(data_flat, recon, image_shape)
        results.append({
            "method": "Kernel PCA (RBF)",
            "avg_mse": np.mean(metrics["mse"]),
            "avg_psnr": np.mean(metrics["psnr"]),
            "avg_ssim": np.mean(metrics["ssim"]),
            "time_seconds": elapsed,
        })
    except Exception as e:
        results.append({
            "method": "Kernel PCA (RBF)",
            "avg_mse": None,
            "avg_psnr": None,
            "avg_ssim": None,
            "time_seconds": time.time() - t0,
        })

    # Sparse PCA
    t0 = time.time()
    try:
        spca = SparsePCAEngine(n_components=min(n_components, 20), alpha=1.0)
        spca.fit(data_flat)
        recon = np.clip(spca.reconstruct(data_flat), 0, 1)
        elapsed = time.time() - t0
        metrics = compute_batch_metrics(data_flat, recon, image_shape)
        results.append({
            "method": "Sparse PCA",
            "avg_mse": np.mean(metrics["mse"]),
            "avg_psnr": np.mean(metrics["psnr"]),
            "avg_ssim": np.mean(metrics["ssim"]),
            "time_seconds": elapsed,
        })
    except Exception as e:
        results.append({
            "method": "Sparse PCA",
            "avg_mse": None,
            "avg_psnr": None,
            "avg_ssim": None,
            "time_seconds": time.time() - t0,
        })

    # Probabilistic PCA
    t0 = time.time()
    ppca = ProbabilisticPCAEngine(n_components=n_components)
    ppca.fit(data_flat)
    recon = np.clip(ppca.reconstruct(data_flat), 0, 1)
    elapsed = time.time() - t0
    metrics = compute_batch_metrics(data_flat, recon, image_shape)
    results.append({
        "method": "Probabilistic PCA",
        "avg_mse": np.mean(metrics["mse"]),
        "avg_psnr": np.mean(metrics["psnr"]),
        "avg_ssim": np.mean(metrics["ssim"]),
        "time_seconds": elapsed,
    })

    # Robust PCA
    t0 = time.time()
    try:
        rpca = RobustPCA(max_iter=50)
        rpca.fit(data_flat)
        recon = np.clip(rpca.get_low_rank(), 0, 1)
        elapsed = time.time() - t0
        metrics = compute_batch_metrics(data_flat, recon, image_shape)
        results.append({
            "method": "Robust PCA (RPCA)",
            "avg_mse": np.mean(metrics["mse"]),
            "avg_psnr": np.mean(metrics["psnr"]),
            "avg_ssim": np.mean(metrics["ssim"]),
            "time_seconds": elapsed,
        })
    except Exception as e:
        results.append({
            "method": "Robust PCA (RPCA)",
            "avg_mse": None,
            "avg_psnr": None,
            "avg_ssim": None,
            "time_seconds": time.time() - t0,
        })

    # 2D-PCA
    t0 = time.time()
    pca2d = PCA2D(n_components=min(n_components, 30))
    pca2d.fit(images)
    recon_2d = np.clip(pca2d.reconstruct(images), 0, 1)
    elapsed = time.time() - t0
    recon_flat = recon_2d.reshape(data_flat.shape)
    metrics = compute_batch_metrics(data_flat, recon_flat, image_shape)
    results.append({
        "method": "2D-PCA",
        "avg_mse": np.mean(metrics["mse"]),
        "avg_psnr": np.mean(metrics["psnr"]),
        "avg_ssim": np.mean(metrics["ssim"]),
        "time_seconds": elapsed,
    })

    return pd.DataFrame(results)

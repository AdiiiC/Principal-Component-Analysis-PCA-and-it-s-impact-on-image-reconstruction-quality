"""
PCA Denoising Experiment: Use PCA reconstruction as a denoiser.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from pca_core import PCAEngine, add_noise, compute_all_metrics
from pca_core.metrics import compute_batch_metrics


def run_denoising_experiment(
    data_flat: np.ndarray,
    images: np.ndarray,
    image_shape: tuple = (64, 64),
    noise_types: List[str] = None,
    noise_levels: List[float] = None,
    n_components: int = 40,
) -> pd.DataFrame:
    """
    Test PCA as a denoiser across noise types and levels.

    Args:
        data_flat: (N, D) clean flattened images
        images: (N, H, W) clean images
        image_shape: Image dimensions
        noise_types: List of noise types to test
        noise_levels: List of noise intensities
        n_components: Number of PCA components for reconstruction

    Returns:
        DataFrame with denoising results
    """
    if noise_types is None:
        noise_types = ["gaussian", "salt_pepper", "occlusion"]
    if noise_levels is None:
        noise_levels = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    # Fit PCA on clean data
    engine = PCAEngine(n_components=n_components)
    engine.fit(data_flat)

    results = []

    for noise_type in noise_types:
        for noise_level in noise_levels:
            # Add noise
            if noise_type == "occlusion":
                noisy_images = add_noise(images, noise_type=noise_type, noise_level=noise_level)
                noisy_flat = noisy_images.reshape(noisy_images.shape[0], -1)
            else:
                noisy_flat = add_noise(data_flat, noise_type=noise_type, noise_level=noise_level)

            # Denoise via PCA reconstruction
            denoised = engine.reconstruct(noisy_flat)
            denoised = np.clip(denoised, 0, 1)

            # Metrics: compare denoised against ORIGINAL (not noisy)
            metrics = compute_batch_metrics(data_flat, denoised, image_shape)

            # Also compute metrics for noisy images (baseline)
            noisy_metrics = compute_batch_metrics(data_flat, noisy_flat, image_shape)

            results.append({
                "noise_type": noise_type,
                "noise_level": noise_level,
                "noisy_psnr": np.mean(noisy_metrics["psnr"]),
                "noisy_ssim": np.mean(noisy_metrics["ssim"]),
                "denoised_psnr": np.mean(metrics["psnr"]),
                "denoised_ssim": np.mean(metrics["ssim"]),
                "psnr_improvement": np.mean(metrics["psnr"]) - np.mean(noisy_metrics["psnr"]),
                "ssim_improvement": np.mean(metrics["ssim"]) - np.mean(noisy_metrics["ssim"]),
            })

    return pd.DataFrame(results)


def denoise_single_image(
    clean_image: np.ndarray,
    engine: PCAEngine,
    noise_type: str = "gaussian",
    noise_level: float = 0.2,
    image_shape: tuple = (64, 64),
) -> Dict[str, np.ndarray]:
    """
    Denoise a single image. Returns dict with noisy, denoised, and error images.
    """
    clean_flat = clean_image.reshape(1, -1)

    if noise_type == "occlusion":
        noisy = add_noise(clean_image.reshape(1, *image_shape), noise_type=noise_type, noise_level=noise_level)
        noisy_flat = noisy.reshape(1, -1)
    else:
        noisy_flat = add_noise(clean_flat, noise_type=noise_type, noise_level=noise_level)

    denoised = engine.reconstruct(noisy_flat)
    denoised = np.clip(denoised, 0, 1)

    return {
        "clean": clean_image.reshape(image_shape),
        "noisy": noisy_flat.reshape(image_shape),
        "denoised": denoised.reshape(image_shape),
        "error": np.abs(clean_image.reshape(image_shape) - denoised.reshape(image_shape)),
    }

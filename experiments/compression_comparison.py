"""
PCA vs JPEG Compression Comparison.
"""

import numpy as np
import pandas as pd
from io import BytesIO
from typing import List
from PIL import Image
from pca_core import PCAEngine
from pca_core.metrics import compute_batch_metrics
from pca_core.utils import quantize_coefficients, compute_file_size_bytes


def run_compression_comparison(
    data_flat: np.ndarray,
    image_shape: tuple = (64, 64),
    n_components_list: List[int] = None,
    bits_per_coeff: List[int] = None,
    jpeg_qualities: List[int] = None,
) -> pd.DataFrame:
    """
    Compare PCA compression vs JPEG at equivalent compression ratios.

    Args:
        data_flat: (N, D) flattened images in [0, 1]
        image_shape: Image dimensions
        n_components_list: PCA component counts
        bits_per_coeff: Quantization bit depths
        jpeg_qualities: JPEG quality levels

    Returns:
        DataFrame with method, file_size, ssim, psnr for each configuration
    """
    if n_components_list is None:
        n_components_list = [5, 10, 20, 30, 40, 60, 80]
    if bits_per_coeff is None:
        bits_per_coeff = [8]
    if jpeg_qualities is None:
        jpeg_qualities = [5, 10, 20, 30, 50, 70, 90]

    n_samples, n_features = data_flat.shape
    original_size = n_features  # per image, in bytes (assuming 8-bit)
    results = []

    # PCA compression at different component counts and quantizations
    for k in n_components_list:
        engine = PCAEngine(n_components=k)
        engine.fit(data_flat)

        for bits in bits_per_coeff:
            # Transform to PCA space
            Z = engine.transform(data_flat)

            # Quantize coefficients
            Z_quantized = quantize_coefficients(Z, n_bits=bits)

            # Reconstruct from quantized coefficients
            reconstructed = engine.inverse_transform(Z_quantized)
            reconstructed = np.clip(reconstructed, 0, 1)

            # Estimate compressed size per image
            per_image_size = compute_file_size_bytes(k, n_features, 1, bits)

            # Metrics
            metrics = compute_batch_metrics(data_flat, reconstructed, image_shape)

            results.append({
                "method": f"PCA-{k}c-{bits}bit",
                "n_components": k,
                "bits": bits,
                "compressed_size_bytes": per_image_size,
                "compression_ratio": original_size / per_image_size,
                "avg_psnr": np.mean(metrics["psnr"]),
                "avg_ssim": np.mean(metrics["ssim"]),
                "avg_mse": np.mean(metrics["mse"]),
            })

    # JPEG compression at different quality levels
    for quality in jpeg_qualities:
        psnr_list, ssim_list, sizes = [], [], []

        for i in range(min(n_samples, 100)):  # Limit for speed
            img = (data_flat[i].reshape(image_shape) * 255).astype(np.uint8)
            pil_img = Image.fromarray(img, mode="L")

            # Compress to JPEG
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG", quality=quality)
            jpeg_size = buffer.tell()
            sizes.append(jpeg_size)

            # Decompress
            buffer.seek(0)
            decompressed = np.array(Image.open(buffer), dtype=np.float64) / 255.0

            from pca_core.metrics import compute_all_metrics
            m = compute_all_metrics(data_flat[i], decompressed.flatten(), image_shape)
            psnr_list.append(m["psnr"])
            ssim_list.append(m["ssim"])

        results.append({
            "method": f"JPEG-q{quality}",
            "n_components": None,
            "bits": None,
            "compressed_size_bytes": np.mean(sizes),
            "compression_ratio": original_size / np.mean(sizes),
            "avg_psnr": np.mean(psnr_list),
            "avg_ssim": np.mean(ssim_list),
            "avg_mse": None,
        })

    return pd.DataFrame(results)

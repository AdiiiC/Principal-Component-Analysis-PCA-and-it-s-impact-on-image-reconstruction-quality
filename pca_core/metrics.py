"""
Image quality metrics for PCA reconstruction evaluation.
"""

import numpy as np
from typing import Dict, Optional
from skimage.metrics import structural_similarity as ssim


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute Mean Squared Error between original and reconstructed images."""
    return float(np.mean((original - reconstructed) ** 2))


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray, max_val: Optional[float] = None) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        original: Original image (any shape)
        reconstructed: Reconstructed image (same shape)
        max_val: Maximum pixel value. Auto-detected if None.
    """
    if max_val is None:
        max_val = max(original.max(), 1.0)

    mse = compute_mse(original, reconstructed)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(max_val / np.sqrt(mse)))


def compute_ssim(original: np.ndarray, reconstructed: np.ndarray, data_range: Optional[float] = None) -> float:
    """
    Compute Structural Similarity Index (SSIM).

    Args:
        original: Original image (H, W) or flattened
        reconstructed: Reconstructed image (same shape)
        data_range: Range of pixel values. Auto-detected if None.
    """
    if original.ndim == 1:
        side = int(np.sqrt(original.shape[0]))
        original = original.reshape(side, side)
        reconstructed = reconstructed.reshape(side, side)

    if data_range is None:
        data_range = max(original.max(), 1.0) - min(original.min(), 0.0)

    return float(ssim(original, reconstructed, data_range=data_range))


def compute_all_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    image_shape: tuple = (64, 64),
) -> Dict[str, float]:
    """
    Compute all quality metrics at once.

    Returns dict with keys: 'mse', 'psnr', 'ssim'
    """
    orig_img = original.reshape(image_shape) if original.ndim == 1 else original
    recon_img = reconstructed.reshape(image_shape) if reconstructed.ndim == 1 else reconstructed

    # Clip to valid range
    recon_img = np.clip(recon_img, 0, 1)

    return {
        "mse": compute_mse(orig_img, recon_img),
        "psnr": compute_psnr(orig_img, recon_img, max_val=1.0),
        "ssim": compute_ssim(orig_img, recon_img, data_range=1.0),
    }


def compute_batch_metrics(
    originals: np.ndarray,
    reconstructed: np.ndarray,
    image_shape: tuple = (64, 64),
) -> Dict[str, np.ndarray]:
    """
    Compute metrics for a batch of images.

    Returns dict with arrays of per-image metrics.
    """
    n = originals.shape[0]
    mse_vals = np.zeros(n)
    psnr_vals = np.zeros(n)
    ssim_vals = np.zeros(n)

    for i in range(n):
        metrics = compute_all_metrics(originals[i], reconstructed[i], image_shape)
        mse_vals[i] = metrics["mse"]
        psnr_vals[i] = metrics["psnr"]
        ssim_vals[i] = metrics["ssim"]

    return {"mse": mse_vals, "psnr": psnr_vals, "ssim": ssim_vals}


def compute_compression_ratio(
    n_features: int, n_components: int, n_samples: int
) -> float:
    """
    Compute compression ratio.

    Original storage: n_samples * n_features
    Compressed storage: n_samples * n_components + n_components * n_features (codes + basis)
    """
    original_size = n_samples * n_features
    compressed_size = n_samples * n_components + n_components * n_features
    return original_size / compressed_size


def compute_perceptual_distance(
    original: np.ndarray,
    reconstructed: np.ndarray,
    device: str = "cpu",
) -> float:
    """
    Compute perceptual distance using VGG16 features.

    Args:
        original: (H, W) grayscale image normalized [0, 1]
        reconstructed: (H, W) grayscale image normalized [0, 1]
        device: 'cpu' or 'cuda'

    Returns:
        L2 distance in VGG16 feature space
    """
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms

    # Load pretrained VGG16
    vgg = models.vgg16(weights="IMAGENET1K_V1").features[:16].to(device).eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def to_tensor(img):
        if img.ndim == 2:
            img_3ch = np.stack([img, img, img], axis=-1)
        else:
            img_3ch = img
        img_uint8 = (np.clip(img_3ch, 0, 1) * 255).astype(np.uint8)
        return transform(img_uint8).unsqueeze(0).to(device)

    with torch.no_grad():
        feat_orig = vgg(to_tensor(original))
        feat_recon = vgg(to_tensor(reconstructed))

    return float(torch.nn.functional.mse_loss(feat_orig, feat_recon).item())

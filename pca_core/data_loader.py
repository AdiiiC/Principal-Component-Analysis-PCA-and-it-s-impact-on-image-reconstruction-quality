"""
Data loading utilities for PCA image reconstruction.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_olivetti(shuffle: bool = True, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the Olivetti Faces dataset.

    Returns:
        images: (400, 64, 64) array of face images
        data_flat: (400, 4096) flattened image vectors
        labels: (400,) subject IDs (0-39)
    """
    from sklearn.datasets import fetch_olivetti_faces

    dataset = fetch_olivetti_faces(shuffle=shuffle, random_state=random_state)
    images = dataset.images
    data_flat = dataset.data
    labels = dataset.target
    return images, data_flat, labels


def load_custom_images(
    folder_path: str,
    target_size: Tuple[int, int] = (64, 64),
    grayscale: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load custom images from a folder.

    Args:
        folder_path: Path to folder containing images
        target_size: Resize all images to this size
        grayscale: Convert to grayscale if True

    Returns:
        images: (N, H, W) array
        data_flat: (N, H*W) flattened array
    """
    from PIL import Image

    folder = Path(folder_path)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = sorted(f for f in folder.iterdir() if f.suffix.lower() in extensions)

    if not image_files:
        raise ValueError(f"No image files found in {folder_path}")

    images_list = []
    for img_path in image_files:
        img = Image.open(img_path)
        if grayscale:
            img = img.convert("L")
        img = img.resize(target_size, Image.LANCZOS)
        img_array = np.array(img, dtype=np.float64) / 255.0
        images_list.append(img_array)

    images = np.array(images_list)
    data_flat = images.reshape(images.shape[0], -1)
    return images, data_flat


def add_noise(
    images: np.ndarray,
    noise_type: str = "gaussian",
    noise_level: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Add noise to images for denoising experiments.

    Args:
        images: (N, H, W) or (N, D) array of images
        noise_type: 'gaussian', 'salt_pepper', or 'occlusion'
        noise_level: Intensity of noise (0 to 1)
        random_state: Random seed

    Returns:
        Noisy images with same shape as input
    """
    rng = np.random.RandomState(random_state)
    noisy = images.copy()

    if noise_type == "gaussian":
        noise = rng.normal(0, noise_level, images.shape)
        noisy = np.clip(images + noise, 0, 1)

    elif noise_type == "salt_pepper":
        mask = rng.random(images.shape)
        noisy[mask < noise_level / 2] = 0.0
        noisy[mask > 1 - noise_level / 2] = 1.0

    elif noise_type == "occlusion":
        if images.ndim == 2:
            raise ValueError("Occlusion noise requires 3D images (N, H, W)")
        n, h, w = images.shape
        block_size = int(h * noise_level)
        for i in range(n):
            y = rng.randint(0, h - block_size)
            x = rng.randint(0, w - block_size)
            noisy[i, y : y + block_size, x : x + block_size] = 0.0

    else:
        raise ValueError(f"Unknown noise type: {noise_type}. Use 'gaussian', 'salt_pepper', or 'occlusion'")

    return noisy


def load_lfw(
    n_samples: Optional[int] = 200,
    target_size: Tuple[int, int] = (64, 64),
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load LFW (Labeled Faces in the Wild) dataset for cross-dataset testing.

    Returns:
        images: (N, 64, 64) array
        data_flat: (N, 4096) flattened array
        labels: (N,) target labels
    """
    from sklearn.datasets import fetch_lfw_people
    from PIL import Image

    lfw = fetch_lfw_people(min_faces_per_person=5, resize=0.4)
    images_raw = lfw.images
    labels = lfw.target

    if n_samples and n_samples < len(images_raw):
        rng = np.random.RandomState(random_state)
        indices = rng.choice(len(images_raw), n_samples, replace=False)
        images_raw = images_raw[indices]
        labels = labels[indices]

    # Resize to target size and normalize
    images_list = []
    for img in images_raw:
        pil_img = Image.fromarray((img).astype(np.uint8))
        pil_img = pil_img.resize(target_size, Image.LANCZOS)
        images_list.append(np.array(pil_img, dtype=np.float64) / 255.0)

    images = np.array(images_list)
    data_flat = images.reshape(images.shape[0], -1)
    return images, data_flat, labels

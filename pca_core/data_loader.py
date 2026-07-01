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


def prefilter_noisy(
    images: np.ndarray,
    noise_type: str,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Light spatial pre-filter applied before PCA projection to improve denoising.

    PCA reconstruction is a least-squares (L2) projection, so it is extremely
    sensitive to impulsive outliers. Salt-&-pepper pixels are extreme outliers
    that dominate the projection and corrupt the whole reconstruction. A small
    median filter removes those spikes while preserving edges, letting the PCA
    subspace recover far more structure (large PSNR/SSIM gain).

    Args:
        images: (N, D) flattened images or (N, H, W) image stack in [0, 1].
        noise_type: 'gaussian', 'salt_pepper', or 'occlusion'.
        image_shape: (H, W) used to reshape flattened rows.

    Returns:
        Pre-filtered images with the same shape as the input.
    """
    from scipy.ndimage import median_filter

    flat_input = images.ndim == 2 and images.shape[1] == int(np.prod(image_shape))
    stack = images.reshape(-1, *image_shape) if flat_input else images

    # A 3x3 median only helps when the image is large enough to keep structure
    # after filtering. On tiny images (e.g. 8x8 digits) it destroys detail and
    # hurts every metric, so skip it there.
    large_enough = min(image_shape) >= 16
    impulsive = noise_type in ("salt_pepper", "occlusion")

    if impulsive and large_enough:
        # Median cleanly removes impulsive spikes/blocks before the L2 projection.
        filtered = np.stack([median_filter(im, size=3) for im in stack])
    else:
        # Gaussian noise lives in the low-variance directions PCA already
        # discards, so the raw PCA projection is (near) optimal - leave it be.
        filtered = stack

    filtered = np.clip(filtered, 0, 1)
    return filtered.reshape(images.shape) if flat_input else filtered


def load_lfw(
    n_samples: Optional[int] = 200,
    target_size: Tuple[int, int] = (64, 64),
    min_faces_per_person: int = 20,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load LFW (Labeled Faces in the Wild) dataset for cross-dataset testing.

    scikit-learn returns LFW pixels already scaled to [0, 1], so we rescale to
    uint8 for PIL resizing and then back to [0, 1].

    Returns:
        images: (N, 64, 64) array in [0, 1]
        data_flat: (N, 4096) flattened array
        labels: (N,) target labels
    """
    from sklearn.datasets import fetch_lfw_people
    from PIL import Image

    lfw = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=0.5)
    images_raw = lfw.images  # float32 in [0, 1]
    labels = lfw.target

    if n_samples and n_samples < len(images_raw):
        rng = np.random.RandomState(random_state)
        indices = rng.choice(len(images_raw), n_samples, replace=False)
        images_raw = images_raw[indices]
        labels = labels[indices]

    # Resize to target size and normalize back to [0, 1]
    images_list = []
    for img in images_raw:
        pil_img = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        pil_img = pil_img.resize(target_size, Image.LANCZOS)
        images_list.append(np.array(pil_img, dtype=np.float64) / 255.0)

    images = np.array(images_list)
    data_flat = images.reshape(images.shape[0], -1)
    return images, data_flat, labels


def load_digits_dataset(
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the scikit-learn handwritten digits dataset (8x8 grayscale, 0-9).

    Pixel intensities are natively in [0, 16]; they are rescaled to [0, 1].

    Returns:
        images: (1797, 8, 8) array in [0, 1]
        data_flat: (1797, 64) flattened array
        labels: (1797,) digit labels 0-9
    """
    from sklearn.datasets import load_digits

    dataset = load_digits()
    images = dataset.images.astype(np.float64) / 16.0
    data_flat = dataset.data.astype(np.float64) / 16.0
    labels = dataset.target
    return images, data_flat, labels


# Registry of datasets exposed to the app / API.
IMAGE_DATASETS = {
    "olivetti": {
        "label": "Olivetti Faces",
        "shape": (64, 64),
        "description": "400 grayscale portraits of 40 subjects (AT&T Laboratories Cambridge).",
        "kind": "faces",
        "requires_download": False,
    },
    "lfw": {
        "label": "Labeled Faces in the Wild",
        "shape": (64, 64),
        "description": "Cropped photographs of public figures, resized to 64×64.",
        "kind": "faces",
        "requires_download": True,
    },
    "digits": {
        "label": "Handwritten Digits",
        "shape": (8, 8),
        "description": "1,797 downsampled 8×8 scans of the digits 0–9.",
        "kind": "digits",
        "requires_download": False,
    },
}


def load_dataset(
    name: str,
    max_samples: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Load one of the registered datasets by name.

    Args:
        name: One of the keys in IMAGE_DATASETS.
        max_samples: Optionally cap the number of samples for responsiveness.
        random_state: Seed for shuffling / subsampling.

    Returns:
        images, data_flat, labels, image_shape
    """
    key = name.lower()
    if key not in IMAGE_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(IMAGE_DATASETS)}")

    if key == "olivetti":
        images, data_flat, labels = load_olivetti(random_state=random_state)
    elif key == "digits":
        images, data_flat, labels = load_digits_dataset(random_state=random_state)
    elif key == "lfw":
        images, data_flat, labels = load_lfw(
            n_samples=max_samples, random_state=random_state
        )
        max_samples = None  # already applied during load
    else:  # pragma: no cover - guarded above
        raise ValueError(f"Unknown dataset '{name}'")

    shape = IMAGE_DATASETS[key]["shape"]

    if max_samples and len(data_flat) > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(data_flat), max_samples, replace=False)
        images, data_flat, labels = images[idx], data_flat[idx], labels[idx]

    return images, data_flat, labels, shape

"""
Face Arithmetic in PCA Latent Space.
Interpolation, averaging, and outlier synthesis.
"""

import numpy as np
from typing import List, Tuple
from pca_core import PCAEngine


def run_face_arithmetic(
    data_flat: np.ndarray,
    labels: np.ndarray,
    image_shape: tuple = (64, 64),
    n_components: int = 40,
) -> dict:
    """
    Perform face arithmetic operations in PCA latent space.

    Returns:
        Dict with interpolation frames, averaged faces, and outlier info
    """
    engine = PCAEngine(n_components=n_components)
    engine.fit(data_flat)
    Z = engine.transform(data_flat)

    results = {}

    # 1. Face interpolation between two subjects
    results["interpolations"] = interpolate_faces(
        data_flat, Z, engine, labels, image_shape
    )

    # 2. Canonical face per subject (average in latent space)
    results["canonical_faces"] = compute_canonical_faces(
        data_flat, Z, engine, labels, image_shape
    )

    # 3. Outlier face (furthest from mean in PCA space)
    results["outlier"] = find_outlier_face(data_flat, Z, engine, image_shape)

    # 4. Random walk in latent space
    results["random_walk"] = random_latent_walk(Z, engine, image_shape)

    return results


def interpolate_faces(
    data_flat: np.ndarray,
    Z: np.ndarray,
    engine: PCAEngine,
    labels: np.ndarray,
    image_shape: tuple,
    n_steps: int = 10,
    face_idx_a: int = 0,
    face_idx_b: int = 10,
) -> dict:
    """
    Linearly interpolate between two faces in latent space.
    """
    z_a = Z[face_idx_a]
    z_b = Z[face_idx_b]

    alphas = np.linspace(0, 1, n_steps)
    interpolated_images = []

    for alpha in alphas:
        z_interp = (1 - alpha) * z_a + alpha * z_b
        recon = engine.inverse_transform(z_interp.reshape(1, -1))
        recon_img = np.clip(recon.reshape(image_shape), 0, 1)
        interpolated_images.append(recon_img)

    return {
        "face_a": data_flat[face_idx_a].reshape(image_shape),
        "face_b": data_flat[face_idx_b].reshape(image_shape),
        "interpolations": interpolated_images,
        "alphas": alphas,
    }


def compute_canonical_faces(
    data_flat: np.ndarray,
    Z: np.ndarray,
    engine: PCAEngine,
    labels: np.ndarray,
    image_shape: tuple,
) -> dict:
    """
    Compute canonical (average) face for each subject by averaging latent vectors.
    """
    unique_labels = np.unique(labels)
    canonical_faces = {}

    for label in unique_labels:
        mask = labels == label
        z_mean = Z[mask].mean(axis=0)
        recon = engine.inverse_transform(z_mean.reshape(1, -1))
        canonical_faces[int(label)] = np.clip(recon.reshape(image_shape), 0, 1)

    return canonical_faces


def find_outlier_face(
    data_flat: np.ndarray,
    Z: np.ndarray,
    engine: PCAEngine,
    image_shape: tuple,
) -> dict:
    """
    Find the face furthest from the mean in PCA space.
    """
    z_mean = Z.mean(axis=0)
    distances = np.linalg.norm(Z - z_mean, axis=1)
    outlier_idx = np.argmax(distances)

    return {
        "index": int(outlier_idx),
        "distance": float(distances[outlier_idx]),
        "image": data_flat[outlier_idx].reshape(image_shape),
        "mean_face": engine.inverse_transform(z_mean.reshape(1, -1)).reshape(image_shape),
    }


def random_latent_walk(
    Z: np.ndarray,
    engine: PCAEngine,
    image_shape: tuple,
    n_steps: int = 20,
    step_size: float = 0.5,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Random walk in latent space starting from the mean, generating novel faces.
    """
    rng = np.random.RandomState(seed)
    z_current = Z.mean(axis=0).copy()
    z_std = Z.std(axis=0)

    walk_images = []
    for _ in range(n_steps):
        direction = rng.randn(Z.shape[1]) * z_std
        z_current = z_current + step_size * direction
        recon = engine.inverse_transform(z_current.reshape(1, -1))
        walk_images.append(np.clip(recon.reshape(image_shape), 0, 1))

    return walk_images

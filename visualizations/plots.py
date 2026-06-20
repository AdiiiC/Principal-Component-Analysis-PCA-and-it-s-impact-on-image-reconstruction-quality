"""
Comprehensive plotting functions for PCA image reconstruction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple


def plot_reconstruction_grid(
    originals: np.ndarray,
    reconstructed: np.ndarray,
    image_shape: tuple = (64, 64),
    n_faces: int = 5,
    title: str = "Original vs Reconstructed",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot side-by-side original and reconstructed faces."""
    fig, axes = plt.subplots(n_faces, 2, figsize=(4, 2 * n_faces),
                             subplot_kw={"xticks": [], "yticks": []},
                             gridspec_kw=dict(hspace=0.05, wspace=0.05))

    for i in range(n_faces):
        orig_img = originals[i].reshape(image_shape)
        recon_img = np.clip(reconstructed[i].reshape(image_shape), 0, 1)
        axes[i, 0].imshow(orig_img, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow(recon_img, cmap="gray", vmin=0, vmax=1)

    axes[0, 0].set_title("Original", fontsize=12)
    axes[0, 1].set_title("Reconstructed", fontsize=12)
    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_eigenfaces(
    components: np.ndarray,
    image_shape: tuple = (64, 64),
    n_eigenfaces: int = 16,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the top eigenfaces (principal components)."""
    n_cols = 4
    n_rows = (n_eigenfaces + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows),
                             subplot_kw={"xticks": [], "yticks": []})
    axes = axes.flatten()

    for i in range(n_eigenfaces):
        eigenface = components[i].reshape(image_shape)
        axes[i].imshow(eigenface, cmap="seismic", interpolation="nearest")
        axes[i].set_title(f"PC {i+1}", fontsize=10)

    for i in range(n_eigenfaces, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Eigenfaces (Principal Components)", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_mean_face(
    mean_face: np.ndarray,
    image_shape: tuple = (64, 64),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the mean face."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(mean_face.reshape(image_shape), cmap="gray")
    ax.set_title("Mean Face", fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_scree(
    explained_variance_ratio: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scree plot showing individual explained variance per component."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, len(explained_variance_ratio) + 1)
    ax.bar(x, explained_variance_ratio, alpha=0.7, color="steelblue")
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Explained Variance Ratio", fontsize=12)
    ax.set_title("Scree Plot", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cumulative_variance(
    explained_variance_ratio: np.ndarray,
    thresholds: List[float] = [0.90, 0.95, 0.99],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Cumulative explained variance with threshold lines."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cumulative = np.cumsum(explained_variance_ratio)
    x = np.arange(1, len(cumulative) + 1)
    ax.plot(x, cumulative, "b-o", markersize=3, linewidth=2)

    colors = ["green", "orange", "red"]
    for thresh, color in zip(thresholds, colors):
        ax.axhline(y=thresh, color=color, linestyle="--", alpha=0.7, label=f"{thresh*100:.0f}%")
        # Find component where threshold is reached
        idx = np.searchsorted(cumulative, thresh)
        if idx < len(cumulative):
            ax.axvline(x=idx + 1, color=color, linestyle=":", alpha=0.5)
            ax.annotate(f"k={idx+1}", xy=(idx + 1, thresh), fontsize=9, color=color)

    ax.set_xlabel("Number of Components", fontsize=12)
    ax.set_ylabel("Cumulative Explained Variance", fontsize=12)
    ax.set_title("Cumulative Explained Variance Ratio", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_component_sweep_metrics(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot MSE, PSNR, SSIM, and compression ratio vs components."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MSE
    axes[0, 0].plot(results_df["n_components"], results_df["avg_mse"], "r-o", markersize=4)
    axes[0, 0].set_xlabel("Number of Components")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].set_title("Mean Squared Error")
    axes[0, 0].grid(True, alpha=0.3)

    # PSNR
    axes[0, 1].plot(results_df["n_components"], results_df["avg_psnr"], "b-o", markersize=4)
    axes[0, 1].set_xlabel("Number of Components")
    axes[0, 1].set_ylabel("PSNR (dB)")
    axes[0, 1].set_title("Peak Signal-to-Noise Ratio")
    axes[0, 1].grid(True, alpha=0.3)

    # SSIM
    axes[1, 0].plot(results_df["n_components"], results_df["avg_ssim"], "g-o", markersize=4)
    axes[1, 0].set_xlabel("Number of Components")
    axes[1, 0].set_ylabel("SSIM")
    axes[1, 0].set_title("Structural Similarity Index")
    axes[1, 0].grid(True, alpha=0.3)

    # Compression Ratio
    axes[1, 1].plot(results_df["n_components"], results_df["compression_ratio"], "m-o", markersize=4)
    axes[1, 1].set_xlabel("Number of Components")
    axes[1, 1].set_ylabel("Compression Ratio")
    axes[1, 1].set_title("Compression Ratio")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Reconstruction Quality vs Number of Components", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_error_heatmap(
    original: np.ndarray,
    reconstructed: np.ndarray,
    image_shape: tuple = (64, 64),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot per-pixel reconstruction error heatmap."""
    orig_img = original.reshape(image_shape)
    recon_img = np.clip(reconstructed.reshape(image_shape), 0, 1)
    error = np.abs(orig_img - recon_img)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(recon_img, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    im = axes[2].imshow(error, cmap="hot", vmin=0, vmax=error.max())
    axes[2].set_title("Pixel Error Map")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pca_scatter_2d(
    Z: np.ndarray,
    labels: np.ndarray,
    pc_x: int = 0,
    pc_y: int = 1,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """2D scatter plot of PCA projections colored by subject."""
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        ax.scatter(Z[mask, pc_x], Z[mask, pc_y], alpha=0.6, s=30, label=f"S{label}")

    ax.set_xlabel(f"PC{pc_x+1}", fontsize=12)
    ax.set_ylabel(f"PC{pc_y+1}", fontsize=12)
    ax.set_title(f"PCA Projection (PC{pc_x+1} vs PC{pc_y+1})", fontsize=14)
    ax.grid(True, alpha=0.3)

    if len(unique_labels) <= 15:
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pca_scatter_3d(
    Z: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """3D scatter plot of first 3 PCA components."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        ax.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2], alpha=0.6, s=20, label=f"S{label}")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("3D PCA Projection", fontsize=14)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_denoising_comparison(
    results: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot clean, noisy, denoised, and error images."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ["Clean", "Noisy", "Denoised", "Error Map"]
    cmaps = ["gray", "gray", "gray", "hot"]
    keys = ["clean", "noisy", "denoised", "error"]

    for ax, title, cmap, key in zip(axes, titles, cmaps, keys):
        ax.imshow(results[key], cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_method_comparison_bar(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing PCA methods."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    methods = results_df["method"]
    x = np.arange(len(methods))

    # PSNR
    axes[0].barh(x, results_df["avg_psnr"].fillna(0), color="steelblue", alpha=0.8)
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(methods, fontsize=9)
    axes[0].set_xlabel("PSNR (dB)")
    axes[0].set_title("PSNR Comparison")

    # SSIM
    axes[1].barh(x, results_df["avg_ssim"].fillna(0), color="seagreen", alpha=0.8)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(methods, fontsize=9)
    axes[1].set_xlabel("SSIM")
    axes[1].set_title("SSIM Comparison")

    # Time
    axes[2].barh(x, results_df["time_seconds"].fillna(0), color="coral", alpha=0.8)
    axes[2].set_yticks(x)
    axes[2].set_yticklabels(methods, fontsize=9)
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_title("Computation Time")

    fig.suptitle("PCA Method Comparison", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_face_recognition_accuracy(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot face recognition accuracy vs components."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df["n_components"], results_df["accuracy"], "b-o", label="Accuracy", markersize=5)
    ax.plot(results_df["n_components"], results_df["f1_macro"], "r--s", label="F1 (macro)", markersize=5)

    ax.set_xlabel("Number of Components", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Face Recognition Performance vs PCA Components", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_anomaly_scores(
    results: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot anomaly score distributions and ROC."""
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution of scores
    axes[0].hist(results["normal_scores"], bins=30, alpha=0.7, label="Normal", color="blue", density=True)
    axes[0].hist(results["anomaly_only_scores"], bins=30, alpha=0.7, label="Anomaly", color="red", density=True)
    axes[0].axvline(results["threshold"], color="black", linestyle="--", label=f"Threshold")
    axes[0].set_xlabel("Reconstruction Error")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Anomaly Score Distribution")
    axes[0].legend()

    # ROC curve
    fpr, tpr, _ = roc_curve(results["y_true"], results["anomaly_scores"])
    axes[1].plot(fpr, tpr, "b-", linewidth=2, label=f'ROC (AUC={results["roc_auc"]:.3f})')
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_compression_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot PCA vs JPEG: file size vs quality."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pca_mask = results_df["method"].str.startswith("PCA")
    jpeg_mask = results_df["method"].str.startswith("JPEG")

    # SSIM vs file size
    if pca_mask.any():
        axes[0].scatter(results_df.loc[pca_mask, "compressed_size_bytes"],
                       results_df.loc[pca_mask, "avg_ssim"], c="blue", s=80, label="PCA", alpha=0.8)
    if jpeg_mask.any():
        axes[0].scatter(results_df.loc[jpeg_mask, "compressed_size_bytes"],
                       results_df.loc[jpeg_mask, "avg_ssim"], c="red", s=80, label="JPEG", alpha=0.8)
    axes[0].set_xlabel("Compressed Size (bytes)")
    axes[0].set_ylabel("SSIM")
    axes[0].set_title("Quality vs Size")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PSNR vs file size
    if pca_mask.any():
        axes[1].scatter(results_df.loc[pca_mask, "compressed_size_bytes"],
                       results_df.loc[pca_mask, "avg_psnr"], c="blue", s=80, label="PCA", alpha=0.8)
    if jpeg_mask.any():
        axes[1].scatter(results_df.loc[jpeg_mask, "compressed_size_bytes"],
                       results_df.loc[jpeg_mask, "avg_psnr"], c="red", s=80, label="JPEG", alpha=0.8)
    axes[1].set_xlabel("Compressed Size (bytes)")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("PSNR vs Size")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("PCA vs JPEG Compression", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_information_theory(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot entropy and mutual information per component."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(results_df["component"], results_df["entropy"], color="steelblue", alpha=0.8)
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Entropy (bits)")
    axes[0].set_title("Coefficient Entropy per Component")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(results_df["component"], results_df["mutual_information_with_identity"],
               color="coral", alpha=0.8)
    axes[1].set_xlabel("Component")
    axes[1].set_ylabel("MI (bits)")
    axes[1].set_title("Mutual Information with Subject Identity")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Information-Theoretic Analysis", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cross_dataset(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot cross-dataset generalization results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(results_df["n_components"], results_df["train_psnr"], "b-o", label="Train (Olivetti)", markersize=4)
    axes[0].plot(results_df["n_components"], results_df["test_psnr"], "r-s", label="Test (LFW)", markersize=4)
    axes[0].set_xlabel("Number of Components")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR: Same vs Cross Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results_df["n_components"], results_df["train_ssim"], "b-o", label="Train (Olivetti)", markersize=4)
    axes[1].plot(results_df["n_components"], results_df["test_ssim"], "r-s", label="Test (LFW)", markersize=4)
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM: Same vs Cross Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Cross-Dataset Generalization", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_sparse_vs_dense_components(
    dense_components: np.ndarray,
    sparse_components: np.ndarray,
    image_shape: tuple = (64, 64),
    n_show: int = 5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side comparison of dense vs sparse PCA components."""
    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6),
                             subplot_kw={"xticks": [], "yticks": []})

    for i in range(n_show):
        axes[0, i].imshow(dense_components[i].reshape(image_shape), cmap="seismic")
        axes[0, i].set_title(f"Dense PC{i+1}", fontsize=9)
        axes[1, i].imshow(sparse_components[i].reshape(image_shape), cmap="seismic")
        axes[1, i].set_title(f"Sparse PC{i+1}", fontsize=9)

    axes[0, 0].set_ylabel("Standard PCA", fontsize=11)
    axes[1, 0].set_ylabel("Sparse PCA", fontsize=11)
    fig.suptitle("Dense vs Sparse Components (Interpretability)", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_uncertainty_map(
    original: np.ndarray,
    reconstructed: np.ndarray,
    uncertainty: np.ndarray,
    image_shape: tuple = (64, 64),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot original, reconstruction, and uncertainty heatmap."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original.reshape(image_shape), cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(np.clip(reconstructed.reshape(image_shape), 0, 1), cmap="gray")
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    im = axes[2].imshow(uncertainty.reshape(image_shape), cmap="inferno")
    axes[2].set_title("Uncertainty Map")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    fig.suptitle("Probabilistic PCA - Reconstruction Uncertainty", fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_face_interpolation(
    interpolation_data: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot face interpolation sequence."""
    images = interpolation_data["interpolations"]
    n = len(images)
    fig, axes = plt.subplots(1, n + 2, figsize=(2 * (n + 2), 2.5),
                             subplot_kw={"xticks": [], "yticks": []})

    axes[0].imshow(interpolation_data["face_a"], cmap="gray")
    axes[0].set_title("Face A", fontsize=9)

    for i, img in enumerate(images):
        axes[i + 1].imshow(img, cmap="gray")
        axes[i + 1].set_title(f"α={interpolation_data['alphas'][i]:.1f}", fontsize=8)

    axes[-1].imshow(interpolation_data["face_b"], cmap="gray")
    axes[-1].set_title("Face B", fontsize=9)

    fig.suptitle("Face Interpolation in PCA Latent Space", fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_canonical_faces(
    canonical_faces: dict,
    n_show: int = 10,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot canonical (averaged) faces per subject."""
    subjects = sorted(canonical_faces.keys())[:n_show]
    n_cols = min(5, len(subjects))
    n_rows = (len(subjects) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows),
                             subplot_kw={"xticks": [], "yticks": []})
    axes = np.atleast_2d(axes).flatten()

    for i, subj in enumerate(subjects):
        axes[i].imshow(canonical_faces[subj], cmap="gray")
        axes[i].set_title(f"Subject {subj}", fontsize=9)

    for i in range(len(subjects), len(axes)):
        axes[i].axis("off")

    fig.suptitle("Canonical Faces (Latent Space Average)", fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_incremental_convergence(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot how incremental PCA quality converges over batches."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(results_df["n_samples_seen"], results_df["avg_psnr"], "b-o", markersize=5)
    axes[0].set_xlabel("Samples Seen")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR Convergence")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results_df["n_samples_seen"], results_df["avg_ssim"], "g-o", markersize=5)
    axes[1].set_xlabel("Samples Seen")
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM Convergence")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Incremental PCA - Quality vs Data Seen", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

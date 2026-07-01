"""
PCA Engine implementations: Standard, Incremental, Kernel, Sparse, Probabilistic, Robust, 2D.
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler


class PCAEngine:
    """
    Standard PCA with manual eigendecomposition and sklearn SVD backend.

    Supports both custom implementation (for educational purposes) and
    sklearn-backed implementation (for production use).
    """

    def __init__(
        self,
        n_components: int = 40,
        method: str = "sklearn",
        scaling: str = "center",
    ):
        """
        Args:
            n_components: Number of principal components to retain
            method: 'manual' for eigendecomposition, 'sklearn' for SVD-based
            scaling: Preprocessing applied before PCA:
                - 'center'   : subtract the per-pixel mean only (default). This makes
                               PCA the MSE-optimal rank-k linear reconstruction
                               (Eckart-Young), giving the best PSNR/MSE for a given k.
                - 'standard' : subtract mean and divide by per-pixel std (whitening
                               of inputs). Kept for backward compatibility; tends to
                               waste components on low-variance pixels.
                - 'none'     : no centering or scaling.
        """
        self.n_components = n_components
        self.method = method
        self.scaling = scaling
        if scaling == "standard":
            self.scaler = StandardScaler()
        elif scaling == "center":
            self.scaler = StandardScaler(with_mean=True, with_std=False)
        elif scaling == "none":
            self.scaler = StandardScaler(with_mean=False, with_std=False)
        else:
            raise ValueError(
                f"Unknown scaling '{scaling}'. Use 'center', 'standard', or 'none'."
            )
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self._sklearn_pca = None

    def fit(self, X: np.ndarray) -> "PCAEngine":
        """
        Fit PCA on data matrix X (n_samples, n_features).
        """
        X_scaled = self.scaler.fit_transform(X)

        if self.method == "sklearn":
            from sklearn.decomposition import PCA

            # svd_solver="auto" silently switches to *randomized* (approximate)
            # SVD for wide matrices like flattened images, so the recovered
            # components - and therefore the reconstruction - are not quite the
            # true top-k subspace. "full" performs an exact SVD, giving the
            # provably MSE-optimal rank-k reconstruction (Eckart-Young) and the
            # best achievable PSNR/SSIM for a given k. These datasets are small
            # enough that the exact solve stays fast.
            self._sklearn_pca = PCA(
                n_components=self.n_components,
                svd_solver="full",
                random_state=42,
            )
            self._sklearn_pca.fit(X_scaled)
            self.components_ = self._sklearn_pca.components_
            self.explained_variance_ = self._sklearn_pca.explained_variance_
            self.explained_variance_ratio_ = self._sklearn_pca.explained_variance_ratio_
            self.mean_ = self._sklearn_pca.mean_
        else:
            # Manual eigendecomposition (educational)
            cov_matrix = np.cov(X_scaled, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Sort descending
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]

            self.components_ = eigenvectors[:, : self.n_components].T
            self.explained_variance_ = eigenvalues[: self.n_components]
            total_var = np.sum(eigenvalues)
            self.explained_variance_ratio_ = self.explained_variance_ / total_var
            self.mean_ = np.mean(X_scaled, axis=0)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data onto principal components."""
        X_scaled = self.scaler.transform(X)
        if self.method == "sklearn":
            return self._sklearn_pca.transform(X_scaled)
        else:
            return (X_scaled - self.mean_) @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct from PCA space back to original space."""
        if self.method == "sklearn":
            X_scaled = self._sklearn_pca.inverse_transform(Z)
        else:
            X_scaled = Z @ self.components_ + self.mean_
        return self.scaler.inverse_transform(X_scaled)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Full pipeline: project then reconstruct."""
        Z = self.transform(X)
        return self.inverse_transform(Z)

    def get_eigenfaces(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """Return eigenfaces reshaped as images."""
        return self.components_.reshape(-1, *image_shape)

    def cumulative_variance(self) -> np.ndarray:
        """Return cumulative explained variance ratio."""
        return np.cumsum(self.explained_variance_ratio_)


class IncrementalPCAEngine:
    """
    Incremental PCA for streaming / large datasets that don't fit in memory.
    """

    def __init__(self, n_components: int = 40, batch_size: int = 50):
        self.n_components = n_components
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self._ipca = None

    def fit(self, X: np.ndarray) -> "IncrementalPCAEngine":
        from sklearn.decomposition import IncrementalPCA

        X_scaled = self.scaler.fit_transform(X)
        # Ensure batch_size > n_components (sklearn requirement)
        effective_batch = max(self.batch_size, self.n_components + 1)
        self._ipca = IncrementalPCA(n_components=self.n_components, batch_size=effective_batch)
        self._ipca.fit(X_scaled)
        return self

    def partial_fit(self, X_batch: np.ndarray) -> "IncrementalPCAEngine":
        """Incrementally update PCA with a new batch."""
        from sklearn.decomposition import IncrementalPCA

        if self._ipca is None:
            self._ipca = IncrementalPCA(n_components=self.n_components, batch_size=self.batch_size)
        X_scaled = self.scaler.transform(X_batch)
        self._ipca.partial_fit(X_scaled)
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        Z = self._ipca.transform(X_scaled)
        X_recon = self._ipca.inverse_transform(Z)
        return self.scaler.inverse_transform(X_recon)

    @property
    def explained_variance_ratio_(self):
        return self._ipca.explained_variance_ratio_ if self._ipca else None

    @property
    def components_(self):
        return self._ipca.components_ if self._ipca else None


class KernelPCAEngine:
    """
    Kernel PCA for non-linear dimensionality reduction.
    """

    def __init__(self, n_components: int = 40, kernel: str = "rbf", gamma: float = None):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.scaler = StandardScaler()
        self._kpca = None

    def fit(self, X: np.ndarray) -> "KernelPCAEngine":
        from sklearn.decomposition import KernelPCA

        X_scaled = self.scaler.fit_transform(X)
        self._kpca = KernelPCA(
            n_components=self.n_components,
            kernel=self.kernel,
            gamma=self.gamma,
            fit_inverse_transform=True,
            random_state=42,
        )
        self._kpca.fit(X_scaled)
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        Z = self._kpca.transform(X_scaled)
        X_recon = self._kpca.inverse_transform(Z)
        return self.scaler.inverse_transform(X_recon)


class SparsePCAEngine:
    """
    Sparse PCA with L1 penalty for interpretable components.
    """

    def __init__(self, n_components: int = 20, alpha: float = 1.0):
        self.n_components = n_components
        self.alpha = alpha
        self.scaler = StandardScaler()
        self._spca = None

    def fit(self, X: np.ndarray) -> "SparsePCAEngine":
        from sklearn.decomposition import SparsePCA

        X_scaled = self.scaler.fit_transform(X)
        self._spca = SparsePCA(
            n_components=self.n_components,
            alpha=self.alpha,
            random_state=42,
            max_iter=100,
        )
        self._spca.fit(X_scaled)
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        Z = self._spca.transform(X_scaled)
        X_recon = Z @ self._spca.components_ + self._spca.mean_
        return self.scaler.inverse_transform(X_recon)

    @property
    def components_(self):
        return self._spca.components_ if self._spca else None

    def get_sparse_eigenfaces(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """Return sparse components reshaped as images."""
        return self._spca.components_.reshape(-1, *image_shape)


class ProbabilisticPCAEngine:
    """
    Probabilistic PCA (PPCA) with uncertainty estimation.
    Models: x = Wz + mu + epsilon, epsilon ~ N(0, sigma^2 I)
    """

    def __init__(self, n_components: int = 40):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self._ppca = None

    def fit(self, X: np.ndarray) -> "ProbabilisticPCAEngine":
        from sklearn.decomposition import FactorAnalysis

        X_scaled = self.scaler.fit_transform(X)
        self._ppca = FactorAnalysis(n_components=self.n_components, random_state=42)
        self._ppca.fit(X_scaled)
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        Z = self._ppca.transform(X_scaled)
        X_recon = Z @ self._ppca.components_ + self._ppca.mean_
        return self.scaler.inverse_transform(X_recon)

    def uncertainty_map(self, X: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Compute per-pixel reconstruction uncertainty.
        Returns uncertainty maps shaped (N, H, W).
        """
        X_scaled = self.scaler.transform(X)
        noise_variance = self._ppca.noise_variance_

        # noise_variance_ is per-feature for FactorAnalysis
        if np.isscalar(noise_variance):
            noise_var_diag = noise_variance * np.ones(X_scaled.shape[1])
        else:
            noise_var_diag = noise_variance

        W = self._ppca.components_  # (n_components, n_features)

        # Reconstruction variance: approximate per-pixel uncertainty
        # Based on the residual variance not captured by the model
        recon_var = noise_var_diag  # Per-feature noise variance

        # Scale back to original space
        uncertainty = np.sqrt(recon_var) * self.scaler.scale_
        # Tile for all samples
        uncertainty_maps = np.tile(uncertainty, (X.shape[0], 1)).reshape(-1, *image_shape)
        return uncertainty_maps

    @property
    def noise_variance(self):
        return self._ppca.noise_variance_ if self._ppca else None


class RobustPCA:
    """
    Robust PCA via Inexact Augmented Lagrange Multiplier (IALM).
    Decomposes matrix D into Low-rank L + Sparse S.
    Effective against outliers and occlusions.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-7, lam: Optional[float] = None):
        self.max_iter = max_iter
        self.tol = tol
        self.lam = lam
        self.L_ = None  # Low-rank component
        self.S_ = None  # Sparse component

    def _shrink(self, X: np.ndarray, tau: float) -> np.ndarray:
        """Soft-thresholding operator."""
        return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

    def _svd_threshold(self, X: np.ndarray, tau: float) -> np.ndarray:
        """Singular value thresholding."""
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        s_thresh = np.maximum(s - tau, 0)
        return U @ np.diag(s_thresh) @ Vt

    def fit(self, D: np.ndarray) -> "RobustPCA":
        """
        Decompose D = L + S where L is low-rank and S is sparse.

        Args:
            D: (n_samples, n_features) data matrix
        """
        m, n = D.shape
        lam = self.lam if self.lam else 1.0 / np.sqrt(max(m, n))

        # Initialize
        S = np.zeros_like(D)
        Y = np.zeros_like(D)
        mu = 1.25 / np.linalg.norm(D, ord=2)
        mu_bar = mu * 1e7
        rho = 1.5
        norm_D = np.linalg.norm(D, "fro")

        for _ in range(self.max_iter):
            # Update L
            L = self._svd_threshold(D - S + Y / mu, 1.0 / mu)
            # Update S
            S = self._shrink(D - L + Y / mu, lam / mu)
            # Update Y
            residual = D - L - S
            Y = Y + mu * residual
            mu = min(rho * mu, mu_bar)

            # Check convergence
            if np.linalg.norm(residual, "fro") / norm_D < self.tol:
                break

        self.L_ = L
        self.S_ = S
        return self

    def get_low_rank(self) -> np.ndarray:
        """Return the low-rank (clean) component."""
        return self.L_

    def get_sparse(self) -> np.ndarray:
        """Return the sparse (noise/outlier) component."""
        return self.S_


class PCA2D:
    """
    2D-PCA: operates directly on image matrices instead of flattened vectors.
    Preserves spatial structure and is computationally more efficient.
    """

    def __init__(self, n_components: int = 20):
        self.n_components = n_components
        self.mean_image_ = None
        self.projection_ = None

    def fit(self, images: np.ndarray) -> "PCA2D":
        """
        Fit 2D-PCA on image matrices.

        Args:
            images: (N, H, W) array of images
        """
        n, h, w = images.shape
        self.mean_image_ = np.mean(images, axis=0)
        centered = images - self.mean_image_

        # Compute image covariance matrix G_t = (1/N) * sum(A_i^T * A_i)
        G = np.zeros((w, w))
        for i in range(n):
            G += centered[i].T @ centered[i]
        G /= n

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(G)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        self.projection_ = eigenvectors[:, : self.n_components]
        self.explained_variance_ = eigenvalues[sorted_indices][: self.n_components]
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        """Project images onto 2D principal components."""
        centered = images - self.mean_image_
        # Each image A -> Y = A * W (H x d)
        projected = np.array([img @ self.projection_ for img in centered])
        return projected

    def reconstruct(self, images: np.ndarray) -> np.ndarray:
        """Reconstruct images from 2D-PCA projections."""
        projected = self.transform(images)
        # Reconstruct: A_hat = Y * W^T + mean
        reconstructed = np.array([p @ self.projection_.T for p in projected])
        return reconstructed + self.mean_image_

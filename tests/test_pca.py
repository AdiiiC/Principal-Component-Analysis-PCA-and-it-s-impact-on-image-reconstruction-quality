"""
Unit tests for PCA Image Reconstruction project.
"""

import numpy as np
import pytest
from pca_core import PCAEngine, IncrementalPCAEngine, SparsePCAEngine, ProbabilisticPCAEngine
from pca_core.pca_engine import RobustPCA, PCA2D
from pca_core.metrics import compute_mse, compute_psnr, compute_ssim, compute_all_metrics
from pca_core.data_loader import load_olivetti, add_noise
from pca_core.utils import get_compression_ratio, entropy, mutual_information


@pytest.fixture
def sample_data():
    """Small synthetic dataset for fast tests."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 50, 100
    X = rng.random((n_samples, n_features))
    labels = np.repeat(np.arange(10), 5)
    image_shape = (10, 10)
    return X, labels, image_shape


@pytest.fixture
def olivetti_data():
    """Load real Olivetti data (cached)."""
    images, data_flat, labels = load_olivetti()
    return images, data_flat, labels


class TestPCAEngine:
    """Tests for standard PCA engine."""

    def test_fit_sklearn(self, sample_data):
        X, _, _ = sample_data
        engine = PCAEngine(n_components=10, method="sklearn")
        engine.fit(X)
        assert engine.components_ is not None
        assert engine.explained_variance_ratio_ is not None
        assert len(engine.explained_variance_ratio_) == 10

    def test_fit_manual(self, sample_data):
        X, _, _ = sample_data
        engine = PCAEngine(n_components=10, method="manual")
        engine.fit(X)
        assert engine.components_ is not None
        assert engine.components_.shape == (10, X.shape[1])

    def test_perfect_reconstruction_full_rank(self, sample_data):
        """PCA with all components should perfectly reconstruct data."""
        X, _, _ = sample_data
        n_components = min(X.shape) - 1
        engine = PCAEngine(n_components=n_components, method="sklearn")
        engine.fit(X)
        reconstructed = engine.reconstruct(X)
        np.testing.assert_allclose(X, reconstructed, atol=1e-10)

    def test_cumulative_variance_sums_correctly(self, sample_data):
        X, _, _ = sample_data
        engine = PCAEngine(n_components=10, method="sklearn")
        engine.fit(X)
        cum_var = engine.cumulative_variance()
        assert cum_var[-1] <= 1.0
        assert np.all(np.diff(cum_var) >= 0)  # Monotonically increasing

    def test_compression_reduces_dimensionality(self, sample_data):
        X, _, _ = sample_data
        engine = PCAEngine(n_components=5)
        engine.fit(X)
        Z = engine.transform(X)
        assert Z.shape == (X.shape[0], 5)

    def test_reconstruct_output_shape(self, sample_data):
        X, _, _ = sample_data
        engine = PCAEngine(n_components=10)
        engine.fit(X)
        recon = engine.reconstruct(X)
        assert recon.shape == X.shape


class TestIncrementalPCA:
    def test_incremental_fit(self, sample_data):
        X, _, _ = sample_data
        ipca = IncrementalPCAEngine(n_components=10, batch_size=10)
        ipca.fit(X)
        assert ipca.components_ is not None
        recon = ipca.reconstruct(X)
        assert recon.shape == X.shape


class TestPCA2D:
    def test_2d_pca_fit_reconstruct(self):
        rng = np.random.RandomState(42)
        images = rng.random((30, 10, 10))
        pca2d = PCA2D(n_components=5)
        pca2d.fit(images)
        recon = pca2d.reconstruct(images)
        assert recon.shape == images.shape


class TestRobustPCA:
    def test_decomposition(self, sample_data):
        X, _, _ = sample_data
        rpca = RobustPCA(max_iter=20)
        rpca.fit(X)
        L = rpca.get_low_rank()
        S = rpca.get_sparse()
        # L + S should approximately equal X
        np.testing.assert_allclose(L + S, X, atol=0.1)


class TestMetrics:
    def test_mse_identical(self):
        img = np.random.random((64, 64))
        assert compute_mse(img, img) == 0.0

    def test_psnr_identical(self):
        img = np.random.random((64, 64))
        assert compute_psnr(img, img) == float("inf")

    def test_ssim_identical(self):
        img = np.random.random((64, 64))
        assert compute_ssim(img, img) == pytest.approx(1.0, abs=1e-5)

    def test_mse_range(self):
        img1 = np.zeros((10, 10))
        img2 = np.ones((10, 10))
        assert compute_mse(img1, img2) == 1.0

    def test_psnr_range(self):
        img1 = np.zeros((10, 10))
        img2 = np.ones((10, 10)) * 0.5
        psnr = compute_psnr(img1, img2, max_val=1.0)
        assert psnr > 0

    def test_all_metrics_returns_dict(self):
        img = np.random.random(4096)
        noisy = img + np.random.random(4096) * 0.1
        metrics = compute_all_metrics(img, noisy, (64, 64))
        assert "mse" in metrics
        assert "psnr" in metrics
        assert "ssim" in metrics


class TestDataLoader:
    def test_load_olivetti(self, olivetti_data):
        images, data_flat, labels = olivetti_data
        assert images.shape == (400, 64, 64)
        assert data_flat.shape == (400, 4096)
        assert labels.shape == (400,)
        assert len(np.unique(labels)) == 40

    def test_add_gaussian_noise(self, olivetti_data):
        _, data_flat, _ = olivetti_data
        noisy = add_noise(data_flat[:5], noise_type="gaussian", noise_level=0.1)
        assert noisy.shape == data_flat[:5].shape
        assert not np.allclose(noisy, data_flat[:5])

    def test_add_salt_pepper_noise(self, olivetti_data):
        _, data_flat, _ = olivetti_data
        noisy = add_noise(data_flat[:5], noise_type="salt_pepper", noise_level=0.1)
        assert noisy.shape == data_flat[:5].shape

    def test_add_occlusion(self, olivetti_data):
        images, _, _ = olivetti_data
        noisy = add_noise(images[:5], noise_type="occlusion", noise_level=0.3)
        assert noisy.shape == images[:5].shape


class TestUtils:
    def test_compression_ratio(self):
        cr = get_compression_ratio(4096, 40, 400)
        assert cr > 1.0  # Should compress

    def test_entropy_uniform(self):
        # Uniform distribution should have higher entropy than concentrated
        rng = np.random.RandomState(42)
        uniform = rng.uniform(0, 1, 1000)
        concentrated = rng.normal(0.5, 0.01, 1000)
        h_uniform = entropy(uniform)
        h_concentrated = entropy(concentrated)
        assert h_uniform > h_concentrated

    def test_mutual_information_dependent_higher(self):
        # Correlated variables should have higher MI than independent
        rng = np.random.RandomState(42)
        x = rng.random(1000)
        y_independent = rng.random(1000)
        y_dependent = x + rng.normal(0, 0.1, 1000)
        mi_indep = mutual_information(x, y_independent)
        mi_dep = mutual_information(x, y_dependent)
        assert mi_dep > mi_indep


class TestExperiments:
    def test_component_sweep(self, olivetti_data):
        _, data_flat, _ = olivetti_data
        from experiments import run_component_sweep
        results = run_component_sweep(data_flat, (64, 64), component_range=[5, 10, 20])
        assert len(results) == 3
        assert "avg_psnr" in results.columns
        assert "avg_ssim" in results.columns

    def test_face_recognition(self, olivetti_data):
        _, data_flat, labels = olivetti_data
        from experiments import run_face_recognition_experiment
        results = run_face_recognition_experiment(
            data_flat, labels, components_range=[10, 20], test_size=0.3
        )
        assert len(results) == 2
        assert all(results["accuracy"] > 0)

    def test_anomaly_detection(self, olivetti_data):
        _, data_flat, labels = olivetti_data
        from experiments import run_anomaly_detection
        results = run_anomaly_detection(data_flat, labels, n_components=20)
        assert results["roc_auc"] > 0.5  # Better than random

    def test_face_arithmetic(self, olivetti_data):
        _, data_flat, labels = olivetti_data
        from experiments import run_face_arithmetic
        results = run_face_arithmetic(data_flat, labels, n_components=20)
        assert "interpolations" in results
        assert "canonical_faces" in results
        assert "outlier" in results

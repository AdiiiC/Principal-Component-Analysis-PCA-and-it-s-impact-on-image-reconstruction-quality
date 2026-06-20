"""
PCA Image Reconstruction Explorer - Interactive Streamlit App
=============================================================
Launch: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

st.set_page_config(page_title="PCA Image Reconstruction Explorer", layout="wide")


@st.cache_resource
def load_data():
    """Load and cache Olivetti faces dataset."""
    from pca_core.data_loader import load_olivetti
    images, data_flat, labels = load_olivetti()
    return images, data_flat, labels


@st.cache_resource
def fit_pca(data_flat, n_components):
    """Fit PCA with given components."""
    from pca_core import PCAEngine
    engine = PCAEngine(n_components=n_components)
    engine.fit(data_flat)
    return engine


def main():
    st.title("🔬 PCA Image Reconstruction Explorer")
    st.markdown("Interactive exploration of Principal Component Analysis for face image compression and reconstruction.")

    images, data_flat, labels = load_data()
    image_shape = images[0].shape

    # Sidebar controls
    st.sidebar.header("Controls")
    page = st.sidebar.radio(
        "Select Page",
        [
            "Reconstruction",
            "Eigenfaces",
            "Component Sweep",
            "Denoising",
            "Face Arithmetic",
            "Anomaly Detection",
            "Method Comparison",
            "Upload Your Image",
        ],
    )

    if page == "Reconstruction":
        page_reconstruction(data_flat, images, labels, image_shape)
    elif page == "Eigenfaces":
        page_eigenfaces(data_flat, image_shape)
    elif page == "Component Sweep":
        page_sweep(data_flat, image_shape)
    elif page == "Denoising":
        page_denoising(data_flat, images, image_shape)
    elif page == "Face Arithmetic":
        page_arithmetic(data_flat, labels, image_shape)
    elif page == "Anomaly Detection":
        page_anomaly(data_flat, labels, image_shape)
    elif page == "Method Comparison":
        page_methods(data_flat, images, image_shape)
    elif page == "Upload Your Image":
        page_upload(data_flat, image_shape)


def page_reconstruction(data_flat, images, labels, image_shape):
    """Interactive reconstruction page."""
    st.header("Interactive Reconstruction")

    col1, col2 = st.columns(2)
    with col1:
        k = st.slider("Number of Components (k)", min_value=1, max_value=150, value=40, step=1)
    with col2:
        face_idx = st.slider("Face Index", min_value=0, max_value=len(data_flat) - 1, value=0)

    engine = fit_pca(data_flat, min(k, data_flat.shape[0] - 1))
    reconstructed = np.clip(engine.reconstruct(data_flat[face_idx:face_idx+1]), 0, 1)

    from pca_core.metrics import compute_all_metrics
    metrics = compute_all_metrics(data_flat[face_idx], reconstructed[0], image_shape)

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PSNR", f"{metrics['psnr']:.2f} dB")
    col2.metric("SSIM", f"{metrics['ssim']:.4f}")
    col3.metric("MSE", f"{metrics['mse']:.6f}")
    from pca_core.utils import get_compression_ratio
    cr = get_compression_ratio(data_flat.shape[1], k, 1)
    col4.metric("Compression Ratio", f"{cr:.2f}x")

    # Display images
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(data_flat[face_idx].reshape(image_shape), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader(f"Reconstructed (k={k})")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(reconstructed[0].reshape(image_shape), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig)
        plt.close()

    with col3:
        st.subheader("Error Map")
        error = np.abs(data_flat[face_idx].reshape(image_shape) - reconstructed[0].reshape(image_shape))
        fig, ax = plt.subplots(figsize=(3, 3))
        im = ax.imshow(error, cmap="hot")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)
        plt.close()

    # Cumulative variance
    st.subheader("Explained Variance")
    cum_var = engine.cumulative_variance()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(range(1, len(cum_var) + 1), cum_var, "b-")
    ax.axvline(x=k, color="r", linestyle="--", label=f"k={k} ({cum_var[min(k-1, len(cum_var)-1)]*100:.1f}%)")
    ax.set_xlabel("Components")
    ax.set_ylabel("Cumulative Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()


def page_eigenfaces(data_flat, image_shape):
    """Eigenfaces visualization page."""
    st.header("Eigenfaces (Principal Components)")

    n_show = st.slider("Number of eigenfaces to show", 4, 36, 16, 4)
    engine = fit_pca(data_flat, n_show)

    cols_per_row = 4
    n_rows = (n_show + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(12, 3 * n_rows))
    axes = axes.flatten()

    for i in range(n_show):
        eigenface = engine.components_[i] if engine.method == "sklearn" else engine.components_[i]
        axes[i].imshow(eigenface.reshape(image_shape), cmap="seismic")
        axes[i].set_title(f"PC{i+1} ({engine.explained_variance_ratio_[i]*100:.1f}%)", fontsize=9)
        axes[i].axis("off")

    for i in range(n_show, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Mean face
    st.subheader("Mean Face")
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(engine.scaler.mean_.reshape(image_shape), cmap="gray")
    ax.axis("off")
    st.pyplot(fig)
    plt.close()


def page_sweep(data_flat, image_shape):
    """Component sweep analysis page."""
    st.header("Quality vs Number of Components")

    max_k = st.slider("Maximum components", 10, 200, 100, 10)

    with st.spinner("Running component sweep..."):
        from experiments import run_component_sweep
        results = run_component_sweep(
            data_flat, image_shape,
            component_range=list(range(1, max_k + 1, max(1, max_k // 20))),
        )

    from visualizations import plot_component_sweep_metrics
    fig = plot_component_sweep_metrics(results)
    st.pyplot(fig)
    plt.close()

    st.dataframe(results, use_container_width=True)


def page_denoising(data_flat, images, image_shape):
    """Denoising experiment page."""
    st.header("PCA as a Denoiser")

    col1, col2, col3 = st.columns(3)
    with col1:
        noise_type = st.selectbox("Noise Type", ["gaussian", "salt_pepper", "occlusion"])
    with col2:
        noise_level = st.slider("Noise Level", 0.05, 0.5, 0.2, 0.05)
    with col3:
        k = st.slider("PCA Components", 5, 100, 40, 5, key="denoise_k")

    face_idx = st.slider("Face Index", 0, len(data_flat) - 1, 0, key="denoise_face")

    from pca_core import PCAEngine, add_noise
    engine = fit_pca(data_flat, k)

    # Add noise and denoise
    if noise_type == "occlusion":
        noisy_img = add_noise(images[face_idx:face_idx+1], noise_type=noise_type, noise_level=noise_level)
        noisy_flat = noisy_img.reshape(1, -1)
    else:
        noisy_flat = add_noise(data_flat[face_idx:face_idx+1], noise_type=noise_type, noise_level=noise_level)

    denoised = np.clip(engine.reconstruct(noisy_flat), 0, 1)

    from pca_core.metrics import compute_all_metrics
    noisy_metrics = compute_all_metrics(data_flat[face_idx], noisy_flat[0], image_shape)
    denoised_metrics = compute_all_metrics(data_flat[face_idx], denoised[0], image_shape)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Noisy PSNR", f"{noisy_metrics['psnr']:.2f} dB")
    col2.metric("Denoised PSNR", f"{denoised_metrics['psnr']:.2f} dB")
    col3.metric("Noisy SSIM", f"{noisy_metrics['ssim']:.4f}")
    col4.metric("Denoised SSIM", f"{denoised_metrics['ssim']:.4f}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(data_flat[face_idx].reshape(image_shape), cmap="gray")
        ax.axis("off")
        st.pyplot(fig)
        plt.close()
    with col2:
        st.subheader("Noisy")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(noisy_flat[0].reshape(image_shape), cmap="gray")
        ax.axis("off")
        st.pyplot(fig)
        plt.close()
    with col3:
        st.subheader("Denoised")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(denoised[0].reshape(image_shape), cmap="gray")
        ax.axis("off")
        st.pyplot(fig)
        plt.close()


def page_arithmetic(data_flat, labels, image_shape):
    """Face arithmetic page."""
    st.header("Face Arithmetic in Latent Space")

    col1, col2, col3 = st.columns(3)
    with col1:
        face_a = st.slider("Face A Index", 0, len(data_flat) - 1, 0)
    with col2:
        face_b = st.slider("Face B Index", 0, len(data_flat) - 1, 10)
    with col3:
        k = st.slider("Components", 10, 100, 40, 5, key="arith_k")

    from pca_core import PCAEngine
    engine = fit_pca(data_flat, k)
    Z = engine.transform(data_flat)

    # Interpolation
    st.subheader("Face Interpolation")
    n_steps = 8
    alphas = np.linspace(0, 1, n_steps)

    fig, axes = plt.subplots(1, n_steps, figsize=(2 * n_steps, 2.5), subplot_kw={"xticks": [], "yticks": []})
    for i, alpha in enumerate(alphas):
        z_interp = (1 - alpha) * Z[face_a] + alpha * Z[face_b]
        recon = engine.inverse_transform(z_interp.reshape(1, -1))
        axes[i].imshow(np.clip(recon.reshape(image_shape), 0, 1), cmap="gray")
        axes[i].set_title(f"α={alpha:.1f}", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Mean face
    st.subheader("Mean Face (Latent Space Average)")
    z_mean = Z.mean(axis=0)
    mean_recon = engine.inverse_transform(z_mean.reshape(1, -1))
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(np.clip(mean_recon.reshape(image_shape), 0, 1), cmap="gray")
    ax.axis("off")
    st.pyplot(fig)
    plt.close()


def page_anomaly(data_flat, labels, image_shape):
    """Anomaly detection page."""
    st.header("Anomaly Detection via Reconstruction Error")

    k = st.slider("PCA Components", 10, 80, 40, 5, key="anomaly_k")

    with st.spinner("Running anomaly detection..."):
        from experiments import run_anomaly_detection
        results = run_anomaly_detection(data_flat, labels, image_shape, n_components=k)

    col1, col2 = st.columns(2)
    col1.metric("ROC AUC", f"{results['roc_auc']:.4f}")
    col2.metric("PR AUC", f"{results['pr_auc']:.4f}")

    from visualizations import plot_anomaly_scores
    fig = plot_anomaly_scores(results)
    st.pyplot(fig)
    plt.close()


def page_methods(data_flat, images, image_shape):
    """Method comparison page."""
    st.header("PCA Method Comparison")

    k = st.slider("Components for comparison", 10, 60, 40, 5, key="method_k")

    with st.spinner("Comparing methods (this may take a moment)..."):
        from experiments import run_method_comparison
        results = run_method_comparison(data_flat, images, image_shape, n_components=k)

    st.dataframe(results, use_container_width=True)

    from visualizations import plot_method_comparison_bar
    fig = plot_method_comparison_bar(results)
    st.pyplot(fig)
    plt.close()


def page_upload(data_flat, image_shape):
    """Upload and reconstruct custom image."""
    st.header("Upload Your Own Image")
    st.markdown("Upload a grayscale face image to see PCA reconstruction.")

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded is not None:
        k = st.slider("Components", 5, 100, 40, 5, key="upload_k")

        img = Image.open(uploaded).convert("L").resize(image_shape)
        img_array = np.array(img, dtype=np.float64) / 255.0
        img_flat = img_array.flatten().reshape(1, -1)

        engine = fit_pca(data_flat, k)
        reconstructed = np.clip(engine.reconstruct(img_flat), 0, 1)

        from pca_core.metrics import compute_all_metrics
        metrics = compute_all_metrics(img_flat[0], reconstructed[0], image_shape)

        col1, col2, col3 = st.columns(3)
        col1.metric("PSNR", f"{metrics['psnr']:.2f} dB")
        col2.metric("SSIM", f"{metrics['ssim']:.4f}")
        col3.metric("MSE", f"{metrics['mse']:.6f}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded")
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(img_array, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            st.pyplot(fig)
            plt.close()
        with col2:
            st.subheader(f"Reconstructed (k={k})")
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(reconstructed[0].reshape(image_shape), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            st.pyplot(fig)
            plt.close()

        # Download reconstructed image
        recon_img = (reconstructed[0].reshape(image_shape) * 255).astype(np.uint8)
        pil_recon = Image.fromarray(recon_img, mode="L")
        buf = io.BytesIO()
        pil_recon.save(buf, format="PNG")
        st.download_button("Download Reconstructed Image", buf.getvalue(), "reconstructed.png", "image/png")


if __name__ == "__main__":
    main()

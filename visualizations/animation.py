"""
Progressive reconstruction animation generator.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple


def create_progressive_animation(
    original: np.ndarray,
    data_flat: np.ndarray,
    image_shape: tuple = (64, 64),
    components_sequence: List[int] = None,
    output_path: str = "outputs/progressive_reconstruction.gif",
    fps: int = 2,
) -> str:
    """
    Create an animated GIF/MP4 showing progressive reconstruction as components increase.

    Args:
        original: Single image (flattened or 2D)
        data_flat: Full dataset for fitting PCA
        image_shape: Image dimensions
        components_sequence: List of component counts for each frame
        output_path: Path to save animation
        fps: Frames per second

    Returns:
        Path to saved animation file
    """
    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from pca_core import PCAEngine

    if components_sequence is None:
        components_sequence = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100]

    original_img = original.reshape(image_shape)
    max_k = max(components_sequence)

    # Fit PCA with max components
    engine = PCAEngine(n_components=min(max_k, data_flat.shape[0] - 1), method="sklearn")
    engine.fit(data_flat)

    frames = []

    for k in components_sequence:
        # Reconstruct with k components by zeroing out higher components
        X_input = original.reshape(1, -1)
        X_scaled = engine.scaler.transform(X_input)

        if engine._sklearn_pca is not None:
            Z_full = engine._sklearn_pca.transform(X_scaled)
            Z_truncated = Z_full.copy()
            Z_truncated[:, k:] = 0
            recon_scaled = engine._sklearn_pca.inverse_transform(Z_truncated)
        else:
            Z_full = (X_scaled - engine.mean_) @ engine.components_.T
            Z_truncated = Z_full.copy()
            Z_truncated[:, k:] = 0
            recon_scaled = Z_truncated @ engine.components_ + engine.mean_

        recon = engine.scaler.inverse_transform(recon_scaled)
        recon_img = np.clip(recon.reshape(image_shape), 0, 1)

        # Create frame with matplotlib using Agg backend
        fig = plt.Figure(figsize=(6, 3), dpi=100)
        canvas = FigureCanvasAgg(fig)
        axes = fig.subplots(1, 2)
        axes[0].imshow(original_img, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Original", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(recon_img, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"k = {k} components", fontsize=11)
        axes[1].axis("off")

        fig.suptitle("Progressive PCA Reconstruction", fontsize=12, y=0.98)
        fig.tight_layout()

        canvas.draw()
        buf = canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3].copy()
        frames.append(frame)

    # Save animation
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output_path.endswith(".mp4"):
        imageio.mimwrite(str(output), frames, fps=fps, codec="libx264")
    else:
        imageio.mimwrite(str(output), frames, fps=fps, loop=0)

    print(f"Animation saved: {output}")
    return str(output)

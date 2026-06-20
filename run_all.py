"""
PCA Image Reconstruction & Analysis - Main Entry Point
=======================================================
Unified CLI to run all experiments and generate all outputs.

Usage:
    python run_all.py                     # Run all experiments
    python run_all.py --experiment sweep   # Run specific experiment
    python run_all.py --list              # List available experiments
"""

import sys
import argparse
import logging
import yaml
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_output_dir(config: dict) -> Path:
    """Create output directory."""
    out_dir = Path(config["visualization"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_dataset(config: dict):
    """Load dataset based on config."""
    from pca_core.data_loader import load_olivetti

    logger.info("Loading Olivetti Faces dataset...")
    images, data_flat, labels = load_olivetti(
        shuffle=config["dataset"]["shuffle"],
        random_state=config["dataset"]["random_state"],
    )
    logger.info(f"Loaded {images.shape[0]} images of shape {images.shape[1:]}")
    return images, data_flat, labels


def run_basic_reconstruction(data_flat, images, labels, config, out_dir):
    """Basic PCA reconstruction with eigenfaces and variance plots."""
    from pca_core import PCAEngine
    from visualizations import (
        plot_reconstruction_grid,
        plot_eigenfaces,
        plot_mean_face,
        plot_scree,
        plot_cumulative_variance,
        plot_error_heatmap,
        plot_pca_scatter_2d,
        plot_pca_scatter_3d,
    )

    logger.info("=== Basic PCA Reconstruction ===")
    n_components = config["pca"]["n_components"]
    image_shape = tuple(config["dataset"]["target_size"])

    engine = PCAEngine(n_components=n_components, method=config["pca"]["method"])
    engine.fit(data_flat)
    reconstructed = np.clip(engine.reconstruct(data_flat), 0, 1)

    # Reconstruction grid
    plot_reconstruction_grid(
        data_flat, reconstructed, image_shape,
        n_faces=config["visualization"]["n_display_faces"],
        title=f"PCA Reconstruction (k={n_components})",
        save_path=str(out_dir / "reconstruction_grid.png"),
    )

    # Eigenfaces
    plot_eigenfaces(
        engine.components_ if engine.method == "sklearn" else engine.components_,
        image_shape,
        n_eigenfaces=config["visualization"]["n_eigenfaces"],
        save_path=str(out_dir / "eigenfaces.png"),
    )

    # Mean face
    mean_face = engine.scaler.mean_
    plot_mean_face(mean_face, image_shape, save_path=str(out_dir / "mean_face.png"))

    # Scree plot
    plot_scree(engine.explained_variance_ratio_, save_path=str(out_dir / "scree_plot.png"))

    # Cumulative variance
    plot_cumulative_variance(
        engine.explained_variance_ratio_,
        thresholds=config["pca"]["variance_thresholds"],
        save_path=str(out_dir / "cumulative_variance.png"),
    )

    # Error heatmap for first face
    plot_error_heatmap(
        data_flat[0], reconstructed[0], image_shape,
        save_path=str(out_dir / "error_heatmap.png"),
    )

    # PCA scatter plots
    Z = engine.transform(data_flat)
    plot_pca_scatter_2d(Z, labels, pc_x=0, pc_y=1, save_path=str(out_dir / "pca_scatter_2d.png"))
    plot_pca_scatter_3d(Z, labels, save_path=str(out_dir / "pca_scatter_3d.png"))

    logger.info("Basic reconstruction complete.")
    return engine


def run_sweep(data_flat, images, labels, config, out_dir):
    """Component sweep experiment."""
    from experiments import run_component_sweep
    from visualizations import plot_component_sweep_metrics

    logger.info("=== Component Sweep ===")
    sweep_cfg = config["pca"]["sweep_range"]
    component_range = list(range(sweep_cfg["start"], sweep_cfg["stop"], sweep_cfg["step"]))
    image_shape = tuple(config["dataset"]["target_size"])

    results = run_component_sweep(data_flat, image_shape, component_range, config["pca"]["method"])
    results.to_csv(str(out_dir / "component_sweep_results.csv"), index=False)
    logger.info(f"Sweep results:\n{results.to_string(index=False)}")

    plot_component_sweep_metrics(results, save_path=str(out_dir / "component_sweep_metrics.png"))
    logger.info("Component sweep complete.")
    return results


def run_denoising(data_flat, images, labels, config, out_dir):
    """Denoising experiment."""
    from experiments import run_denoising_experiment
    from experiments.denoising import denoise_single_image
    from pca_core import PCAEngine
    from visualizations import plot_denoising_comparison

    logger.info("=== Denoising Experiment ===")
    image_shape = tuple(config["dataset"]["target_size"])
    denoise_cfg = config["denoising"]

    results = run_denoising_experiment(
        data_flat, images, image_shape,
        noise_types=denoise_cfg["noise_types"],
        noise_levels=denoise_cfg["noise_levels"],
        n_components=config["pca"]["n_components"],
    )
    results.to_csv(str(out_dir / "denoising_results.csv"), index=False)
    logger.info(f"Denoising summary:\n{results.to_string(index=False)}")

    # Visual example
    engine = PCAEngine(n_components=config["pca"]["n_components"])
    engine.fit(data_flat)

    for noise_type in ["gaussian", "salt_pepper"]:
        result_imgs = denoise_single_image(
            data_flat[0], engine, noise_type=noise_type, noise_level=0.2, image_shape=image_shape
        )
        plot_denoising_comparison(result_imgs, save_path=str(out_dir / f"denoising_{noise_type}.png"))

    logger.info("Denoising experiment complete.")
    return results


def run_recognition(data_flat, images, labels, config, out_dir):
    """Face recognition experiment."""
    from experiments import run_face_recognition_experiment
    from visualizations import plot_face_recognition_accuracy

    logger.info("=== Face Recognition ===")
    recog_cfg = config["face_recognition"]
    components_range = list(range(
        recog_cfg["components_range"]["start"],
        recog_cfg["components_range"]["stop"],
        recog_cfg["components_range"]["step"],
    ))

    results = run_face_recognition_experiment(
        data_flat, labels, components_range,
        classifier=recog_cfg["classifier"],
        test_size=recog_cfg["test_size"],
    )
    results.to_csv(str(out_dir / "face_recognition_results.csv"), index=False)
    logger.info(f"Recognition results:\n{results.to_string(index=False)}")

    plot_face_recognition_accuracy(results, save_path=str(out_dir / "face_recognition_accuracy.png"))
    logger.info("Face recognition experiment complete.")
    return results


def run_anomaly(data_flat, images, labels, config, out_dir):
    """Anomaly detection experiment."""
    from experiments import run_anomaly_detection
    from visualizations import plot_anomaly_scores

    logger.info("=== Anomaly Detection ===")
    anomaly_cfg = config["anomaly_detection"]

    results = run_anomaly_detection(
        data_flat, labels,
        image_shape=tuple(config["dataset"]["target_size"]),
        n_components=config["pca"]["n_components"],
        contamination=anomaly_cfg["contamination"],
        threshold_percentile=anomaly_cfg["threshold_percentile"],
    )
    logger.info(f"Anomaly Detection - ROC AUC: {results['roc_auc']:.4f}, PR AUC: {results['pr_auc']:.4f}")

    plot_anomaly_scores(results, save_path=str(out_dir / "anomaly_detection.png"))
    logger.info("Anomaly detection complete.")
    return results


def run_compression(data_flat, images, labels, config, out_dir):
    """Compression comparison experiment."""
    from experiments import run_compression_comparison
    from visualizations import plot_compression_comparison

    logger.info("=== Compression Comparison (PCA vs JPEG) ===")
    comp_cfg = config["compression"]

    results = run_compression_comparison(
        data_flat,
        image_shape=tuple(config["dataset"]["target_size"]),
        bits_per_coeff=comp_cfg["bits_per_coefficient"],
        jpeg_qualities=comp_cfg["jpeg_qualities"],
    )
    results.to_csv(str(out_dir / "compression_comparison.csv"), index=False)
    logger.info(f"Compression results:\n{results.to_string(index=False)}")

    plot_compression_comparison(results, save_path=str(out_dir / "compression_comparison.png"))
    logger.info("Compression comparison complete.")
    return results


def run_info_theory(data_flat, images, labels, config, out_dir):
    """Information theory analysis."""
    from experiments import run_information_theory_analysis
    from visualizations import plot_information_theory

    logger.info("=== Information Theory Analysis ===")
    info_cfg = config["information_theory"]

    results = run_information_theory_analysis(
        data_flat, labels,
        n_components=config["pca"]["n_components"],
        n_bins_entropy=info_cfg["n_bins_entropy"],
        n_bins_mi=info_cfg["n_bins_mi"],
    )
    results.to_csv(str(out_dir / "information_theory.csv"), index=False)
    logger.info(f"Info theory:\n{results.to_string(index=False)}")

    plot_information_theory(results, save_path=str(out_dir / "information_theory.png"))
    logger.info("Information theory analysis complete.")
    return results


def run_methods(data_flat, images, labels, config, out_dir):
    """Method comparison (all PCA variants)."""
    from experiments import run_method_comparison
    from visualizations import plot_method_comparison_bar

    logger.info("=== PCA Method Comparison ===")

    results = run_method_comparison(
        data_flat, images,
        image_shape=tuple(config["dataset"]["target_size"]),
        n_components=config["pca"]["n_components"],
    )
    results.to_csv(str(out_dir / "method_comparison.csv"), index=False)
    logger.info(f"Method comparison:\n{results.to_string(index=False)}")

    plot_method_comparison_bar(results, save_path=str(out_dir / "method_comparison.png"))
    logger.info("Method comparison complete.")
    return results


def run_arithmetic(data_flat, images, labels, config, out_dir):
    """Face arithmetic in latent space."""
    from experiments import run_face_arithmetic
    from visualizations import plot_face_interpolation, plot_canonical_faces

    logger.info("=== Face Arithmetic ===")

    results = run_face_arithmetic(
        data_flat, labels,
        image_shape=tuple(config["dataset"]["target_size"]),
        n_components=config["pca"]["n_components"],
    )

    plot_face_interpolation(results["interpolations"], save_path=str(out_dir / "face_interpolation.png"))
    plot_canonical_faces(results["canonical_faces"], save_path=str(out_dir / "canonical_faces.png"))
    logger.info("Face arithmetic complete.")
    return results


def run_incremental(data_flat, images, labels, config, out_dir):
    """Incremental PCA streaming experiment."""
    from experiments import run_incremental_experiment
    from visualizations import plot_incremental_convergence

    logger.info("=== Incremental PCA (Streaming) ===")

    results = run_incremental_experiment(
        data_flat,
        image_shape=tuple(config["dataset"]["target_size"]),
        n_components=config["incremental_pca"]["n_components"],
        batch_size=config["incremental_pca"]["batch_size"],
    )
    results.to_csv(str(out_dir / "incremental_pca_results.csv"), index=False)

    plot_incremental_convergence(results, save_path=str(out_dir / "incremental_convergence.png"))
    logger.info("Incremental PCA complete.")
    return results


def run_animation(data_flat, images, labels, config, out_dir):
    """Progressive reconstruction animation."""
    from visualizations import create_progressive_animation

    logger.info("=== Progressive Reconstruction Animation ===")
    anim_cfg = config["animation"]

    output_path = str(out_dir / f"progressive_reconstruction.{anim_cfg['output_format']}")
    create_progressive_animation(
        data_flat[0],
        data_flat,
        image_shape=tuple(config["dataset"]["target_size"]),
        components_sequence=anim_cfg["components_sequence"],
        output_path=output_path,
        fps=anim_cfg["fps"],
    )
    logger.info("Animation complete.")


def run_sparse_comparison(data_flat, images, labels, config, out_dir):
    """Sparse PCA vs Standard PCA component visualization."""
    from pca_core import PCAEngine, SparsePCAEngine
    from visualizations import plot_sparse_vs_dense_components

    logger.info("=== Sparse PCA Components ===")
    image_shape = tuple(config["dataset"]["target_size"])
    n_comp = config["sparse_pca"]["n_components"]

    # Dense components
    engine = PCAEngine(n_components=n_comp)
    engine.fit(data_flat)
    dense_components = engine.components_ if engine.method == "sklearn" else engine.components_

    # Sparse components
    spca = SparsePCAEngine(n_components=n_comp, alpha=config["sparse_pca"]["alpha"])
    spca.fit(data_flat)
    sparse_components = spca.components_

    plot_sparse_vs_dense_components(
        dense_components, sparse_components, image_shape,
        n_show=min(5, n_comp),
        save_path=str(out_dir / "sparse_vs_dense_components.png"),
    )
    logger.info("Sparse comparison complete.")


def run_probabilistic(data_flat, images, labels, config, out_dir):
    """Probabilistic PCA with uncertainty maps."""
    from pca_core import ProbabilisticPCAEngine
    from visualizations import plot_uncertainty_map

    logger.info("=== Probabilistic PCA ===")
    image_shape = tuple(config["dataset"]["target_size"])
    n_comp = config["probabilistic_pca"]["n_components"]

    ppca = ProbabilisticPCAEngine(n_components=n_comp)
    ppca.fit(data_flat)
    reconstructed = np.clip(ppca.reconstruct(data_flat), 0, 1)
    uncertainty_maps = ppca.uncertainty_map(data_flat[:5], image_shape)

    for i in range(3):
        plot_uncertainty_map(
            data_flat[i], reconstructed[i], uncertainty_maps[i], image_shape,
            save_path=str(out_dir / f"uncertainty_map_{i}.png"),
        )
    logger.info(f"Noise variance (mean): {np.mean(ppca.noise_variance):.6f}")
    logger.info("Probabilistic PCA complete.")


# Experiment registry
EXPERIMENTS = {
    "basic": run_basic_reconstruction,
    "sweep": run_sweep,
    "denoising": run_denoising,
    "recognition": run_recognition,
    "anomaly": run_anomaly,
    "compression": run_compression,
    "info_theory": run_info_theory,
    "methods": run_methods,
    "arithmetic": run_arithmetic,
    "incremental": run_incremental,
    "animation": run_animation,
    "sparse": run_sparse_comparison,
    "probabilistic": run_probabilistic,
}


def main():
    parser = argparse.ArgumentParser(description="PCA Image Reconstruction & Analysis")
    parser.add_argument("--experiment", "-e", type=str, default="all",
                       help="Experiment to run (or 'all')")
    parser.add_argument("--config", "-c", type=str, default="config.yaml",
                       help="Path to config file")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available experiments")
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for name in EXPERIMENTS:
            print(f"  - {name}")
        print("  - all (run everything)")
        return

    config = load_config(args.config)
    out_dir = setup_output_dir(config)
    images, data_flat, labels = load_dataset(config)

    if args.experiment == "all":
        for name, func in EXPERIMENTS.items():
            try:
                func(data_flat, images, labels, config, out_dir)
            except Exception as e:
                logger.error(f"Experiment '{name}' failed: {e}")
    else:
        if args.experiment not in EXPERIMENTS:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available: {list(EXPERIMENTS.keys())}")
            sys.exit(1)
        EXPERIMENTS[args.experiment](data_flat, images, labels, config, out_dir)

    logger.info(f"\nAll outputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()

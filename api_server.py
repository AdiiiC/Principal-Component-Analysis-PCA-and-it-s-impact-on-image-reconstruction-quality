"""
PCA Image Reconstruction - FastAPI Endpoint
============================================
Launch: uvicorn api_server:app --reload --port 8000

Endpoints:
    POST /reconstruct        - Reconstruct an uploaded image
    POST /denoise           - Denoise an uploaded image  
    GET  /eigenfaces        - Get eigenfaces as images
    GET  /metrics           - Get reconstruction metrics for all component counts
    GET  /health            - Health check
"""

import io
import base64
import numpy as np
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pca_core.data_loader import load_dataset, IMAGE_DATASETS
from pca_core import PCAEngine, add_noise, prefilter_noisy
from pca_core.metrics import compute_all_metrics
from pca_core.utils import get_compression_ratio

app = FastAPI(
    title="PCA Image Reconstruction API",
    description="API for PCA-based image compression, reconstruction, and denoising",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_DATASET = "olivetti"

# Cap sample counts per dataset so PCA fits stay responsive.
DATASET_MAX_SAMPLES = {"olivetti": None, "digits": None, "lfw": 800}

# Lazily-populated caches.
_datasets: dict = {}          # name -> {"images", "data_flat", "labels", "shape"}
_engines: dict = {}           # (name, k) -> PCAEngine


def _validate_dataset(dataset: str) -> str:
    key = dataset.lower()
    if key not in IMAGE_DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown dataset '{dataset}'. Available: {list(IMAGE_DATASETS)}",
        )
    return key


def get_data(dataset: str = DEFAULT_DATASET) -> dict:
    """Load a dataset lazily and cache it in memory."""
    key = _validate_dataset(dataset)
    if key not in _datasets:
        images, data_flat, labels, shape = load_dataset(
            key, max_samples=DATASET_MAX_SAMPLES.get(key)
        )
        _datasets[key] = {
            "images": images,
            "data_flat": data_flat,
            "labels": labels,
            "shape": tuple(shape),
        }
    return _datasets[key]


def max_components_for(data_flat: np.ndarray) -> int:
    """The largest number of components a dataset can support."""
    return int(min(300, data_flat.shape[0] - 1, data_flat.shape[1]))


def get_engine(dataset: str, n_components: int) -> PCAEngine:
    """Get or create a cached PCA engine for a dataset + component count."""
    key = _validate_dataset(dataset)
    data = get_data(key)
    k = max(1, min(n_components, max_components_for(data["data_flat"])))
    cache_key = (key, k)
    if cache_key not in _engines:
        engine = PCAEngine(n_components=k)
        engine.fit(data["data_flat"])
        _engines[cache_key] = engine
    return _engines[cache_key]


def image_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy image to base64 PNG string."""
    img_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode="L")
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": "PCA Image Reconstruction"}


@app.get("/datasets")
async def list_datasets():
    """List the datasets available to train/explore, without loading them."""
    return {
        "default": DEFAULT_DATASET,
        "datasets": [
            {
                "id": key,
                "label": meta["label"],
                "description": meta["description"],
                "kind": meta["kind"],
                "image_shape": list(meta["shape"]),
                "requires_download": meta["requires_download"],
            }
            for key, meta in IMAGE_DATASETS.items()
        ],
    }


@app.get("/dataset_info")
async def dataset_info(dataset: str = Query(DEFAULT_DATASET)):
    """Return metadata about a dataset for the frontend."""
    key = _validate_dataset(dataset)
    data = get_data(key)
    data_flat = data["data_flat"]
    meta = IMAGE_DATASETS[key]
    return {
        "id": key,
        "n_samples": int(len(data_flat)),
        "n_features": int(data_flat.shape[1]),
        "image_shape": list(data["shape"]),
        "n_subjects": int(len(np.unique(data["labels"]))),
        "max_components": max_components_for(data_flat),
        "dataset": meta["label"],
        "kind": meta["kind"],
        "description": meta["description"],
    }


@app.post("/reconstruct")
async def reconstruct(
    file: UploadFile = File(...),
    k: int = Query(40, ge=1, le=300, description="Number of PCA components"),
    dataset: str = Query(DEFAULT_DATASET),
):
    """
    Reconstruct an uploaded image using PCA trained on the chosen dataset.

    Returns the reconstructed image (base64) and quality metrics.
    """
    try:
        data = get_data(dataset)
        shape = data["shape"]
        contents = await file.read()
        # PIL resize expects (width, height); dataset shape is (rows, cols).
        # LANCZOS matches the training data loader for a cleaner, consistent target.
        img = (
            Image.open(io.BytesIO(contents))
            .convert("L")
            .resize((shape[1], shape[0]), Image.LANCZOS)
        )
        img_array = np.array(img, dtype=np.float64) / 255.0
        img_flat = img_array.flatten().reshape(1, -1)

        engine = get_engine(dataset, k)
        reconstructed = np.clip(engine.reconstruct(img_flat), 0, 1)

        metrics = compute_all_metrics(img_flat[0], reconstructed[0], shape)
        compression_ratio = get_compression_ratio(img_flat.shape[1], engine.n_components, 1)

        return JSONResponse({
            "reconstructed_image": image_to_base64(reconstructed[0].reshape(shape)),
            "original_image": image_to_base64(img_array),
            "n_components": engine.n_components,
            "metrics": {
                "psnr": round(metrics["psnr"], 4),
                "ssim": round(metrics["ssim"], 4),
                "mse": round(metrics["mse"], 6),
            },
            "compression_ratio": round(compression_ratio, 2),
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/denoise")
async def denoise(
    file: UploadFile = File(...),
    k: int = Query(40, ge=1, le=300),
    noise_type: str = Query("gaussian", pattern="^(gaussian|salt_pepper)$"),
    noise_level: float = Query(0.2, ge=0.01, le=0.9),
    dataset: str = Query(DEFAULT_DATASET),
):
    """
    Add noise to uploaded image, then denoise via PCA reconstruction.
    """
    try:
        data = get_data(dataset)
        shape = data["shape"]
        contents = await file.read()
        img = (
            Image.open(io.BytesIO(contents))
            .convert("L")
            .resize((shape[1], shape[0]), Image.LANCZOS)
        )
        img_array = np.array(img, dtype=np.float64) / 255.0
        img_flat = img_array.flatten().reshape(1, -1)

        # Add noise
        noisy_flat = add_noise(img_flat, noise_type=noise_type, noise_level=noise_level)

        # Denoise: pre-filter impulsive noise so PCA's L2 projection isn't
        # dominated by outliers, then project onto the clean top-k subspace.
        prefiltered = prefilter_noisy(noisy_flat, noise_type, shape)
        engine = get_engine(dataset, k)
        denoised = np.clip(engine.reconstruct(prefiltered), 0, 1)

        noisy_metrics = compute_all_metrics(img_flat[0], noisy_flat[0], shape)
        denoised_metrics = compute_all_metrics(img_flat[0], denoised[0], shape)

        return JSONResponse({
            "original_image": image_to_base64(img_array),
            "noisy_image": image_to_base64(noisy_flat[0].reshape(shape)),
            "denoised_image": image_to_base64(denoised[0].reshape(shape)),
            "noise_type": noise_type,
            "noise_level": noise_level,
            "n_components": engine.n_components,
            "noisy_metrics": {
                "psnr": round(noisy_metrics["psnr"], 4),
                "ssim": round(noisy_metrics["ssim"], 4),
            },
            "denoised_metrics": {
                "psnr": round(denoised_metrics["psnr"], 4),
                "ssim": round(denoised_metrics["ssim"], 4),
            },
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/eigenfaces")
async def get_eigenfaces(
    k: int = Query(16, ge=1, le=100, description="Number of eigenfaces to return"),
    dataset: str = Query(DEFAULT_DATASET),
):
    """Return the leading principal components as base64 images."""
    engine = get_engine(dataset, k)
    shape = get_data(dataset)["shape"]
    n = engine.n_components
    components = engine.components_

    eigenfaces = []
    for i in range(n):
        ef = components[i].reshape(shape)
        # Normalize to [0, 1] for display
        ef_norm = (ef - ef.min()) / (ef.max() - ef.min() + 1e-10)
        eigenfaces.append({
            "component": i + 1,
            "image": image_to_base64(ef_norm),
            "explained_variance_ratio": float(engine.explained_variance_ratio_[i]),
        })

    return JSONResponse({
        "n_components": n,
        "eigenfaces": eigenfaces,
        "cumulative_variance": float(engine.cumulative_variance()[n - 1]),
    })


@app.get("/metrics")
async def get_metrics_sweep(
    max_k: int = Query(100, ge=5, le=300),
    step: int = Query(5, ge=1, le=20),
    dataset: str = Query(DEFAULT_DATASET),
):
    """Get reconstruction metrics across component counts."""
    data = get_data(dataset)
    data_flat = data["data_flat"]
    shape = data["shape"]
    cap = max_components_for(data_flat)
    sample = data_flat[:50]

    component_range = sorted({min(k, cap) for k in range(1, max_k + 1, step)})

    from pca_core.metrics import compute_batch_metrics
    results = []
    for k in component_range:
        engine = get_engine(dataset, k)
        reconstructed = np.clip(engine.reconstruct(sample), 0, 1)
        metrics = compute_batch_metrics(sample, reconstructed, shape)
        results.append({
            "n_components": k,
            "avg_psnr": round(float(np.mean(metrics["psnr"])), 4),
            "avg_ssim": round(float(np.mean(metrics["ssim"])), 4),
            "avg_mse": round(float(np.mean(metrics["mse"])), 6),
            "compression_ratio": round(get_compression_ratio(data_flat.shape[1], k, 1), 2),
        })

    return JSONResponse({"sweep_results": results})


@app.get("/sample/{index}")
async def get_sample(
    index: int,
    k: int = Query(40, ge=1, le=300),
    dataset: str = Query(DEFAULT_DATASET),
):
    """Get a sample image from the dataset and its reconstruction."""
    data = get_data(dataset)
    data_flat = data["data_flat"]
    labels = data["labels"]
    shape = data["shape"]

    if index < 0 or index >= len(data_flat):
        raise HTTPException(status_code=404, detail=f"Index must be 0-{len(data_flat)-1}")

    engine = get_engine(dataset, k)
    reconstructed = np.clip(engine.reconstruct(data_flat[index:index+1]), 0, 1)
    metrics = compute_all_metrics(data_flat[index], reconstructed[0], shape)

    compression_ratio = get_compression_ratio(data_flat.shape[1], engine.n_components, 1)
    return JSONResponse({
        "index": index,
        "subject_id": int(labels[index]),
        "original_image": image_to_base64(data_flat[index].reshape(shape)),
        "reconstructed_image": image_to_base64(reconstructed[0].reshape(shape)),
        "n_components": engine.n_components,
        "compression_ratio": round(compression_ratio, 2),
        "metrics": {
            "psnr": round(metrics["psnr"], 4),
            "ssim": round(metrics["ssim"], 4),
            "mse": round(metrics["mse"], 6),
        },
    })


@app.get("/denoise/sample/{index}")
async def denoise_sample(
    index: int,
    k: int = Query(40, ge=1, le=300),
    noise_type: str = Query("gaussian", pattern="^(gaussian|salt_pepper)$"),
    noise_level: float = Query(0.2, ge=0.01, le=0.9),
    dataset: str = Query(DEFAULT_DATASET),
):
    """Add noise to a dataset sample then denoise it via PCA reconstruction."""
    data = get_data(dataset)
    data_flat = data["data_flat"]
    shape = data["shape"]

    if index < 0 or index >= len(data_flat):
        raise HTTPException(status_code=404, detail=f"Index must be 0-{len(data_flat)-1}")

    original_flat = data_flat[index:index + 1]
    noisy_flat = add_noise(original_flat, noise_type=noise_type, noise_level=noise_level)

    # Pre-filter impulsive noise before the PCA projection (see /denoise).
    prefiltered = prefilter_noisy(noisy_flat, noise_type, shape)
    engine = get_engine(dataset, k)
    denoised = np.clip(engine.reconstruct(prefiltered), 0, 1)

    noisy_metrics = compute_all_metrics(original_flat[0], noisy_flat[0], shape)
    denoised_metrics = compute_all_metrics(original_flat[0], denoised[0], shape)

    return JSONResponse({
        "index": index,
        "original_image": image_to_base64(original_flat[0].reshape(shape)),
        "noisy_image": image_to_base64(noisy_flat[0].reshape(shape)),
        "denoised_image": image_to_base64(denoised[0].reshape(shape)),
        "noise_type": noise_type,
        "noise_level": noise_level,
        "n_components": engine.n_components,
        "noisy_metrics": {
            "psnr": round(noisy_metrics["psnr"], 4),
            "ssim": round(noisy_metrics["ssim"], 4),
        },
        "denoised_metrics": {
            "psnr": round(denoised_metrics["psnr"], 4),
            "ssim": round(denoised_metrics["ssim"], 4),
        },
    })

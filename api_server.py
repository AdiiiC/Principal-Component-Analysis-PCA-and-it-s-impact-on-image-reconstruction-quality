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
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pca_core.data_loader import load_olivetti
from pca_core import PCAEngine, add_noise
from pca_core.metrics import compute_all_metrics
from pca_core.utils import get_compression_ratio

app = FastAPI(
    title="PCA Image Reconstruction API",
    description="API for PCA-based image compression, reconstruction, and denoising",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_images = None
_data_flat = None
_labels = None
_engines = {}
IMAGE_SHAPE = (64, 64)


def get_data():
    """Load dataset lazily."""
    global _images, _data_flat, _labels
    if _data_flat is None:
        _images, _data_flat, _labels = load_olivetti()
    return _images, _data_flat, _labels


def get_engine(n_components: int) -> PCAEngine:
    """Get or create a cached PCA engine."""
    if n_components not in _engines:
        _, data_flat, _ = get_data()
        engine = PCAEngine(n_components=n_components)
        engine.fit(data_flat)
        _engines[n_components] = engine
    return _engines[n_components]


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


@app.post("/reconstruct")
async def reconstruct(
    file: UploadFile = File(...),
    k: int = Query(40, ge=1, le=300, description="Number of PCA components"),
):
    """
    Reconstruct an uploaded image using PCA.

    Returns the reconstructed image (base64) and quality metrics.
    """
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("L").resize(IMAGE_SHAPE)
        img_array = np.array(img, dtype=np.float64) / 255.0
        img_flat = img_array.flatten().reshape(1, -1)

        engine = get_engine(k)
        reconstructed = np.clip(engine.reconstruct(img_flat), 0, 1)

        metrics = compute_all_metrics(img_flat[0], reconstructed[0], IMAGE_SHAPE)
        compression_ratio = get_compression_ratio(img_flat.shape[1], k, 1)

        return JSONResponse({
            "reconstructed_image": image_to_base64(reconstructed[0].reshape(IMAGE_SHAPE)),
            "original_image": image_to_base64(img_array),
            "n_components": k,
            "metrics": {
                "psnr": round(metrics["psnr"], 4),
                "ssim": round(metrics["ssim"], 4),
                "mse": round(metrics["mse"], 6),
            },
            "compression_ratio": round(compression_ratio, 2),
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/denoise")
async def denoise(
    file: UploadFile = File(...),
    k: int = Query(40, ge=1, le=300),
    noise_type: str = Query("gaussian", regex="^(gaussian|salt_pepper)$"),
    noise_level: float = Query(0.2, ge=0.01, le=0.9),
):
    """
    Add noise to uploaded image, then denoise via PCA reconstruction.
    """
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("L").resize(IMAGE_SHAPE)
        img_array = np.array(img, dtype=np.float64) / 255.0
        img_flat = img_array.flatten().reshape(1, -1)

        # Add noise
        noisy_flat = add_noise(img_flat, noise_type=noise_type, noise_level=noise_level)

        # Denoise
        engine = get_engine(k)
        denoised = np.clip(engine.reconstruct(noisy_flat), 0, 1)

        noisy_metrics = compute_all_metrics(img_flat[0], noisy_flat[0], IMAGE_SHAPE)
        denoised_metrics = compute_all_metrics(img_flat[0], denoised[0], IMAGE_SHAPE)

        return JSONResponse({
            "original_image": image_to_base64(img_array),
            "noisy_image": image_to_base64(noisy_flat[0].reshape(IMAGE_SHAPE)),
            "denoised_image": image_to_base64(denoised[0].reshape(IMAGE_SHAPE)),
            "noise_type": noise_type,
            "noise_level": noise_level,
            "n_components": k,
            "noisy_metrics": {
                "psnr": round(noisy_metrics["psnr"], 4),
                "ssim": round(noisy_metrics["ssim"], 4),
            },
            "denoised_metrics": {
                "psnr": round(denoised_metrics["psnr"], 4),
                "ssim": round(denoised_metrics["ssim"], 4),
            },
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/eigenfaces")
async def get_eigenfaces(
    k: int = Query(16, ge=1, le=100, description="Number of eigenfaces to return"),
):
    """Return eigenfaces as base64 images."""
    engine = get_engine(k)
    components = engine.components_ if engine.method == "sklearn" else engine.components_

    eigenfaces = []
    for i in range(k):
        ef = components[i].reshape(IMAGE_SHAPE)
        # Normalize to [0, 1] for display
        ef_norm = (ef - ef.min()) / (ef.max() - ef.min() + 1e-10)
        eigenfaces.append({
            "component": i + 1,
            "image": image_to_base64(ef_norm),
            "explained_variance_ratio": float(engine.explained_variance_ratio_[i]),
        })

    return JSONResponse({
        "n_components": k,
        "eigenfaces": eigenfaces,
        "cumulative_variance": float(engine.cumulative_variance()[k - 1]),
    })


@app.get("/metrics")
async def get_metrics_sweep(
    max_k: int = Query(100, ge=5, le=300),
    step: int = Query(5, ge=1, le=20),
):
    """Get reconstruction metrics across component counts."""
    _, data_flat, _ = get_data()
    component_range = list(range(1, max_k + 1, step))

    results = []
    for k in component_range:
        engine = get_engine(k)
        reconstructed = np.clip(engine.reconstruct(data_flat[:50]), 0, 1)

        from pca_core.metrics import compute_batch_metrics
        metrics = compute_batch_metrics(data_flat[:50], reconstructed, IMAGE_SHAPE)

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
):
    """Get a sample face from the dataset and its reconstruction."""
    _, data_flat, labels = get_data()

    if index < 0 or index >= len(data_flat):
        raise HTTPException(status_code=404, detail=f"Index must be 0-{len(data_flat)-1}")

    engine = get_engine(k)
    reconstructed = np.clip(engine.reconstruct(data_flat[index:index+1]), 0, 1)
    metrics = compute_all_metrics(data_flat[index], reconstructed[0], IMAGE_SHAPE)

    return JSONResponse({
        "index": index,
        "subject_id": int(labels[index]),
        "original_image": image_to_base64(data_flat[index].reshape(IMAGE_SHAPE)),
        "reconstructed_image": image_to_base64(reconstructed[0].reshape(IMAGE_SHAPE)),
        "n_components": k,
        "metrics": {
            "psnr": round(metrics["psnr"], 4),
            "ssim": round(metrics["ssim"], 4),
            "mse": round(metrics["mse"], 6),
        },
    })

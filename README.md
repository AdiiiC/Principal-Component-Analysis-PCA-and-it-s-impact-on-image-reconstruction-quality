# PCA Image Reconstruction & Explainability Suite

A comprehensive research-grade exploration of **Principal Component Analysis (PCA)** for image compression, reconstruction, denoising, face recognition, anomaly detection, and more — with interactive visualizations, a REST API, and a deployable Streamlit app.

---

## Features

| Category | What's Included |
|----------|----------------|
| **Core PCA** | Standard (SVD), Manual (Eigendecomposition), Incremental, Kernel, Sparse, Probabilistic, Robust (RPCA), 2D-PCA |
| **Metrics** | MSE, PSNR, SSIM, Perceptual (VGG16), Compression Ratio, Explained Variance |
| **Experiments** | Component sweep, denoising, face recognition, anomaly detection, PCA vs JPEG, information theory, cross-dataset generalization, method comparison, incremental streaming |
| **Advanced** | Face arithmetic (interpolation, averaging, outlier synthesis), progressive reconstruction animation, uncertainty maps, sparse vs dense interpretability |
| **Deployment** | Streamlit interactive app, FastAPI REST endpoint |
| **Engineering** | Unit tests, YAML config, argparse CLI, modular architecture |

---

## Project Structure

```
├── pca_core/                  # Core reusable module
│   ├── __init__.py
│   ├── data_loader.py         # Dataset loading (Olivetti, LFW, custom)
│   ├── pca_engine.py          # All PCA implementations
│   ├── metrics.py             # MSE, PSNR, SSIM, perceptual
│   └── utils.py               # Compression, quantization, entropy, MI
├── experiments/               # All experiment implementations
│   ├── component_sweep.py
│   ├── denoising.py
│   ├── face_recognition.py
│   ├── anomaly_detection.py
│   ├── compression_comparison.py
│   ├── information_theory.py
│   ├── cross_dataset.py
│   ├── method_comparison.py
│   ├── incremental_streaming.py
│   └── face_arithmetic.py
├── visualizations/            # All plotting and animation
│   ├── plots.py
│   └── animation.py
├── tests/                     # Unit tests
│   └── test_pca.py
├── run_all.py                 # Unified CLI entry point
├── streamlit_app.py           # Interactive web app
├── api_server.py              # FastAPI REST endpoint
├── config.yaml                # Centralized configuration
├── requirements.txt           # Dependencies
├── main.py - main6.py        # Original scripts (preserved)
└── README.md
```

---

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run All Experiments
```bash
python run_all.py
```
Outputs (plots, CSVs, animations) are saved to `outputs/`.

### Run Specific Experiment
```bash
python run_all.py --experiment basic
python run_all.py --experiment sweep
python run_all.py --experiment denoising
python run_all.py --experiment recognition
python run_all.py --experiment anomaly
python run_all.py --experiment compression
python run_all.py --experiment methods
python run_all.py --experiment arithmetic
python run_all.py --experiment animation
python run_all.py --experiment sparse
python run_all.py --experiment probabilistic
python run_all.py --experiment incremental
python run_all.py --experiment info_theory
```

### Launch Interactive App
```bash
streamlit run streamlit_app.py
```

### Launch API Server
```bash
uvicorn api_server:app --reload --port 8000
```
API docs at: http://localhost:8000/docs

### Run Tests
```bash
pytest
```

---

## Mathematical Foundation

PCA finds the orthogonal directions of maximum variance in the data:

$$C = \frac{1}{n} X^T X$$

The principal components are the eigenvectors of the covariance matrix $C$, sorted by eigenvalue magnitude. The proportion of variance captured by the first $k$ components is:

$$\text{Variance Captured} = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}$$

Reconstruction from $k$ components:
$$\hat{x} = \bar{x} + \sum_{i=1}^{k} z_i \cdot v_i$$

where $z_i = (x - \bar{x})^T v_i$ are the PCA coefficients and $v_i$ are eigenvectors.

---

## Key Experiments

### 1. Component Sweep
Evaluates MSE, PSNR, SSIM, and compression ratio as $k$ varies from 1 to 100.

### 2. PCA as a Denoiser
Tests reconstruction as a denoising method against Gaussian, salt & pepper, and occlusion noise.

### 3. Face Recognition
Uses PCA projections as features for SVM/k-NN classification, plotting accuracy vs $k$.

### 4. Anomaly Detection
Uses reconstruction error as an anomaly score, with ROC-AUC evaluation on synthetic outliers.

### 5. PCA vs JPEG
Direct file-size vs quality comparison between PCA compression and JPEG at matched bit rates.

### 6. Face Arithmetic
Interpolation, averaging, and random walks in latent space — demonstrating generative capabilities.

### 7. Robust PCA
L+S decomposition for handling corrupted data, separating low-rank signal from sparse noise.

### 8. Information Theory
Entropy and mutual information per component, revealing which PCs carry identity information.

### 9. Progressive Animation
Animated GIF showing reconstruction improving as components are added one-by-one.

### 10. Method Comparison
Benchmarks Standard, Incremental, Kernel, Sparse, Probabilistic, Robust, and 2D-PCA.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reconstruct` | Upload image → get PCA reconstruction + metrics |
| POST | `/denoise` | Upload image → add noise → denoise via PCA |
| GET | `/eigenfaces?k=16` | Get eigenfaces as base64 images |
| GET | `/metrics?max_k=100` | Metrics sweep across component counts |
| GET | `/sample/{index}?k=40` | Get dataset sample + reconstruction |
| GET | `/health` | Health check |

---

## Configuration

All hyperparameters are in `config.yaml`:
- Dataset selection and parameters
- PCA variant settings (components, kernel, alpha, etc.)
- Sweep ranges and noise levels
- Visualization preferences
- Animation settings
- API/Streamlit configuration

---

## Technologies

- **NumPy / SciPy** — Core linear algebra
- **scikit-learn** — PCA implementations, classification, preprocessing
- **scikit-image** — SSIM metric
- **matplotlib** — Static visualizations
- **PyTorch / torchvision** — VGG16 perceptual metrics
- **Streamlit** — Interactive web app
- **FastAPI** — REST API
- **imageio** — Animation generation
- **pandas** — Results tabulation

---

## License

MIT

# Eigenlab — PCA Image Reconstruction Explorer (Web)

A high-end React frontend for the PCA image-reconstruction research suite. It replaces the
Streamlit app with a modern, animated single-page application that talks to the existing
FastAPI backend (`api_server.py`).

Built with **Vite + React + TypeScript + Tailwind CSS + Framer Motion + Recharts**.

## Datasets

Switch training data live from the bar under the nav; every view re-fits PCA on the chosen set:

- **Olivetti Faces** — 400 grayscale portraits, 64×64 (bundled/cached, offline).
- **Labeled Faces in the Wild (LFW)** — cropped photos of public figures resized to 64×64
  (downloaded by scikit-learn on first use, then cached).
- **Handwritten Digits** — 1,797 downsampled 8×8 scans of the digits 0–9 (bundled, offline).

## Features

- **Overview** — animated hero with live dataset stats.
- **Reconstruction** — sweep `k` components and face index, live PSNR / SSIM / MSE / compression,
  a client-side error heatmap, and a draggable before/after compare slider.
- **Eigenfaces** — gallery of principal components with per-component explained variance.
- **Analytics** — quality-vs-components and compression-vs-quality charts.
- **Denoising** — add Gaussian or salt-and-pepper noise and recover it through the PCA subspace.
- **Your Image** — drag-and-drop upload, reconstruct against the learned eigenspace, and download.

## Running locally

You need **two** processes: the Python API and the web dev server.

### 1. Start the API (from the project root)

```bash
pip install -r requirements.txt          # once
uvicorn api_server:app --reload --port 8000
```

### 2. Start the web app (from this `web/` folder)

```bash
npm install                              # once
npm run dev
```

Open http://localhost:5173. In development, all `/api/*` requests are proxied to the API on
port 8000 (see `vite.config.ts`), so there are no CORS issues.

## Production build

```bash
npm run build      # outputs static assets to dist/
npm run preview    # serve the production build locally
```

When deploying, either serve `dist/` behind a reverse proxy that forwards `/api` to the
FastAPI service, or set the API base URL in `src/lib/api.ts`.

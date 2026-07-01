// Typed client for the PCA FastAPI backend (proxied under /api in dev).

const BASE = '/api'

export type Metrics = { psnr: number; ssim: number; mse: number }

export type DatasetInfo = {
  id: string
  n_samples: number
  n_features: number
  image_shape: [number, number]
  n_subjects: number
  max_components: number
  dataset: string
  kind: string
  description: string
}

export type DatasetSummary = {
  id: string
  label: string
  description: string
  kind: string
  image_shape: [number, number]
  requires_download: boolean
}

export type SampleResult = {
  index: number
  subject_id: number
  original_image: string
  reconstructed_image: string
  n_components: number
  compression_ratio: number
  metrics: Metrics
}

export type ReconstructResult = {
  original_image: string
  reconstructed_image: string
  n_components: number
  compression_ratio: number
  metrics: Metrics
}

export type Eigenface = {
  component: number
  image: string
  explained_variance_ratio: number
}

export type EigenfacesResult = {
  n_components: number
  eigenfaces: Eigenface[]
  cumulative_variance: number
}

export type SweepRow = {
  n_components: number
  avg_psnr: number
  avg_ssim: number
  avg_mse: number
  compression_ratio: number
}

export type DenoiseResult = {
  index?: number
  original_image: string
  noisy_image: string
  denoised_image: string
  noise_type: string
  noise_level: number
  n_components: number
  noisy_metrics: { psnr: number; ssim: number }
  denoised_metrics: { psnr: number; ssim: number }
}

export const pngSrc = (b64: string) => `data:image/png;base64,${b64}`

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { signal })
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText)
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => get<{ status: string }>(`/health`),

  datasets: (signal?: AbortSignal) =>
    get<{ default: string; datasets: DatasetSummary[] }>(`/datasets`, signal),

  datasetInfo: (dataset: string, signal?: AbortSignal) =>
    get<DatasetInfo>(`/dataset_info?dataset=${dataset}`, signal),

  sample: (index: number, k: number, dataset: string, signal?: AbortSignal) =>
    get<SampleResult>(`/sample/${index}?k=${k}&dataset=${dataset}`, signal),

  eigenfaces: (k: number, dataset: string, signal?: AbortSignal) =>
    get<EigenfacesResult>(`/eigenfaces?k=${k}&dataset=${dataset}`, signal),

  metrics: (maxK: number, step: number, dataset: string, signal?: AbortSignal) =>
    get<{ sweep_results: SweepRow[] }>(
      `/metrics?max_k=${maxK}&step=${step}&dataset=${dataset}`,
      signal,
    ),

  denoiseSample: (
    index: number,
    k: number,
    noiseType: string,
    noiseLevel: number,
    dataset: string,
    signal?: AbortSignal,
  ) =>
    get<DenoiseResult>(
      `/denoise/sample/${index}?k=${k}&noise_type=${noiseType}&noise_level=${noiseLevel}&dataset=${dataset}`,
      signal,
    ),

  reconstructUpload: async (
    file: File,
    k: number,
    dataset: string,
  ): Promise<ReconstructResult> => {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(`${BASE}/reconstruct?k=${k}&dataset=${dataset}`, {
      method: 'POST',
      body: form,
    })
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`)
    return res.json()
  },

  denoiseUpload: async (
    file: File,
    k: number,
    noiseType: string,
    noiseLevel: number,
    dataset: string,
  ): Promise<DenoiseResult> => {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(
      `${BASE}/denoise?k=${k}&noise_type=${noiseType}&noise_level=${noiseLevel}&dataset=${dataset}`,
      { method: 'POST', body: form },
    )
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`)
    return res.json()
  },
}

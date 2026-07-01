import { useState } from 'react'
import { motion } from 'framer-motion'
import { Shuffle, TrendingUp } from 'lucide-react'
import { api, pngSrc, type DatasetInfo } from '../lib/api'
import { useAsync, useDebounced } from '../lib/hooks'
import { ImageFrame } from '../components/imagery'
import {
  ErrorNote,
  Panel,
  SectionHeading,
  Segmented,
  Slider,
  Spinner,
  StatCard,
} from '../components/primitives'

type NoiseType = 'gaussian' | 'salt_pepper'

export function Denoise({ info }: { info: DatasetInfo }) {
  const maxK = Math.min(150, info.max_components)
  const [noiseType, setNoiseType] = useState<NoiseType>('gaussian')
  const [noiseLevel, setNoiseLevel] = useState(0.25)
  const [k, setK] = useState(40)
  const [index, setIndex] = useState(0)

  const dLevel = useDebounced(noiseLevel, 220)
  const dK = useDebounced(k, 220)
  const dIndex = useDebounced(index, 220)

  const { data, loading, error } = useAsync(
    (signal) => api.denoiseSample(dIndex, dK, noiseType, dLevel, info.id, signal),
    [dIndex, dK, noiseType, dLevel, info.id],
  )

  const gainP = data ? data.denoised_metrics.psnr - data.noisy_metrics.psnr : 0
  const gainS = data ? data.denoised_metrics.ssim - data.noisy_metrics.ssim : 0

  return (
    <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6">
      <SectionHeading
        eyebrow="Denoising"
        title="PCA as a denoiser"
        description="Noise mostly lives in the low-variance directions PCA discards. Projecting a corrupted image onto the top-k subspace and back can clean it up — sometimes dramatically."
      />

      <div className="grid gap-6 lg:grid-cols-[340px_1fr]">
        <Panel className="h-fit">
          <div className="space-y-6">
            <div>
              <p className="mb-2 text-sm font-medium text-slate-300">Noise type</p>
              <Segmented<NoiseType>
                value={noiseType}
                onChange={setNoiseType}
                options={[
                  { value: 'gaussian', label: 'Gaussian' },
                  { value: 'salt_pepper', label: 'Salt & Pepper' },
                ]}
              />
            </div>
            <Slider
              label="Noise level"
              value={noiseLevel}
              min={0.05}
              max={0.9}
              step={0.05}
              onChange={setNoiseLevel}
            />
            <Slider label="Components (k)" value={k} min={1} max={maxK} onChange={setK} />
            <Slider label="Sample index" value={index} min={0} max={info.n_samples - 1} onChange={setIndex} />
            <button
              className="btn-ghost w-full"
              onClick={() => setIndex(Math.floor(Math.random() * info.n_samples))}
            >
              <Shuffle size={16} /> Random sample
            </button>

            <div className="grid grid-cols-2 gap-3 pt-1">
              <StatCard label="Noisy PSNR" value={data ? data.noisy_metrics.psnr.toFixed(2) : '—'} unit="dB" accent="pink" />
              <StatCard label="Denoised PSNR" value={data ? data.denoised_metrics.psnr.toFixed(2) : '—'} unit="dB" accent="cyan" />
            </div>
            {data && (
              <div className="flex items-center gap-2 rounded-xl border border-emerald-400/20 bg-emerald-400/10 px-3 py-2 text-sm text-emerald-300">
                <TrendingUp size={16} />
                <span>
                  {gainP >= 0 ? '+' : ''}
                  {gainP.toFixed(2)} dB PSNR · {gainS >= 0 ? '+' : ''}
                  {gainS.toFixed(3)} SSIM recovered
                </span>
              </div>
            )}
          </div>
        </Panel>

        <Panel>
          {error ? (
            <ErrorNote message={error} />
          ) : loading && !data ? (
            <Spinner label="Adding noise & denoising…" />
          ) : data ? (
            <motion.div
              key={`${data.index}-${data.n_components}-${data.noise_type}-${data.noise_level}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
              className="grid gap-6 sm:grid-cols-3"
            >
              <ImageFrame src={pngSrc(data.original_image)} caption="Original" glow="cyan" />
              <ImageFrame
                src={pngSrc(data.noisy_image)}
                caption="Noisy"
                badge={`${Math.round(data.noise_level * 100)}%`}
                glow="pink"
              />
              <ImageFrame
                src={pngSrc(data.denoised_image)}
                caption="Denoised"
                badge={`k=${data.n_components}`}
                glow="violet"
              />
            </motion.div>
          ) : null}
        </Panel>
      </div>
    </div>
  )
}

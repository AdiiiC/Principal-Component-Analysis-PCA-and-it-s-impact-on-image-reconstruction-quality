import { useCallback, useState } from 'react'
import { motion } from 'framer-motion'
import { Shuffle } from 'lucide-react'
import { api, pngSrc, type DatasetInfo } from '../lib/api'
import { useAsync, useDebounced } from '../lib/hooks'
import { CompareSlider, ErrorMap, ImageFrame } from '../components/imagery'
import { ErrorNote, Panel, SectionHeading, Slider, Spinner, StatCard } from '../components/primitives'

export function Reconstruct({ info }: { info: DatasetInfo }) {
  const maxK = Math.min(150, info.max_components)
  const [k, setK] = useState(40)
  const [index, setIndex] = useState(0)

  const dK = useDebounced(k, 200)
  const dIndex = useDebounced(index, 200)

  const { data, loading, error } = useAsync(
    (signal) => api.sample(dIndex, dK, info.id, signal),
    [dIndex, dK, info.id],
  )

  const randomize = useCallback(() => {
    setIndex(Math.floor(Math.random() * info.n_samples))
  }, [info.n_samples])

  return (
    <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6">
      <SectionHeading
        eyebrow="Reconstruction"
        title="Rebuild a face from k components"
        description="PCA keeps only the top-k directions of variance. Fewer components means smaller storage but a blurrier reconstruction. Drag to find the sweet spot."
      />

      <div className="grid gap-6 lg:grid-cols-[340px_1fr]">
        {/* Controls */}
        <Panel className="h-fit">
          <div className="space-y-7">
            <Slider label="Components (k)" value={k} min={1} max={maxK} onChange={setK} />
            <Slider
              label="Sample index"
              value={index}
              min={0}
              max={info.n_samples - 1}
              onChange={setIndex}
            />
            <button className="btn-ghost w-full" onClick={randomize}>
              <Shuffle size={16} /> Random sample
            </button>

            <div className="grid grid-cols-2 gap-3 pt-2">
              <StatCard
                label="PSNR"
                value={data ? data.metrics.psnr.toFixed(2) : '—'}
                unit="dB"
                accent="cyan"
              />
              <StatCard
                label="SSIM"
                value={data ? data.metrics.ssim.toFixed(4) : '—'}
                accent="violet"
              />
              <StatCard
                label="MSE"
                value={data ? data.metrics.mse.toFixed(5) : '—'}
                accent="pink"
              />
              <StatCard
                label="Compression"
                value={data ? `${data.compression_ratio}×` : '—'}
                accent="emerald"
              />
            </div>
            {data && (
              <p className="text-xs text-slate-500">
                Class {data.subject_id} · storing {data.n_components} coefficients instead of{' '}
                {info.n_features} pixels.
              </p>
            )}
          </div>
        </Panel>

        {/* Display */}
        <Panel>
          {error ? (
            <ErrorNote message={error} />
          ) : loading && !data ? (
            <Spinner label="Fitting PCA…" />
          ) : data ? (
            <motion.div
              key={`${data.index}-${data.n_components}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
              className="space-y-8"
            >
              <div className="grid gap-6 sm:grid-cols-3">
                <ImageFrame src={pngSrc(data.original_image)} caption="Original" glow="cyan" />
                <ImageFrame
                  src={pngSrc(data.reconstructed_image)}
                  caption="Reconstructed"
                  badge={`k=${data.n_components}`}
                  glow="violet"
                />
                <ErrorMap
                  originalSrc={pngSrc(data.original_image)}
                  reconstructedSrc={pngSrc(data.reconstructed_image)}
                />
              </div>

              <div className="flex flex-col items-center gap-3 border-t border-line pt-8">
                <p className="text-sm font-medium text-paper-300">Drag to compare</p>
                <CompareSlider
                  beforeSrc={pngSrc(data.original_image)}
                  afterSrc={pngSrc(data.reconstructed_image)}
                />
              </div>
            </motion.div>
          ) : null}
        </Panel>
      </div>
    </div>
  )
}

import { useState } from 'react'
import { motion } from 'framer-motion'
import { api, pngSrc, type DatasetInfo } from '../lib/api'
import { useAsync, useDebounced } from '../lib/hooks'
import { ErrorNote, Panel, SectionHeading, Slider, Spinner, StatCard } from '../components/primitives'

export function Eigenfaces({ info }: { info: DatasetInfo }) {
  const maxShow = Math.min(64, info.max_components)
  const [count, setCount] = useState(24)
  const dCount = useDebounced(count, 220)

  const { data, loading, error } = useAsync(
    (signal) => api.eigenfaces(dCount, info.id, signal),
    [dCount, info.id],
  )

  return (
    <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6">
      <SectionHeading
        eyebrow="Principal Components"
        title={info.kind === 'faces' ? 'The basis faces PCA learns' : 'The basis patterns PCA learns'}
        description="Every reconstruction is a weighted sum of these principal components. The first few capture the broadest variation; later ones encode fine detail."
      />

      <div className="grid gap-6 lg:grid-cols-[340px_1fr]">
        <Panel className="h-fit">
          <div className="space-y-7">
            <Slider
              label="Components to show"
              value={count}
              min={4}
              max={maxShow}
              step={2}
              onChange={setCount}
            />
            <div className="grid grid-cols-2 gap-3">
              <StatCard
                label="Cumulative variance"
                value={data ? `${(data.cumulative_variance * 100).toFixed(1)}` : '—'}
                unit="%"
                accent="cyan"
                hint={`with ${data?.n_components ?? count} PCs`}
              />
              <StatCard
                label="Top PC variance"
                value={
                  data && data.eigenfaces[0]
                    ? `${(data.eigenfaces[0].explained_variance_ratio * 100).toFixed(1)}`
                    : '—'
                }
                unit="%"
                accent="violet"
              />
            </div>
            <p className="text-xs text-slate-500">
              Components are colour-normalised for display. Brighter/darker regions show where each
              component pushes pixel intensity.
            </p>
          </div>
        </Panel>

        <Panel>
          {error ? (
            <ErrorNote message={error} />
          ) : loading && !data ? (
            <Spinner label="Computing eigenfaces…" />
          ) : data ? (
            <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-6">
              {data.eigenfaces.map((ef, i) => (
                <motion.div
                  key={ef.component}
                  initial={{ opacity: 0, scale: 0.85 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: Math.min(i * 0.015, 0.4), duration: 0.3 }}
                  className="group relative"
                >
                  <div className="overflow-hidden rounded-md border border-line bg-ink-950 p-0.5 transition-colors group-hover:border-line-strong">
                    <img
                      src={pngSrc(ef.image)}
                      alt={`PC ${ef.component}`}
                      className="pixelated aspect-square w-full rounded-sm object-cover"
                    />
                  </div>
                  <div className="mt-1.5 flex items-center justify-between px-0.5">
                    <span className="font-mono text-[11px] text-paper-400">PC{ef.component}</span>
                    <span className="font-mono text-[11px] text-accent-cyan">
                      {(ef.explained_variance_ratio * 100).toFixed(1)}%
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>
          ) : null}
        </Panel>
      </div>
    </div>
  )
}

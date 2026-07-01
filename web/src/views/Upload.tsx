import { useCallback, useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { Download, ImagePlus, Loader2 } from 'lucide-react'
import { api, pngSrc, type DatasetInfo, type ReconstructResult } from '../lib/api'
import { useDebounced } from '../lib/hooks'
import { ErrorMap, ImageFrame } from '../components/imagery'
import { ErrorNote, Panel, SectionHeading, Slider, StatCard } from '../components/primitives'

export function Upload({ info }: { info: DatasetInfo }) {
  const maxK = Math.min(150, info.max_components)
  const [file, setFile] = useState<File | null>(null)
  const [k, setK] = useState(60)
  const [result, setResult] = useState<ReconstructResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const dK = useDebounced(k, 300)

  const run = useCallback(
    async (f: File, kk: number) => {
      setLoading(true)
      setError(null)
      try {
        const res = await api.reconstructUpload(f, kk, info.id)
        setResult(res)
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Upload failed')
      } finally {
        setLoading(false)
      }
    },
    [info.id],
  )

  useEffect(() => {
    if (file) run(file, dK)
  }, [file, dK, run])

  const onSelect = (f: File | undefined) => {
    if (!f) return
    if (!f.type.startsWith('image/')) {
      setError('Please choose an image file (jpg, png, bmp).')
      return
    }
    setError(null)
    setFile(f)
  }

  const download = () => {
    if (!result) return
    const a = document.createElement('a')
    a.href = pngSrc(result.reconstructed_image)
    a.download = `pca_reconstruction_k${result.n_components}.png`
    a.click()
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6">
      <SectionHeading
        eyebrow="Your Image"
        title="Project your own image into eigenspace"
        description={`Upload any image. It's converted to ${info.image_shape[0]}×${info.image_shape[1]} grayscale and reconstructed from the components learned on the ${info.dataset} dataset — a great way to see what PCA has (and hasn't) captured.`}
      />

      <div className="grid gap-6 lg:grid-cols-[340px_1fr]">
        <Panel className="h-fit">
          <div className="space-y-6">
            <div
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault()
                setDragOver(true)
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault()
                setDragOver(false)
                onSelect(e.dataTransfer.files?.[0])
              }}
              className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-8 text-center transition-colors ${
                dragOver
                  ? 'border-accent-violet bg-accent-violet/10'
                  : 'border-white/15 bg-white/[0.02] hover:border-white/30'
              }`}
            >
              <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-accent-indigo to-accent-violet shadow-glow">
                <ImagePlus size={22} />
              </span>
              <div>
                <p className="text-sm font-medium text-slate-200">
                  {file ? file.name : 'Drop an image or click to browse'}
                </p>
                <p className="mt-0.5 text-xs text-slate-500">JPG, PNG or BMP</p>
              </div>
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => onSelect(e.target.files?.[0])}
              />
            </div>

            <Slider label="Components (k)" value={k} min={1} max={maxK} onChange={setK} />

            <div className="grid grid-cols-2 gap-3">
              <StatCard label="PSNR" value={result ? result.metrics.psnr.toFixed(2) : '—'} unit="dB" accent="cyan" />
              <StatCard label="SSIM" value={result ? result.metrics.ssim.toFixed(4) : '—'} accent="violet" />
              <StatCard label="MSE" value={result ? result.metrics.mse.toFixed(5) : '—'} accent="pink" />
              <StatCard label="Compression" value={result ? `${result.compression_ratio}×` : '—'} accent="emerald" />
            </div>

            <button className="btn-primary w-full" onClick={download} disabled={!result}>
              <Download size={16} /> Download reconstruction
            </button>
          </div>
        </Panel>

        <Panel>
          {error ? (
            <ErrorNote message={error} />
          ) : !file ? (
            <div className="flex h-full min-h-[300px] flex-col items-center justify-center gap-3 text-center text-slate-500">
              <ImagePlus size={40} className="opacity-40" />
              <p>Upload an image to see its PCA reconstruction.</p>
            </div>
          ) : loading && !result ? (
            <div className="flex min-h-[300px] items-center justify-center gap-3 text-slate-400">
              <Loader2 size={22} className="animate-spin" /> Reconstructing…
            </div>
          ) : result ? (
            <motion.div
              key={result.n_components}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
              className="grid gap-6 sm:grid-cols-3"
            >
              <ImageFrame
                src={pngSrc(result.original_image)}
                caption={`Input (${info.image_shape[0]}×${info.image_shape[1]})`}
                glow="cyan"
              />
              <ImageFrame
                src={pngSrc(result.reconstructed_image)}
                caption="Reconstructed"
                badge={`k=${result.n_components}`}
                glow="violet"
              />
              <ErrorMap
                originalSrc={pngSrc(result.original_image)}
                reconstructedSrc={pngSrc(result.reconstructed_image)}
              />
            </motion.div>
          ) : null}
        </Panel>
      </div>
    </div>
  )
}

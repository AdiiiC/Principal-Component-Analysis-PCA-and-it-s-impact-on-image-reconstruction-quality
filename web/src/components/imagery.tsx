import { useEffect, useRef, useState } from 'react'

/** A framed, upscaled grayscale image tile with a caption. */
export function ImageFrame({
  src,
  caption,
  badge,
  glow = 'violet',
}: {
  src: string
  caption?: string
  badge?: string
  glow?: 'violet' | 'cyan' | 'pink' | 'none'
}) {
  const ring: Record<string, string> = {
    violet: 'shadow-glow',
    cyan: 'shadow-glow-cyan',
    pink: 'shadow-[0_0_40px_-12px_rgba(236,72,153,0.5)]',
    none: '',
  }
  return (
    <div className="flex flex-col items-center gap-2">
      <div className={`relative overflow-hidden rounded-2xl border border-white/10 bg-black/40 ${ring[glow]}`}>
        <img src={src} alt={caption ?? 'image'} className="pixelated h-44 w-44 object-cover sm:h-52 sm:w-52" />
        {badge && (
          <span className="absolute left-2 top-2 rounded-md bg-black/60 px-2 py-0.5 font-mono text-[11px] text-accent-cyan backdrop-blur">
            {badge}
          </span>
        )}
      </div>
      {caption && <span className="text-sm font-medium text-slate-400">{caption}</span>}
    </div>
  )
}

/**
 * Computes a per-pixel absolute-difference "hot" heatmap between two images
 * entirely on the client using canvas.
 */
export function ErrorMap({
  originalSrc,
  reconstructedSrc,
  caption = 'Error map',
}: {
  originalSrc: string
  reconstructedSrc: string
  caption?: string
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [maxErr, setMaxErr] = useState<number>(0)

  useEffect(() => {
    let cancelled = false
    const load = (src: string) =>
      new Promise<HTMLImageElement>((resolve, reject) => {
        const img = new Image()
        img.onload = () => resolve(img)
        img.onerror = reject
        img.src = src
      })

    Promise.all([load(originalSrc), load(reconstructedSrc)]).then(([a, b]) => {
      if (cancelled) return
      const w = a.width
      const h = a.height
      const off = document.createElement('canvas')
      off.width = w
      off.height = h
      const octx = off.getContext('2d', { willReadFrequently: true })!
      octx.drawImage(a, 0, 0)
      const da = octx.getImageData(0, 0, w, h).data
      octx.clearRect(0, 0, w, h)
      octx.drawImage(b, 0, 0)
      const db = octx.getImageData(0, 0, w, h).data

      const diff = new Float32Array(w * h)
      let localMax = 1e-6
      for (let i = 0; i < w * h; i++) {
        const d = Math.abs(da[i * 4] - db[i * 4]) / 255
        diff[i] = d
        if (d > localMax) localMax = d
      }

      const canvas = canvasRef.current
      if (!canvas) return
      canvas.width = w
      canvas.height = h
      const ctx = canvas.getContext('2d')!
      const out = ctx.createImageData(w, h)
      for (let i = 0; i < w * h; i++) {
        const [r, g, bl] = hot(diff[i] / localMax)
        out.data[i * 4] = r
        out.data[i * 4 + 1] = g
        out.data[i * 4 + 2] = bl
        out.data[i * 4 + 3] = 255
      }
      ctx.putImageData(out, 0, 0)
      setMaxErr(localMax)
    })

    return () => {
      cancelled = true
    }
  }, [originalSrc, reconstructedSrc])

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-black/40 shadow-[0_0_40px_-12px_rgba(236,72,153,0.5)]">
        <canvas ref={canvasRef} className="pixelated h-44 w-44 sm:h-52 sm:w-52" />
        <span className="absolute bottom-2 right-2 rounded-md bg-black/60 px-2 py-0.5 font-mono text-[11px] text-accent-pink backdrop-blur">
          max {(maxErr * 100).toFixed(1)}%
        </span>
      </div>
      {caption && <span className="text-sm font-medium text-slate-400">{caption}</span>}
    </div>
  )
}

/** "Hot" colormap: black -> red -> yellow -> white. Input t in [0,1]. */
function hot(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t))
  const r = Math.min(1, x / 0.4)
  const g = Math.min(1, Math.max(0, (x - 0.4) / 0.4))
  const b = Math.min(1, Math.max(0, (x - 0.8) / 0.2))
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)]
}

/** Draggable before/after comparison slider. */
export function CompareSlider({
  beforeSrc,
  afterSrc,
  beforeLabel = 'Original',
  afterLabel = 'Reconstructed',
}: {
  beforeSrc: string
  afterSrc: string
  beforeLabel?: string
  afterLabel?: string
}) {
  const [pos, setPos] = useState(50)
  const containerRef = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)

  const move = (clientX: number) => {
    const el = containerRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const p = ((clientX - rect.left) / rect.width) * 100
    setPos(Math.min(100, Math.max(0, p)))
  }

  useEffect(() => {
    const onMove = (e: MouseEvent) => dragging.current && move(e.clientX)
    const onUp = () => (dragging.current = false)
    const onTouch = (e: TouchEvent) => dragging.current && move(e.touches[0].clientX)
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    window.addEventListener('touchmove', onTouch)
    window.addEventListener('touchend', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      window.removeEventListener('touchmove', onTouch)
      window.removeEventListener('touchend', onUp)
    }
  }, [])

  return (
    <div
      ref={containerRef}
      className="relative aspect-square w-full max-w-sm select-none overflow-hidden rounded-2xl border border-white/10 bg-black/40 shadow-glow"
      onMouseDown={(e) => {
        dragging.current = true
        move(e.clientX)
      }}
      onTouchStart={(e) => {
        dragging.current = true
        move(e.touches[0].clientX)
      }}
    >
      <img src={afterSrc} alt={afterLabel} className="pixelated absolute inset-0 h-full w-full object-cover" />
      <div className="absolute inset-0 overflow-hidden" style={{ width: `${pos}%` }}>
        <img
          src={beforeSrc}
          alt={beforeLabel}
          className="pixelated h-full object-cover"
          style={{ width: containerRef.current?.clientWidth ?? '100%' }}
        />
      </div>
      <span className="absolute left-2 top-2 rounded-md bg-black/60 px-2 py-0.5 text-[11px] text-slate-200 backdrop-blur">
        {beforeLabel}
      </span>
      <span className="absolute right-2 top-2 rounded-md bg-black/60 px-2 py-0.5 text-[11px] text-slate-200 backdrop-blur">
        {afterLabel}
      </span>
      <div className="absolute inset-y-0 w-0.5 bg-white/80" style={{ left: `${pos}%` }}>
        <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/30 bg-gradient-to-br from-accent-cyan to-accent-violet p-2 shadow-glow">
          <div className="h-3 w-3 rounded-full bg-white" />
        </div>
      </div>
    </div>
  )
}

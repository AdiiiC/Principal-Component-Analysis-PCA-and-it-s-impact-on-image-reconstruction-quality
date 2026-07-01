import { motion } from 'framer-motion'
import { ArrowRight, Grid3x3, Image, LineChart, Sparkles, Upload, Waves } from 'lucide-react'
import type { ViewId } from '../components/Nav'
import type { DatasetInfo } from '../lib/api'

const features: {
  id: ViewId
  title: string
  desc: string
  icon: typeof Image
}[] = [
  {
    id: 'reconstruct',
    title: 'Interactive Reconstruction',
    desc: 'Sweep the number of components and watch a face rebuild in real time with PSNR, SSIM and a live error heatmap.',
    icon: Image,
  },
  {
    id: 'eigenfaces',
    title: 'Eigenfaces Gallery',
    desc: 'Visualise the principal components — the ghostly basis faces that PCA learns from the dataset.',
    icon: Grid3x3,
  },
  {
    id: 'analytics',
    title: 'Quality Analytics',
    desc: 'Chart reconstruction quality and compression ratio against the number of retained components.',
    icon: LineChart,
  },
  {
    id: 'denoise',
    title: 'Denoising Lab',
    desc: 'Corrupt an image with noise, then use the PCA subspace as a denoiser and measure the recovery.',
    icon: Waves,
  },
  {
    id: 'upload',
    title: 'Bring Your Own Image',
    desc: 'Upload any face and project it onto the learned eigenspace to see how well it compresses.',
    icon: Upload,
  },
]

const fade = {
  hidden: { opacity: 0, y: 24 },
  show: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.06, duration: 0.5, ease: [0.22, 1, 0.36, 1] as const },
  }),
}

export function Overview({
  setView,
  info,
}: {
  setView: (v: ViewId) => void
  info: DatasetInfo | null
}) {
  const stats = [
    { label: 'Samples', value: info?.n_samples ?? '—' },
    { label: 'Dimensions', value: info ? `${info.image_shape[0]}×${info.image_shape[1]}` : '—' },
    { label: 'Classes', value: info?.n_subjects ?? '—' },
    { label: 'Pixels / image', value: info?.n_features ?? '—' },
  ]

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6">
      {/* Hero */}
      <section className="relative pt-16 pb-14 text-center sm:pt-24">
        <motion.span
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="tag mx-auto mb-6"
        >
          <Sparkles size={14} className="text-accent" />
          Principal Component Analysis · Explorable
        </motion.span>

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          className="mx-auto max-w-4xl text-5xl font-extrabold leading-[1.05] tracking-tight text-paper-100 sm:text-7xl"
        >
          See how <span className="accent-text">PCA rebuilds</span> an image from almost nothing
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.6 }}
          className="mx-auto mt-6 max-w-2xl text-lg text-paper-400"
        >
          An interactive lab for image compression, reconstruction and denoising. Drag a slider, keep a
          handful of principal components, and watch reconstruction quality trade off against size.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.6 }}
          className="mt-9 flex flex-wrap items-center justify-center gap-3"
        >
          <button className="btn-primary text-base" onClick={() => setView('reconstruct')}>
            Start exploring <ArrowRight size={18} />
          </button>
          <button className="btn-ghost text-base" onClick={() => setView('eigenfaces')}>
            View eigenfaces
          </button>
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.6 }}
          className="mx-auto mt-16 grid max-w-3xl grid-cols-2 gap-3 sm:grid-cols-4"
        >
          {stats.map((s) => (
            <div key={s.label} className="card px-4 py-5">
              <p className="font-mono text-3xl font-bold text-paper-100">{s.value}</p>
              <p className="mt-1 text-xs uppercase tracking-wider text-paper-400">{s.label}</p>
            </div>
          ))}
        </motion.div>
      </section>

      {/* Feature grid */}
      <section className="pb-20">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((f, i) => {
            const Icon = f.icon
            return (
              <motion.button
                key={f.id}
                custom={i}
                variants={fade}
                initial="hidden"
                whileInView="show"
                viewport={{ once: true, margin: '-40px' }}
                whileHover={{ y: -4 }}
                onClick={() => setView(f.id)}
                className="group card p-6 text-left transition-colors hover:border-line-strong"
              >
                <span className="mb-4 inline-flex h-11 w-11 items-center justify-center rounded-md border border-line-strong bg-ink-900 text-accent">
                  <Icon size={20} />
                </span>
                <h3 className="flex items-center gap-1.5 text-lg font-semibold text-paper-100">
                  {f.title}
                  <ArrowRight
                    size={16}
                    className="translate-x-0 text-accent opacity-0 transition-all group-hover:translate-x-1 group-hover:opacity-100"
                  />
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-paper-400">{f.desc}</p>
              </motion.button>
            )
          })}
        </div>
      </section>
    </div>
  )
}

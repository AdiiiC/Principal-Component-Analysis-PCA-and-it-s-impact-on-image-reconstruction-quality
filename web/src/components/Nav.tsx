import { motion } from 'framer-motion'
import { Boxes, Grid3x3, Image, LineChart, Sparkles, Upload, Waves } from 'lucide-react'
import type { ComponentType } from 'react'

export type ViewId =
  | 'overview'
  | 'reconstruct'
  | 'eigenfaces'
  | 'analytics'
  | 'denoise'
  | 'upload'

export const NAV_ITEMS: { id: ViewId; label: string; icon: ComponentType<{ size?: number }> }[] = [
  { id: 'overview', label: 'Overview', icon: Sparkles },
  { id: 'reconstruct', label: 'Reconstruct', icon: Image },
  { id: 'eigenfaces', label: 'Eigenfaces', icon: Grid3x3 },
  { id: 'analytics', label: 'Analytics', icon: LineChart },
  { id: 'denoise', label: 'Denoise', icon: Waves },
  { id: 'upload', label: 'Your Image', icon: Upload },
]

export function Nav({
  view,
  setView,
  online,
}: {
  view: ViewId
  setView: (v: ViewId) => void
  online: boolean | null
}) {
  return (
    <header className="sticky top-0 z-40 border-b border-white/10 bg-ink-950/70 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <button
          onClick={() => setView('overview')}
          className="flex items-center gap-2.5 text-left"
        >
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-accent-indigo to-accent-violet shadow-glow">
            <Boxes size={18} />
          </span>
          <span className="hidden sm:block">
            <span className="block text-sm font-bold leading-tight text-white">Eigenlab</span>
            <span className="block text-[11px] leading-tight text-slate-400">PCA Reconstruction</span>
          </span>
        </button>

        <nav className="flex items-center gap-1 overflow-x-auto rounded-2xl border border-white/10 bg-white/5 p-1">
          {NAV_ITEMS.map((item) => {
            const active = item.id === view
            const Icon = item.icon
            return (
              <button
                key={item.id}
                onClick={() => setView(item.id)}
                className={`relative flex items-center gap-1.5 whitespace-nowrap rounded-xl px-3 py-2 text-sm font-medium transition-colors ${
                  active ? 'text-white' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {active && (
                  <motion.span
                    layoutId="nav-active"
                    className="absolute inset-0 rounded-xl bg-gradient-to-r from-accent-indigo/80 to-accent-violet/80"
                    transition={{ type: 'spring', stiffness: 400, damping: 32 }}
                  />
                )}
                <span className="relative z-10 flex items-center gap-1.5">
                  <Icon size={15} />
                  <span className="hidden md:inline">{item.label}</span>
                </span>
              </button>
            )
          })}
        </nav>

        <div className="hidden items-center gap-2 lg:flex">
          <span
            className={`h-2 w-2 rounded-full ${
              online === null ? 'bg-slate-500' : online ? 'bg-emerald-400 shadow-[0_0_10px_#34d399]' : 'bg-rose-500'
            }`}
          />
          <span className="text-xs text-slate-400">
            {online === null ? 'Connecting' : online ? 'API online' : 'API offline'}
          </span>
        </div>
      </div>
    </header>
  )
}

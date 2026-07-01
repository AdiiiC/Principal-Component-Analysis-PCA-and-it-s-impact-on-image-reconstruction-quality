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
    <header className="sticky top-0 z-40 border-b border-line bg-ink-950/90 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <button
          onClick={() => setView('overview')}
          className="flex items-center gap-2.5 text-left"
        >
          <span className="flex h-9 w-9 items-center justify-center rounded-md border border-line-strong bg-ink-800 text-accent">
            <Boxes size={18} />
          </span>
          <span className="hidden sm:block">
            <span className="block text-sm font-bold leading-tight text-paper-100">Eigenlab</span>
            <span className="block text-[11px] leading-tight text-paper-400">PCA Reconstruction</span>
          </span>
        </button>

        <nav className="flex items-center gap-1 overflow-x-auto rounded-md border border-line bg-ink-900 p-1">
          {NAV_ITEMS.map((item) => {
            const active = item.id === view
            const Icon = item.icon
            return (
              <button
                key={item.id}
                onClick={() => setView(item.id)}
                className={`relative flex items-center gap-1.5 whitespace-nowrap rounded-sm px-3 py-2 text-sm font-medium transition-colors ${
                  active ? 'text-ink-950' : 'text-paper-400 hover:text-paper-100'
                }`}
              >
                {active && (
                  <motion.span
                    layoutId="nav-active"
                    className="absolute inset-0 rounded-sm bg-accent"
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
            className={`h-1.5 w-1.5 rounded-full ${
              online === null ? 'bg-paper-600' : online ? 'bg-accent-sage' : 'bg-rose-400'
            }`}
          />
          <span className="text-xs text-paper-400">
            {online === null ? 'Connecting' : online ? 'API online' : 'API offline'}
          </span>
        </div>
      </div>
    </header>
  )
}

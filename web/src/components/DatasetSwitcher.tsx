import { motion } from 'framer-motion'
import { Database, ScanFace, Hash, CloudDownload } from 'lucide-react'
import type { DatasetSummary } from '../lib/api'

const kindIcon: Record<string, typeof ScanFace> = {
  faces: ScanFace,
  digits: Hash,
}

export function DatasetSwitcher({
  datasets,
  current,
  onChange,
}: {
  datasets: DatasetSummary[]
  current: string
  onChange: (id: string) => void
}) {
  return (
    <div className="sticky top-[57px] z-30 border-b border-line bg-ink-950/85 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-3 px-4 py-2.5 sm:px-6">
        <span className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-paper-400">
          <Database size={14} /> Dataset
        </span>
        <div className="flex flex-wrap items-center gap-1.5">
          {datasets.map((d) => {
            const active = d.id === current
            const Icon = kindIcon[d.kind] ?? Database
            return (
              <button
                key={d.id}
                onClick={() => onChange(d.id)}
                title={d.description}
                className={`relative flex items-center gap-1.5 rounded-sm px-3 py-1.5 text-sm font-medium transition-colors ${
                  active ? 'text-ink-950' : 'text-paper-400 hover:text-paper-100'
                }`}
              >
                {active && (
                  <motion.span
                    layoutId="dataset-active"
                    className="absolute inset-0 rounded-sm bg-accent"
                    transition={{ type: 'spring', stiffness: 400, damping: 32 }}
                  />
                )}
                <span className="relative z-10 flex items-center gap-1.5">
                  <Icon size={14} />
                  {d.label}
                  {d.requires_download && (
                    <CloudDownload size={12} className="opacity-60" />
                  )}
                </span>
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}

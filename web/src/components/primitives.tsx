import type { ReactNode } from 'react'
import { motion } from 'framer-motion'

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  suffix,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  suffix?: string
  onChange: (v: number) => void
}) {
  return (
    <div className="w-full">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-medium text-slate-300">{label}</span>
        <span className="rounded-lg bg-white/5 px-2 py-0.5 font-mono text-sm text-accent-cyan">
          {value}
          {suffix ?? ''}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  )
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: string }[]
  value: T
  onChange: (v: T) => void
}) {
  return (
    <div className="inline-flex rounded-xl border border-white/10 bg-white/5 p-1">
      {options.map((opt) => {
        const active = opt.value === value
        return (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            className={`relative rounded-lg px-3.5 py-1.5 text-sm font-medium transition-colors ${
              active ? 'text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {active && (
              <motion.span
                layoutId="segmented-active"
                className="absolute inset-0 rounded-lg bg-gradient-to-r from-accent-indigo to-accent-violet"
                transition={{ type: 'spring', stiffness: 400, damping: 32 }}
              />
            )}
            <span className="relative z-10">{opt.label}</span>
          </button>
        )
      })}
    </div>
  )
}

export function StatCard({
  label,
  value,
  unit,
  accent = 'violet',
  hint,
}: {
  label: string
  value: string | number
  unit?: string
  accent?: 'violet' | 'cyan' | 'pink' | 'emerald'
  hint?: string
}) {
  const ring: Record<string, string> = {
    violet: 'from-accent-violet/20 to-transparent',
    cyan: 'from-accent-cyan/20 to-transparent',
    pink: 'from-accent-pink/20 to-transparent',
    emerald: 'from-emerald-400/20 to-transparent',
  }
  const text: Record<string, string> = {
    violet: 'text-accent-violet',
    cyan: 'text-accent-cyan',
    pink: 'text-accent-pink',
    emerald: 'text-emerald-400',
  }
  return (
    <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-white/[0.03] p-4">
      <div className={`pointer-events-none absolute -right-6 -top-6 h-24 w-24 rounded-full bg-gradient-to-br ${ring[accent]} blur-2xl`} />
      <p className="text-xs font-medium uppercase tracking-wider text-slate-400">{label}</p>
      <p className="mt-1 font-mono text-2xl font-semibold text-white">
        {value}
        {unit && <span className={`ml-1 text-sm ${text[accent]}`}>{unit}</span>}
      </p>
      {hint && <p className="mt-0.5 text-xs text-slate-500">{hint}</p>}
    </div>
  )
}

export function Panel({
  children,
  className = '',
}: {
  children: ReactNode
  className?: string
}) {
  return <div className={`card p-5 sm:p-6 ${className}`}>{children}</div>
}

export function SectionHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string
  title: string
  description?: string
}) {
  return (
    <div className="mb-8 max-w-2xl">
      <span className="pill mb-3">{eyebrow}</span>
      <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">{title}</h2>
      {description && <p className="mt-3 text-slate-400">{description}</p>}
    </div>
  )
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-10 text-slate-400">
      <div className="h-9 w-9 animate-spin rounded-full border-2 border-white/10 border-t-accent-violet" />
      {label && <p className="text-sm">{label}</p>}
    </div>
  )
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-200">
      <p className="font-semibold">Something went wrong</p>
      <p className="mt-1 text-rose-300/80">{message}</p>
      <p className="mt-2 text-xs text-rose-300/60">
        Make sure the API is running: <code className="font-mono">uvicorn api_server:app --port 8000</code>
      </p>
    </div>
  )
}

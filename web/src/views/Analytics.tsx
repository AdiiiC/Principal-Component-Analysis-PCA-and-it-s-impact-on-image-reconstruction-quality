import { useState } from 'react'
import {
  Area,
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api, type DatasetInfo } from '../lib/api'
import { useAsync, useDebounced } from '../lib/hooks'
import { ErrorNote, Panel, SectionHeading, Slider, Spinner } from '../components/primitives'

const tooltipStyle = {
  background: 'rgba(10,12,27,0.95)',
  border: '1px solid rgba(255,255,255,0.12)',
  borderRadius: 12,
  color: '#e5e7f0',
  fontSize: 12,
}

export function Analytics({ info }: { info: DatasetInfo }) {
  const [maxK, setMaxK] = useState(Math.min(120, info.max_components))
  const [step, setStep] = useState(6)
  const dMaxK = useDebounced(maxK, 300)
  const dStep = useDebounced(step, 300)

  const { data, loading, error } = useAsync(
    (signal) => api.metrics(Math.min(dMaxK, info.max_components), dStep, info.id, signal),
    [dMaxK, dStep, info.id],
  )

  const rows = data?.sweep_results ?? []

  return (
    <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6">
      <SectionHeading
        eyebrow="Analytics"
        title="Quality vs. compression trade-off"
        description="As you add components, reconstruction quality rises with diminishing returns while the compression ratio falls. The elbow of these curves is the practical sweet spot."
      />

      <Panel className="mb-6">
        <div className="grid gap-8 sm:grid-cols-2">
          <Slider label="Max components" value={maxK} min={20} max={info.max_components} step={10} onChange={setMaxK} />
          <Slider label="Step size" value={step} min={1} max={20} onChange={setStep} />
        </div>
      </Panel>

      {error ? (
        <Panel>
          <ErrorNote message={error} />
        </Panel>
      ) : loading && !rows.length ? (
        <Panel>
          <Spinner label="Running component sweep…" />
        </Panel>
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel>
            <h3 className="mb-1 text-lg font-semibold text-white">Reconstruction quality</h3>
            <p className="mb-4 text-sm text-slate-400">PSNR (dB) and SSIM vs. components</p>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={rows} margin={{ top: 8, right: 12, left: -8, bottom: 0 }}>
                <defs>
                  <linearGradient id="psnrFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#22d3ee" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="n_components" stroke="#64748b" fontSize={12} />
                <YAxis yAxisId="l" stroke="#22d3ee" fontSize={12} />
                <YAxis yAxisId="r" orientation="right" stroke="#8b5cf6" fontSize={12} domain={[0, 1]} />
                <Tooltip contentStyle={tooltipStyle} />
                <Area yAxisId="l" type="monotone" dataKey="avg_psnr" name="PSNR (dB)" stroke="#22d3ee" strokeWidth={2} fill="url(#psnrFill)" />
                <Line yAxisId="r" type="monotone" dataKey="avg_ssim" name="SSIM" stroke="#8b5cf6" strokeWidth={2} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </Panel>

          <Panel>
            <h3 className="mb-1 text-lg font-semibold text-white">Compression vs. quality</h3>
            <p className="mb-4 text-sm text-slate-400">Compression ratio (bars) against PSNR (line)</p>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={rows} margin={{ top: 8, right: 12, left: -8, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="n_components" stroke="#64748b" fontSize={12} />
                <YAxis yAxisId="l" stroke="#ec4899" fontSize={12} />
                <YAxis yAxisId="r" orientation="right" stroke="#22d3ee" fontSize={12} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar yAxisId="l" dataKey="compression_ratio" name="Compression ×" fill="#ec4899" opacity={0.55} radius={[4, 4, 0, 0]} />
                <Line yAxisId="r" type="monotone" dataKey="avg_psnr" name="PSNR (dB)" stroke="#22d3ee" strokeWidth={2} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </Panel>
        </div>
      )}
    </div>
  )
}

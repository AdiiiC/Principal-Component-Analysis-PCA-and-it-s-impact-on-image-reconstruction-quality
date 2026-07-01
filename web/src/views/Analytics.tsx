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
  background: '#1f1c16',
  border: '1px solid rgba(233,225,206,0.18)',
  borderRadius: 6,
  color: '#ece7dc',
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
            <h3 className="mb-1 text-lg font-semibold text-paper-100">Reconstruction quality</h3>
            <p className="mb-4 text-sm text-paper-400">PSNR (dB) and SSIM vs. components</p>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={rows} margin={{ top: 8, right: 12, left: -8, bottom: 0 }}>
                <defs>
                  <linearGradient id="psnrFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#7e93a6" stopOpacity={0.28} />
                    <stop offset="100%" stopColor="#7e93a6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(233,225,206,0.08)" />
                <XAxis dataKey="n_components" stroke="#837c6d" fontSize={12} />
                <YAxis yAxisId="l" stroke="#7e93a6" fontSize={12} />
                <YAxis yAxisId="r" orientation="right" stroke="#cf9b52" fontSize={12} domain={[0, 1]} />
                <Tooltip contentStyle={tooltipStyle} />
                <Area yAxisId="l" type="monotone" dataKey="avg_psnr" name="PSNR (dB)" stroke="#7e93a6" strokeWidth={2} fill="url(#psnrFill)" />
                <Line yAxisId="r" type="monotone" dataKey="avg_ssim" name="SSIM" stroke="#cf9b52" strokeWidth={2} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </Panel>

          <Panel>
            <h3 className="mb-1 text-lg font-semibold text-paper-100">Compression vs. quality</h3>
            <p className="mb-4 text-sm text-paper-400">Compression ratio (bars) against PSNR (line)</p>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={rows} margin={{ top: 8, right: 12, left: -8, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(233,225,206,0.08)" />
                <XAxis dataKey="n_components" stroke="#837c6d" fontSize={12} />
                <YAxis yAxisId="l" stroke="#bd7355" fontSize={12} />
                <YAxis yAxisId="r" orientation="right" stroke="#7e93a6" fontSize={12} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar yAxisId="l" dataKey="compression_ratio" name="Compression ×" fill="#bd7355" opacity={0.6} radius={[2, 2, 0, 0]} />
                <Line yAxisId="r" type="monotone" dataKey="avg_psnr" name="PSNR (dB)" stroke="#7e93a6" strokeWidth={2} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </Panel>
        </div>
      )}
    </div>
  )
}

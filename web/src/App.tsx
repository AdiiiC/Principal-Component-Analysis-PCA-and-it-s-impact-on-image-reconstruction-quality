import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Background } from './components/Background'
import { DatasetSwitcher } from './components/DatasetSwitcher'
import { Nav, type ViewId } from './components/Nav'
import { Spinner } from './components/primitives'
import { api, type DatasetInfo, type DatasetSummary } from './lib/api'
import { Analytics } from './views/Analytics'
import { Denoise } from './views/Denoise'
import { Eigenfaces } from './views/Eigenfaces'
import { Overview } from './views/Overview'
import { Reconstruct } from './views/Reconstruct'
import { Upload } from './views/Upload'

export default function App() {
  const [view, setView] = useState<ViewId>('overview')
  const [datasets, setDatasets] = useState<DatasetSummary[]>([])
  const [dataset, setDataset] = useState<string>('olivetti')
  const [info, setInfo] = useState<DatasetInfo | null>(null)
  const [online, setOnline] = useState<boolean | null>(null)
  const [bootError, setBootError] = useState<string | null>(null)

  // Load the list of available datasets once.
  useEffect(() => {
    let active = true
    api
      .datasets()
      .then((d) => {
        if (!active) return
        setDatasets(d.datasets)
        setDataset(d.default)
        setOnline(true)
      })
      .catch((e: unknown) => {
        if (!active) return
        setOnline(false)
        setBootError(e instanceof Error ? e.message : 'Failed to reach API')
      })
    return () => {
      active = false
    }
  }, [])

  // Load metadata whenever the selected dataset changes.
  useEffect(() => {
    let active = true
    setInfo(null)
    api
      .datasetInfo(dataset)
      .then((d) => {
        if (!active) return
        setInfo(d)
        setOnline(true)
      })
      .catch((e: unknown) => {
        if (!active) return
        setOnline(false)
        setBootError(e instanceof Error ? e.message : 'Failed to reach API')
      })
    return () => {
      active = false
    }
  }, [dataset])

  const changeView = (v: ViewId) => {
    setView(v)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className="min-h-screen">
      <Background />
      <Nav view={view} setView={changeView} online={online} />
      {datasets.length > 0 && (
        <DatasetSwitcher datasets={datasets} current={dataset} onChange={setDataset} />
      )}

      <main className="min-h-[70vh]">
        <AnimatePresence mode="wait">
          <motion.div
            key={`${view}-${dataset}`}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
          >
            {view === 'overview' && <Overview setView={changeView} info={info} />}

            {view !== 'overview' &&
              (info ? (
                <>
                  {view === 'reconstruct' && <Reconstruct info={info} />}
                  {view === 'eigenfaces' && <Eigenfaces info={info} />}
                  {view === 'analytics' && <Analytics info={info} />}
                  {view === 'denoise' && <Denoise info={info} />}
                  {view === 'upload' && <Upload info={info} />}
                </>
              ) : (
                <div className="mx-auto max-w-md px-4 py-24 text-center">
                  {online === false ? (
                    <div className="card p-8">
                      <h2 className="text-xl font-semibold text-white">API not reachable</h2>
                      <p className="mt-2 text-sm text-slate-400">Start the backend, then reload:</p>
                      <pre className="mt-4 overflow-x-auto rounded-xl bg-black/50 p-3 text-left font-mono text-xs text-accent-cyan">
uvicorn api_server:app --reload --port 8000
                      </pre>
                      {bootError && <p className="mt-3 text-xs text-rose-400/70">{bootError}</p>}
                    </div>
                  ) : (
                    <Spinner label="Loading dataset…" />
                  )}
                </div>
              ))}
          </motion.div>
        </AnimatePresence>
      </main>

      <footer className="border-t border-white/10 py-8">
        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-3 px-4 text-sm text-slate-500 sm:flex-row sm:px-6">
          <p>
            <span className="font-semibold text-slate-300">Eigenlab</span> · PCA image reconstruction
            explorer
          </p>
          <p>
            {info?.dataset ?? 'Olivetti Faces'} · React + FastAPI · scikit-learn PCA
          </p>
        </div>
      </footer>
    </div>
  )
}

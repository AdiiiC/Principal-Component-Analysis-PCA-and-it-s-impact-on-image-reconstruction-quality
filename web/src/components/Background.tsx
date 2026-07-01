/** Ambient background: a subtle hairline grid with a warm vignette. */
export function Background() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-ink-950">
      <div className="absolute inset-0 grid-bg opacity-70" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,transparent_55%,#0d0c0a_100%)]" />
    </div>
  )
}

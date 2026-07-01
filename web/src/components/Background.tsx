import { motion } from 'framer-motion'

/** Ambient animated background: gradient blobs + subtle grid + vignette. */
export function Background() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-60" />

      <motion.div
        className="absolute -left-32 -top-32 h-[38rem] w-[38rem] rounded-full bg-accent-indigo/25 blur-[120px]"
        animate={{ x: [0, 60, 0], y: [0, 40, 0] }}
        transition={{ duration: 18, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="absolute -right-40 top-20 h-[34rem] w-[34rem] rounded-full bg-accent-violet/25 blur-[120px]"
        animate={{ x: [0, -50, 0], y: [0, 60, 0] }}
        transition={{ duration: 22, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="absolute bottom-0 left-1/3 h-[30rem] w-[30rem] rounded-full bg-accent-cyan/20 blur-[130px]"
        animate={{ x: [0, 40, 0], y: [0, -30, 0] }}
        transition={{ duration: 26, repeat: Infinity, ease: 'easeInOut' }}
      />

      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_40%,#05060f_100%)]" />
    </div>
  )
}

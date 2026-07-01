/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: {
          950: '#12110e',
          900: '#181611',
          800: '#1f1c16',
          700: '#2a261e',
        },
        line: {
          DEFAULT: 'rgba(233,225,206,0.10)',
          strong: 'rgba(233,225,206,0.18)',
        },
        paper: {
          100: '#ece7dc',
          300: '#c7c0b2',
          400: '#a29a8a',
          500: '#837c6d',
          600: '#6f685c',
        },
        accent: {
          DEFAULT: '#cf9b52',
          strong: '#e0b06a',
          deep: '#a86a28',
          violet: '#cf9b52',
          indigo: '#cf9b52',
          cyan: '#7e93a6',
          pink: '#bd7355',
          sage: '#84a06e',
        },
        slate: {
          200: '#d6cfc0',
          300: '#c7c0b2',
          400: '#a29a8a',
          500: '#837c6d',
          600: '#6f685c',
          700: '#4a4438',
        },
        emerald: {
          300: '#9bb583',
          400: '#84a06e',
          500: '#6f8b5b',
        },
        rose: {
          200: '#d9a58c',
          300: '#c98a6c',
          400: '#bd7355',
          500: '#a85f42',
        },
      },
      fontFamily: {
        sans: ['"Space Grotesk"', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
      },
      borderRadius: {
        none: '0',
        sm: '3px',
        DEFAULT: '5px',
        md: '6px',
        lg: '8px',
        xl: '8px',
        '2xl': '10px',
        full: '9999px',
      },
      boxShadow: {
        panel: '0 1px 0 rgba(233,225,206,0.03) inset, 0 8px 24px -18px rgba(0,0,0,0.9)',
        lift: '0 12px 30px -20px rgba(0,0,0,0.95)',
      },
    },
  },
  plugins: [],
}

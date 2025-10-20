/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'tower-bg': '#0f0f0f',
        'tower-bg-elevated': '#1a1a1a',
        'tower-bg-card': '#1e1e1e',
        'tower-bg-hover': '#252525',
        'tower-accent-primary': '#3b82f6',
        'tower-accent-secondary': '#8b5cf6',
        'tower-accent-success': '#10b981',
        'tower-accent-warning': '#f59e0b',
        'tower-accent-danger': '#ef4444',
        'tower-text-primary': '#f9fafb',
        'tower-text-secondary': '#d1d5db',
        'tower-text-muted': '#9ca3af',
        'tower-border': '#374151',
        'tower-border-hover': '#4b5563',
      }
    },
  },
  plugins: [],
}

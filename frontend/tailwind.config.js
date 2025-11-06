/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Claude Code exact colors
        'tower-bg': '#0d0d0d',           // Claude main bg
        'tower-bg-elevated': '#1a1a1a',  // Claude sidebar/elevated
        'tower-bg-card': '#1e1e1e',      // Claude card background
        'tower-bg-hover': '#2a2a2a',     // Claude hover states
        'tower-accent-primary': '#ff6b35', // Claude orange accent
        'tower-accent-secondary': '#4a9eff', // Claude blue
        'tower-accent-success': '#00d4aa',   // Claude success green
        'tower-accent-warning': '#f59e0b',   // Warning amber
        'tower-accent-danger': '#ff4757',    // Claude error red
        'tower-text-primary': '#e4e4e7',     // Claude primary text
        'tower-text-secondary': '#a1a1aa',   // Claude secondary text
        'tower-text-muted': '#71717a',       // Claude muted text
        'tower-border': '#3f3f46',           // Claude border
        'tower-border-hover': '#52525b',     // Claude border hover
      }
    },
  },
  plugins: [],
}

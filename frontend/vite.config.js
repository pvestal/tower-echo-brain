import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  base: '/static/dist/',
  server: {
    port: 5173,
    host: '0.0.0.0'
  },
  build: {
    outDir: '../static/dist',
    emptyOutDir: true
  }
})

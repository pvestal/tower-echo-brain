import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  base: '/echo-brain/',
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src')
    }
  },
  server: {
    port: 5173,
    host: '0.0.0.0',
    proxy: {
      '/api/echo': {
        target: 'http://localhost:8309',
        changeOrigin: true
      },
      '/api/calendar': {
        target: 'http://localhost:8309',
        changeOrigin: true
      },
      '/api/google': {
        target: 'http://localhost:8309',
        changeOrigin: true
      },
      '/api/workers': {
        target: 'http://localhost:8309',
        changeOrigin: true
      }
    }
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    assetsDir: 'assets'
  }
})
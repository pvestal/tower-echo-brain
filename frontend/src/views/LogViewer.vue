<template>
  <div class="log-viewer-page">
    <div class="header">
      <h1>SYSTEM LOGS</h1>
      <div class="controls">
        <select v-model="selectedService" @change="fetchLogs" class="service-select">
          <option value="tower-echo-brain">Echo Brain</option>
          <option value="tower-anime-production">Anime Production</option>
          <option value="tower-dashboard">Dashboard</option>
          <option value="tower-kb">Knowledge Base</option>
          <option value="tower-auth">Auth Service</option>
        </select>
        <input
          v-model.number="lineCount"
          type="number"
          min="10"
          max="1000"
          class="line-input"
          placeholder="Lines"
        />
        <button @click="fetchLogs" :disabled="loading" class="btn-refresh">
          {{ loading ? 'Loading...' : 'Refresh' }}
        </button>
        <label class="auto-refresh">
          <input type="checkbox" v-model="autoRefresh" />
          Auto (2s)
        </label>
        <button @click="clearLogs" class="btn-clear">Clear</button>
      </div>
    </div>

    <div class="log-container">
      <div class="log-stats">
        <span>Service: {{ selectedService }}</span>
        <span>Lines: {{ logs.split('\n').filter(l => l).length }}</span>
        <span>Last Updated: {{ lastUpdated }}</span>
      </div>

      <div class="log-content" ref="logContent">
        <pre v-if="logs">{{ logs }}</pre>
        <div v-else class="no-logs">No logs available</div>
      </div>
    </div>

    <div v-if="error" class="error-banner">
      {{ error }}
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import axios from 'axios'

const selectedService = ref('tower-echo-brain')
const lineCount = ref(100)
const logs = ref('')
const loading = ref(false)
const autoRefresh = ref(false)
const error = ref(null)
const lastUpdated = ref(new Date().toLocaleTimeString())
const logContent = ref(null)

let refreshInterval = null

async function fetchLogs() {
  loading.value = true
  error.value = null

  try {
    // Try to fetch logs via Echo Brain API first
    const response = await axios.post('https://vestal-garcia.duckdns.org/api/echo/admin/logs', {
      service: selectedService.value,
      lines: lineCount.value
    }, { timeout: 5000 })

    if (response.data.logs) {
      logs.value = response.data.logs
    } else {
      // Fallback: Use direct journalctl command through a custom endpoint
      logs.value = await fetchSystemLogs()
    }

    lastUpdated.value = new Date().toLocaleTimeString()

    // Auto-scroll to bottom if autoRefresh is on
    if (autoRefresh.value) {
      await nextTick()
      if (logContent.value) {
        logContent.value.scrollTop = logContent.value.scrollHeight
      }
    }

  } catch (err) {
    error.value = `Failed to fetch logs: ${err.message}`
    // Try direct system log fetch as fallback
    try {
      logs.value = await fetchSystemLogs()
      error.value = null
    } catch (fallbackErr) {
      logs.value = `Unable to fetch logs. Error: ${fallbackErr.message}\n\nMake sure the service is running and accessible.`
    }
  } finally {
    loading.value = false
  }
}

async function fetchSystemLogs() {
  // This would need a backend endpoint that runs journalctl
  // For now, we'll simulate with a placeholder
  return `[Simulated logs for ${selectedService.value}]
${new Date().toISOString()} INFO Starting service...
${new Date().toISOString()} INFO Service ${selectedService.value} initialized
${new Date().toISOString()} INFO Listening on port...
${new Date().toISOString()} INFO Health check passed
${new Date().toISOString()} DEBUG Processing request...

Note: Direct log access requires backend endpoint implementation.
To view real logs, run: sudo journalctl -u ${selectedService.value} -n ${lineCount.value}`
}

function clearLogs() {
  logs.value = ''
  lastUpdated.value = new Date().toLocaleTimeString()
}

function startAutoRefresh() {
  refreshInterval = setInterval(() => {
    if (autoRefresh.value) {
      fetchLogs()
    }
  }, 2000)
}

onMounted(() => {
  fetchLogs()
  startAutoRefresh()
})

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
  }
})
</script>

<style scoped>
.log-viewer-page {
  background: #0a0a0f;
  color: #e2e8f0;
  min-height: 100vh;
  padding: 20px;
  font-family: 'Roboto Mono', monospace;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.header {
  padding: 20px;
  background: #151520;
  border: 1px solid #2d3748;
  border-radius: 4px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 15px;
}

h1 {
  margin: 0;
  font-size: 24px;
  color: #2a7de1;
}

.controls {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
}

.service-select {
  background: #0a0a0f;
  border: 1px solid #2d3748;
  border-radius: 4px;
  padding: 8px 12px;
  color: #e2e8f0;
  font-family: inherit;
  font-size: 14px;
  cursor: pointer;
}

.service-select:focus {
  outline: none;
  border-color: #2a7de1;
}

.line-input {
  background: #0a0a0f;
  border: 1px solid #2d3748;
  border-radius: 4px;
  padding: 8px;
  color: #e2e8f0;
  font-family: inherit;
  font-size: 14px;
  width: 80px;
}

.line-input:focus {
  outline: none;
  border-color: #2a7de1;
}

.btn-refresh, .btn-clear {
  background: #2a7de1;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
  font-family: inherit;
  font-size: 14px;
  transition: background 150ms;
}

.btn-refresh:hover:not(:disabled),
.btn-clear:hover {
  background: #1e6dd0;
}

.btn-refresh:disabled {
  background: #475569;
  cursor: not-allowed;
}

.btn-clear {
  background: #475569;
}

.btn-clear:hover {
  background: #334155;
}

.auto-refresh {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 14px;
  color: #94a3b8;
}

.log-container {
  flex: 1;
  background: #151520;
  border: 1px solid #2d3748;
  border-radius: 4px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.log-stats {
  padding: 10px 20px;
  background: #0a0a0f;
  border-bottom: 1px solid #2d3748;
  display: flex;
  gap: 30px;
  font-size: 12px;
  color: #94a3b8;
}

.log-stats span {
  text-transform: uppercase;
}

.log-content {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
  background: #0a0a0f;
}

.log-content pre {
  margin: 0;
  font-size: 12px;
  line-height: 1.6;
  color: #94a3b8;
  white-space: pre-wrap;
  word-break: break-all;
}

/* Color code different log levels */
.log-content pre {
  color: #e2e8f0;
}

/* Highlight patterns in logs */
.log-content :deep(.error),
.log-content :deep(.ERROR) {
  color: #ef4444;
  font-weight: 500;
}

.log-content :deep(.warn),
.log-content :deep(.WARNING) {
  color: #f59e0b;
}

.log-content :deep(.info),
.log-content :deep(.INFO) {
  color: #2a7de1;
}

.log-content :deep(.debug),
.log-content :deep(.DEBUG) {
  color: #475569;
}

.log-content :deep(.success) {
  color: #19b37b;
}

.no-logs {
  color: #475569;
  text-align: center;
  padding: 40px;
  font-size: 14px;
}

.error-banner {
  padding: 15px 20px;
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid #ef4444;
  border-radius: 4px;
  color: #ef4444;
  font-size: 14px;
}

/* Scrollbar styling */
.log-content::-webkit-scrollbar {
  width: 8px;
}

.log-content::-webkit-scrollbar-track {
  background: #0a0a0f;
}

.log-content::-webkit-scrollbar-thumb {
  background: #2d3748;
  border-radius: 4px;
}

.log-content::-webkit-scrollbar-thumb:hover {
  background: #475569;
}

@media (max-width: 768px) {
  .header {
    flex-direction: column;
    align-items: stretch;
  }

  .controls {
    justify-content: space-between;
  }
}
</style>
<template>
  <div class="health-check">
    <div class="header">
      <h1>ECHO BRAIN STATUS</h1>
      <div class="controls">
        <button @click="refresh" :disabled="loading" class="btn-refresh">
          {{ loading ? 'Checking...' : 'Refresh' }}
        </button>
        <label class="auto-refresh">
          <input type="checkbox" v-model="autoRefresh" />
          Auto (5s)
        </label>
      </div>
    </div>

    <div class="section">
      <h2>Services</h2>
      <div class="service-grid">
        <div v-for="service in services" :key="service.name" class="service-item">
          <span class="status-indicator" :class="service.status"></span>
          <span class="service-name">{{ service.name }} ({{ service.port }})</span>
          <span class="service-status">{{ service.statusText }}</span>
        </div>
      </div>
    </div>

    <div class="section">
      <h2>Verification Tests <button @click="runTests" :disabled="testRunning" class="btn-small">Run All</button></h2>
      <div class="test-grid">
        <div v-for="test in verificationTests" :key="test.name" class="test-item">
          <span class="test-status" :class="test.status">{{ test.icon }}</span>
          <div class="test-details">
            <div class="test-name">{{ test.name }}</div>
            <div class="test-result" :class="test.status">{{ test.result }}</div>
            <pre v-if="test.output" class="test-output">{{ test.output }}</pre>
          </div>
        </div>
      </div>
    </div>

    <div class="section">
      <h2>Recent Logs</h2>
      <div class="log-viewer">
        <pre>{{ logs }}</pre>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import axios from 'axios'

const loading = ref(false)
const testRunning = ref(false)
const autoRefresh = ref(false)
const logs = ref('')

const services = ref([
  { name: 'Echo Brain', port: 8309, status: 'unknown', statusText: 'CHECKING' },
  { name: 'Anime Production', port: 8328, status: 'unknown', statusText: 'CHECKING' },
  { name: 'ComfyUI', port: 8188, status: 'unknown', statusText: 'CHECKING' },
  { name: 'Ollama', port: 11434, status: 'unknown', statusText: 'CHECKING' },
  { name: 'Qdrant', port: 6333, status: 'unknown', statusText: 'CHECKING' },
  { name: 'Redis', port: 6379, status: 'unknown', statusText: 'CHECKING' }
])

const verificationTests = ref([
  {
    name: 'Model Routing',
    status: 'pending',
    icon: '○',
    result: 'Not tested',
    output: ''
  },
  {
    name: 'Context Contamination',
    status: 'pending',
    icon: '○',
    result: 'Not tested',
    output: ''
  },
  {
    name: 'Theater Integration',
    status: 'pending',
    icon: '○',
    result: 'Not tested',
    output: ''
  },
  {
    name: 'WebSocket Connection',
    status: 'pending',
    icon: '○',
    result: 'Not tested',
    output: ''
  }
])

let refreshInterval = null

async function checkService(service) {
  try {
    const endpoints = {
      'Echo Brain': 'https://vestal-garcia.duckdns.org/api/coordination/services',
      'Anime Production': 'https://vestal-garcia.duckdns.org/api/anime/health',
      'ComfyUI': 'https://vestal-garcia.duckdns.org/api/comfyui/system_stats',
      'Ollama': 'https://vestal-garcia.duckdns.org/api/ollama/tags',
      'Qdrant': 'https://vestal-garcia.duckdns.org/api/qdrant/collections',
      'Redis': null // No HTTP endpoint
    }

    if (service.name === 'Redis') {
      // Check via Echo Brain's Redis connection
      const response = await axios.get('https://vestal-garcia.duckdns.org/api/echo/db/stats', { timeout: 2000 })
      service.status = response.data.redis_connected ? 'running' : 'stopped'
      service.statusText = response.data.redis_connected ? 'RUNNING' : 'NOT CONNECTED'
    } else {
      const endpoint = endpoints[service.name]
      await axios.get(endpoint, { timeout: 2000 })
      service.status = 'running'
      service.statusText = 'RUNNING'
    }
  } catch (error) {
    service.status = 'stopped'
    service.statusText = error.code === 'ECONNREFUSED' ? 'NOT RUNNING' : 'ERROR'
  }
}

async function refresh() {
  loading.value = true

  // Check all services
  await Promise.all(services.value.map(checkService))

  // Get logs
  try {
    const response = await axios.post('https://vestal-garcia.duckdns.org/api/echo/admin/logs', {
      lines: 20
    })
    logs.value = response.data.logs || 'Unable to fetch logs'
  } catch (error) {
    logs.value = `Error fetching logs: ${error.message}`
  }

  loading.value = false
}

async function runTests() {
  testRunning.value = true

  // Test 1: Model Routing
  const modelTest = verificationTests.value[0]
  modelTest.status = 'testing'
  modelTest.icon = '⟳'
  try {
    const response = await axios.post('https://vestal-garcia.duckdns.org/api/echo/chat', {
      query: "What model are you using? Just return the model name.",
      temperature: 0
    }, { timeout: 30000 })

    const modelName = response.data.response || response.data.message || ''
    if (modelName.includes('deepseek-coder-v2') || modelName.includes('deepseek-r1')) {
      modelTest.status = 'passed'
      modelTest.icon = '✓'
      modelTest.result = `Model: ${modelName.match(/deepseek-[^\s]+/)[0]}`
    } else {
      modelTest.status = 'failed'
      modelTest.icon = '✗'
      modelTest.result = 'Wrong model selected'
      modelTest.output = modelName
    }
  } catch (error) {
    modelTest.status = 'failed'
    modelTest.icon = '✗'
    modelTest.result = 'API Error'
    modelTest.output = error.message
  }

  // Test 2: Context Contamination
  const contextTest = verificationTests.value[1]
  contextTest.status = 'testing'
  contextTest.icon = '⟳'
  try {
    const response = await axios.post('https://vestal-garcia.duckdns.org/api/echo/chat', {
      query: 'Return exactly: {"test": 1}',
      temperature: 0
    }, { timeout: 30000 })

    const output = response.data.response || response.data.message || ''
    if (output.trim() === '{"test": 1}') {
      contextTest.status = 'passed'
      contextTest.icon = '✓'
      contextTest.result = 'CLEAN - No contamination'
    } else {
      contextTest.status = 'failed'
      contextTest.icon = '✗'
      contextTest.result = 'CONTAMINATED - Got narrative wrapper'
      contextTest.output = output.substring(0, 200)
    }
  } catch (error) {
    contextTest.status = 'failed'
    contextTest.icon = '✗'
    contextTest.result = 'API Error'
    contextTest.output = error.message
  }

  // Test 3: Theater Integration
  const theaterTest = verificationTests.value[2]
  theaterTest.status = 'testing'
  theaterTest.icon = '⟳'
  try {
    const response = await axios.get('https://vestal-garcia.duckdns.org/api/theater/agents', { timeout: 5000 })
    if (response.data && Array.isArray(response.data.agents)) {
      theaterTest.status = 'passed'
      theaterTest.icon = '✓'
      theaterTest.result = `${response.data.agents.length} agents registered`
    } else {
      theaterTest.status = 'warning'
      theaterTest.icon = '⚠'
      theaterTest.result = 'Theater endpoint exists but no agents'
    }
  } catch (error) {
    theaterTest.status = 'failed'
    theaterTest.icon = '✗'
    theaterTest.result = 'NOT INTEGRATED'
    theaterTest.output = 'Theater files not wired up'
  }

  // Test 4: WebSocket
  const wsTest = verificationTests.value[3]
  wsTest.status = 'testing'
  wsTest.icon = '⟳'
  try {
    const ws = new WebSocket('wss://vestal-garcia.duckdns.org/echo-brain/ws/echo')
    await new Promise((resolve, reject) => {
      ws.onopen = () => {
        wsTest.status = 'passed'
        wsTest.icon = '✓'
        wsTest.result = 'WebSocket connected'
        ws.close()
        resolve()
      }
      ws.onerror = () => {
        wsTest.status = 'failed'
        wsTest.icon = '✗'
        wsTest.result = 'WebSocket connection failed'
        reject()
      }
      setTimeout(() => reject(new Error('Timeout')), 3000)
    })
  } catch (error) {
    wsTest.status = 'failed'
    wsTest.icon = '✗'
    wsTest.result = 'WebSocket not available'
  }

  testRunning.value = false
}

onMounted(() => {
  refresh()

  refreshInterval = setInterval(() => {
    if (autoRefresh.value) {
      refresh()
    }
  }, 5000)
})

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
  }
})
</script>

<style scoped>
.health-check {
  background: #0a0a0f;
  color: #e2e8f0;
  min-height: 100vh;
  padding: 20px;
  font-family: 'Roboto Mono', monospace;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  background: #151520;
  border: 1px solid #2d3748;
  border-radius: 4px;
  margin-bottom: 20px;
}

h1 {
  margin: 0;
  font-size: 24px;
  color: #2a7de1;
}

h2 {
  font-size: 16px;
  text-transform: uppercase;
  color: #94a3b8;
  margin-bottom: 15px;
}

.controls {
  display: flex;
  gap: 15px;
  align-items: center;
}

.btn-refresh, .btn-small {
  background: #2a7de1;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
  font-family: inherit;
  transition: background 150ms;
}

.btn-refresh:hover, .btn-small:hover {
  background: #1e6dd0;
}

.btn-refresh:disabled, .btn-small:disabled {
  background: #475569;
  cursor: not-allowed;
}

.btn-small {
  padding: 4px 12px;
  font-size: 12px;
  margin-left: 10px;
}

.auto-refresh {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 14px;
}

.section {
  background: #151520;
  border: 1px solid #2d3748;
  border-radius: 4px;
  padding: 20px;
  margin-bottom: 20px;
}

.service-grid {
  display: grid;
  gap: 10px;
}

.service-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px;
  background: #0a0a0f;
  border: 1px solid #2d3748;
  border-radius: 4px;
}

.status-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: #475569;
}

.status-indicator.running {
  background: #19b37b;
  animation: pulse 2s infinite;
}

.status-indicator.stopped {
  background: #ef4444;
}

.status-indicator.unknown {
  background: #f59e0b;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.service-name {
  flex: 1;
  font-weight: 500;
}

.service-status {
  font-size: 12px;
  text-transform: uppercase;
  color: #94a3b8;
}

.test-grid {
  display: grid;
  gap: 15px;
}

.test-item {
  display: flex;
  gap: 15px;
  padding: 15px;
  background: #0a0a0f;
  border: 1px solid #2d3748;
  border-radius: 4px;
}

.test-status {
  font-size: 20px;
  width: 30px;
  text-align: center;
}

.test-status.passed { color: #19b37b; }
.test-status.failed { color: #ef4444; }
.test-status.warning { color: #f59e0b; }
.test-status.pending { color: #475569; }
.test-status.testing {
  color: #2a7de1;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.test-details {
  flex: 1;
}

.test-name {
  font-weight: 500;
  margin-bottom: 5px;
}

.test-result {
  font-size: 14px;
  color: #94a3b8;
}

.test-result.passed { color: #19b37b; }
.test-result.failed { color: #ef4444; }
.test-result.warning { color: #f59e0b; }

.test-output {
  margin-top: 10px;
  padding: 10px;
  background: #0a0a0f;
  border: 1px solid #2d3748;
  border-radius: 4px;
  font-size: 12px;
  color: #94a3b8;
  overflow-x: auto;
}

.log-viewer {
  background: #0a0a0f;
  border: 1px solid #2d3748;
  border-radius: 4px;
  padding: 15px;
  height: 300px;
  overflow-y: auto;
}

.log-viewer pre {
  margin: 0;
  font-size: 12px;
  line-height: 1.5;
  color: #94a3b8;
}
</style>
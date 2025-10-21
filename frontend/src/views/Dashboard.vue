<template>
  <div class="space-y-6">
    <!-- Metrics Grid -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
      <TowerCard hoverable>
        <h3 class="text-lg font-semibold mb-2 text-tower-text-primary">
          CPU Usage
        </h3>
        <p class="text-3xl font-bold text-tower-accent-primary">
          {{ metrics.cpu_percent || 0 }}%
        </p>
        <p class="text-xs text-tower-text-muted mt-2">Current processor load</p>
      </TowerCard>

      <TowerCard hoverable>
        <h3 class="text-lg font-semibold mb-2 text-tower-text-primary">
          Memory
        </h3>
        <p class="text-3xl font-bold text-tower-accent-success">
          {{ metrics.memory_percent || 0 }}%
        </p>
        <p class="text-xs text-tower-text-muted mt-2">
          {{ metrics.memory_used_gb?.toFixed(1) || 0 }} / {{ metrics.memory_total_gb?.toFixed(0) || 0 }} GB
        </p>
      </TowerCard>

      <TowerCard hoverable>
        <h3 class="text-lg font-semibold mb-2 text-tower-text-primary">
          VRAM (NVIDIA)
        </h3>
        <p class="text-3xl font-bold text-tower-accent-warning">
          {{ metrics.vram_used_gb?.toFixed(1) || 0 }} GB
        </p>
        <p class="text-xs text-tower-text-muted mt-2">
          {{ metrics.vram_total_gb?.toFixed(0) || 0 }} GB total
        </p>
      </TowerCard>

      <TowerCard hoverable>
        <h3 class="text-lg font-semibold mb-2 text-tower-text-primary">
          Services
        </h3>
        <p class="text-3xl font-bold text-tower-accent-primary">
          {{ servicesOnline }}/{{ totalServices }}
        </p>
        <p class="text-xs text-tower-text-muted mt-2">Online services</p>
      </TowerCard>
    </div>

    <!-- System Status -->
    <TowerCard>
      <template #header>
        <h2 class="text-xl font-bold text-tower-text-primary">Service Status</h2>
      </template>
      <div class="space-y-3">
        <div class="status-item" v-for="service in services" :key="service.name">
          <div>
            <div class="text-tower-text-secondary">{{ service.name }}</div>
            <div class="text-xs text-tower-text-muted">{{ service.description }}</div>
          </div>
          <span class="status-badge" :class="service.online ? 'success' : 'offline'">
            {{ service.online ? 'Online' : 'Offline' }}
          </span>
        </div>
      </div>
    </TowerCard>

    <!-- Database Stats -->
    <TowerCard>
      <template #header>
        <div class="flex justify-between items-center">
          <h2 class="text-xl font-bold text-tower-text-primary">Database Status</h2>
          <span class="text-xs text-tower-text-muted">PostgreSQL</span>
        </div>
      </template>
      <div class="space-y-3">
        <div class="status-item">
          <div>
            <div class="text-tower-text-secondary">echo_brain</div>
            <div class="text-xs text-tower-text-muted">Main production database</div>
          </div>
          <span class="text-tower-text-primary font-mono">{{ dbStats.echo_brain || '-' }}</span>
        </div>
        <div class="status-item">
          <div>
            <div class="text-tower-text-secondary">knowledge_base</div>
            <div class="text-xs text-tower-text-muted">KB articles storage</div>
          </div>
          <span class="text-tower-text-primary font-mono">{{ dbStats.knowledge_base || '-' }}</span>
        </div>
        <div class="status-item">
          <div>
            <div class="text-tower-text-secondary">Active Connections</div>
            <div class="text-xs text-tower-text-muted">Current DB connections</div>
          </div>
          <span class="text-tower-text-primary font-mono">{{ dbStats.active_connections || 0 }}</span>
        </div>
        <div class="status-item">
          <div>
            <div class="text-tower-text-secondary">Tables</div>
            <div class="text-xs text-tower-text-muted">echo_brain table count</div>
          </div>
          <span class="text-tower-text-primary font-mono">{{ dbStats.echo_brain_tables || 0 }}</span>
        </div>
      </div>
    </TowerCard>

    <!-- Echo Status -->
    <TowerCard>
      <template #header>
        <h2 class="text-xl font-bold text-tower-text-primary">Echo Brain Status</h2>
      </template>
      <div class="space-y-4">
        <div>
          <h3 class="text-sm font-semibold text-tower-text-secondary mb-2">Current Cognitive Mode</h3>
          <p class="text-tower-text-primary">{{ echoStatus.agentic_persona || 'Loading...' }}</p>
        </div>
        <div>
          <h3 class="text-sm font-semibold text-tower-text-secondary mb-2">Recent Activity</h3>
          <div class="space-y-2">
            <div
              v-if="echoStatus.recent_messages && echoStatus.recent_messages.length > 0"
              v-for="(msg, idx) in echoStatus.recent_messages.slice(0, 5)"
              :key="idx"
              class="text-sm p-2 bg-tower-bg-elevated rounded"
            >
              <span class="text-tower-text-secondary font-mono text-xs">{{ msg.time }}:</span>
              <span class="text-tower-text-primary ml-2">{{ msg.text }}</span>
            </div>
            <div v-else class="text-sm text-tower-text-muted">
              No recent activity
            </div>
          </div>
        </div>
      </div>
    </TowerCard>

    <!-- Error Display -->
    <TowerCard v-if="errorMessage" class="border-tower-accent-danger">
      <template #header>
        <h2 class="text-xl font-bold text-tower-accent-danger">Connection Error</h2>
      </template>
      <p class="text-tower-text-secondary">{{ errorMessage }}</p>
      <p class="text-xs text-tower-text-muted mt-2">
        Check that Echo Brain API is running on port 8309
      </p>
    </TowerCard>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { TowerCard } from '@tower/ui-components'
import axios from 'axios'

// Configure axios base URL
const API_BASE = 'http://192.168.50.135:8309'

// Reactive state with default values
const metrics = ref({
  cpu_percent: 0,
  memory_percent: 0,
  memory_used_gb: 0,
  memory_total_gb: 0,
  vram_used_gb: 0,
  vram_total_gb: 0
})

const services = ref([
  {
    name: 'Echo Brain',
    description: 'Main AI orchestration service',
    endpoint: 'http://192.168.50.135:8309/api/echo/health',
    online: false
  },
  {
    name: 'Knowledge Base',
    description: 'Article storage and retrieval',
    endpoint: 'https://192.168.50.135/api/kb/articles?limit=1',
    online: false
  },
  {
    name: 'ComfyUI',
    description: 'Image/video generation',
    endpoint: 'http://192.168.50.135:8188/',
    online: false
  }
])

const servicesOnline = ref(0)
const totalServices = ref(services.value.length)

const dbStats = ref({
  echo_brain: null,
  knowledge_base: null,
  active_connections: 0,
  echo_brain_tables: 0
})

const echoStatus = ref({
  agentic_persona: 'Connecting...',
  recent_messages: []
})

const errorMessage = ref('')

// Fetch functions
async function fetchMetrics() {
  try {
    const response = await axios.get(`${API_BASE}/api/echo/system/metrics`)
    metrics.value = response.data
    errorMessage.value = ''
  } catch (error) {
    console.error('Failed to fetch metrics:', error)
    errorMessage.value = 'Failed to connect to Echo Brain API'
  }
}

async function fetchServices() {
  let onlineCount = 0
  for (const service of services.value) {
    try {
      const response = await axios.get(service.endpoint)
      service.online = response.status === 200
      if (service.online) onlineCount++
    } catch {
      service.online = false
    }
  }
  servicesOnline.value = onlineCount
}

async function fetchDatabaseStats() {
  try {
    const response = await axios.get(`${API_BASE}/api/echo/db/stats`)
    dbStats.value = response.data
  } catch (error) {
    console.error('Failed to fetch database stats:', error)
  }
}

async function fetchEchoStatus() {
  try {
    const response = await axios.get(`${API_BASE}/api/echo/status`)
    echoStatus.value = response.data
  } catch (error) {
    console.error('Failed to fetch Echo status:', error)
    echoStatus.value.agentic_persona = 'Connection error'
  }
}

// Initialize and setup polling
onMounted(() => {
  // Initial fetch
  fetchMetrics()
  fetchServices()
  fetchDatabaseStats()
  fetchEchoStatus()

  // Auto-refresh
  setInterval(fetchMetrics, 5000)        // 5 seconds
  setInterval(fetchServices, 30000)      // 30 seconds
  setInterval(fetchDatabaseStats, 60000) // 1 minute
  setInterval(fetchEchoStatus, 10000)    // 10 seconds
})
</script>

<style scoped>
.status-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 0;
  border-bottom: 1px solid var(--tower-border);
}

.status-item:last-child {
  border-bottom: none;
}

.status-badge {
  padding: 0.25rem 0.75rem;
  border-radius: 0.25rem;
  font-size: 0.875rem;
  font-weight: 500;
  white-space: nowrap;
}

.status-badge.success {
  background-color: rgba(16, 185, 129, 0.1);
  color: var(--tower-accent-success);
}

.status-badge.offline {
  background-color: rgba(107, 114, 128, 0.1);
  color: var(--tower-text-muted);
}
</style>

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

    <!-- Service Status Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      <TowerCard v-for="service in services" :key="service.name" hoverable>
        <div class="flex items-center justify-between mb-2">
          <h3 class="font-semibold text-tower-text-primary">{{ service.name }}</h3>
          <span class="status-badge" :class="service.online ? 'success' : 'offline'">
            {{ service.online ? 'Online' : 'Offline' }}
          </span>
        </div>
        <p class="text-xs text-tower-text-muted mb-2">{{ service.description }}</p>
        <p v-if="service.port" class="text-xs text-tower-text-secondary font-mono">Port: {{ service.port }}</p>
        <a v-if="service.ui && service.online" :href="service.ui" target="_blank" class="text-xs text-tower-accent-primary hover:underline">
          Open UI →
        </a>
      </TowerCard>
    </div>

    <!-- Database Stats -->
    <TowerCard>
      <template #header>
        <div class="flex justify-between items-center">
          <h2 class="text-xl font-bold text-tower-text-primary">Database Status</h2>
          <span class="text-xs text-tower-text-muted">PostgreSQL</span>
        </div>
      </template>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div class="stat-item">
          <div class="text-tower-text-secondary">echo_brain</div>
          <div class="text-xs text-tower-text-muted">Main production database</div>
          <span class="text-tower-text-primary font-mono font-bold">{{ dbStats.echo_brain || '-' }}</span>
        </div>
        <div class="stat-item">
          <div class="text-tower-text-secondary">knowledge_base</div>
          <div class="text-xs text-tower-text-muted">KB articles storage</div>
          <span class="text-tower-text-primary font-mono font-bold">{{ dbStats.knowledge_base || '-' }}</span>
        </div>
        <div class="stat-item">
          <div class="text-tower-text-secondary">Active Connections</div>
          <div class="text-xs text-tower-text-muted">Current DB connections</div>
          <span class="text-tower-text-primary font-mono font-bold">{{ dbStats.active_connections || 0 }}</span>
        </div>
        <div class="stat-item">
          <div class="text-tower-text-secondary">Tables</div>
          <div class="text-xs text-tower-text-muted">echo_brain table count</div>
          <span class="text-tower-text-primary font-mono font-bold">{{ dbStats.echo_brain_tables || 0 }}</span>
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
const API_BASE = 'http://***REMOVED***:8309'

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
  // Core Infrastructure
  {
    name: 'Echo Brain',
    description: 'AI orchestration & conversation',
    endpoint: 'http://***REMOVED***:8309/api/echo/health',
    ui: 'http://***REMOVED***:8309/static/dist/',
    port: 8309,
    online: false
  },
  {
    name: 'Knowledge Base',
    description: 'Article storage & retrieval',
    endpoint: 'https://***REMOVED***/api/kb/articles?limit=1',
    ui: null,
    port: 8307,
    online: false
  },
  {
    name: 'HashiCorp Vault',
    description: 'Secrets management',
    endpoint: null,
    ui: 'http://***REMOVED***:8200/ui',
    port: 8200,
    online: false
  },
  
  // Media & Generation
  {
    name: 'ComfyUI',
    description: 'Image/video generation',
    endpoint: 'http://***REMOVED***:8188/',
    ui: 'http://***REMOVED***:8188/',
    port: 8188,
    online: false
  },
  {
    name: 'Anime Production',
    description: 'Anime generation pipeline',
    endpoint: 'http://***REMOVED***:8328/api/health',
    ui: 'http://***REMOVED***:8328/project_manager',
    port: 8328,
    online: false
  },
  {
    name: 'Music Production',
    description: 'Music generation service',
    endpoint: 'http://***REMOVED***:8316/api/music-prod/health',
    ui: null,
    port: 8316,
    online: false
  },
  {
    name: 'Jellyfin Media',
    description: 'Media server & streaming',
    endpoint: null,
    ui: 'http://***REMOVED***:8096/',
    port: 8096,
    online: false
  },
  
  // Communication
  {
    name: 'Telegram Bot',
    description: '@PatricksEchobot',
    endpoint: null,
    ui: null,
    port: null,
    online: false
  },
  {
    name: 'Voice WebSocket',
    description: 'Voice streaming service',
    endpoint: 'http://***REMOVED***:8312/api/voice/health',
    ui: null,
    port: 8312,
    online: false
  },
  
  // Integration Services
  {
    name: 'Apple Music',
    description: 'Music integration API',
    endpoint: 'http://***REMOVED***:8315/api/music/health',
    ui: null,
    port: 8315,
    online: false
  },
  {
    name: 'Plaid Financial',
    description: 'Bank account integration',
    endpoint: 'http://***REMOVED***:8089/api/plaid/health',
    ui: 'http://***REMOVED***:8089/plaid/auth',
    port: 8089,
    online: false
  },
  {
    name: 'Auth Service',
    description: 'OAuth & authentication',
    endpoint: 'http://***REMOVED***:8088/api/auth/health',
    ui: null,
    port: 8088,
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
    if (!service.endpoint) {
      // No health endpoint - check if UI is accessible or assume offline
      service.online = false
      continue
    }
    
    try {
      const response = await axios.get(service.endpoint, { timeout: 3000 })
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
.stat-item {
  padding: 1rem;
  background: var(--tower-bg-elevated);
  border-radius: 0.5rem;
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

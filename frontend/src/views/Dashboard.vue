<template>
  <div class="space-y-6">
    <!-- Metrics Grid -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
      <TowerCard hoverable>
        <h3 class="text-lg font-semibold mb-2 text-tower-text-primary">
          CPU Usage
        </h3>
        <p class="text-3xl font-bold text-tower-accent-primary">
          {{ metrics.cpu_percent }}%
        </p>
      </TowerCard>

      <TowerCard hoverable>
        <h3 class="text-lg font-semibold mb-2 text-tower-text-primary">
          Memory
        </h3>
        <p class="text-3xl font-bold text-tower-accent-success">
          {{ metrics.memory_percent }}%
        </p>
      </TowerCard>

      <TowerCard hoverable>
        <h3 class="text-lg font-semibold mb-2 text-tower-text-primary">
          VRAM
        </h3>
        <p class="text-3xl font-bold text-tower-accent-warning">
          {{ metrics.vram_used_gb }}/{{ metrics.vram_total_gb }} GB
        </p>
      </TowerCard>

      <TowerCard hoverable>
        <h3 class="text-lg font-semibold mb-2 text-tower-text-primary">
          Services Online
        </h3>
        <p class="text-3xl font-bold text-tower-accent-primary">
          {{ servicesOnline }}/{{ totalServices }}
        </p>
      </TowerCard>
    </div>

    <!-- System Status -->
    <TowerCard>
      <template #header>
        <h2 class="text-xl font-bold text-tower-text-primary">System Status</h2>
      </template>
      <div class="space-y-3">
        <div class="status-item" v-for="service in services" :key="service.name">
          <span class="text-tower-text-secondary">{{ service.name }}</span>
          <span class="status-badge" :class="service.online ? 'success' : 'offline'">
            {{ service.online ? 'Online' : 'Offline' }}
          </span>
        </div>
      </div>
    </TowerCard>

    <!-- Database Stats -->
    <TowerCard>
      <template #header>
        <h2 class="text-xl font-bold text-tower-text-primary">Database Status</h2>
      </template>
      <div class="space-y-3">
        <div class="status-item">
          <span class="text-tower-text-secondary">echo_brain</span>
          <span class="text-tower-text-primary">{{ dbStats.echo_brain || '-' }}</span>
        </div>
        <div class="status-item">
          <span class="text-tower-text-secondary">knowledge_base</span>
          <span class="text-tower-text-primary">{{ dbStats.knowledge_base || '-' }}</span>
        </div>
        <div class="status-item">
          <span class="text-tower-text-secondary">Active Connections</span>
          <span class="text-tower-text-primary">{{ dbStats.active_connections || 0 }}</span>
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
          <h3 class="text-sm font-semibold text-tower-text-secondary mb-2">Current Persona</h3>
          <p class="text-tower-text-primary">{{ echoStatus.agentic_persona || 'Default mode' }}</p>
        </div>
        <div>
          <h3 class="text-sm font-semibold text-tower-text-secondary mb-2">Recent Activity</h3>
          <div class="space-y-2">
            <div
              v-for="(msg, idx) in echoStatus.recent_messages?.slice(0, 3)"
              :key="idx"
              class="text-sm text-tower-text-muted"
            >
              <span class="text-tower-text-secondary">{{ msg.time }}:</span>
              {{ msg.text }}
            </div>
          </div>
        </div>
      </div>
    </TowerCard>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { TowerCard } from '@tower/ui-components'
import axios from 'axios'

// Reactive state
const metrics = ref({
  cpu_percent: 0,
  memory_percent: 0,
  vram_used_gb: 0,
  vram_total_gb: 0
})

const services = ref([
  { name: 'Echo Brain', endpoint: '/api/echo/health', online: false },
  { name: 'Knowledge Base', endpoint: '/api/kb/stats', online: false },
  { name: 'ComfyUI', endpoint: '/api/comfyui/system_stats', online: false }
])

const servicesOnline = ref(0)
const totalServices = ref(services.value.length)

const dbStats = ref({
  echo_brain: null,
  knowledge_base: null,
  active_connections: 0
})

const echoStatus = ref({
  agentic_persona: 'Loading...',
  recent_messages: []
})

// Fetch functions
async function fetchMetrics() {
  try {
    const response = await axios.get('/api/echo/system/metrics')
    metrics.value = response.data
  } catch (error) {
    console.error('Failed to fetch metrics:', error)
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
    const response = await axios.get('/api/echo/db/stats')
    dbStats.value = response.data
  } catch (error) {
    console.error('Failed to fetch database stats:', error)
  }
}

async function fetchEchoStatus() {
  try {
    const response = await axios.get('/api/echo/status')
    echoStatus.value = response.data
  } catch (error) {
    console.error('Failed to fetch Echo status:', error)
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
  setInterval(fetchMetrics, 5000)       // 5 seconds
  setInterval(fetchServices, 30000)     // 30 seconds
  setInterval(fetchDatabaseStats, 60000) // 1 minute
  setInterval(fetchEchoStatus, 10000)   // 10 seconds
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

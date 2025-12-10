<template>
  <div class="improvement-visualization">
    <div class="header">
      <h1>Echo Brain Improvement System</h1>
      <p>Continuous learning and spatial intelligence monitoring</p>
    </div>

    <!-- Knowledge Sources -->
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-header">
          <span class="metric-label">Claude Conversations</span>
        </div>
        <div class="metric-value">{{ claudeConversations.toLocaleString() }}</div>
        <div class="metric-progress">
          <div class="progress-bar">
            <div class="progress-fill" :style="`width: ${claudeProgress}%`"></div>
          </div>
          <span class="progress-text">{{ claudeProgress }}% of 12,248</span>
        </div>
      </div>

      <div class="metric-card">
        <div class="metric-header">
          <span class="metric-label">Learning Facts</span>
        </div>
        <div class="metric-value">{{ learningFacts.toLocaleString() }}</div>
        <div class="metric-subtext">Qdrant vectors</div>
      </div>

      <div class="metric-card">
        <div class="metric-header">
          <span class="metric-label">Error Rate</span>
        </div>
        <div class="metric-value" :class="errorRateClass">{{ errorRate }}%</div>
        <div class="metric-progress">
          <div class="progress-bar">
            <div class="progress-fill error" :style="`width: ${errorRate}%`"></div>
          </div>
        </div>
      </div>

      <div class="metric-card">
        <div class="metric-header">
          <span class="metric-label">Response Time</span>
        </div>
        <div class="metric-value">{{ avgResponseTime }}s</div>
        <div class="metric-subtext">avg processing</div>
      </div>
    </div>

    <!-- Spatial Reasoning Visualization -->
    <div class="spatial-section">
      <h2 class="section-header">Spatial Intelligence</h2>
      <div class="spatial-grid">

        <!-- Knowledge Graph -->
        <div class="graph-container">
          <h3 class="panel-header">Knowledge Graph</h3>
          <canvas ref="graphCanvas" width="400" height="300"></canvas>
          <div class="graph-stats">
            <span>{{ graphNodes }} nodes</span>
            <span>•</span>
            <span>{{ graphEdges }} edges</span>
            <span>•</span>
            <span>{{ graphServices }} services</span>
          </div>
        </div>

        <!-- Codebase Coverage -->
        <div class="coverage-container">
          <h3 class="panel-header">Codebase Understanding</h3>
          <div class="coverage-visual">
            <div class="coverage-grid">
              <div v-for="i in 100" :key="i"
                   class="coverage-cell"
                   :class="{ 'covered': i <= coveragePercent }">
              </div>
            </div>
          </div>
          <div class="coverage-stats">
            <div>Tower Codebase: {{ coveragePercent }}% analyzed</div>
            <div>{{ filesAnalyzed.toLocaleString() }} / 141,957 files</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Vision Models Status -->
    <div class="models-section">
      <h2 class="section-header">Vision & Spatial Models</h2>
      <div class="models-grid">
        <div v-for="model in visionModels" :key="model" class="model-card">
          <div class="model-icon">[VL]</div>
          <div class="model-name">{{ model }}</div>
          <div class="model-status">Active</div>
        </div>
        <div v-if="visionModels.length === 0" class="model-card inactive">
          <div class="model-icon">[X]</div>
          <div class="model-name">No Vision Models</div>
          <div class="model-status">Install Qwen3-VL or LLaVA</div>
        </div>
      </div>
    </div>

    <!-- Memory Systems -->
    <div class="memory-section">
      <h2 class="section-header">Memory Systems</h2>
      <div class="activity-grid">
        <div class="memory-card">
          <h3 class="memory-title">PostgreSQL</h3>
          <div class="memory-stats">
            <div class="stat-row">
              <span>Conversations:</span>
              <strong>{{ dbConversations }}</strong>
            </div>
            <div class="stat-row">
              <span>Retention:</span>
              <strong>Indefinite</strong>
            </div>
            <div class="stat-row">
              <span>Database Size:</span>
              <strong>{{ dbSize }}</strong>
            </div>
          </div>
        </div>

        <div class="memory-card">
          <h3 class="memory-title">Qdrant Vectors</h3>
          <div class="memory-stats">
            <div class="stat-row">
              <span>Collections:</span>
              <strong>{{ qdrantCollections }}</strong>
            </div>
            <div class="stat-row">
              <span>Total Vectors:</span>
              <strong>{{ qdrantVectors.toLocaleString() }}</strong>
            </div>
            <div class="stat-row">
              <span>Embedding:</span>
              <strong>768D BERT</strong>
            </div>
          </div>
        </div>

        <div class="memory-card">
          <h3 class="memory-title">Learning Service</h3>
          <div class="memory-stats">
            <div class="stat-row">
              <span>Status:</span>
              <strong :class="learningStatusClass">{{ learningStatus }}</strong>
            </div>
            <div class="stat-row">
              <span>Last Update:</span>
              <strong>{{ lastLearningUpdate }}</strong>
            </div>
            <div class="stat-row">
              <span>Patterns:</span>
              <strong>{{ patternsFound }}</strong>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Recent Activity -->
    <div class="activity-section">
      <h2 class="section-header">Recent Improvements</h2>
      <div class="activity-log">
        <div v-for="query in recentQueries" :key="query.id" class="activity-item">
          <div class="activity-query">{{ query.query }}</div>
          <div class="activity-meta">
            <span class="activity-frequency">{{ query.frequency }}x</span>
            <span class="activity-time">{{ query.timestamp }}</span>
          </div>
        </div>
        <div v-if="recentQueries.length === 0" class="activity-empty">
          No recent repeated queries found
        </div>
      </div>
    </div>

    <!-- Action Buttons -->
    <div class="actions-section">
      <button @click="triggerImprovement" class="action-button primary">
        Trigger Improvement Cycle
      </button>
      <button @click="refreshMetrics" class="action-button">
        Refresh Metrics
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'

// State
const metrics = ref({
  knowledge_sources: {
    claude_conversations: 0,
    learning_facts: 0
  },
  performance: {
    error_rate: 0,
    avg_response_time: 0
  },
  repeated_queries: []
})

const status = ref({
  continuous_improvement: false,
  spatial_reasoning: false,
  claude_memory: false,
  models_available: []
})

const knowledgeGraph = ref({
  stats: {
    total_nodes: 0,
    total_edges: 0,
    services: 0
  }
})

const dbStats = ref({
  conversations: 0,
  db_size: '0 MB'
})

const qdrantStats = ref({
  collections: 0,
  total_vectors: 0
})

// Canvas ref
const graphCanvas = ref(null)
let refreshInterval = null

// Computed properties
const claudeConversations = computed(() => metrics.value.knowledge_sources?.claude_conversations || 0)
const claudeProgress = computed(() => ((claudeConversations.value / 12248) * 100).toFixed(1))
const learningFacts = computed(() => metrics.value.knowledge_sources?.learning_facts || 0)
const errorRate = computed(() => (metrics.value.performance?.error_rate * 100 || 0).toFixed(1))
const avgResponseTime = computed(() => (metrics.value.performance?.avg_response_time || 0).toFixed(2))
const errorRateClass = computed(() => errorRate.value > 10 ? 'text-red-400' : 'text-green-400')

const graphNodes = computed(() => knowledgeGraph.value.stats?.total_nodes || 0)
const graphEdges = computed(() => knowledgeGraph.value.stats?.total_edges || 0)
const graphServices = computed(() => knowledgeGraph.value.stats?.services || 0)

const visionModels = computed(() => status.value.models_available || [])
const coveragePercent = computed(() => Math.min(100, Math.round((graphNodes.value / 1420) * 100)))
const filesAnalyzed = computed(() => Math.round(coveragePercent.value * 1419.57))

const dbConversations = computed(() => dbStats.value.conversations || 0)
const dbSize = computed(() => dbStats.value.db_size || '0 MB')
const qdrantCollections = computed(() => qdrantStats.value.collections || 0)
const qdrantVectors = computed(() => qdrantStats.value.total_vectors || 0)

const learningStatus = computed(() => status.value.continuous_improvement ? 'Active' : 'Inactive')
const learningStatusClass = computed(() => status.value.continuous_improvement ? 'text-green-400' : 'text-red-400')
const lastLearningUpdate = ref('Never')
const patternsFound = ref(0)

const recentQueries = computed(() => {
  return (metrics.value.repeated_queries || []).map((q, i) => ({
    id: i,
    query: q.query?.substring(0, 100) + (q.query?.length > 100 ? '...' : ''),
    frequency: q.frequency,
    timestamp: 'recent'
  }))
})

// Methods
async function fetchMetrics() {
  try {
    // Fetch improvement metrics
    const metricsRes = await fetch('/api/echo/improvement/metrics')
    if (metricsRes.ok) {
      metrics.value = await metricsRes.json()
    }

    // Fetch status
    const statusRes = await fetch('/api/echo/improvement/status')
    if (statusRes.ok) {
      status.value = await statusRes.json()
    }

    // Fetch knowledge graph
    const graphRes = await fetch('/api/echo/improvement/knowledge-graph')
    if (graphRes.ok) {
      knowledgeGraph.value = await graphRes.json()
      drawKnowledgeGraph()
    }

    // Fetch database stats
    const dbRes = await fetch('/api/echo/db/stats')
    if (dbRes.ok) {
      const data = await dbRes.json()
      dbStats.value = {
        conversations: data.conversations || 0,
        db_size: data.db_size ? `${(parseInt(data.db_size) / 1048576).toFixed(1)} MB` : '0 MB'
      }
    }

    // Fetch Qdrant stats
    await fetchQdrantStats()

  } catch (error) {
    console.error('Error fetching metrics:', error)
  }
}

async function fetchQdrantStats() {
  try {
    const res = await fetch('http://localhost:6333/collections')
    if (res.ok) {
      const data = await res.json()
      qdrantStats.value.collections = data.result?.collections?.length || 0

      // Calculate total vectors
      let totalVectors = 0
      for (const collection of data.result?.collections || []) {
        try {
          const colRes = await fetch(`http://localhost:6333/collections/${collection.name}`)
          const colData = await colRes.json()
          if (colData.result?.points_count) {
            totalVectors += colData.result.points_count
          }
        } catch (e) {
          console.error(`Error fetching collection ${collection.name}:`, e)
        }
      }
      qdrantStats.value.total_vectors = totalVectors
    }
  } catch (error) {
    console.error('Error fetching Qdrant stats:', error)
  }
}

function drawKnowledgeGraph() {
  if (!graphCanvas.value) return

  const ctx = graphCanvas.value.getContext('2d')
  const width = graphCanvas.value.width
  const height = graphCanvas.value.height

  // Clear canvas
  ctx.fillStyle = '#0d1117'
  ctx.fillRect(0, 0, width, height)

  // Draw nodes and connections
  const nodes = Math.min(graphNodes.value, 50)
  const centerX = width / 2
  const centerY = height / 2

  // Draw connections
  ctx.strokeStyle = '#21262d'
  ctx.lineWidth = 1

  for (let i = 0; i < nodes; i++) {
    const angle1 = (i / nodes) * Math.PI * 2
    const x1 = centerX + Math.cos(angle1) * 100
    const y1 = centerY + Math.sin(angle1) * 100

    // Connect to nearby nodes
    for (let j = 1; j <= 2; j++) {
      const nextIdx = (i + j) % nodes
      const angle2 = (nextIdx / nodes) * Math.PI * 2
      const x2 = centerX + Math.cos(angle2) * 100
      const y2 = centerY + Math.sin(angle2) * 100

      ctx.beginPath()
      ctx.moveTo(x1, y1)
      ctx.lineTo(x2, y2)
      ctx.stroke()
    }
  }

  // Draw nodes
  for (let i = 0; i < nodes; i++) {
    const angle = (i / nodes) * Math.PI * 2
    const x = centerX + Math.cos(angle) * 100
    const y = centerY + Math.sin(angle) * 100

    ctx.fillStyle = i < 5 ? '#2f81f7' : '#8b949e'
    ctx.beginPath()
    ctx.arc(x, y, 4, 0, Math.PI * 2)
    ctx.fill()
  }

  // Draw central node
  ctx.fillStyle = '#7c3aed'
  ctx.beginPath()
  ctx.arc(centerX, centerY, 8, 0, Math.PI * 2)
  ctx.fill()

  // Add text
  ctx.fillStyle = '#8b949e'
  ctx.font = '12px monospace'
  ctx.textAlign = 'center'
  ctx.fillText('Echo Brain', centerX, centerY + 130)
}

async function triggerImprovement() {
  try {
    const res = await fetch('/api/echo/improvement/trigger', {
      method: 'POST'
    })
    if (res.ok) {
      await fetchMetrics()
    }
  } catch (error) {
    console.error('Error triggering improvement:', error)
  }
}

function refreshMetrics() {
  fetchMetrics()
}

// Lifecycle
onMounted(() => {
  fetchMetrics()
  refreshInterval = setInterval(fetchMetrics, 5000)
})

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
  }
})
</script>

<style scoped>
/* Claude Code / Tower Console Theme */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

.improvement-visualization {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Monaco, Inconsolata, "Roboto Mono", Consolas, "Liberation Mono", Menlo, monospace;
  background: #0d1117;
  color: #f0f6fc;
  line-height: 1.6;
  padding: 1rem;
  min-height: 100vh;
}

.header {
  border-bottom: 1px solid #21262d;
  padding-bottom: 1rem;
  margin-bottom: 1.5rem;
}

.header h1 {
  font-size: 1.25rem;
  font-weight: 600;
  color: #f0f6fc;
  margin-bottom: 0.25rem;
}

.header p {
  color: #8b949e;
  font-size: 0.75rem;
}

.section-header {
  font-size: 0.875rem;
  font-weight: 600;
  color: #f0f6fc;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid #21262d;
}

/* Metrics Grid */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.metric-card {
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 0.375rem;
  padding: 1.25rem;
}

.metric-value {
  font-size: 2rem;
  font-weight: 600;
  color: #f0f6fc;
  margin-bottom: 0.5rem;
}

.metric-label {
  font-size: 0.75rem;
  color: #8b949e;
  text-transform: uppercase;
}

.metric-subtext {
  font-size: 0.65rem;
  color: #8b949e;
  margin-top: 0.5rem;
}

.progress-bar {
  margin-top: 0.75rem;
  height: 4px;
  background: #21262d;
  border-radius: 2px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: #8b949e;
  transition: width 0.3s ease;
}

.progress-fill.error {
  background: #8b949e;
}

.progress-text {
  font-size: 0.65rem;
  color: #8b949e;
  margin-top: 0.25rem;
}

/* Spatial Section */
.spatial-section {
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 0.5rem;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}

.spatial-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
}

@media (max-width: 768px) {
  .spatial-grid {
    grid-template-columns: 1fr;
  }
}

/* Panel Styles */
.graph-container, .coverage-container {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 0.375rem;
  padding: 1rem;
}

.panel-header {
  font-size: 0.875rem;
  font-weight: 600;
  color: #f0f6fc;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid #21262d;
}

.graph-stats, .coverage-stats {
  margin-top: 1rem;
  font-size: 0.75rem;
  color: #8b949e;
  display: flex;
  gap: 0.5rem;
  justify-content: center;
}

/* Coverage Grid */
.coverage-grid {
  display: grid;
  grid-template-columns: repeat(10, 1fr);
  gap: 2px;
  padding: 1rem;
}

.coverage-cell {
  width: 100%;
  aspect-ratio: 1;
  background: #21262d;
  border-radius: 2px;
  transition: all 0.3s ease;
}

.coverage-cell.covered {
  background: #8b949e;
}

/* Models Section */
.models-section {
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 0.5rem;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}

.models-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

.model-card {
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 0.375rem;
  padding: 1rem;
  text-align: center;
  transition: all 0.3s ease;
}

.model-card:hover {
  background: rgba(33, 38, 45, 0.5);
}

.model-card.inactive {
  opacity: 0.6;
}

.model-icon {
  font-size: 2rem;
  margin-bottom: 0.5rem;
}

.model-name {
  font-size: 0.875rem;
  font-weight: 600;
  color: #f0f6fc;
  margin-bottom: 0.25rem;
}

.model-status {
  font-size: 0.75rem;
  color: #8b949e;
}

/* Memory Section */
.memory-section {
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 0.5rem;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}

.activity-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
}

.memory-card {
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 0.375rem;
  padding: 1rem;
}

.memory-title {
  font-size: 0.875rem;
  font-weight: 600;
  color: #f0f6fc;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid #21262d;
}

.memory-stats {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.stat-row {
  display: flex;
  justify-content: space-between;
  font-size: 0.75rem;
}

.stat-row span {
  color: #8b949e;
}

.stat-row strong {
  color: #f0f6fc;
}

/* Activity Section */
.activity-section {
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 0.5rem;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}

.activity-log {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 0.375rem;
  max-height: 300px;
  overflow-y: auto;
}

.activity-item {
  padding: 0.75rem;
  border-bottom: 1px solid #21262d;
}

.activity-item:hover {
  background: rgba(33, 38, 45, 0.5);
}

.activity-item:last-child {
  border-bottom: none;
}

.activity-query {
  font-size: 0.75rem;
  color: #f0f6fc;
  margin-bottom: 0.25rem;
}

.activity-meta {
  display: flex;
  gap: 1rem;
  font-size: 0.65rem;
  color: #8b949e;
}

.activity-frequency {
  color: #8b949e;
  font-weight: 600;
}

.activity-empty {
  text-align: center;
  color: #8b949e;
  padding: 2rem;
  font-size: 0.75rem;
}

/* Action Buttons */
.actions-section {
  display: flex;
  gap: 1rem;
  justify-content: center;
  margin-top: 2rem;
}

.action-button {
  padding: 0.75rem 1.5rem;
  background: #21262d;
  border: 1px solid #30363d;
  color: #f0f6fc;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  font-family: inherit;
  cursor: pointer;
  transition: all 0.2s ease;
}

.action-button:hover {
  background: rgba(33, 38, 45, 0.8);
}

.action-button.primary {
  background: #21262d;
  border-color: #30363d;
}

.action-button.primary:hover {
  background: rgba(33, 38, 45, 0.8);
}

/* Canvas */
canvas {
  width: 100%;
  height: auto;
  max-height: 300px;
}

/* Text colors for status */
.text-green-400 {
  color: #8b949e;
}

.text-red-400 {
  color: #8b949e;
}
</style>
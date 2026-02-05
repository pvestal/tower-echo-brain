<template>
  <div class="logs-view">
    <!-- Filters -->
    <div class="card mb-3">
      <h3>Log Filters</h3>
      <div class="grid grid-4 gap-2">
        <select v-model="selectedService" class="w-full">
          <option value="">All Services</option>
          <option value="postgres">PostgreSQL</option>
          <option value="ollama">Ollama</option>
          <option value="qdrant">Qdrant</option>
          <option value="mcp">MCP Server</option>
          <option value="comfyui">ComfyUI</option>
        </select>

        <select v-model="selectedLevel" class="w-full">
          <option value="">All Levels</option>
          <option value="DEBUG">Debug</option>
          <option value="INFO">Info</option>
          <option value="WARNING">Warning</option>
          <option value="ERROR">Error</option>
        </select>

        <input
          v-model="searchQuery"
          placeholder="Search logs..."
          class="w-full"
          @keyup.enter="searchLogs"
        />

        <button @click="refreshLogs" class="btn btn-primary">
          {{ loading ? 'Loading...' : 'Refresh' }}
        </button>
      </div>
    </div>

    <!-- Real-time Toggle -->
    <div class="card mb-3">
      <div class="flex justify-between items-center">
        <h3>System Logs</h3>
        <div class="flex gap-2">
          <button @click="toggleRealtime" class="btn">
            Real-time: {{ realtime ? 'ON' : 'OFF' }}
          </button>
          <button @click="clearLogs" class="btn">Clear</button>
          <button @click="exportLogs" class="btn">Export</button>
        </div>
      </div>
    </div>

    <!-- Logs Display -->
    <div class="card logs-container">
      <div v-if="logs.length === 0" class="text-center text-muted">
        No logs to display
      </div>
      <div v-else class="logs-scroll">
        <div
          v-for="(log, index) in logs"
          :key="index"
          :class="['log-entry', `log-${log.level?.toLowerCase()}`]"
        >
          <div class="log-header">
            <span class="log-time">{{ formatTime(log.timestamp) }}</span>
            <span :class="['log-level', `level-${log.level?.toLowerCase()}`]">
              {{ log.level }}
            </span>
            <span class="log-service">{{ log.service }}</span>
          </div>
          <div class="log-message">{{ log.message }}</div>
          <div v-if="log.details" class="log-details">
            <pre>{{ JSON.stringify(log.details, null, 2) }}</pre>
          </div>
        </div>
      </div>
    </div>

    <!-- History -->
    <div class="card mt-3">
      <h3>Activity History</h3>
      <div class="grid grid-2 gap-2">
        <div>
          <h4 class="text-xs text-muted mb-2">Recent API Calls</h4>
          <div class="history-list">
            <div v-for="call in recentCalls" :key="call.id" class="history-item">
              <span class="text-mono text-xs">{{ call.method }} {{ call.path }}</span>
              <span :class="['status', 'text-xs', statusClass(call.status)]">
                {{ call.status }}
              </span>
              <span class="text-xs text-muted">{{ formatTime(call.timestamp) }}</span>
            </div>
          </div>
        </div>

        <div>
          <h4 class="text-xs text-muted mb-2">Recent Questions</h4>
          <div class="history-list">
            <div v-for="q in recentQuestions" :key="q.id" class="history-item">
              <div class="text-sm">{{ q.question.substring(0, 50) }}...</div>
              <div class="text-xs text-muted">
                {{ formatTime(q.timestamp) }} - {{ q.response_time }}ms
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue';
import { systemApi } from '@/api/echoApi';

const logs = ref<any[]>([]);
const loading = ref(false);
const realtime = ref(false);
const selectedService = ref('');
const selectedLevel = ref('');
const searchQuery = ref('');
const recentCalls = ref<any[]>([]);
const recentQuestions = ref<any[]>([]);

let realtimeInterval: number | null = null;

const refreshLogs = async () => {
  loading.value = true;
  try {
    const response = await systemApi.logs();
    logs.value = response.data.logs || [];

    // Get activity history - endpoint not implemented yet
    // const metricsResponse = await systemApi.metricsHistory();
    recentCalls.value = [];
    recentQuestions.value = [];
  } catch (error) {
    console.error('Failed to fetch logs:', error);
  } finally {
    loading.value = false;
  }
};

const searchLogs = () => {
  if (!searchQuery.value) {
    refreshLogs();
    return;
  }

  logs.value = logs.value.filter(log =>
    log.message?.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
    log.service?.toLowerCase().includes(searchQuery.value.toLowerCase())
  );
};

const toggleRealtime = () => {
  realtime.value = !realtime.value;

  if (realtime.value) {
    realtimeInterval = window.setInterval(refreshLogs, 2000);
  } else if (realtimeInterval) {
    clearInterval(realtimeInterval);
    realtimeInterval = null;
  }
};

const clearLogs = () => {
  logs.value = [];
};

const exportLogs = () => {
  const data = JSON.stringify(logs.value, null, 2);
  const blob = new Blob([data], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `echo-brain-logs-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
};

const formatTime = (timestamp: string | Date) => {
  return new Date(timestamp).toLocaleTimeString();
};

const statusClass = (status: number) => {
  if (status >= 200 && status < 300) return 'healthy';
  if (status >= 400 && status < 500) return 'caution';
  return 'down';
};

onMounted(() => {
  refreshLogs();
});

onUnmounted(() => {
  if (realtimeInterval) {
    clearInterval(realtimeInterval);
  }
});
</script>

<style scoped>
.logs-container {
  max-height: 500px;
  overflow-y: auto;
}

.logs-scroll {
  font-family: monospace;
  font-size: 0.75rem;
}

.log-entry {
  padding: 0.5rem;
  border-bottom: 1px solid #21262d;
}

.log-entry:hover {
  background: rgba(47, 129, 247, 0.05);
}

.log-header {
  display: flex;
  gap: 1rem;
  margin-bottom: 0.25rem;
}

.log-time {
  color: #8b949e;
}

.log-level {
  font-weight: 600;
  text-transform: uppercase;
}

.level-debug { color: #8b949e; }
.level-info { color: #58a6ff; }
.level-warning { color: #d29922; }
.level-error { color: #f85149; }

.log-service {
  color: #79c0ff;
}

.log-message {
  color: #f0f6fc;
  line-height: 1.4;
}

.log-details {
  margin-top: 0.5rem;
  padding: 0.5rem;
  background: #0d1117;
  border-radius: 0.25rem;
}

.log-details pre {
  font-size: 0.7rem;
  color: #8b949e;
  margin: 0;
}

.history-list {
  max-height: 200px;
  overflow-y: auto;
}

.history-item {
  padding: 0.5rem;
  border-left: 2px solid #21262d;
  margin-bottom: 0.5rem;
}

.history-item:hover {
  border-left-color: #2f81f7;
  background: rgba(47, 129, 247, 0.05);
}

.log-error {
  background: rgba(248, 81, 73, 0.1);
  border-left: 3px solid #f85149;
}

.log-warning {
  background: rgba(210, 153, 34, 0.1);
  border-left: 3px solid #d29922;
}
</style>
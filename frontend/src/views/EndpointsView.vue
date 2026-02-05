<template>
  <div class="endpoints-view">
    <div class="card mb-3">
      <h3>Endpoint Tester</h3>

      <div class="grid grid-2 gap-4 mb-3">
        <div>
          <label class="text-xs text-muted">Method</label>
          <select v-model="selectedMethod" class="w-full">
            <option>GET</option>
            <option>POST</option>
            <option>PUT</option>
            <option>DELETE</option>
          </select>
        </div>

        <div>
          <label class="text-xs text-muted">Endpoint Path</label>
          <input
            v-model="endpointPath"
            placeholder="/api/health/"
            class="w-full"
          />
        </div>
      </div>

      <div v-if="selectedMethod !== 'GET'" class="mb-3">
        <label class="text-xs text-muted">Request Body (JSON)</label>
        <textarea
          v-model="requestBody"
          placeholder='{"key": "value"}'
          rows="4"
          class="w-full text-mono"
        ></textarea>
      </div>

      <div class="flex gap-2">
        <button @click="testEndpoint" class="btn btn-primary" :disabled="testing">
          {{ testing ? 'Testing...' : 'Test Endpoint' }}
        </button>
        <button @click="loadPreset" class="btn">Load Preset</button>
        <button @click="clearAll" class="btn">Clear</button>
      </div>
    </div>

    <div class="card mb-3">
      <h3>Quick Tests</h3>
      <div class="grid grid-3 gap-2">
        <button
          v-for="endpoint in quickEndpoints"
          :key="endpoint.path"
          @click="quickTest(endpoint)"
          class="btn text-xs"
        >
          {{ endpoint.desc }}
        </button>
      </div>
    </div>

    <div class="card mb-3" v-if="allEndpoints.length > 0">
      <h3>All Available Endpoints ({{ allEndpoints.length }})</h3>
      <div class="table-wrapper">
        <table class="table">
          <thead>
            <tr>
              <th>Method</th>
              <th>Path</th>
              <th>Description</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="endpoint in allEndpoints" :key="endpoint.path">
              <td>
                <span :class="['status', methodClass(endpoint.method)]">
                  {{ endpoint.method }}
                </span>
              </td>
              <td class="text-mono text-xs">{{ endpoint.path }}</td>
              <td class="text-xs">{{ endpoint.desc }}</td>
              <td>
                <button @click="testPreset(endpoint)" class="btn text-xs">
                  Test
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="card" v-if="response">
      <h3>Response</h3>

      <div class="mb-2">
        <span :class="['status', statusClass(response.status)]">
          {{ response.status }} {{ response.statusText }}
        </span>
        <span class="text-xs text-muted ml-2">
          {{ response.duration }}ms
        </span>
      </div>

      <div v-if="response.headers" class="mb-2">
        <details>
          <summary class="text-xs text-muted cursor-pointer">Headers</summary>
          <pre class="text-xs">{{ response.headers }}</pre>
        </details>
      </div>

      <div>
        <label class="text-xs text-muted">Response Body</label>
        <pre>{{ formatResponse(response.data) }}</pre>
      </div>
    </div>

    <div class="card mt-3" v-if="testHistory.length > 0">
      <h3>Test History</h3>
      <div class="space-y-2">
        <div v-for="(test, index) in testHistory" :key="index" class="history-item">
          <div class="flex justify-between">
            <span class="text-mono text-xs">
              {{ test.method }} {{ test.path }}
            </span>
            <span :class="['status', 'text-xs', statusClass(test.status)]">
              {{ test.status }}
            </span>
          </div>
          <div class="text-xs text-muted">
            {{ formatTime(test.timestamp) }} - {{ test.duration }}ms
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { testEndpoint as apiTest, getAllEndpoints } from '@/api/echoApi';

const selectedMethod = ref('GET');
const endpointPath = ref('');
const requestBody = ref('');
const testing = ref(false);
const response = ref<any>(null);
const testHistory = ref<any[]>([]);
const allEndpoints = ref<any[]>([]);

const quickEndpoints = [
  { method: 'GET', path: '/api/health/', desc: 'Health' },
  { method: 'GET', path: '/api/system/status', desc: 'System' },
  { method: 'GET', path: '/api/memory/status', desc: 'Memory' },
  { method: 'GET', path: '/api/echo/status', desc: 'Echo' },
  { method: 'GET', path: '/mcp/health', desc: 'MCP' },
  { method: 'GET', path: '/api/self-test/quick', desc: 'Self-Test' }
];

onMounted(() => {
  allEndpoints.value = getAllEndpoints();
});

const testEndpoint = async () => {
  if (!endpointPath.value) return;

  testing.value = true;
  response.value = null;
  const startTime = Date.now();

  try {
    let data = undefined;
    if (selectedMethod.value !== 'GET' && requestBody.value) {
      try {
        data = JSON.parse(requestBody.value);
      } catch {
        data = requestBody.value;
      }
    }

    const res = await apiTest(selectedMethod.value, endpointPath.value, data);
    const duration = Date.now() - startTime;

    response.value = {
      status: res.status,
      statusText: res.statusText,
      data: res.data,
      headers: res.headers,
      duration
    };

    testHistory.value.unshift({
      method: selectedMethod.value,
      path: endpointPath.value,
      status: res.status,
      timestamp: new Date(),
      duration
    });

    if (testHistory.value.length > 10) {
      testHistory.value = testHistory.value.slice(0, 10);
    }
  } catch (error: any) {
    const duration = Date.now() - startTime;
    response.value = {
      status: error.response?.status || 0,
      statusText: error.response?.statusText || 'Network Error',
      data: error.response?.data || { error: error.message },
      duration
    };

    testHistory.value.unshift({
      method: selectedMethod.value,
      path: endpointPath.value,
      status: error.response?.status || 0,
      timestamp: new Date(),
      duration
    });
  } finally {
    testing.value = false;
  }
};

const quickTest = (endpoint: any) => {
  selectedMethod.value = endpoint.method;
  endpointPath.value = endpoint.path;
  requestBody.value = '';
  testEndpoint();
};

const testPreset = (endpoint: any) => {
  selectedMethod.value = endpoint.method;
  endpointPath.value = endpoint.path;
  requestBody.value = endpoint.body || '';
  if (endpoint.params) {
    endpointPath.value += endpoint.params;
  }
  testEndpoint();
};

const loadPreset = () => {
  const presets = [
    { method: 'POST', path: '/api/memory/search', body: '{"query": "test", "limit": 5}' },
    { method: 'POST', path: '/api/ask', body: '{"question": "What is Echo Brain?"}' },
    { method: 'POST', path: '/api/intelligence/think', body: '{"query": "test query"}' }
  ];
  const preset = presets[Math.floor(Math.random() * presets.length)];
  if (preset) {
    selectedMethod.value = preset.method;
    endpointPath.value = preset.path;
    requestBody.value = preset.body;
  }
};

const clearAll = () => {
  endpointPath.value = '';
  requestBody.value = '';
  response.value = null;
};

const formatResponse = (data: any) => {
  if (typeof data === 'string') return data;
  return JSON.stringify(data, null, 2);
};

const formatTime = (timestamp: Date) => {
  return new Date(timestamp).toLocaleTimeString();
};

const statusClass = (status: number) => {
  if (status >= 200 && status < 300) return 'healthy';
  if (status >= 400 && status < 500) return 'caution';
  return 'down';
};

const methodClass = (method: string) => {
  const classes: any = {
    GET: 'healthy',
    POST: 'caution',
    PUT: 'caution',
    DELETE: 'down'
  };
  return classes[method] || '';
};
</script>

<style scoped>
.table-wrapper {
  max-height: 400px;
  overflow-y: auto;
}

.history-item {
  padding: 0.5rem;
  border-left: 2px solid #21262d;
  margin-left: 0.5rem;
}

.history-item:hover {
  border-left-color: #2f81f7;
  background: rgba(47, 129, 247, 0.05);
}

.cursor-pointer {
  cursor: pointer;
}

.ml-2 {
  margin-left: 0.5rem;
}
</style>
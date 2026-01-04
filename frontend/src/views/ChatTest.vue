<template>
  <div class="chat-test">
    <div class="header">
      <h1>ECHO TEST</h1>
    </div>

    <div class="test-panel">
      <div class="query-section">
        <label>Query:</label>
        <div class="query-input">
          <input
            v-model="query"
            @keyup.enter="send"
            placeholder='Return: {"test": 1}'
            :disabled="loading"
          />
          <button @click="send" :disabled="loading" class="btn-send">
            {{ loading ? 'Sending...' : 'Send' }}
          </button>
        </div>
      </div>

      <div class="presets">
        <span>Quick tests:</span>
        <button @click="setJsonTest" class="preset">
          JSON Test
        </button>
        <button @click="query = 'What is 2+2?'" class="preset">
          Simple Math
        </button>
        <button @click="query = 'What model are you using?'" class="preset">
          Model Check
        </button>
        <button @click="query = 'Generate a Python hello world function'" class="preset">
          Code Gen
        </button>
      </div>

      <div v-if="response" class="response-section">
        <h3>Response:</h3>
        <div class="response-box">
          <pre class="response-text">{{ response.text }}</pre>

          <div class="response-meta">
            <div class="meta-item">
              <span class="meta-label">Model:</span>
              <span class="meta-value">{{ response.model }}</span>
            </div>
            <div class="meta-item">
              <span class="meta-label">Time:</span>
              <span class="meta-value">{{ response.time }}ms</span>
            </div>
            <div class="meta-item">
              <span class="meta-label">Status:</span>
              <span class="meta-value" :class="response.contaminated ? 'failed' : 'passed'">
                {{ response.contaminated ? '✗ CONTAMINATED' : '✓ CLEAN' }}
                {{ response.contaminated ? '(narrative detected)' : '(no contamination)' }}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div v-if="error" class="error-box">
        <h3>Error:</h3>
        <pre>{{ error }}</pre>
      </div>
    </div>

    <div class="history-panel">
      <h2>Test History</h2>
      <div v-if="history.length === 0" class="no-history">No tests run yet</div>
      <div v-else class="history-list">
        <div
          v-for="(item, index) in history"
          :key="index"
          class="history-item"
          @click="loadFromHistory(item)"
        >
          <div class="history-query">{{ item.query }}</div>
          <div class="history-meta">
            <span :class="item.contaminated ? 'status-bad' : 'status-good'">
              {{ item.contaminated ? 'CONTAMINATED' : 'CLEAN' }}
            </span>
            <span class="history-time">{{ item.time }}ms</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

const query = ref('Return exactly: {"test": 1}')
const response = ref(null)
const error = ref(null)
const loading = ref(false)
const history = ref([])

function checkContamination(text) {
  // Check for narrative contamination patterns
  const narrativePatterns = [
    /the scene/i,
    /cyber.?goblin/i,
    /mei/i,
    /anime/i,
    /character/i,
    /narrative/i,
    /story/i,
    /In this .* scene/i,
    /The .* unfolds/i
  ]

  return narrativePatterns.some(pattern => pattern.test(text))
}

async function send() {
  if (!query.value.trim()) return

  loading.value = true
  error.value = null
  response.value = null

  const startTime = Date.now()

  try {
    const result = await axios.post('https://vestal-garcia.duckdns.org/api/echo/chat', {
      query: query.value,
      temperature: 0,
      max_tokens: 500
    }, {
      timeout: 120000
    })

    const responseTime = Date.now() - startTime
    const responseText = result.data.response || result.data.message || JSON.stringify(result.data)
    const contaminated = checkContamination(responseText)

    response.value = {
      text: responseText,
      model: result.data.model || 'unknown',
      time: responseTime,
      contaminated
    }

    // Add to history
    history.value.unshift({
      query: query.value,
      response: responseText,
      model: result.data.model || 'unknown',
      time: responseTime,
      contaminated
    })

    // Keep only last 10 tests
    if (history.value.length > 10) {
      history.value = history.value.slice(0, 10)
    }

  } catch (err) {
    error.value = err.response?.data?.detail || err.message || 'Unknown error'
  } finally {
    loading.value = false
  }
}

function loadFromHistory(item) {
  query.value = item.query
  response.value = {
    text: item.response,
    model: item.model,
    time: item.time,
    contaminated: item.contaminated
  }
}

function setJsonTest() {
  query.value = 'Return exactly: {"test": 1}'
}
</script>

<style scoped>
.chat-test {
  background: #0a0a0f;
  color: #e2e8f0;
  min-height: 100vh;
  padding: 20px;
  font-family: 'Roboto Mono', monospace;
  display: grid;
  grid-template-columns: 1fr 400px;
  gap: 20px;
}

.header {
  grid-column: 1 / -1;
  padding: 20px;
  background: #151520;
  border: 1px solid #2d3748;
  border-radius: 4px;
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
  margin: 0 0 15px;
}

h3 {
  font-size: 14px;
  text-transform: uppercase;
  color: #94a3b8;
  margin: 0 0 10px;
}

.test-panel {
  background: #151520;
  border: 1px solid #2d3748;
  border-radius: 4px;
  padding: 20px;
}

.query-section {
  margin-bottom: 20px;
}

.query-section label {
  display: block;
  margin-bottom: 10px;
  color: #94a3b8;
  text-transform: uppercase;
  font-size: 12px;
}

.query-input {
  display: flex;
  gap: 10px;
}

input {
  flex: 1;
  background: #0a0a0f;
  border: 1px solid #2d3748;
  border-radius: 4px;
  padding: 10px;
  color: #e2e8f0;
  font-family: inherit;
  font-size: 14px;
}

input:focus {
  outline: none;
  border-color: #2a7de1;
}

input:disabled {
  opacity: 0.5;
}

.btn-send {
  background: #2a7de1;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 4px;
  cursor: pointer;
  font-family: inherit;
  transition: background 150ms;
}

.btn-send:hover:not(:disabled) {
  background: #1e6dd0;
}

.btn-send:disabled {
  background: #475569;
  cursor: not-allowed;
}

.presets {
  display: flex;
  gap: 10px;
  align-items: center;
  margin-bottom: 20px;
  flex-wrap: wrap;
}

.presets span {
  color: #94a3b8;
  font-size: 12px;
  text-transform: uppercase;
}

.preset {
  background: transparent;
  border: 1px solid #2d3748;
  color: #94a3b8;
  padding: 4px 12px;
  border-radius: 4px;
  cursor: pointer;
  font-family: inherit;
  font-size: 12px;
  transition: all 150ms;
}

.preset:hover {
  border-color: #2a7de1;
  color: #2a7de1;
}

.response-section {
  margin-top: 30px;
  padding-top: 20px;
  border-top: 1px solid #2d3748;
}

.response-box {
  background: #0a0a0f;
  border: 1px solid #2d3748;
  border-radius: 4px;
  overflow: hidden;
}

.response-text {
  padding: 15px;
  margin: 0;
  color: #e2e8f0;
  font-size: 13px;
  line-height: 1.5;
  max-height: 400px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-word;
}

.response-meta {
  padding: 15px;
  background: #151520;
  border-top: 1px solid #2d3748;
  display: flex;
  gap: 30px;
}

.meta-item {
  display: flex;
  gap: 8px;
  align-items: baseline;
}

.meta-label {
  font-size: 12px;
  color: #94a3b8;
  text-transform: uppercase;
}

.meta-value {
  color: #e2e8f0;
  font-size: 13px;
}

.meta-value.passed {
  color: #19b37b;
  font-weight: 500;
}

.meta-value.failed {
  color: #ef4444;
  font-weight: 500;
}

.error-box {
  margin-top: 20px;
  padding: 15px;
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid #ef4444;
  border-radius: 4px;
}

.error-box pre {
  margin: 0;
  color: #ef4444;
  font-size: 13px;
}

.history-panel {
  background: #151520;
  border: 1px solid #2d3748;
  border-radius: 4px;
  padding: 20px;
}

.no-history {
  color: #475569;
  font-size: 14px;
  text-align: center;
  padding: 40px 20px;
}

.history-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.history-item {
  padding: 12px;
  background: #0a0a0f;
  border: 1px solid #2d3748;
  border-radius: 4px;
  cursor: pointer;
  transition: border-color 150ms;
}

.history-item:hover {
  border-color: #2a7de1;
}

.history-query {
  color: #e2e8f0;
  font-size: 13px;
  margin-bottom: 5px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.history-meta {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
}

.status-good {
  color: #19b37b;
  text-transform: uppercase;
}

.status-bad {
  color: #ef4444;
  text-transform: uppercase;
}

.history-time {
  color: #475569;
}

@media (max-width: 1200px) {
  .chat-test {
    grid-template-columns: 1fr;
  }

  .history-panel {
    order: -1;
  }
}
</style>
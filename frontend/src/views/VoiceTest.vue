<template>
  <div class="voice-test-container">
    <h2>Echo Brain Voice System Test</h2>

    <!-- Status Overview -->
    <div class="test-section">
      <h3>System Status</h3>
      <div class="status-grid">
        <div class="status-item">
          <span class="label">Voice Service:</span>
          <span :class="['value', voiceStatus.initialized ? 'success' : 'error']">
            {{ voiceStatus.initialized ? 'OK - Initialized' : 'ERR - Not Initialized' }}
          </span>
        </div>
        <div class="status-item">
          <span class="label">STT Model:</span>
          <span :class="['value', voiceStatus.stt?.loaded ? 'success' : 'warning']">
            {{ voiceStatus.stt?.model }} ({{ voiceStatus.stt?.device }})
          </span>
        </div>
        <div class="status-item">
          <span class="label">TTS Model:</span>
          <span :class="['value', voiceStatus.tts?.loaded ? 'success' : 'warning']">
            {{ voiceStatus.tts?.model }}
          </span>
        </div>
        <div class="status-item">
          <span class="label">WebSocket:</span>
          <span :class="['value', wsConnected ? 'success' : 'error']">
            {{ wsConnected ? 'OK - Connected' : 'ERR - Disconnected' }}
          </span>
        </div>
      </div>
    </div>

    <!-- API Tests -->
    <div class="test-section">
      <h3>API Contract Tests</h3>
      <div class="test-grid">
        <!-- Status Test -->
        <div class="test-item">
          <button @click="testStatus" :disabled="testing.status">
            Test /status
          </button>
          <span :class="['result', testResults.status]">
            {{ testResults.status || 'Not tested' }}
          </span>
        </div>

        <!-- Voices Test -->
        <div class="test-item">
          <button @click="testVoices" :disabled="testing.voices">
            Test /voices
          </button>
          <span :class="['result', testResults.voices]">
            {{ testResults.voices || 'Not tested' }}
          </span>
        </div>

        <!-- TTS Test -->
        <div class="test-item">
          <button @click="testTTS" :disabled="testing.tts">
            Test TTS
          </button>
          <span :class="['result', testResults.tts]">
            {{ testResults.tts || 'Not tested' }}
          </span>
        </div>

        <!-- STT Test -->
        <div class="test-item">
          <button @click="testSTT" :disabled="testing.stt || !audioBlob">
            Test STT
          </button>
          <span :class="['result', testResults.stt]">
            {{ testResults.stt || 'Record audio first' }}
          </span>
        </div>

        <!-- Full Chat Loop Test -->
        <div class="test-item">
          <button @click="testChatLoop" :disabled="testing.chat || !audioBlob">
            Test Chat Loop
          </button>
          <span :class="['result', testResults.chat]">
            {{ testResults.chat || 'Record audio first' }}
          </span>
        </div>

        <!-- WebSocket Test -->
        <div class="test-item">
          <button @click="testWebSocket" :disabled="testing.ws">
            Test WebSocket
          </button>
          <span :class="['result', testResults.ws]">
            {{ testResults.ws || 'Not tested' }}
          </span>
        </div>
      </div>
    </div>

    <!-- Audio Recording -->
    <div class="test-section">
      <h3>Audio Recording for Tests</h3>
      <div class="recording-controls">
        <button
          @click="toggleRecording"
          :class="['record-btn', { recording: isRecording }]"
        >
          {{ isRecording ? 'Stop Recording' : 'Start Recording' }}
        </button>
        <div v-if="audioBlob" class="audio-preview">
          <audio :src="audioUrl" controls></audio>
          <span class="audio-size">{{ (audioBlob.size / 1024).toFixed(1) }} KB</span>
        </div>
      </div>
    </div>

    <!-- Test Results Log -->
    <div class="test-section">
      <h3>Test Log</h3>
      <div class="test-log">
        <div v-for="(log, idx) in testLog" :key="idx" class="log-entry" :class="log.type">
          <span class="timestamp">{{ log.time }}</span>
          <span class="message">{{ log.message }}</span>
          <pre v-if="log.data" class="log-data">{{ JSON.stringify(log.data, null, 2) }}</pre>
        </div>
      </div>
    </div>

    <!-- Performance Metrics -->
    <div class="test-section">
      <h3>Performance Metrics</h3>
      <div class="metrics-grid">
        <div class="metric">
          <span class="label">Avg STT Time:</span>
          <span class="value">{{ metrics.avgSTT }}ms</span>
        </div>
        <div class="metric">
          <span class="label">Avg TTS Time:</span>
          <span class="value">{{ metrics.avgTTS }}ms</span>
        </div>
        <div class="metric">
          <span class="label">Avg Chat Time:</span>
          <span class="value">{{ metrics.avgChat }}ms</span>
        </div>
        <div class="metric">
          <span class="label">Total Tests:</span>
          <span class="value">{{ metrics.totalTests }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'

// Props
const props = defineProps<{
  apiBase?: string
}>()

const API_BASE = computed(() => props.apiBase || window.location.origin)
const WS_URL = computed(() => {
  const base = API_BASE.value.replace(/^http/, 'ws')
  return `${base}/api/echo/voice/ws`
})

// State
const voiceStatus = ref<any>({})
const wsConnected = ref(false)
const isRecording = ref(false)
const audioBlob = ref<Blob | null>(null)
const audioUrl = ref('')

const testing = ref({
  status: false,
  voices: false,
  tts: false,
  stt: false,
  chat: false,
  ws: false
})

const testResults = ref<Record<string, string>>({})
const testLog = ref<Array<any>>([])

const metrics = ref({
  avgSTT: 0,
  avgTTS: 0,
  avgChat: 0,
  totalTests: 0
})

// Recording
let mediaRecorder: MediaRecorder | null = null
let audioChunks: Blob[] = []

// Logging
function log(message: string, type: 'info' | 'success' | 'error' | 'warning' = 'info', data?: any) {
  testLog.value.unshift({
    time: new Date().toLocaleTimeString(),
    message,
    type,
    data
  })
}

// Tests
async function testStatus() {
  testing.value.status = true
  log('Testing /api/echo/voice/status...')

  try {
    const start = performance.now()
    const res = await fetch(`${API_BASE.value}/api/echo/voice/status`)
    const data = await res.json()
    const time = performance.now() - start

    voiceStatus.value = data
    testResults.value.status = `OK ${time.toFixed(0)}ms`
    log(`Status check successful (${time.toFixed(0)}ms)`, 'success', data)
  } catch (err) {
    testResults.value.status = 'FAIL'
    log(`Status check failed: ${err}`, 'error')
  } finally {
    testing.value.status = false
  }
}

async function testVoices() {
  testing.value.voices = true
  log('Testing /api/echo/voice/voices...')

  try {
    const start = performance.now()
    const res = await fetch(`${API_BASE.value}/api/echo/voice/voices`)
    const data = await res.json()
    const time = performance.now() - start

    const voiceCount = data.installed.length + data.suggested.length
    testResults.value.voices = `OK ${voiceCount} voices (${time.toFixed(0)}ms)`
    log(`Found ${voiceCount} voices (${time.toFixed(0)}ms)`, 'success', data)
  } catch (err) {
    testResults.value.voices = 'FAIL'
    log(`Voices check failed: ${err}`, 'error')
  } finally {
    testing.value.voices = false
  }
}

async function testTTS() {
  testing.value.tts = true
  log('Testing Text-to-Speech...')

  try {
    const start = performance.now()
    const res = await fetch(`${API_BASE.value}/api/echo/voice/synthesize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: 'Echo Brain voice system operational.',
        length_scale: 1.0
      })
    })

    if (!res.ok) throw new Error(`HTTP ${res.status}`)

    const blob = await res.blob()
    const time = performance.now() - start

    // Update metrics
    metrics.value.avgTTS = ((metrics.value.avgTTS * metrics.value.totalTests + time) / (metrics.value.totalTests + 1))
    metrics.value.totalTests++

    testResults.value.tts = `OK ${(blob.size/1024).toFixed(1)}KB (${time.toFixed(0)}ms)`
    log(`TTS successful: ${(blob.size/1024).toFixed(1)}KB audio generated in ${time.toFixed(0)}ms`, 'success')

    // Play the audio
    const url = URL.createObjectURL(blob)
    const audio = new Audio(url)
    audio.play()
  } catch (err) {
    testResults.value.tts = 'FAIL'
    log(`TTS failed: ${err}`, 'error')
  } finally {
    testing.value.tts = false
  }
}

async function testSTT() {
  if (!audioBlob.value) return

  testing.value.stt = true
  log('Testing Speech-to-Text...')

  try {
    const formData = new FormData()
    formData.append('file', audioBlob.value, 'test.webm')
    formData.append('sample_rate', '16000')

    const start = performance.now()
    const res = await fetch(`${API_BASE.value}/api/echo/voice/transcribe`, {
      method: 'POST',
      body: formData
    })

    if (!res.ok) throw new Error(`HTTP ${res.status}`)

    const data = await res.json()
    const time = performance.now() - start

    // Update metrics
    metrics.value.avgSTT = ((metrics.value.avgSTT * metrics.value.totalTests + time) / (metrics.value.totalTests + 1))
    metrics.value.totalTests++

    testResults.value.stt = `OK "${data.text}" (${time.toFixed(0)}ms)`
    log(`STT successful in ${time.toFixed(0)}ms`, 'success', data)
  } catch (err) {
    testResults.value.stt = 'FAIL'
    log(`STT failed: ${err}`, 'error')
  } finally {
    testing.value.stt = false
  }
}

async function testChatLoop() {
  if (!audioBlob.value) return

  testing.value.chat = true
  log('Testing full voice chat loop...')

  try {
    const formData = new FormData()
    formData.append('file', audioBlob.value, 'test.webm')
    formData.append('sample_rate', '16000')

    const start = performance.now()
    const res = await fetch(`${API_BASE.value}/api/echo/voice/chat`, {
      method: 'POST',
      body: formData
    })

    if (!res.ok) throw new Error(`HTTP ${res.status}`)

    const data = await res.json()
    const time = performance.now() - start

    // Update metrics
    metrics.value.avgChat = ((metrics.value.avgChat * metrics.value.totalTests + time) / (metrics.value.totalTests + 1))
    metrics.value.totalTests++

    testResults.value.chat = `OK Complete (${time.toFixed(0)}ms)`
    log(`Chat loop successful in ${time.toFixed(0)}ms`, 'success', {
      transcript: data.transcript,
      response: data.response_text,
      timings: data.timings
    })

    // Play response
    if (data.audio_base64) {
      const audioData = atob(data.audio_base64)
      const arrayBuffer = new ArrayBuffer(audioData.length)
      const view = new Uint8Array(arrayBuffer)
      for (let i = 0; i < audioData.length; i++) {
        view[i] = audioData.charCodeAt(i)
      }
      const blob = new Blob([arrayBuffer], { type: 'audio/wav' })
      const url = URL.createObjectURL(blob)
      const audio = new Audio(url)
      audio.play()
    }
  } catch (err) {
    testResults.value.chat = 'FAIL'
    log(`Chat loop failed: ${err}`, 'error')
  } finally {
    testing.value.chat = false
  }
}

async function testWebSocket() {
  testing.value.ws = true
  log('Testing WebSocket connection...')

  try {
    const ws = new WebSocket(WS_URL.value)

    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        ws.close()
        reject(new Error('Connection timeout'))
      }, 5000)

      ws.onopen = () => {
        clearTimeout(timeout)
        wsConnected.value = true
        log('WebSocket connected', 'success')

        // Test ping
        ws.send(JSON.stringify({ type: 'ping' }))
      }

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data)
        if (msg.type === 'pong') {
          testResults.value.ws = 'OK Connected & responding'
          log('WebSocket ping/pong successful', 'success')
          ws.close()
          resolve(true)
        }
      }

      ws.onerror = () => {
        clearTimeout(timeout)
        reject(new Error('WebSocket error'))
      }
    })
  } catch (err) {
    testResults.value.ws = 'FAIL'
    wsConnected.value = false
    log(`WebSocket test failed: ${err}`, 'error')
  } finally {
    testing.value.ws = false
  }
}

// Recording functions
async function toggleRecording() {
  if (isRecording.value) {
    stopRecording()
  } else {
    await startRecording()
  }
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    mediaRecorder = new MediaRecorder(stream)
    audioChunks = []

    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data)
    }

    mediaRecorder.onstop = () => {
      audioBlob.value = new Blob(audioChunks, { type: 'audio/webm' })
      audioUrl.value = URL.createObjectURL(audioBlob.value)
      log(`Recorded ${(audioBlob.value.size / 1024).toFixed(1)}KB of audio`, 'success')
    }

    mediaRecorder.start()
    isRecording.value = true
    log('Recording started', 'info')
  } catch (err) {
    log(`Failed to start recording: ${err}`, 'error')
  }
}

function stopRecording() {
  if (mediaRecorder) {
    mediaRecorder.stop()
    mediaRecorder.stream.getTracks().forEach(track => track.stop())
    isRecording.value = false
    log('Recording stopped', 'info')
  }
}

// Auto-test on mount
onMounted(() => {
  testStatus()
})
</script>

<style scoped>
.voice-test-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: 'JetBrains Mono', monospace;
  color: #c8d6e5;
}

h2 {
  color: #60a5fa;
  border-bottom: 2px solid #1e293b;
  padding-bottom: 10px;
  margin-bottom: 20px;
}

h3 {
  color: #94a3b8;
  font-size: 1.1rem;
  margin-bottom: 15px;
}

.test-section {
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 20px;
}

.status-grid, .test-grid, .metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 15px;
}

.status-item, .test-item, .metric {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px;
  background: #1a1a2e;
  border-radius: 6px;
  border: 1px solid #2a2a3e;
}

.label {
  color: #64748b;
  font-size: 0.9rem;
}

.value {
  font-weight: 600;
}

.value.success { color: #10b981; }
.value.warning { color: #f59e0b; }
.value.error { color: #ef4444; }

button {
  background: #1d4ed8;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.9rem;
  transition: all 0.2s;
}

button:hover:not(:disabled) {
  background: #2563eb;
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.record-btn {
  padding: 12px 24px;
  font-size: 1rem;
}

.record-btn.recording {
  background: #ef4444;
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

.result {
  flex: 1;
  text-align: right;
  font-size: 0.9rem;
}

.recording-controls {
  display: flex;
  gap: 20px;
  align-items: center;
}

.audio-preview {
  display: flex;
  gap: 15px;
  align-items: center;
}

.audio-size {
  color: #64748b;
  font-size: 0.9rem;
}

.test-log {
  max-height: 300px;
  overflow-y: auto;
  background: #0a0a0a;
  border: 1px solid #1e293b;
  border-radius: 6px;
  padding: 10px;
}

.log-entry {
  margin-bottom: 10px;
  padding: 8px;
  border-left: 3px solid #475569;
  background: #111827;
  font-size: 0.85rem;
}

.log-entry.success { border-left-color: #10b981; }
.log-entry.error { border-left-color: #ef4444; }
.log-entry.warning { border-left-color: #f59e0b; }

.timestamp {
  color: #475569;
  margin-right: 10px;
  font-size: 0.8rem;
}

.log-data {
  margin-top: 8px;
  padding: 8px;
  background: #000;
  border-radius: 4px;
  font-size: 0.75rem;
  overflow-x: auto;
}
</style>
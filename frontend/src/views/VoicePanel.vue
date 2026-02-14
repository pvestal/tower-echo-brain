<template>
  <div class="voice-panel" :class="{ 'voice-active': isRecording, 'voice-speaking': isSpeaking }">

    <!-- Header -->
    <div class="voice-header">
      <div class="voice-title">
        <svg class="voice-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
          <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
          <line x1="12" y1="19" x2="12" y2="23"/>
          <line x1="8" y1="23" x2="16" y2="23"/>
        </svg>
        <span>Voice</span>
      </div>

      <div class="voice-status-row">
        <span class="status-dot" :class="statusClass"></span>
        <span class="status-text">{{ statusText }}</span>
      </div>

      <button class="voice-mute-btn" :class="{ muted: isMuted }" @click="isMuted = !isMuted" :title="isMuted ? 'Unmute' : 'Mute'">
        <svg v-if="!isMuted" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="16" height="16">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
          <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/>
        </svg>
        <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="16" height="16">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
          <line x1="23" y1="9" x2="17" y2="15"/>
          <line x1="17" y1="9" x2="23" y2="15"/>
        </svg>
      </button>

      <button class="voice-settings-btn" @click="showSettings = !showSettings" title="Voice settings">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="16" height="16">
          <circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
        </svg>
      </button>
    </div>

    <!-- Settings Panel (collapsible) -->
    <div class="voice-settings" v-if="showSettings">
      <div class="setting-row">
        <label>Voice</label>
        <select v-model="selectedVoice" @change="onVoiceChange">
          <option v-for="v in availableVoices" :key="v.name" :value="v.name">{{ v.name }}</option>
        </select>
      </div>
      <div class="setting-row">
        <label>Speed</label>
        <input type="range" min="0.5" max="2.0" step="0.1" v-model.number="speechSpeed" />
        <span class="setting-value">{{ speechSpeed.toFixed(1) }}x</span>
      </div>
      <div class="setting-row">
        <label>Volume</label>
        <input type="range" min="0" max="100" step="1" v-model.number="volumePercent" />
        <span class="setting-value">{{ volumePercent }}%</span>
      </div>
      <div class="setting-row">
        <label>Auto-play</label>
        <label class="toggle">
          <input type="checkbox" v-model="autoPlay" />
          <span class="toggle-slider"></span>
        </label>
      </div>
      <div class="setting-row">
        <label>Language</label>
        <select v-model="language">
          <option value="en">English</option>
          <option value="es">Spanish</option>
          <option value="ja">Japanese</option>
          <option value="de">German</option>
          <option value="fr">French</option>
        </select>
      </div>
    </div>

    <!-- Waveform Visualization -->
    <div class="waveform-container" ref="waveformContainer">
      <canvas ref="waveformCanvas" class="waveform-canvas"></canvas>
      <div class="waveform-overlay" v-if="!isRecording && !isSpeaking && !isProcessing">
        <span class="waveform-hint">{{ wsConnected ? 'Hold to speak' : 'Connecting...' }}</span>
      </div>
      <div class="waveform-overlay processing" v-if="isProcessing">
        <div class="processing-dots">
          <span></span><span></span><span></span>
        </div>
        <span class="waveform-hint">Thinking...</span>
      </div>
    </div>

    <!-- Push-to-Talk Button -->
    <div class="ptt-container">
      <button
        class="ptt-button"
        :class="{ recording: isRecording, disabled: !wsConnected || isProcessing }"
        @pointerdown.prevent="startRecording"
        @pointerup.prevent="stopRecording"
        @pointerleave="stopRecording"
        @touchstart.prevent="startRecording"
        @touchend.prevent="stopRecording"
        :disabled="!wsConnected || isProcessing"
      >
        <div class="ptt-ring ring-1"></div>
        <div class="ptt-ring ring-2"></div>
        <div class="ptt-ring ring-3"></div>
        <div class="ptt-core">
          <svg v-if="!isRecording" viewBox="0 0 24 24" fill="currentColor" width="28" height="28">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2" fill="none" stroke="currentColor" stroke-width="2"/>
            <line x1="12" y1="19" x2="12" y2="23" stroke="currentColor" stroke-width="2"/>
          </svg>
          <svg v-else viewBox="0 0 24 24" fill="currentColor" width="28" height="28">
            <rect x="6" y="6" width="12" height="12" rx="2"/>
          </svg>
        </div>
      </button>
      <div class="ptt-label">{{ pttLabel }}</div>
    </div>

    <!-- Conversation Thread -->
    <div class="voice-conversation" ref="conversationEl">
      <div
        v-for="(msg, idx) in conversation"
        :key="idx"
        class="voice-msg"
        :class="msg.role"
      >
        <div class="msg-role">{{ msg.role === 'user' ? 'You' : 'Echo' }}</div>
        <!-- User messages: plain text. Assistant messages: rich rendering -->
        <div v-if="msg.role === 'user'" class="msg-text">{{ msg.text }}</div>
        <EchoMessage
          v-else
          :text="msg.text"
          :query-type="msg.queryType"
          :confidence="msg.confidence"
          :sources="msg.sources"
          :actions-taken="msg.actionsTaken"
          :execution-time-ms="msg.executionTimeMs"
        />
        <div class="msg-meta" v-if="msg.timings">
          <span v-if="msg.timings.stt_ms" title="Speech-to-text time">STT {{ Math.round(msg.timings.stt_ms) }}ms</span>
          <span v-if="msg.timings.chat_ms" title="Reasoning time">Think {{ Math.round(msg.timings.chat_ms) }}ms</span>
          <span v-if="msg.timings.tts_ms" title="Text-to-speech time">TTS {{ Math.round(msg.timings.tts_ms) }}ms</span>
          <span v-if="msg.timings.total_ms" class="msg-meta-total" title="Total round-trip time">Total {{ Math.round(msg.timings.total_ms) }}ms</span>
        </div>
        <!-- Collapsible STT debug for user messages -->
        <details v-if="msg.role === 'user' && (msg.sttLanguage || msg.sttConfidence != null)" class="msg-debug">
          <summary>STT Debug</summary>
          <div class="debug-row" v-if="msg.sttLanguage"><span class="debug-key">Language</span><span class="debug-val">{{ msg.sttLanguage }}</span></div>
          <div class="debug-row" v-if="msg.sttConfidence != null"><span class="debug-key">Confidence</span><span class="debug-val">{{ (msg.sttConfidence * 100).toFixed(1) }}%</span></div>
        </details>
        <!-- Collapsible debug panel for assistant messages -->
        <details v-if="msg.role === 'assistant' && msg.timings" class="msg-debug">
          <summary>Debug</summary>
          <div class="debug-row" v-if="msg.timings.total_ms"><span class="debug-key">Total round-trip</span><span class="debug-val">{{ Math.round(msg.timings.total_ms) }}ms</span></div>
          <div class="debug-row" v-if="msg.audioBytesReceived"><span class="debug-key">Audio sent</span><span class="debug-val">{{ (msg.audioBytesReceived / 1024).toFixed(1) }} KB</span></div>
          <div class="debug-row" v-if="msg.queryType"><span class="debug-key">Query type</span><span class="debug-val">{{ msg.queryType }}</span></div>
          <div class="debug-row" v-if="msg.confidence != null"><span class="debug-key">Confidence</span><span class="debug-val">{{ (msg.confidence * 100).toFixed(1) }}%</span></div>
          <div class="debug-row" v-if="msg.sources && msg.sources.length"><span class="debug-key">Sources</span><span class="debug-val">{{ msg.sources.join(', ') }}</span></div>
          <div class="debug-row" v-if="msg.executionTimeMs"><span class="debug-key">Reasoning engine</span><span class="debug-val">{{ msg.executionTimeMs }}ms</span></div>
        </details>
        <!-- Replay button for assistant messages -->
        <button
          v-if="msg.role === 'assistant' && msg.audioData"
          class="replay-btn"
          @click="playAudio(msg.audioData)"
          title="Replay"
        >
          <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
            <polygon points="5 3 19 12 5 21 5 3"/>
          </svg>
        </button>
      </div>

      <div v-if="currentTranscript" class="voice-msg user partial">
        <div class="msg-role">You</div>
        <div class="msg-text">{{ currentTranscript }}</div>
      </div>
    </div>

    <!-- Connection indicator -->
    <div class="ws-status" :class="{ connected: wsConnected }">
      <span class="ws-dot"></span>
      {{ wsConnected ? 'Connected' : 'Disconnected' }}
      <span v-if="wsLatencyMs != null" class="ws-latency">{{ wsLatencyMs }}ms</span>
      <span v-if="reconnectCount > 0" class="ws-reconnects">{{ reconnectCount }} reconnect{{ reconnectCount > 1 ? 's' : '' }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, nextTick, computed } from 'vue'
import EchoMessage from '@/components/EchoMessage.vue'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface VoiceMessage {
  role: 'user' | 'assistant'
  text: string
  timings?: Record<string, number>
  audioData?: string  // base64 WAV for replay
  queryType?: string
  confidence?: number
  sources?: string[]
  actionsTaken?: any[]
  executionTimeMs?: number
  // STT debug metadata (user messages)
  sttLanguage?: string
  sttConfidence?: number
  // Audio debug metadata (assistant messages)
  audioBytesReceived?: number
}

interface VoiceConfig {
  name: string
  description?: string
}

// ---------------------------------------------------------------------------
// Props & configuration
// ---------------------------------------------------------------------------

const props = defineProps<{
  /** Base URL for the Echo Brain API (e.g. https://vestal-garcia.duckdns.org) */
  apiBase?: string
}>()

const API_BASE = computed(() => props.apiBase || window.location.origin)
const WS_URL = computed(() => {
  const base = API_BASE.value.replace(/^http/, 'ws')
  return `${base}/api/echo/voice/ws`
})

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

// Connection
const wsConnected = ref(false)
const reconnectCount = ref(0)
const wsLatencyMs = ref<number | null>(null)
let ws: WebSocket | null = null
let reconnectTimer: ReturnType<typeof setTimeout> | null = null
let pingSentAt: number | null = null

// Recording
const isRecording = ref(false)
const isProcessing = ref(false)
const isSpeaking = ref(false)
let mediaStream: MediaStream | null = null
let audioContext: AudioContext | null = null
let analyserNode: AnalyserNode | null = null
let processorNode: ScriptProcessorNode | null = null

// Mute & Volume
const isMuted = ref(false)
const volumePercent = ref(100)
let gainNode: GainNode | null = null

// UI
const showSettings = ref(false)
const conversation = ref<VoiceMessage[]>([])
const currentTranscript = ref('')
const selectedVoice = ref('en_US-lessac-medium')
const speechSpeed = ref(1.0)
const autoPlay = ref(true)
const language = ref('en')
const availableVoices = ref<VoiceConfig[]>([])

// DOM refs
const waveformCanvas = ref<HTMLCanvasElement | null>(null)
const waveformContainer = ref<HTMLDivElement | null>(null)
const conversationEl = ref<HTMLDivElement | null>(null)

// Waveform animation
let animationFrameId: number | null = null
let waveformData = new Uint8Array(128)

// Audio playback
let playbackAudioContext: AudioContext | null = null

// ---------------------------------------------------------------------------
// Computed
// ---------------------------------------------------------------------------

const statusClass = computed(() => {
  if (isRecording.value) return 'recording'
  if (isProcessing.value) return 'processing'
  if (isSpeaking.value) return 'speaking'
  if (wsConnected.value) return 'ready'
  return 'disconnected'
})

const statusText = computed(() => {
  if (isRecording.value) return 'Listening...'
  if (isProcessing.value) return 'Processing...'
  if (isSpeaking.value) return 'Speaking...'
  if (wsConnected.value) return 'Ready'
  return 'Connecting...'
})

const pttLabel = computed(() => {
  if (isRecording.value) return 'Release to send'
  if (isProcessing.value) return 'Processing...'
  if (!wsConnected.value) return 'Connecting...'
  return 'Hold to speak'
})

// ---------------------------------------------------------------------------
// WebSocket Connection
// ---------------------------------------------------------------------------

function connectWebSocket() {
  if (ws?.readyState === WebSocket.OPEN) return

  ws = new WebSocket(WS_URL.value)

  ws.onopen = () => {
    wsConnected.value = true
    console.log('[Voice] WebSocket connected')
    // Send initial config
    ws?.send(JSON.stringify({
      type: 'config',
      language: language.value,
      sample_rate: 16000,
    }))
  }

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data)
    handleServerMessage(msg)
  }

  ws.onclose = () => {
    wsConnected.value = false
    console.log('[Voice] WebSocket disconnected')
    scheduleReconnect()
  }

  ws.onerror = (err) => {
    console.error('[Voice] WebSocket error:', err)
    wsConnected.value = false
  }
}

function scheduleReconnect() {
  if (reconnectTimer) clearTimeout(reconnectTimer)
  reconnectCount.value++
  reconnectTimer = setTimeout(() => {
    console.log('[Voice] Attempting reconnect...')
    connectWebSocket()
  }, 3000)
}

function handleServerMessage(msg: any) {
  switch (msg.type) {
    case 'status':
      if (msg.state === 'listening') {
        isProcessing.value = false
        isSpeaking.value = false
      } else if (msg.state === 'processing') {
        isProcessing.value = true
      } else if (msg.state === 'speaking') {
        isSpeaking.value = true
      }
      break

    case 'transcript':
      if (msg.final) {
        // Add user message to conversation with STT metadata
        conversation.value.push({
          role: 'user',
          text: msg.text,
          sttLanguage: msg.language,
          sttConfidence: msg.confidence,
        })
        currentTranscript.value = ''
      } else {
        currentTranscript.value = msg.text
      }
      scrollToBottom()
      break

    case 'response':
      isProcessing.value = false
      // Add assistant message with debug metadata
      conversation.value.push({
        role: 'assistant',
        text: msg.text,
        timings: msg.timings,
        audioData: msg.audio,
        queryType: msg.query_type,
        confidence: msg.confidence,
        sources: msg.sources,
        actionsTaken: msg.actions_taken,
        executionTimeMs: msg.execution_time_ms,
        audioBytesReceived: msg.audio_bytes_received,
      })
      scrollToBottom()

      // Auto-play audio response
      if (autoPlay.value && msg.audio) {
        playAudio(msg.audio)
      } else {
        isSpeaking.value = false
      }
      break

    case 'error':
      console.error('[Voice] Server error:', msg.message)
      isProcessing.value = false
      isSpeaking.value = false
      break

    case 'pong':
      if (pingSentAt) {
        wsLatencyMs.value = Math.round(performance.now() - pingSentAt)
        pingSentAt = null
      }
      break
  }
}

// ---------------------------------------------------------------------------
// Audio Recording (Microphone)
// ---------------------------------------------------------------------------

async function startRecording() {
  if (isRecording.value || isProcessing.value || !wsConnected.value) return

  try {
    // Request microphone access
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      }
    })

    // Create audio processing pipeline
    audioContext = new AudioContext({ sampleRate: 16000 })
    // Ensure AudioContext is running (browsers may suspend even from user gesture)
    if (audioContext.state === 'suspended') {
      await audioContext.resume()
    }
    const source = audioContext.createMediaStreamSource(mediaStream)

    // Analyser for waveform visualization
    analyserNode = audioContext.createAnalyser()
    analyserNode.fftSize = 256
    source.connect(analyserNode)

    // Processor to capture raw PCM and send over WebSocket
    processorNode = audioContext.createScriptProcessor(4096, 1, 1)
    processorNode.onaudioprocess = (e) => {
      if (!isRecording.value || !ws || ws.readyState !== WebSocket.OPEN) return

      const pcmData = e.inputBuffer.getChannelData(0)
      // Convert Float32 → Int16 PCM
      const int16 = new Int16Array(pcmData.length)
      for (let i = 0; i < pcmData.length; i++) {
        const sample = pcmData[i] ?? 0
        const s = Math.max(-1, Math.min(1, sample))
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF
      }

      // Send as base64 chunk
      const bytes = new Uint8Array(int16.buffer)
      const base64 = btoa(String.fromCharCode(...Array.from(bytes)))
      ws.send(JSON.stringify({
        type: 'audio_chunk',
        data: base64,
        sample_rate: 16000,
      }))
    }

    source.connect(processorNode)
    processorNode.connect(audioContext.destination)

    isRecording.value = true
    startWaveformAnimation()

  } catch (err) {
    console.error('[Voice] Microphone access failed:', err)
  }
}

function stopRecording() {
  if (!isRecording.value) return

  isRecording.value = false
  stopWaveformAnimation()

  // Send end signal
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'audio_end' }))
    isProcessing.value = true
  }

  // Cleanup audio nodes
  if (processorNode) {
    processorNode.disconnect()
    processorNode = null
  }
  if (analyserNode) {
    analyserNode.disconnect()
    analyserNode = null
  }
  if (audioContext) {
    audioContext.close()
    audioContext = null
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop())
    mediaStream = null
  }
}

// ---------------------------------------------------------------------------
// Audio Playback
// ---------------------------------------------------------------------------

async function playAudio(base64Audio: string) {
  if (isMuted.value) {
    isSpeaking.value = false
    return
  }

  isSpeaking.value = true

  try {
    if (!playbackAudioContext) {
      playbackAudioContext = new AudioContext()
    }

    // Ensure persistent GainNode
    if (!gainNode) {
      gainNode = playbackAudioContext.createGain()
      gainNode.connect(playbackAudioContext.destination)
    }
    gainNode.gain.value = volumePercent.value / 100

    const audioData = Uint8Array.from(atob(base64Audio), c => c.charCodeAt(0))
    const audioBuffer = await playbackAudioContext.decodeAudioData(audioData.buffer.slice(0))

    const source = playbackAudioContext.createBufferSource()
    source.buffer = audioBuffer
    source.playbackRate.value = speechSpeed.value
    source.connect(gainNode)

    source.onended = () => {
      isSpeaking.value = false
    }

    source.start()
  } catch (err) {
    console.error('[Voice] Playback failed:', err)
    isSpeaking.value = false
  }
}

// ---------------------------------------------------------------------------
// Waveform Visualization
// ---------------------------------------------------------------------------

function startWaveformAnimation() {
  const canvas = waveformCanvas.value
  if (!canvas) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const draw = () => {
    if (!analyserNode || !isRecording.value) {
      drawIdleWaveform(ctx, canvas)
      return
    }

    animationFrameId = requestAnimationFrame(draw)

    analyserNode.getByteTimeDomainData(waveformData)

    const { width, height } = canvas
    ctx.clearRect(0, 0, width, height)

    // Gradient stroke
    const gradient = ctx.createLinearGradient(0, 0, width, 0)
    gradient.addColorStop(0, '#3b82f680')
    gradient.addColorStop(0.5, '#60a5fa')
    gradient.addColorStop(1, '#3b82f680')

    ctx.lineWidth = 2
    ctx.strokeStyle = gradient
    ctx.beginPath()

    const sliceWidth = width / waveformData.length
    let x = 0

    for (let i = 0; i < waveformData.length; i++) {
      const value = waveformData[i] ?? 128
      const v = value / 128.0
      const y = (v * height) / 2
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
      x += sliceWidth
    }

    ctx.lineTo(width, height / 2)
    ctx.stroke()
  }

  draw()
}

function stopWaveformAnimation() {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId)
    animationFrameId = null
  }
}

function drawIdleWaveform(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) {
  const { width, height } = canvas
  ctx.clearRect(0, 0, width, height)

  ctx.lineWidth = 1
  ctx.strokeStyle = '#ffffff15'
  ctx.beginPath()

  const time = Date.now() / 2000
  for (let x = 0; x < width; x++) {
    const y = height / 2 + Math.sin(x / 30 + time) * 4 + Math.sin(x / 15 + time * 1.5) * 2
    if (x === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.stroke()

  animationFrameId = requestAnimationFrame(() => {
    if (!isRecording.value) {
      drawIdleWaveform(ctx, canvas)
    }
  })
}

function resizeCanvas() {
  const canvas = waveformCanvas.value
  const container = waveformContainer.value
  if (!canvas || !container) return

  canvas.width = container.clientWidth
  canvas.height = container.clientHeight
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function scrollToBottom() {
  await nextTick()
  if (conversationEl.value) {
    conversationEl.value.scrollTop = conversationEl.value.scrollHeight
  }
}

async function fetchVoices() {
  try {
    const res = await fetch(`${API_BASE.value}/api/echo/voice/voices`)
    const data = await res.json()
    availableVoices.value = [
      ...data.installed.map((v: any) => ({ name: v.name })),
      ...data.suggested.map((v: any) => ({ name: v.name, description: v.description })),
    ]
  } catch (err) {
    console.warn('[Voice] Failed to fetch voices:', err)
    availableVoices.value = [{ name: 'en_US-lessac-medium' }]
  }
}

function onVoiceChange() {
  // Voice change would require backend support — for now just stored locally
  console.log('[Voice] Selected voice:', selectedVoice.value)
}

// Keepalive ping
let pingInterval: ReturnType<typeof setInterval> | null = null

function startPing() {
  pingInterval = setInterval(() => {
    if (ws?.readyState === WebSocket.OPEN) {
      pingSentAt = performance.now()
      ws.send(JSON.stringify({ type: 'ping' }))
    }
  }, 30000)
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

onMounted(() => {
  resizeCanvas()
  window.addEventListener('resize', resizeCanvas)
  connectWebSocket()
  startPing()
  fetchVoices()

  // Start idle waveform
  const canvas = waveformCanvas.value
  if (canvas) {
    const ctx = canvas.getContext('2d')
    if (ctx) drawIdleWaveform(ctx, canvas)
  }
})

onUnmounted(() => {
  window.removeEventListener('resize', resizeCanvas)
  stopRecording()
  stopWaveformAnimation()

  if (pingInterval) clearInterval(pingInterval)
  if (reconnectTimer) clearTimeout(reconnectTimer)
  if (ws) ws.close()
  if (playbackAudioContext) playbackAudioContext.close()
})

// Watch language changes → update WebSocket config
watch(language, (newLang) => {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'config', language: newLang }))
  }
})

// Watch volume → update GainNode
watch(volumePercent, (pct) => {
  if (gainNode) {
    gainNode.gain.value = pct / 100
  }
})
</script>

<style scoped>
/* ══════════════════════════════════════════════════
   Echo Brain Voice Panel — Dark industrial aesthetic
   with electric blue accents
   ══════════════════════════════════════════════════ */

.voice-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #0a0f1a;
  border: 1px solid #1a2340;
  border-radius: 12px;
  overflow: hidden;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  color: #c8d6e5;
}

.voice-panel.voice-active {
  border-color: #3b82f6;
  box-shadow: 0 0 20px #3b82f620, inset 0 0 30px #3b82f608;
}

.voice-panel.voice-speaking {
  border-color: #10b981;
  box-shadow: 0 0 20px #10b98120;
}

/* ── Header ── */
.voice-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border-bottom: 1px solid #1a2340;
  background: #0d1424;
}

.voice-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  color: #60a5fa;
}

.voice-icon {
  width: 18px;
  height: 18px;
}

.voice-status-row {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-left: auto;
  font-size: 11px;
}

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #475569;
  transition: background 0.3s;
}

.status-dot.ready { background: #22c55e; }
.status-dot.recording { background: #ef4444; animation: pulse-dot 1s infinite; }
.status-dot.processing { background: #f59e0b; animation: pulse-dot 0.5s infinite; }
.status-dot.speaking { background: #10b981; animation: pulse-dot 0.8s infinite; }
.status-dot.disconnected { background: #64748b; }

.status-text {
  color: #94a3b8;
}

/* ── Mute Button ── */
.voice-mute-btn {
  background: none;
  border: 1px solid #1e293b;
  border-radius: 6px;
  padding: 4px 6px;
  color: #64748b;
  cursor: pointer;
  transition: all 0.2s;
}
.voice-mute-btn:hover {
  color: #94a3b8;
  border-color: #334155;
}
.voice-mute-btn.muted {
  color: #ef4444;
  border-color: #7f1d1d;
  background: #7f1d1d20;
}

.voice-settings-btn {
  background: none;
  border: 1px solid #1e293b;
  border-radius: 6px;
  padding: 4px 6px;
  color: #64748b;
  cursor: pointer;
  transition: all 0.2s;
}
.voice-settings-btn:hover {
  color: #94a3b8;
  border-color: #334155;
}

/* ── Settings Panel ── */
.voice-settings {
  padding: 12px 16px;
  border-bottom: 1px solid #1a2340;
  background: #0b1020;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.setting-row {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 12px;
}

.setting-row label:first-child {
  width: 70px;
  color: #64748b;
  flex-shrink: 0;
}

.setting-row select,
.setting-row input[type="range"] {
  flex: 1;
  background: #131b2e;
  border: 1px solid #1e293b;
  color: #c8d6e5;
  border-radius: 4px;
  padding: 4px 6px;
  font-size: 11px;
  font-family: inherit;
}

.setting-value {
  width: 32px;
  text-align: right;
  color: #60a5fa;
  font-size: 11px;
}

/* Toggle switch */
.toggle {
  position: relative;
  display: inline-block;
  width: 36px;
  height: 20px;
}
.toggle input { opacity: 0; width: 0; height: 0; }
.toggle-slider {
  position: absolute;
  inset: 0;
  background: #1e293b;
  border-radius: 10px;
  cursor: pointer;
  transition: 0.3s;
}
.toggle-slider::before {
  content: '';
  position: absolute;
  left: 3px;
  top: 3px;
  width: 14px;
  height: 14px;
  background: #64748b;
  border-radius: 50%;
  transition: 0.3s;
}
.toggle input:checked + .toggle-slider { background: #1d4ed8; }
.toggle input:checked + .toggle-slider::before {
  transform: translateX(16px);
  background: #60a5fa;
}

/* ── Waveform ── */
.waveform-container {
  position: relative;
  height: 80px;
  background: #080d18;
  border-bottom: 1px solid #1a2340;
  overflow: hidden;
}

.waveform-canvas {
  width: 100%;
  height: 100%;
}

.waveform-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  pointer-events: none;
}

.waveform-hint {
  font-size: 11px;
  color: #475569;
  letter-spacing: 1px;
  text-transform: uppercase;
}

.processing-dots {
  display: flex;
  gap: 6px;
}

.processing-dots span {
  width: 6px;
  height: 6px;
  background: #60a5fa;
  border-radius: 50%;
  animation: bounce-dot 1.2s infinite ease-in-out;
}
.processing-dots span:nth-child(2) { animation-delay: 0.2s; }
.processing-dots span:nth-child(3) { animation-delay: 0.4s; }

/* ── Push-to-Talk Button ── */
.ptt-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px 16px 12px;
  gap: 8px;
}

.ptt-button {
  position: relative;
  width: 72px;
  height: 72px;
  border: none;
  border-radius: 50%;
  background: #131b2e;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  -webkit-user-select: none;
  user-select: none;
  touch-action: none;
}

.ptt-button:hover:not(.disabled) {
  background: #1a2440;
}

.ptt-button.disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.ptt-core {
  position: relative;
  z-index: 2;
  color: #60a5fa;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: color 0.2s;
}

.ptt-button.recording .ptt-core {
  color: #ef4444;
}

/* Animated rings */
.ptt-ring {
  position: absolute;
  inset: -4px;
  border-radius: 50%;
  border: 1px solid transparent;
  transition: all 0.3s;
}

.ptt-button.recording .ring-1 {
  border-color: #3b82f640;
  animation: ring-expand 1.5s infinite;
}
.ptt-button.recording .ring-2 {
  border-color: #3b82f630;
  animation: ring-expand 1.5s infinite 0.3s;
}
.ptt-button.recording .ring-3 {
  border-color: #3b82f620;
  animation: ring-expand 1.5s infinite 0.6s;
}

.ptt-label {
  font-size: 11px;
  color: #475569;
  letter-spacing: 0.5px;
}

/* ── Conversation Thread ── */
.voice-conversation {
  flex: 1;
  overflow-y: auto;
  padding: 12px 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  scrollbar-width: thin;
  scrollbar-color: #1e293b transparent;
}

.voice-msg {
  position: relative;
  padding: 10px 12px;
  border-radius: 8px;
  font-size: 13px;
  line-height: 1.5;
  max-width: 90%;
}

.voice-msg.user {
  align-self: flex-end;
  background: #1a2a4a;
  border: 1px solid #234078;
}

.voice-msg.assistant {
  align-self: flex-start;
  background: #111827;
  border: 1px solid #1e293b;
}

.voice-msg.partial {
  opacity: 0.6;
  border-style: dashed;
}

.msg-role {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 4px;
  color: #475569;
}

.voice-msg.user .msg-role { color: #60a5fa; }
.voice-msg.assistant .msg-role { color: #10b981; }

.msg-text {
  color: #e2e8f0;
}

.msg-meta {
  display: flex;
  gap: 10px;
  margin-top: 6px;
  font-size: 10px;
  color: #475569;
}

.msg-meta-total {
  color: #60a5fa;
  font-weight: 600;
}

/* ── Debug Panels ── */
.msg-debug {
  margin-top: 6px;
  font-size: 10px;
  color: #475569;
}
.msg-debug summary {
  cursor: pointer;
  color: #64748b;
  user-select: none;
  font-size: 10px;
  letter-spacing: 0.3px;
}
.msg-debug summary:hover {
  color: #94a3b8;
}
.msg-debug[open] summary {
  margin-bottom: 4px;
}
.debug-row {
  display: flex;
  justify-content: space-between;
  padding: 1px 0;
  border-bottom: 1px solid #1a234020;
}
.debug-key {
  color: #64748b;
}
.debug-val {
  color: #94a3b8;
  text-align: right;
}

.replay-btn {
  position: absolute;
  top: 8px;
  right: 8px;
  background: #1e293b;
  border: none;
  border-radius: 4px;
  padding: 4px 6px;
  color: #64748b;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.2s;
}

.voice-msg:hover .replay-btn {
  opacity: 1;
}

.replay-btn:hover {
  color: #10b981;
  background: #1a2a4a;
}

/* ── WebSocket Status ── */
.ws-status {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 16px;
  font-size: 10px;
  color: #475569;
  border-top: 1px solid #1a2340;
  background: #0b1020;
}

.ws-dot {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: #ef4444;
}

.ws-status.connected .ws-dot {
  background: #22c55e;
}

.ws-latency {
  margin-left: auto;
  color: #64748b;
  font-size: 9px;
}

.ws-reconnects {
  color: #f59e0b;
  font-size: 9px;
}

/* ── Animations ── */
@keyframes pulse-dot {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

@keyframes bounce-dot {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1); }
}

@keyframes ring-expand {
  0% {
    transform: scale(1);
    opacity: 0.6;
  }
  100% {
    transform: scale(1.6);
    opacity: 0;
  }
}

/* ── Scrollbar ── */
.voice-conversation::-webkit-scrollbar {
  width: 4px;
}
.voice-conversation::-webkit-scrollbar-thumb {
  background: #1e293b;
  border-radius: 2px;
}
</style>
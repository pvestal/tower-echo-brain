<template>
  <div class="voice-simple">
    <div class="voice-header">
      <h2>Echo Brain Voice</h2>
      <div class="status-bar">
        <span :class="['status', statusClass]">{{ status }}</span>
        <span v-if="latency" class="latency">{{ latency }}ms</span>
      </div>
    </div>

    <!-- Main Voice Interface -->
    <div class="voice-main">
      <!-- Visual Feedback -->
      <div class="voice-visualizer" :class="{ active: isRecording, processing: isProcessing }">
        <div class="pulse-ring" v-for="i in 3" :key="i"></div>
        <button
          @click="toggleRecording"
          :disabled="isProcessing"
          class="voice-button"
          :class="{ recording: isRecording }"
        >
          <svg v-if="!isRecording" viewBox="0 0 24 24" fill="currentColor" width="32" height="32">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2" fill="none" stroke="currentColor" stroke-width="2"/>
          </svg>
          <svg v-else viewBox="0 0 24 24" fill="currentColor" width="32" height="32">
            <rect x="6" y="6" width="12" height="12" rx="2"/>
          </svg>
        </button>
      </div>

      <p class="voice-hint">{{ hint }}</p>
    </div>

    <!-- Conversation -->
    <div class="conversation">
      <TransitionGroup name="message">
        <div v-for="msg in messages" :key="msg.id" class="message" :class="msg.role">
          <div class="message-content">
            <div class="message-text">{{ msg.text }}</div>
            <div class="message-meta">
              <span v-if="msg.duration">{{ msg.duration }}ms</span>
              <button v-if="msg.audio" @click="playAudio(msg.audio)" class="replay-btn">
                Replay
              </button>
            </div>
          </div>
        </div>
      </TransitionGroup>
    </div>

    <!-- Settings -->
    <div class="settings">
      <label>
        <input type="checkbox" v-model="autoPlay"> Auto-play responses
      </label>
      <label>
        <input type="checkbox" v-model="isMuted"> Mute audio
      </label>
      <label>
        <input type="checkbox" v-model="continuousMode"> Continuous mode
      </label>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onUnmounted } from 'vue'

interface Message {
  id: number
  role: 'user' | 'assistant'
  text: string
  duration?: number
  audio?: string
}

// State
const isRecording = ref(false)
const isProcessing = ref(false)
const messages = ref<Message[]>([])
const latency = ref<number | null>(null)
const autoPlay = ref(true)
const isMuted = ref(false)
const continuousMode = ref(false)

let mediaRecorder: MediaRecorder | null = null
let audioChunks: Blob[] = []
let messageId = 0

// Computed
const status = computed(() => {
  if (isRecording.value) return 'Recording...'
  if (isProcessing.value) return 'Processing...'
  return 'Ready'
})

const statusClass = computed(() => {
  if (isRecording.value) return 'recording'
  if (isProcessing.value) return 'processing'
  return 'ready'
})

const hint = computed(() => {
  if (isRecording.value) return 'Click to stop recording'
  if (isProcessing.value) return 'Echo Brain is thinking...'
  return 'Click to start speaking'
})

// Recording
async function toggleRecording() {
  if (isRecording.value) {
    await stopRecording()
  } else {
    await startRecording()
  }
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    })

    mediaRecorder = new MediaRecorder(stream)
    audioChunks = []

    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data)
    }

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' })
      await processAudio(audioBlob)
    }

    mediaRecorder.start()
    isRecording.value = true
  } catch (err) {
    console.error('Failed to start recording:', err)
    alert('Microphone access denied')
  }
}

async function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop()
    mediaRecorder.stream.getTracks().forEach(track => track.stop())
    isRecording.value = false
  }
}

// Audio Processing
async function processAudio(audioBlob: Blob) {
  isProcessing.value = true
  const startTime = performance.now()

  try {
    // Create form data
    const formData = new FormData()
    formData.append('file', audioBlob, 'audio.webm')
    formData.append('sample_rate', '16000')

    // Call voice chat endpoint
    const response = await fetch('/api/echo/voice/chat', {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const data = await response.json()
    const endTime = performance.now()
    latency.value = Math.round(endTime - startTime)

    // Add user message
    if (data.transcript) {
      messages.value.unshift({
        id: ++messageId,
        role: 'user',
        text: data.transcript,
        duration: data.timings?.stt_ms
      })
    }

    // Add assistant response
    if (data.response_text) {
      messages.value.unshift({
        id: ++messageId,
        role: 'assistant',
        text: data.response_text,
        duration: data.timings?.total_ms,
        audio: data.audio_base64
      })

      // Auto-play response
      if (autoPlay.value && data.audio_base64) {
        playAudio(data.audio_base64)
      }
    }

    // Continue recording in continuous mode
    if (continuousMode.value) {
      setTimeout(() => startRecording(), 500)
    }

  } catch (err) {
    console.error('Processing error:', err)
    messages.value.unshift({
      id: ++messageId,
      role: 'assistant',
      text: 'Sorry, I had trouble processing that. Please try again.'
    })
  } finally {
    isProcessing.value = false
  }
}

// Audio Playback
function playAudio(base64Audio: string) {
  if (isMuted.value) return
  try {
    const audioData = atob(base64Audio)
    const arrayBuffer = new ArrayBuffer(audioData.length)
    const view = new Uint8Array(arrayBuffer)
    for (let i = 0; i < audioData.length; i++) {
      view[i] = audioData.charCodeAt(i)
    }
    const blob = new Blob([arrayBuffer], { type: 'audio/wav' })
    const url = URL.createObjectURL(blob)
    const audio = new Audio(url)
    audio.play()
    audio.onended = () => URL.revokeObjectURL(url)
  } catch (err) {
    console.error('Playback error:', err)
  }
}

// Cleanup
onUnmounted(() => {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    stopRecording()
  }
})
</script>

<style scoped>
.voice-simple {
  max-width: 600px;
  margin: 0 auto;
  padding: 20px;
  font-family: 'JetBrains Mono', monospace;
}

.voice-header {
  text-align: center;
  margin-bottom: 30px;
}

.voice-header h2 {
  color: #60a5fa;
  margin: 0 0 10px 0;
}

.status-bar {
  display: flex;
  justify-content: center;
  gap: 20px;
  font-size: 0.9rem;
}

.status {
  padding: 4px 12px;
  border-radius: 12px;
  background: #1e293b;
  color: #94a3b8;
  transition: all 0.3s;
}

.status.ready {
  background: #065f46;
  color: #10b981;
}

.status.recording {
  background: #7f1d1d;
  color: #ef4444;
  animation: pulse 1s infinite;
}

.status.processing {
  background: #78350f;
  color: #fbbf24;
}

.latency {
  color: #64748b;
}

.voice-main {
  text-align: center;
  padding: 40px 0;
}

.voice-visualizer {
  position: relative;
  width: 120px;
  height: 120px;
  margin: 0 auto;
}

.pulse-ring {
  position: absolute;
  inset: 0;
  border: 2px solid #3b82f640;
  border-radius: 50%;
  opacity: 0;
}

.voice-visualizer.active .pulse-ring:nth-child(1) {
  animation: pulse-ring 1.5s infinite;
}

.voice-visualizer.active .pulse-ring:nth-child(2) {
  animation: pulse-ring 1.5s infinite 0.3s;
}

.voice-visualizer.active .pulse-ring:nth-child(3) {
  animation: pulse-ring 1.5s infinite 0.6s;
}

.voice-button {
  position: relative;
  width: 100%;
  height: 100%;
  border-radius: 50%;
  background: linear-gradient(135deg, #1e3a8a, #1e40af);
  border: 3px solid #3b82f6;
  color: white;
  cursor: pointer;
  transition: all 0.3s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.voice-button:hover:not(:disabled) {
  transform: scale(1.05);
  box-shadow: 0 0 30px #3b82f640;
}

.voice-button.recording {
  background: linear-gradient(135deg, #7f1d1d, #991b1b);
  border-color: #ef4444;
  animation: pulse 1s infinite;
}

.voice-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.voice-hint {
  margin-top: 20px;
  color: #64748b;
  font-size: 0.9rem;
}

.conversation {
  max-height: 400px;
  overflow-y: auto;
  padding: 20px;
  background: #0f172a;
  border-radius: 12px;
  margin: 20px 0;
}

.message {
  margin-bottom: 16px;
}

.message.user {
  text-align: right;
}

.message.assistant {
  text-align: left;
}

.message-content {
  display: inline-block;
  max-width: 80%;
  padding: 12px 16px;
  border-radius: 12px;
  background: #1e293b;
}

.message.user .message-content {
  background: #1e3a8a;
}

.message-text {
  color: #e2e8f0;
  line-height: 1.5;
}

.message-meta {
  display: flex;
  gap: 10px;
  margin-top: 8px;
  font-size: 0.8rem;
  color: #64748b;
  align-items: center;
}

.message.user .message-meta {
  justify-content: flex-end;
}

.replay-btn {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 6px;
  padding: 2px 8px;
  color: #94a3b8;
  cursor: pointer;
  font-size: 0.8rem;
  transition: all 0.2s;
}

.replay-btn:hover {
  background: #334155;
  color: #e2e8f0;
}

.settings {
  display: flex;
  justify-content: center;
  gap: 30px;
  padding: 20px;
  background: #0f172a;
  border-radius: 12px;
}

.settings label {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #94a3b8;
  font-size: 0.9rem;
  cursor: pointer;
}

.settings input[type="checkbox"] {
  width: 18px;
  height: 18px;
  cursor: pointer;
}

/* Animations */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

@keyframes pulse-ring {
  0% {
    transform: scale(1);
    opacity: 0.6;
  }
  100% {
    transform: scale(1.8);
    opacity: 0;
  }
}

.message-enter-active {
  transition: all 0.3s ease;
}

.message-enter-from {
  opacity: 0;
  transform: translateY(-20px);
}

/* Scrollbar */
.conversation::-webkit-scrollbar {
  width: 6px;
}

.conversation::-webkit-scrollbar-thumb {
  background: #334155;
  border-radius: 3px;
}

.conversation::-webkit-scrollbar-thumb:hover {
  background: #475569;
}
</style>
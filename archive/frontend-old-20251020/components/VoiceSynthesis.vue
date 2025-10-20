<template>
  <div class="voice-synthesis">
    <div class="voice-header">
      <h3>Voice Synthesis</h3>
      <div class="voice-status">
        <span :class="['status-indicator', voiceService.status]"></span>
        {{ voiceService.status }}
      </div>
    </div>

    <div class="voice-controls">
      <!-- Text Input -->
      <div class="text-input-section">
        <textarea 
          v-model="voiceText" 
          placeholder="Enter text to synthesize..."
          :disabled="isGenerating"
          class="voice-textarea"
        ></textarea>
      </div>

      <!-- Voice Settings -->
      <div class="voice-settings">
        <div class="setting-group">
          <label>Voice</label>
          <select v-model="settings.voice" :disabled="isGenerating">
            <option v-for="voice in availableVoices" :key="voice" :value="voice">
              {{ formatVoiceName(voice) }}
            </option>
          </select>
        </div>

        <div class="setting-group">
          <label>Speed: {{ settings.speed }}</label>
          <input 
            type="range" 
            v-model="settings.speed" 
            min="80" 
            max="300" 
            :disabled="isGenerating"
            class="voice-slider"
          />
        </div>

        <div class="setting-group">
          <label>Pitch: {{ settings.pitch }}</label>
          <input 
            type="range" 
            v-model="settings.pitch" 
            min="0" 
            max="99" 
            :disabled="isGenerating"
            class="voice-slider"
          />
        </div>

        <div class="setting-group">
          <label>Volume: {{ settings.volume }}</label>
          <input 
            type="range" 
            v-model="settings.volume" 
            min="0" 
            max="200" 
            :disabled="isGenerating"
            class="voice-slider"
          />
        </div>
      </div>

      <!-- Action Buttons -->
      <div class="voice-actions">
        <button 
          @click="synthesizeAndPlay" 
          :disabled="!voiceText.trim() || isGenerating"
          class="action-btn primary"
        >
          <span v-if="isGenerating" class="spinner"></span>
          {{ isGenerating ? 'Generating...' : 'Speak Text' }}
        </button>

        <button 
          @click="downloadAudio" 
          :disabled="!currentAudioUrl"
          class="action-btn secondary"
        >
          Download Audio
        </button>

        <button 
          @click="clearText" 
          :disabled="isGenerating"
          class="action-btn clear"
        >
          Clear
        </button>
      </div>

      <!-- Audio Player -->
      <div v-if="currentAudioUrl" class="audio-player-section">
        <audio 
          ref="audioPlayer"
          :src="currentAudioUrl" 
          controls 
          class="voice-audio-player"
          @loadstart="onAudioLoad"
          @error="onAudioError"
        ></audio>
      </div>

      <!-- Quick Preset Messages -->
      <div class="preset-messages">
        <h4>Quick Messages</h4>
        <div class="preset-grid">
          <button 
            v-for="preset in presetMessages" 
            :key="preset.id"
            @click="speakPreset(preset.text)"
            :disabled="isGenerating"
            class="preset-btn"
          >
            {{ preset.label }}
          </button>
        </div>
      </div>

      <!-- Voice History -->
      <div class="voice-history">
        <h4>Recent Synthesis</h4>
        <div class="history-list">
          <div 
            v-for="item in voiceHistory.slice(0, 5)" 
            :key="item.id" 
            class="history-item"
          >
            <span class="history-text">{{ item.text.substring(0, 50) }}...</span>
            <button 
              @click="repeatSynthesis(item)"
              :disabled="isGenerating"
              class="repeat-btn"
            >
              Repeat
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'

// Types
interface VoiceSettings {
  voice: string
  speed: number
  pitch: number
  volume: number
}

interface VoiceHistoryItem {
  id: string
  text: string
  settings: VoiceSettings
  timestamp: Date
}

interface PresetMessage {
  id: string
  label: string
  text: string
}

// Reactive State
const voiceText = ref('')
const isGenerating = ref(false)
const currentAudioUrl = ref<string | null>(null)
const audioPlayer = ref<HTMLAudioElement>()

const voiceService = reactive({
  status: 'checking' as 'healthy' | 'unhealthy' | 'checking'
})

const settings = reactive<VoiceSettings>({
  voice: 'en',
  speed: 175,
  pitch: 50,
  volume: 100
})

const availableVoices = ref<string[]>([
  'en', 'en+f3', 'en+m1', 'en+m2', 'en+m3', 'en+m4', 'en+m5', 'en+m6', 'en+m7',
  'en-us', 'en-uk', 'en-au', 'en-ca', 'en-in', 'en-ie', 'en-sc', 'en-za'
])

const voiceHistory = ref<VoiceHistoryItem[]>([])

const presetMessages: PresetMessage[] = [
  { id: 'system-online', label: 'System Online', text: 'Tower systems are online and operational' },
  { id: 'task-complete', label: 'Task Complete', text: 'Task completed successfully' },
  { id: 'error-alert', label: 'Error Alert', text: 'System error detected, please check logs' },
  { id: 'backup-complete', label: 'Backup Complete', text: 'Database backup completed successfully' },
  { id: 'update-ready', label: 'Update Ready', text: 'System update is ready for installation' },
  { id: 'maintenance', label: 'Maintenance Mode', text: 'System entering maintenance mode' }
]

// API Functions
const VOICE_API_BASE = 'http://localhost:8316'

const checkVoiceService = async () => {
  try {
    const response = await fetch(`${VOICE_API_BASE}/health`)
    if (response.ok) {
      voiceService.status = 'healthy'
      const data = await response.json()
      console.log('Voice service:', data)
    } else {
      voiceService.status = 'unhealthy'
    }
  } catch (error) {
    console.error('Voice service check failed:', error)
    voiceService.status = 'unhealthy'
  }
}

const synthesizeAndPlay = async () => {
  if (!voiceText.value.trim()) return
  
  isGenerating.value = true
  
  try {
    const response = await fetch(`${VOICE_API_BASE}/synthesize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: voiceText.value,
        voice: settings.voice,
        speed: settings.speed,
        pitch: settings.pitch,
        volume: settings.volume
      })
    })
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }
    
    // Get audio blob
    const audioBlob = await response.blob()
    
    // Create object URL for audio playback
    if (currentAudioUrl.value) {
      URL.revokeObjectURL(currentAudioUrl.value)
    }
    
    currentAudioUrl.value = URL.createObjectURL(audioBlob)
    
    // Add to history
    addToHistory(voiceText.value, { ...settings })
    
    // Auto-play audio
    setTimeout(() => {
      if (audioPlayer.value) {
        audioPlayer.value.play().catch(console.error)
      }
    }, 100)
    
    showToast('Voice synthesis completed', 'success')
    
  } catch (error) {
    console.error('Voice synthesis failed:', error)
    showToast(`Synthesis failed: ${error.message}`, 'error')
  } finally {
    isGenerating.value = false
  }
}

const speakPreset = async (text: string) => {
  voiceText.value = text
  await synthesizeAndPlay()
}

const repeatSynthesis = async (item: VoiceHistoryItem) => {
  voiceText.value = item.text
  Object.assign(settings, item.settings)
  await synthesizeAndPlay()
}

const downloadAudio = () => {
  if (currentAudioUrl.value) {
    const a = document.createElement('a')
    a.href = currentAudioUrl.value
    a.download = `tower-voice-${Date.now()}.wav`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }
}

const clearText = () => {
  voiceText.value = ''
  if (currentAudioUrl.value) {
    URL.revokeObjectURL(currentAudioUrl.value)
    currentAudioUrl.value = null
  }
}

const addToHistory = (text: string, settingsSnapshot: VoiceSettings) => {
  const historyItem: VoiceHistoryItem = {
    id: `${Date.now()}-${Math.random()}`,
    text,
    settings: { ...settingsSnapshot },
    timestamp: new Date()
  }
  
  voiceHistory.value.unshift(historyItem)
  
  // Keep only last 20 items
  if (voiceHistory.value.length > 20) {
    voiceHistory.value = voiceHistory.value.slice(0, 20)
  }
  
  // Save to localStorage
  localStorage.setItem('tower-voice-history', JSON.stringify(voiceHistory.value))
}

const loadHistory = () => {
  try {
    const saved = localStorage.getItem('tower-voice-history')
    if (saved) {
      voiceHistory.value = JSON.parse(saved).map((item: any) => ({
        ...item,
        timestamp: new Date(item.timestamp)
      }))
    }
  } catch (error) {
    console.error('Failed to load voice history:', error)
  }
}

// Utility Functions
const formatVoiceName = (voice: string): string => {
  const names: Record<string, string> = {
    'en': 'English (Default)',
    'en+f3': 'English Female',
    'en+m1': 'English Male 1',
    'en+m2': 'English Male 2',
    'en+m3': 'English Male 3',
    'en+m4': 'English Male 4',
    'en+m5': 'English Male 5',
    'en+m6': 'English Male 6',
    'en+m7': 'English Male 7',
    'en-us': 'US English',
    'en-uk': 'UK English',
    'en-au': 'Australian English',
    'en-ca': 'Canadian English',
    'en-in': 'Indian English',
    'en-ie': 'Irish English',
    'en-sc': 'Scottish English',
    'en-za': 'South African English'
  }
  return names[voice] || voice
}

const showToast = (message: string, type: string) => {
  const toast = document.createElement('div')
  toast.className = `toast toast-${type}`
  toast.textContent = message
  document.body.appendChild(toast)
  setTimeout(() => {
    if (document.body.contains(toast)) {
      document.body.removeChild(toast)
    }
  }, 3000)
}

const onAudioLoad = () => {
  console.log('Audio loaded successfully')
}

const onAudioError = (event: Event) => {
  console.error('Audio playback error:', event)
  showToast('Audio playback failed', 'error')
}

// Lifecycle
onMounted(async () => {
  await checkVoiceService()
  loadHistory()
})
</script>

<style scoped>
.voice-synthesis {
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 8px;
  padding: 20px;
  color: #ffffff;
}

.voice-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 10px;
  border-bottom: 1px solid #333;
}

.voice-header h3 {
  margin: 0;
  color: #ffffff;
  font-size: 16px;
  font-weight: 600;
}

.voice-status {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}

.status-indicator.healthy { background: #00ff88; }
.status-indicator.unhealthy { background: #ff4444; }
.status-indicator.checking { background: #ffaa00; }

.voice-textarea {
  width: 100%;
  min-height: 100px;
  background: #222;
  border: 1px solid #444;
  color: #e0e0e0;
  padding: 12px;
  border-radius: 6px;
  font-family: inherit;
  resize: vertical;
  margin-bottom: 15px;
}

.voice-textarea:focus {
  outline: none;
  border-color: #00ff88;
  box-shadow: 0 0 10px rgba(0, 255, 136, 0.2);
}

.voice-settings {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 20px;
}

.setting-group {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.setting-group label {
  font-size: 12px;
  color: #ccc;
  font-weight: 500;
}

.setting-group select {
  background: #222;
  border: 1px solid #444;
  color: #e0e0e0;
  padding: 8px;
  border-radius: 4px;
  font-family: inherit;
}

.voice-slider {
  background: #333;
  height: 4px;
  border-radius: 2px;
  outline: none;
  opacity: 0.8;
  transition: opacity 0.2s;
}

.voice-slider:hover {
  opacity: 1;
}

.voice-slider::-webkit-slider-thumb {
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #00ff88;
  cursor: pointer;
}

.voice-actions {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}

.action-btn {
  padding: 10px 20px;
  border: none;
  border-radius: 6px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: 8px;
}

.action-btn.primary {
  background: #00ff88;
  color: #000;
}

.action-btn.primary:hover:not(:disabled) {
  background: #00cc66;
  transform: translateY(-1px);
}

.action-btn.secondary {
  background: #2a2a2a;
  color: #fff;
  border: 1px solid #555;
}

.action-btn.secondary:hover:not(:disabled) {
  background: #3a3a3a;
}

.action-btn.clear {
  background: #ff4444;
  color: #fff;
}

.action-btn.clear:hover:not(:disabled) {
  background: #cc3333;
}

.action-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none;
}

.spinner {
  width: 12px;
  height: 12px;
  border: 2px solid #333;
  border-top: 2px solid #fff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.audio-player-section {
  margin-bottom: 20px;
  padding: 15px;
  background: #222;
  border-radius: 6px;
  border: 1px solid #444;
}

.voice-audio-player {
  width: 100%;
  background: #333;
}

.preset-messages h4,
.voice-history h4 {
  margin: 0 0 10px 0;
  color: #ccc;
  font-size: 14px;
  font-weight: 500;
}

.preset-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 8px;
  margin-bottom: 20px;
}

.preset-btn {
  background: #2a2a2a;
  color: #fff;
  border: 1px solid #555;
  padding: 8px 12px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s ease;
}

.preset-btn:hover:not(:disabled) {
  background: #3a3a3a;
  border-color: #777;
}

.preset-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.history-list {
  max-height: 150px;
  overflow-y: auto;
}

.history-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px;
  background: #222;
  border: 1px solid #333;
  border-radius: 4px;
  margin-bottom: 6px;
}

.history-text {
  flex: 1;
  font-size: 12px;
  color: #ccc;
}

.repeat-btn {
  background: #333;
  color: #fff;
  border: 1px solid #555;
  padding: 4px 8px;
  border-radius: 3px;
  cursor: pointer;
  font-size: 11px;
}

.repeat-btn:hover:not(:disabled) {
  background: #444;
}

.repeat-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Toast Styles */
.toast {
  position: fixed;
  top: 20px;
  right: 20px;
  padding: 12px 20px;
  border-radius: 6px;
  color: #fff;
  font-weight: 500;
  z-index: 1000;
  animation: slideIn 0.3s ease;
}

.toast-success { background: #00ff88; color: #000; }
.toast-error { background: #ff4444; }
.toast-warning { background: #ffaa00; color: #000; }

@keyframes slideIn {
  from { transform: translateX(100%); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}

/* Responsive */
@media (max-width: 768px) {
  .voice-settings {
    grid-template-columns: 1fr;
  }
  
  .voice-actions {
    flex-direction: column;
  }
  
  .preset-grid {
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  }
}
</style>
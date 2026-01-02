<template>
  <div class="space-y-6">
    <TowerCard>
      <template #header>
        <h2 class="text-2xl font-bold text-tower-text-primary">Voice Interface</h2>
      </template>
      
      <div class="text-center space-y-6 py-8">
        <div class="voice-indicator" :class="{ active: isRecording }">
          <div class="indicator-wave"></div>
        </div>

        <p class="text-tower-text-secondary">
          {{ isRecording ? 'Listening...' : 'Click to start speaking' }}
        </p>

        <TowerButton 
          size="lg"
          :variant="isRecording ? 'danger' : 'primary'"
          @click="toggleRecording"
        >
          {{ isRecording ? 'Stop' : 'Start' }}
        </TowerButton>
      </div>
    </TowerCard>

    <TowerCard v-if="transcript">
      <template #header>
        <h3 class="text-lg font-semibold text-tower-text-primary">Transcript</h3>
      </template>
      <p class="text-tower-text-secondary">{{ transcript }}</p>
    </TowerCard>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const isRecording = ref(false)
const transcript = ref('')

const toggleRecording = () => {
  isRecording.value = !isRecording.value
  if (!isRecording.value) {
    transcript.value = 'Voice recording not yet implemented - coming soon!'
  }
}
</script>

<style scoped>
.voice-indicator {
  width: 200px;
  height: 200px;
  margin: 0 auto;
  border-radius: 50%;
  background: radial-gradient(circle, var(--tower-accent-primary), transparent);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s;
}

.voice-indicator.active {
  background: radial-gradient(circle, var(--tower-accent-danger), transparent);
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.1); }
}

.indicator-wave {
  width: 100px;
  height: 100px;
  border-radius: 50%;
  background-color: var(--tower-bg-card);
}
</style>

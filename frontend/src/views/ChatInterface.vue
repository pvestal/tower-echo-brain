<template>
  <div class="space-y-6">
    <TowerCard>
      <template #header>
        <h2 class="text-2xl font-bold text-tower-text-primary">Chat with Echo</h2>
      </template>
      
      <div class="space-y-4">
        <div class="messages-container h-96 overflow-y-auto space-y-3">
          <div 
            v-for="(msg, idx) in messages" 
            :key="idx"
            class="message"
            :class="msg.role"
          >
            <div class="message-content">
              {{ msg.content }}
            </div>
          </div>
        </div>

        <div class="flex gap-3">
          <TowerInput
            v-model="userMessage"
            placeholder="Type your message..."
            @keyup.enter="sendMessage"
            class="flex-1"
          />
          <TowerButton @click="sendMessage" :loading="loading">
            Send
          </TowerButton>
        </div>
      </div>
    </TowerCard>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { TowerCard, TowerButton, TowerInput } from '@tower/ui-components'
import axios from 'axios'

const messages = ref([])
const userMessage = ref('')
const loading = ref(false)

const sendMessage = async () => {
  if (!userMessage.value.trim()) return

  const msg = userMessage.value
  messages.value.push({ role: 'user', content: msg })
  userMessage.value = ''
  loading.value = true

  try {
    const response = await axios.post('/api/echo/chat', {
      message: msg
    })
    messages.value.push({ role: 'assistant', content: response.data.response })
  } catch (error) {
    messages.value.push({ role: 'error', content: 'Error: ' + error.message })
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.messages-container {
  padding: 1rem;
  background-color: var(--tower-bg-elevated);
  border-radius: 0.5rem;
}

.message {
  padding: 0.75rem 1rem;
  border-radius: 0.5rem;
  max-width: 80%;
}

.message.user {
  background-color: var(--tower-accent-primary);
  color: white;
  margin-left: auto;
}

.message.assistant {
  background-color: var(--tower-bg-card);
  color: var(--tower-text-primary);
}

.message.error {
  background-color: var(--tower-accent-danger);
  color: white;
}
</style>

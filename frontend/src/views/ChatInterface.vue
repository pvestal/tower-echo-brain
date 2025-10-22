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
            <div v-if="msg.retry" class="message-meta">
              <button @click="retryMessage(msg)" class="retry-button">
                Retry
              </button>
            </div>
          </div>
        </div>

        <div v-if="connectionError" class="error-banner">
          ⚠️ Connection issues detected. Messages will auto-retry.
        </div>

        <div class="flex gap-3">
          <TowerInput
            v-model="userMessage"
            placeholder="Type your message..."
            @keyup.enter="sendMessage"
            :disabled="loading"
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
const connectionError = ref(false)

// Retry configuration
const MAX_RETRIES = 3
const RETRY_DELAY = 1000 // Start with 1 second

// Helper function to sleep
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms))

// Send message with retry logic
const sendMessageWithRetry = async (msg, retryCount = 0) => {
  try {
    const response = await axios.post('http://192.168.50.135:8309/api/echo/chat', {
      query: msg,
      user_id: 'web_user',
      conversation_id: 'web_chat_' + Date.now(),
      intelligence_level: 'auto',
      context: {}
    }, {
      timeout: 30000, // 30 second timeout
      headers: {
        'Content-Type': 'application/json'
      }
    })

    connectionError.value = false
    return { success: true, data: response.data }

  } catch (error) {
    console.error('Echo API Error:', error.response?.data || error.message)

    // Check if it's a retryable error
    const isRetryable =
      error.code === 'ECONNABORTED' || // Timeout
      error.code === 'ECONNREFUSED' || // Connection refused
      error.response?.status >= 500 ||  // Server errors
      error.response?.status === 422    // Validation errors can be retried

    if (isRetryable && retryCount < MAX_RETRIES) {
      connectionError.value = true
      const delay = RETRY_DELAY * Math.pow(2, retryCount) // Exponential backoff
      console.log(`Retrying in ${delay}ms... (attempt ${retryCount + 1}/${MAX_RETRIES})`)
      await sleep(delay)
      return sendMessageWithRetry(msg, retryCount + 1)
    }

    // Not retryable or max retries reached
    return {
      success: false,
      error: error.response?.data?.detail || error.message,
      retryable: isRetryable,
      statusCode: error.response?.status
    }
  }
}

const sendMessage = async () => {
  if (!userMessage.value.trim()) return

  const msg = userMessage.value
  messages.value.push({ role: 'user', content: msg })
  userMessage.value = ''
  loading.value = true

  const result = await sendMessageWithRetry(msg)

  if (result.success) {
    messages.value.push({
      role: 'assistant',
      content: result.data.response
    })
  } else {
    const errorMessage = {
      role: 'error',
      content: `Error: ${result.error}${result.statusCode ? ` (${result.statusCode})` : ''}`,
      retry: result.retryable,
      originalMessage: msg
    }
    messages.value.push(errorMessage)
  }

  loading.value = false
}

const retryMessage = async (errorMsg) => {
  if (!errorMsg.originalMessage) return

  // Remove the error message
  const index = messages.value.indexOf(errorMsg)
  if (index > -1) {
    messages.value.splice(index, 1)
  }

  // Retry sending
  loading.value = true
  const result = await sendMessageWithRetry(errorMsg.originalMessage)

  if (result.success) {
    messages.value.push({
      role: 'assistant',
      content: result.data.response
    })
  } else {
    messages.value.push({
      role: 'error',
      content: `Error: ${result.error}${result.statusCode ? ` (${result.statusCode})` : ''}`,
      retry: result.retryable,
      originalMessage: errorMsg.originalMessage
    })
  }

  loading.value = false
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

.message-meta {
  margin-top: 0.5rem;
  font-size: 0.875rem;
}

.retry-button {
  background: rgba(255, 255, 255, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.3);
  padding: 0.25rem 0.75rem;
  border-radius: 0.25rem;
  color: white;
  cursor: pointer;
  transition: all 0.2s;
}

.retry-button:hover {
  background: rgba(255, 255, 255, 0.3);
}

.error-banner {
  background-color: rgba(245, 158, 11, 0.1);
  border: 1px solid rgba(245, 158, 11, 0.3);
  color: #f59e0b;
  padding: 0.75rem 1rem;
  border-radius: 0.5rem;
  font-size: 0.875rem;
}
</style>

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
            <div v-if="msg.requestType" class="request-type-indicator">
              {{ getRequestTypeLabel(msg.requestType) }}
            </div>
            <div v-if="msg.metadata" class="message-metadata">
              <div v-for="(value, key) in msg.metadata" :key="key" class="metadata-item">
                <span class="metadata-key">{{ key }}:</span>
                <span class="metadata-value">{{ value }}</span>
              </div>
            </div>
            <div v-if="msg.retry" class="message-meta">
              <TowerButton @click="retryMessage(msg)" variant="link" size="sm">
                Retry
              </TowerButton>
            </div>
          </div>
        </div>

        <div v-if="connectionError" class="error-banner">
          ⚠️ Connection issues detected. Messages will auto-retry.
        </div>

        <!-- Model Selector -->
        <div class="model-selector mb-4">
          <div class="flex gap-4 items-center">
            <span class="text-sm font-medium text-tower-text-secondary">Model:</span>
            <TowerSelect
              v-model="selectedModel"
              :options="modelOptions"
              placeholder="Default (Auto-select)"
              value-key="value"
              label-key="label"
            />
            <TowerButton @click="refreshModels" variant="secondary" size="sm">
              Refresh
            </TowerButton>
          </div>
        </div>

        <!-- Request Type Selector -->
        <div class="request-type-selector mb-4">
          <div class="flex gap-4 items-center">
            <span class="text-sm font-medium text-tower-text-secondary">Request Type:</span>
            <label class="flex items-center gap-2">
              <input type="radio" v-model="requestType" value="conversation" class="text-tower-accent-primary">
              <span class="text-sm">💬 Conversation</span>
            </label>
            <label class="flex items-center gap-2">
              <input type="radio" v-model="requestType" value="system_command" class="text-tower-accent-primary">
              <span class="text-sm">🔧 System Command</span>
            </label>
            <label class="flex items-center gap-2">
              <input type="radio" v-model="requestType" value="collaboration" class="text-tower-accent-primary">
              <span class="text-sm">🤝 Collaboration</span>
            </label>
          </div>
        </div>

        <!-- Command Options (show when system_command selected) -->
        <div v-if="requestType === 'system_command'" class="command-options mb-4 p-3 bg-tower-bg-elevated rounded-lg">
          <div class="flex gap-4 items-center">
            <label class="flex items-center gap-2">
              <input type="checkbox" v-model="safeMode" class="text-tower-accent-primary">
              <span class="text-sm">Safe Mode</span>
            </label>
            <div class="flex items-center gap-2">
              <span class="text-sm">Timeout:</span>
              <TowerInput
                v-model="timeout"
                type="number"
                min="1"
                max="300"
                class="w-16"
                size="sm"
              />
              <span class="text-sm">sec</span>
            </div>
          </div>
        </div>

        <!-- Collaboration Options (show when collaboration selected) -->
        <div v-if="requestType === 'collaboration'" class="collab-options mb-4 p-3 bg-tower-bg-elevated rounded-lg">
          <div class="flex gap-4 items-center">
            <div class="flex items-center gap-2">
              <span class="text-sm">Workflow:</span>
              <TowerSelect
                v-model="workflowType"
                :options="workflowOptions"
                size="sm"
              />
            </div>
          </div>
        </div>

        <div class="flex gap-3">
          <TowerInput
            v-model="userMessage"
            :placeholder="getPlaceholderText()"
            @keyup.enter="sendMessage"
            :disabled="loading"
            class="flex-1"
          />
          <TowerButton @click="sendMessage" :loading="loading">
            {{ getButtonText() }}
          </TowerButton>
        </div>
      </div>
    </TowerCard>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import axios from 'axios'

const messages = ref([])
const userMessage = ref('')
const loading = ref(false)
const connectionError = ref(false)

// Request type controls
const requestType = ref('conversation')
const selectedModel = ref('')
const availableModels = ref([])
const safeMode = ref(true)
const timeout = ref(30)
const workflowType = ref('sequential')

// Options for TowerSelect components
const modelOptions = computed(() => [
  { value: '', label: 'Default (Auto-select)' },
  ...availableModels.value.map(model => ({
    value: model.name,
    label: `${model.name} (${model.size})`
  }))
])

const workflowOptions = [
  { value: 'sequential', label: 'Sequential' },
  { value: 'parallel', label: 'Parallel' },
  { value: 'consensus', label: 'Consensus' }
]

// Retry configuration
const MAX_RETRIES = 3
const RETRY_DELAY = 1000 // Start with 1 second

// Helper function to sleep
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms))

// Extract response content based on request type
const getResponseContent = (data) => {
  switch (requestType.value) {
    case 'system_command':
      if (data.success === false) {
        return `❌ Command failed (exit code: ${data.exit_code})\n${data.error || data.output}`;
      }
      return `✅ Command executed (exit code: ${data.exit_code})\n${data.output}`;
    case 'collaboration':
      return data.response || data.result || JSON.stringify(data, null, 2);
    default:
      return data.response || data.output || 'No response received';
  }
}

// Extract response metadata
const getResponseMetadata = (data) => {
  const metadata = {
    processing_time: data.processing_time ? `${(data.processing_time * 1000).toFixed(0)}ms` : 'N/A'
  }

  switch (requestType.value) {
    case 'system_command':
      metadata.exit_code = data.exit_code || 'N/A'
      metadata.safety_checks = data.safety_checks ? 'Passed' : 'N/A'
      break;
    case 'collaboration':
      metadata.workflow_type = workflowType.value
      metadata.models_used = data.models_used || 'N/A'
      break;
    default:
      metadata.model_used = data.model_used || 'N/A'
      metadata.intelligence_level = data.intelligence_level || 'N/A'
      break;
  }

  return metadata
}

// Get dynamic placeholder text
const getPlaceholderText = () => {
  switch (requestType.value) {
    case 'system_command':
      return 'Enter system command (e.g., systemctl status, ps aux, curl...)';
    case 'collaboration':
      return 'Describe workflow for multi-LLM collaboration...';
    default:
      return 'Type your message...';
  }
}

// Get dynamic button text
const getButtonText = () => {
  switch (requestType.value) {
    case 'system_command':
      return 'Execute';
    case 'collaboration':
      return 'Collaborate';
    default:
      return 'Send';
  }
}

// Get request type label for display
const getRequestTypeLabel = (type) => {
  switch (type) {
    case 'system_command':
      return '🔧 System Command';
    case 'collaboration':
      return '🤝 Collaboration';
    default:
      return '💬 Conversation';
  }
}

// Get API endpoint based on request type
const getApiEndpoint = () => {
  switch (requestType.value) {
    case 'system_command':
      return 'http://***REMOVED***:8309/api/echo/execute';
    case 'collaboration':
      return 'http://***REMOVED***:8309/api/echo/collaborate';
    default:
      return 'http://***REMOVED***:8309/api/echo/query';
  }
}

// Build payload based on request type
const buildRequestPayload = (msg) => {
  const basePayload = {
    user_id: 'web_user',
    conversation_id: `web_${requestType.value}_${Date.now()}`
  }

  switch (requestType.value) {
    case 'system_command':
      return {
        ...basePayload,
        command: msg,
        safe_mode: safeMode.value,
        timeout: timeout.value
      };
    case 'collaboration':
      return {
        ...basePayload,
        workflow: msg,
        workflow_type: workflowType.value,
        models: ['llama3.1:8b', 'qwen2.5-coder:32b'] // Default models
      };
    default:
      return {
        ...basePayload,
        query: msg,
        intelligence_level: 'auto',
        request_type: 'conversation'
      };
  }
}

// Send message with retry logic
const refreshModels = async () => {
  try {
    const response = await axios.get('/api/echo/models/list')
    availableModels.value = response.data || []
  } catch (error) {
    console.error('Failed to fetch models:', error)
    availableModels.value = []
  }
}

const sendMessageWithRetry = async (msg, retryCount = 0) => {
  try {
    const endpoint = getApiEndpoint()
    const payload = buildRequestPayload(msg)

    // Add selected model to payload if specified
    if (selectedModel.value) {
      payload.model = selectedModel.value
    }

    const response = await axios.post(endpoint, payload, {
      timeout: (requestType.value === 'system_command' ? timeout.value * 1000 : 30000),
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
  messages.value.push({ role: 'user', content: msg, requestType: requestType.value })
  userMessage.value = ''
  loading.value = true

  const result = await sendMessageWithRetry(msg)

  if (result.success) {
    const responseContent = getResponseContent(result.data)
    messages.value.push({
      role: 'assistant',
      content: responseContent,
      metadata: getResponseMetadata(result.data),
      requestType: requestType.value
    })
  } else {
    const errorMessage = {
      role: 'error',
      content: `Error: ${result.error}${result.statusCode ? ` (${result.statusCode})` : ''}`,
      retry: result.retryable,
      originalMessage: msg,
      requestType: requestType.value
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

  // Set request type for retry
  if (errorMsg.requestType) {
    requestType.value = errorMsg.requestType
  }

  // Retry sending
  loading.value = true
  const result = await sendMessageWithRetry(errorMsg.originalMessage)

  if (result.success) {
    const responseContent = getResponseContent(result.data)
    messages.value.push({
      role: 'assistant',
      content: responseContent,
      metadata: getResponseMetadata(result.data),
      requestType: requestType.value
    })
  } else {
    messages.value.push({
      role: 'error',
      content: `Error: ${result.error}${result.statusCode ? ` (${result.statusCode})` : ''}`,
      retry: result.retryable,
      originalMessage: errorMsg.originalMessage,
      requestType: requestType.value
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

/* Request type controls */
.request-type-selector {
  border-bottom: 1px solid var(--tower-border);
  padding-bottom: 1rem;
}

.command-options,
.collab-options {
  border: 1px solid var(--tower-border);
  background-color: var(--tower-bg-elevated);
}

/* Message metadata */
.request-type-indicator {
  font-size: 0.75rem;
  color: var(--tower-text-secondary);
  margin-top: 0.25rem;
  padding: 0.125rem 0.5rem;
  background-color: rgba(47, 129, 247, 0.1);
  border-radius: 0.25rem;
  display: inline-block;
}

.message-metadata {
  margin-top: 0.5rem;
  padding: 0.5rem;
  background-color: rgba(0, 0, 0, 0.1);
  border-radius: 0.25rem;
  font-size: 0.75rem;
}

.metadata-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.25rem;
}

.metadata-key {
  color: var(--tower-text-secondary);
  text-transform: uppercase;
  font-weight: 500;
}

.metadata-value {
  color: var(--tower-text-primary);
  font-family: monospace;
}

/* System command styling */
.message.assistant .message-content {
  font-family: monospace;
  white-space: pre-wrap;
}
</style>


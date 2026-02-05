<template>
  <div class="ask-view">
    <div class="card mb-3">
      <h3>Ask Echo Brain</h3>
      <div class="space-y-4">
        <div>
          <textarea
            v-model="question"
            placeholder="Ask a question..."
            class="w-full"
            rows="3"
            @keydown.meta.enter="submitQuestion"
            @keydown.ctrl.enter="submitQuestion"
          ></textarea>
        </div>
        <div class="flex gap-2">
          <button @click="submitQuestion" class="btn btn-primary" :disabled="loading || !question.trim()">
            {{ loading ? 'Thinking...' : 'Ask' }}
          </button>
          <button @click="toggleStream" class="btn">
            Stream: {{ streamMode ? 'ON' : 'OFF' }}
          </button>
          <button @click="clearHistory" class="btn">Clear</button>
        </div>
      </div>
    </div>

    <div class="card" v-if="history.length > 0">
      <h3>Conversation</h3>
      <div class="space-y-4">
        <div v-for="(item, index) in history" :key="index" class="conversation-item">
          <div class="text-xs text-muted mb-1">{{ formatTime(item.timestamp) }}</div>
          <div class="mb-2">
            <strong>Q:</strong> {{ item.question }}
          </div>
          <div v-if="item.answer" class="response">
            <strong>A:</strong>
            <pre>{{ item.answer }}</pre>
          </div>
          <div v-else-if="item.error" class="text-error">
            Error: {{ item.error }}
          </div>
        </div>
      </div>
    </div>

    <div class="card mt-3" v-if="recentMemories.length > 0">
      <h3>Related Memories</h3>
      <div class="space-y-2">
        <div v-for="(memory, index) in recentMemories" :key="index" class="memory-item">
          <div class="text-xs text-muted">Score: {{ memory.score?.toFixed(3) }}</div>
          <div class="text-sm">{{ memory.content }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { askApi, memoryApi } from '@/api/echoApi';

const question = ref('');
const loading = ref(false);
const streamMode = ref(false);
const history = ref<any[]>([]);
const recentMemories = ref<any[]>([]);

const submitQuestion = async () => {
  if (!question.value.trim() || loading.value) return;

  loading.value = true;
  const q = question.value;
  const entry: any = {
    question: q,
    timestamp: new Date(),
    answer: null,
    error: null
  };

  history.value.unshift(entry);
  question.value = '';

  try {
    if (streamMode.value) {
      await askApi.stream(q);
      // Handle streaming response
      entry.answer = 'Streaming not yet implemented in UI';
    } else {
      const response = await askApi.ask(q);
      entry.answer = response.data.answer || response.data.response || JSON.stringify(response.data);
    }

    // Search for related memories
    try {
      const memResponse = await memoryApi.search(q, 3);
      recentMemories.value = memResponse.data.results || [];
    } catch (e) {
      // Ignore memory search errors
    }
  } catch (error: any) {
    entry.error = error.response?.data?.detail || error.message;
  } finally {
    loading.value = false;
  }
};

const toggleStream = () => {
  streamMode.value = !streamMode.value;
};

const clearHistory = () => {
  history.value = [];
  recentMemories.value = [];
};

const formatTime = (timestamp: Date) => {
  return new Date(timestamp).toLocaleTimeString();
};
</script>

<style scoped>
.conversation-item {
  padding: 1rem;
  border-left: 2px solid #21262d;
  margin-left: 0.5rem;
}

.conversation-item:hover {
  border-left-color: #2f81f7;
  background: rgba(47, 129, 247, 0.05);
}

.memory-item {
  padding: 0.5rem;
  border-left: 2px solid #21262d;
  margin-left: 0.5rem;
  font-size: 0.75rem;
}

.response pre {
  white-space: pre-wrap;
  word-wrap: break-word;
  background: transparent;
  border: none;
  padding: 0;
  margin: 0.5rem 0 0 0;
}

.text-error {
  color: #f85149;
}
</style>
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

            <!-- Debug Information -->
            <div v-if="item.debug" class="debug-info mt-3">
              <details>
                <summary class="cursor-pointer text-xs text-muted">Debug Info</summary>
                <div class="mt-2 text-xs">
                  <div v-if="item.debug.search_terms?.length">
                    <strong>Search Terms:</strong> {{ item.debug.search_terms.join(', ') }}
                  </div>
                  <div v-if="item.debug.steps?.length" class="mt-1">
                    <strong>Steps:</strong>
                    <ul class="ml-4">
                      <li v-for="(step, si) in item.debug.steps" :key="si">{{ step }}</li>
                    </ul>
                  </div>
                  <div v-if="item.debug.total_sources" class="mt-1">
                    <strong>Total Sources:</strong> {{ item.debug.total_sources }}
                  </div>
                </div>
              </details>
            </div>

            <!-- Source Information -->
            <div v-if="item.sources?.length" class="sources-info mt-3">
              <details>
                <summary class="cursor-pointer text-xs text-muted">
                  Sources ({{ item.sources.length }})
                </summary>
                <div class="mt-2 space-y-1">
                  <div v-for="(source, si) in item.sources" :key="si" class="source-item">
                    <span class="source-type" :class="`source-${source.type}`">
                      {{ source.type }}
                    </span>
                    <span class="text-xs">{{ source.content }}</span>
                    <span v-if="source.confidence" class="text-xs text-muted ml-1">
                      ({{ (source.confidence * 100).toFixed(0) }}%)
                    </span>
                  </div>
                </div>
              </details>
            </div>
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
      entry.debug = response.data.debug;
      entry.sources = response.data.sources;
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

.debug-info, .sources-info {
  border-top: 1px solid #21262d;
  padding-top: 0.5rem;
}

.source-item {
  padding: 0.25rem 0;
  font-size: 0.75rem;
}

.source-type {
  display: inline-block;
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 0.65rem;
  font-weight: bold;
  margin-right: 0.5rem;
  text-transform: uppercase;
}

.source-fact {
  background: #2ea043;
  color: white;
}

.source-memory, .source-vector {
  background: #1f6feb;
  color: white;
}

.source-conversation {
  background: #8957e5;
  color: white;
}

.source-core {
  background: #da3633;
  color: white;
}

details summary {
  cursor: pointer;
  user-select: none;
}

details summary:hover {
  color: #58a6ff;
}
</style>
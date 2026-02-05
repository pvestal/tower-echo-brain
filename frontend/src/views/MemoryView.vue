<template>
  <div class="memory-view">
    <div class="card mb-3">
      <h3>Memory Operations</h3>

      <div class="grid grid-2 gap-4 mb-3">
        <div>
          <h4 class="text-xs text-muted mb-2">Search Memory</h4>
          <input
            v-model="searchQuery"
            @keyup.enter="searchMemory"
            placeholder="Search query..."
            class="w-full mb-2"
          />
          <div class="flex gap-2">
            <input
              v-model.number="searchLimit"
              type="number"
              min="1"
              max="100"
              placeholder="Limit"
              class="w-full"
              style="max-width: 100px"
            />
            <button @click="searchMemory" class="btn btn-primary" :disabled="searchLoading">
              {{ searchLoading ? 'Searching...' : 'Search' }}
            </button>
          </div>
        </div>

        <div>
          <h4 class="text-xs text-muted mb-2">Ingest Content</h4>
          <textarea
            v-model="ingestContent"
            placeholder="Content to ingest..."
            rows="3"
            class="w-full mb-2"
          ></textarea>
          <button @click="ingestMemory" class="btn btn-primary" :disabled="ingestLoading">
            {{ ingestLoading ? 'Ingesting...' : 'Ingest' }}
          </button>
        </div>
      </div>

      <div class="grid grid-3 gap-4">
        <button @click="getMemoryStatus" class="btn">
          Get Status
        </button>
        <button @click="getMemoryHealth" class="btn">
          Check Health
        </button>
        <button @click="clearResults" class="btn">
          Clear Results
        </button>
      </div>
    </div>

    <div class="card" v-if="memoryStatus">
      <h3>Memory Status</h3>
      <pre>{{ JSON.stringify(memoryStatus, null, 2) }}</pre>
    </div>

    <div class="card mt-3" v-if="searchResults.length > 0">
      <h3>Search Results ({{ searchResults.length }})</h3>
      <div class="space-y-2">
        <div v-for="(result, index) in searchResults" :key="index" class="result-item">
          <div class="flex justify-between mb-1">
            <span class="text-xs text-muted">Score: {{ result.score?.toFixed(4) }}</span>
            <span class="text-xs text-muted">{{ result.metadata?.source || 'Unknown' }}</span>
          </div>
          <div class="text-sm">{{ result.content || result.text }}</div>
          <div v-if="result.metadata" class="text-xs text-muted mt-1">
            <details>
              <summary>Metadata</summary>
              <pre>{{ JSON.stringify(result.metadata, null, 2) }}</pre>
            </details>
          </div>
        </div>
      </div>
    </div>

    <div class="card mt-3" v-if="lastResponse">
      <h3>Last Response</h3>
      <pre>{{ JSON.stringify(lastResponse, null, 2) }}</pre>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { memoryApi } from '@/api/echoApi';

const searchQuery = ref('');
const searchLimit = ref(10);
const searchLoading = ref(false);
const searchResults = ref<any[]>([]);

const ingestContent = ref('');
const ingestLoading = ref(false);

const memoryStatus = ref<any>(null);
const lastResponse = ref<any>(null);

const searchMemory = async () => {
  if (!searchQuery.value.trim()) return;

  searchLoading.value = true;
  searchResults.value = [];
  lastResponse.value = null;

  try {
    const response = await memoryApi.search(searchQuery.value, searchLimit.value);
    searchResults.value = response.data.results || response.data.memories || [];
    if (!searchResults.value.length) {
      lastResponse.value = response.data;
    }
  } catch (error: any) {
    lastResponse.value = {
      error: error.response?.data?.detail || error.message
    };
  } finally {
    searchLoading.value = false;
  }
};

const ingestMemory = async () => {
  if (!ingestContent.value.trim()) return;

  ingestLoading.value = true;
  lastResponse.value = null;

  try {
    const response = await memoryApi.ingest(ingestContent.value, {
      source: 'manual_ingest',
      timestamp: new Date().toISOString()
    });
    lastResponse.value = response.data;
    ingestContent.value = '';
  } catch (error: any) {
    lastResponse.value = {
      error: error.response?.data?.detail || error.message
    };
  } finally {
    ingestLoading.value = false;
  }
};

const getMemoryStatus = async () => {
  try {
    const response = await memoryApi.status();
    memoryStatus.value = response.data;
  } catch (error: any) {
    lastResponse.value = {
      error: error.response?.data?.detail || error.message
    };
  }
};

const getMemoryHealth = async () => {
  try {
    const response = await memoryApi.health();
    lastResponse.value = response.data;
  } catch (error: any) {
    lastResponse.value = {
      error: error.response?.data?.detail || error.message
    };
  }
};

const clearResults = () => {
  searchResults.value = [];
  memoryStatus.value = null;
  lastResponse.value = null;
};
</script>

<style scoped>
.result-item {
  padding: 0.75rem;
  border: 1px solid #21262d;
  border-radius: 0.375rem;
  background: #0d1117;
}

.result-item:hover {
  border-color: #2f81f7;
}

details {
  cursor: pointer;
}

details summary {
  user-select: none;
  padding: 0.25rem 0;
}

details pre {
  margin-top: 0.5rem;
  font-size: 0.7rem;
}
</style>
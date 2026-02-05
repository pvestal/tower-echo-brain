<script setup lang="ts">
import type { ServiceHealth } from '@/types/echo';

const props = defineProps<{
  service: ServiceHealth;
}>();

const statusClass = (status: string) => {
  return `status ${status}`;
};
</script>

<template>
  <div class="card">
    <div class="flex justify-between items-center mb-1">
      <h3 class="text-sm text-mono">{{ service.name }}</h3>
      <span :class="statusClass(service.status)">
        {{ service.status }}
      </span>
    </div>

    <div class="text-xs text-dim text-mono">
      <div v-if="service.latency_ms">{{ service.latency_ms.toFixed(1) }}ms</div>

      <div v-if="service.name === 'postgres' && service.details" class="mt-2">
        <div>conversations: {{ service.details.conversations || 0 }}</div>
        <div>size: {{ service.details.db_size_mb }}mb</div>
      </div>

      <div v-else-if="service.name === 'qdrant' && service.details" class="mt-2">
        <div>vectors: {{ service.details.vectors_count || 0 }}</div>
        <div>status: {{ service.details.status }}</div>
      </div>

      <div v-else-if="service.name === 'ollama' && service.details" class="mt-2">
        <div>models: {{ service.details.models_available }}</div>
        <div>vram: {{ service.details.gpu_vram_mb }}mb</div>
      </div>

      <div v-if="service.error" class="mt-2" style="color: #444;">
        {{ service.error.substring(0, 50) }}
      </div>
    </div>
  </div>
</template>
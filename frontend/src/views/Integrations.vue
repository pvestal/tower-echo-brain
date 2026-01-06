<template>
  <div>
    <h1 class="text-3xl font-bold mb-8">Integrations</h1>

    <div class="grid grid-cols-2 gap-6">
      <div
        v-for="integration in store.integrations"
        :key="integration.id"
        class="card"
      >
        <div class="flex items-center justify-between mb-4">
          <div class="flex items-center gap-3">
            <span
              class="w-3 h-3 rounded-full"
              :class="{
                'bg-green-500': integration.status === 'connected',
                'bg-red-500': integration.status === 'disconnected',
                'bg-yellow-500': integration.status === 'error'
              }"
            ></span>
            <h3 class="text-lg font-semibold">{{ integration.display_name }}</h3>
          </div>
        </div>

        <div v-if="integration.status === 'connected'" class="space-y-2 text-sm text-gray-400">
          <p v-if="integration.connected_at">
            Connected: {{ formatDate(integration.connected_at) }}
          </p>
          <p v-if="integration.last_sync_at">
            Last sync: {{ formatDate(integration.last_sync_at) }}
          </p>
          <p v-if="integration.scopes?.length">
            Scopes: {{ integration.scopes.join(', ') }}
          </p>
        </div>

        <div v-if="integration.status === 'error'" class="text-red-400 text-sm mt-2">
          {{ integration.error_message }}
        </div>

        <div class="flex gap-2 mt-4">
          <button
            v-if="integration.status === 'connected'"
            @click="syncIntegration(integration.provider)"
            class="btn btn-secondary text-sm"
          >
            Sync Now
          </button>
          <button
            v-if="integration.status === 'connected'"
            @click="disconnectIntegration(integration.provider)"
            class="btn btn-danger text-sm"
          >
            Disconnect
          </button>
          <button
            v-if="integration.status !== 'connected'"
            @click="connectIntegration(integration.provider)"
            class="btn btn-primary text-sm"
          >
            Connect
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useIntegrationsStore } from '@/stores/integrations'

const store = useIntegrationsStore()

function formatDate(date: string) {
  return new Date(date).toLocaleString()
}

async function connectIntegration(provider: string) {
  // Would redirect to OAuth
  alert(`OAuth flow for ${provider} not yet implemented`)
}

async function disconnectIntegration(provider: string) {
  if (confirm(`Disconnect ${provider}?`)) {
    await store.disconnect(provider)
  }
}

async function syncIntegration(provider: string) {
  await store.sync(provider)
  alert(`Sync triggered for ${provider}`)
}

onMounted(() => store.fetchIntegrations())
</script>

<style scoped>
.card {
  @apply bg-gray-800 rounded-xl p-6 shadow-lg;
}

.btn {
  @apply px-4 py-2 rounded-lg font-medium transition-colors;
}

.btn-primary {
  @apply bg-blue-600 hover:bg-blue-700 text-white;
}

.btn-secondary {
  @apply bg-gray-700 hover:bg-gray-600 text-white;
}

.btn-danger {
  @apply bg-red-600 hover:bg-red-700 text-white;
}
</style>
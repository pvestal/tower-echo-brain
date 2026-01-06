<template>
  <div>
    <h1 class="text-3xl font-bold mb-8">What Echo Brain Knows</h1>

    <!-- Stats Cards -->
    <div class="grid grid-cols-4 gap-6 mb-8">
      <div class="card">
        <div class="text-4xl font-bold text-blue-400">{{ knowledgeStore.total }}</div>
        <div class="text-gray-400 mt-1">Facts</div>
      </div>
      <div class="card">
        <div class="text-4xl font-bold text-blue-400">{{ preferencesStore.preferences.length }}</div>
        <div class="text-gray-400 mt-1">Preferences</div>
      </div>
      <div class="card">
        <div class="text-4xl font-bold text-blue-400">{{ integrationsStore.connected.length }}/{{ integrationsStore.integrations.length }}</div>
        <div class="text-gray-400 mt-1">Integrations</div>
      </div>
      <div class="card">
        <div class="text-4xl font-bold text-blue-400">{{ vaultStore.keys.filter(k => k.is_set).length }}</div>
        <div class="text-gray-400 mt-1">API Keys</div>
      </div>
    </div>

    <!-- Recent Knowledge -->
    <div class="card mb-8">
      <h2 class="text-xl font-semibold mb-4 flex items-center gap-2">
        <Database class="w-5 h-5 text-blue-400" />
        Recent Knowledge
      </h2>
      <div class="space-y-2">
        <div
          v-for="fact in recentFacts"
          :key="fact.id"
          class="flex items-center gap-2 text-gray-300 py-2 border-b border-gray-700 last:border-0"
        >
          <span class="text-blue-400">{{ fact.subject }}</span>
          <span class="text-gray-500">{{ fact.predicate }}</span>
          <span class="text-white">{{ fact.object }}</span>
        </div>
        <div v-if="recentFacts.length === 0" class="text-gray-500">
          No facts yet. Start adding knowledge!
        </div>
      </div>
      <RouterLink to="/knowledge" class="text-blue-400 hover:text-blue-300 text-sm mt-4 inline-block">
        View all knowledge →
      </RouterLink>
    </div>

    <!-- Integration Status -->
    <div class="card">
      <h2 class="text-xl font-semibold mb-4 flex items-center gap-2">
        <Link class="w-5 h-5 text-blue-400" />
        Integration Status
      </h2>
      <div class="space-y-3">
        <div
          v-for="integration in integrationsStore.integrations"
          :key="integration.id"
          class="flex items-center justify-between py-2 border-b border-gray-700 last:border-0"
        >
          <div class="flex items-center gap-3">
            <span
              class="w-3 h-3 rounded-full"
              :class="{
                'bg-green-500': integration.status === 'connected',
                'bg-red-500': integration.status === 'disconnected',
                'bg-yellow-500': integration.status === 'error'
              }"
            ></span>
            <span>{{ integration.display_name }}</span>
          </div>
          <span class="text-sm text-gray-500">
            {{ integration.last_sync_at ? `Last sync: ${formatDate(integration.last_sync_at)}` : 'Not connected' }}
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { RouterLink } from 'vue-router'
import { Database, Link } from 'lucide-vue-next'
import { useKnowledgeStore } from '@/stores/knowledge'
import { usePreferencesStore } from '@/stores/preferences'
import { useVaultStore } from '@/stores/vault'
import { useIntegrationsStore } from '@/stores/integrations'

const knowledgeStore = useKnowledgeStore()
const preferencesStore = usePreferencesStore()
const vaultStore = useVaultStore()
const integrationsStore = useIntegrationsStore()

const recentFacts = computed(() => knowledgeStore.facts.slice(0, 10))

function formatDate(date: string) {
  return new Date(date).toLocaleDateString()
}

onMounted(async () => {
  await Promise.all([
    knowledgeStore.fetchFacts(),
    knowledgeStore.fetchSubjects(),
    preferencesStore.fetchPreferences(),
    vaultStore.fetchKeys(),
    integrationsStore.fetchIntegrations()
  ])
})
</script>

<style scoped>
.card {
  @apply bg-gray-800 rounded-xl p-6 shadow-lg;
}
</style>
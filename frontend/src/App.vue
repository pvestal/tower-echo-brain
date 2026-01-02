<template>
  <div class="min-h-screen bg-tower-bg">
    <!-- Simple navbar replacement -->
    <nav style="background: #1a1a1a; padding: 1rem; border-bottom: 1px solid #333;">
      <div style="max-width: 1280px; margin: 0 auto; display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 1.25rem; font-weight: bold; color: #f0f6fc;">Echo Brain</span>
        <div style="display: flex; gap: 1rem;">
          <a
            v-for="tab in tabs"
            :key="tab.id"
            @click="currentTab = tab.id"
            style="padding: 0.5rem 1rem; cursor: pointer; border-radius: 0.375rem; color: #8b949e;"
            :style="currentTab === tab.id ? 'background: #21262d; color: #f0f6fc;' : ''"
          >
            {{ tab.label }}
          </a>
        </div>
        <span :style="wsConnected ? 'color: #3fb950;' : 'color: #da3633;'">
          {{ wsConnected ? '● Connected' : '● Disconnected' }}
        </span>
      </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 py-8">
      <ChatInterface v-if="currentTab === 'chat'" />
      <VoiceInterface v-if="currentTab === 'voice'" />
      <EchoMetrics v-if="currentTab === 'metrics'" />
      <PlaidAuth v-if="currentTab === 'plaid'" />
      <ComponentShowcase v-if="currentTab === 'components'" />
    </main>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import ChatInterface from './views/ChatInterface.vue'
import VoiceInterface from './views/VoiceInterface.vue'
import EchoMetrics from './views/EchoMetrics.vue'
import PlaidAuth from './views/PlaidAuth.vue'
import ComponentShowcase from './views/ComponentShowcase.vue'

const currentTab = ref('chat')
const wsConnected = ref(false)

const tabs = [
  { id: 'chat', label: 'Chat' },
  { id: 'voice', label: 'Voice' },
  { id: 'metrics', label: 'Metrics' },
  { id: 'plaid', label: 'Financial' },
  { id: 'components', label: 'Components' }
]
</script>

<style scoped>
.nav-link {
  color: var(--tower-text-secondary);
  cursor: pointer;
  padding: 0.5rem 1rem;
  border-radius: 0.375rem;
  transition: all 0.2s;
}

.nav-link:hover {
  color: var(--tower-text-primary);
  background-color: var(--tower-bg-hover);
}

.nav-link.active {
  color: var(--tower-accent-primary);
  background-color: rgba(59, 130, 246, 0.1);
}
</style>

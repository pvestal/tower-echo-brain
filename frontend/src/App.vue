<template>
  <div class="min-h-screen bg-tower-bg">
    <TowerNavbar>
      <template #brand>
        <span class="text-xl font-bold">Echo Brain</span>
      </template>
      <template #links>
        <a 
          v-for="tab in tabs" 
          :key="tab.id"
          @click="currentTab = tab.id"
          class="nav-link"
          :class="{ 'active': currentTab === tab.id }"
        >
          {{ tab.label }}
        </a>
      </template>
      <template #actions>
        <TowerWebSocketStatus :connected="wsConnected" />
      </template>
    </TowerNavbar>

    <main class="max-w-7xl mx-auto px-4 py-8">
      <ChatInterface v-if="currentTab === 'chat'" />
      <VoiceInterface v-if="currentTab === 'voice'" />
      <Dashboard v-if="currentTab === 'dashboard'" />
      <PlaidAuth v-if="currentTab === 'plaid'" />
    </main>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { TowerNavbar, TowerWebSocketStatus } from '@tower/ui-components'
import ChatInterface from './views/ChatInterface.vue'
import VoiceInterface from './views/VoiceInterface.vue'
import Dashboard from './views/Dashboard.vue'
import PlaidAuth from './views/PlaidAuth.vue'

const currentTab = ref('chat')
const wsConnected = ref(false)

const tabs = [
  { id: 'chat', label: 'Chat' },
  { id: 'voice', label: 'Voice' },
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'plaid', label: 'Financial' }
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

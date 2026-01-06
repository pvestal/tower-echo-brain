<template>
  <div class="min-h-screen bg-gray-900">
    <!-- Sidebar -->
    <aside class="fixed left-0 top-0 h-full w-64 bg-gray-800 border-r border-gray-700">
      <div class="p-6">
        <h1 class="text-2xl font-bold text-blue-400 flex items-center gap-2">
          <Brain class="w-8 h-8" />
          Echo Brain
        </h1>
        <p class="text-sm text-gray-400 mt-1">Personal Dashboard</p>
      </div>

      <nav class="mt-6">
        <RouterLink
          v-for="item in navItems"
          :key="item.path"
          :to="item.path"
          class="flex items-center gap-3 px-6 py-3 text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
          :class="{ 'bg-gray-700 text-white border-r-2 border-blue-500': $route.path === item.path }"
        >
          <component :is="item.icon" class="w-5 h-5" />
          {{ item.label }}
        </RouterLink>
      </nav>

      <div class="absolute bottom-0 left-0 right-0 p-6 border-t border-gray-700">
        <div class="flex items-center gap-2">
          <span
            class="w-3 h-3 rounded-full"
            :class="systemHealthy ? 'bg-green-500' : 'bg-red-500'"
          ></span>
          <span class="text-sm text-gray-400">
            {{ systemHealthy ? 'System Online' : 'System Offline' }}
          </span>
        </div>
      </div>
    </aside>

    <!-- Main Content -->
    <main class="ml-64 p-8">
      <RouterView />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { RouterLink, RouterView, useRoute } from 'vue-router'
import { Brain, LayoutDashboard, Database, Settings, Key, Link, Activity, MessageSquare, FileText } from 'lucide-vue-next'
import axios from 'axios'

const $route = useRoute()
const systemHealthy = ref(false)

const navItems = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/knowledge', label: 'Knowledge', icon: Database },
  { path: '/preferences', label: 'Preferences', icon: Settings },
  { path: '/vault', label: 'Vault', icon: Key },
  { path: '/integrations', label: 'Integrations', icon: Link },
  // Legacy items
  { path: '/health', label: 'Health Check', icon: Activity },
  { path: '/chat-test', label: 'Chat Test', icon: MessageSquare },
  { path: '/logs', label: 'System Logs', icon: FileText },
]

async function checkHealth() {
  try {
    await axios.get('http://localhost:8309/health', { timeout: 2000 })
    systemHealthy.value = true
  } catch {
    systemHealthy.value = false
  }
}

onMounted(() => {
  checkHealth()
  setInterval(checkHealth, 10000)
})
</script>

<style>
@import './style.css';
</style>
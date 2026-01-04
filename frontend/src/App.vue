<template>
  <div class="app">
    <nav class="app-nav">
      <div class="nav-brand">Echo Brain Monitor</div>
      <div class="nav-links">
        <router-link to="/" class="nav-link" :class="{ active: $route.path === '/' }">
          Health Check
        </router-link>
        <router-link to="/chat-test" class="nav-link" :class="{ active: $route.path === '/chat-test' }">
          Chat Test
        </router-link>
        <router-link to="/logs" class="nav-link" :class="{ active: $route.path === '/logs' }">
          System Logs
        </router-link>
      </div>
      <div class="nav-status">
        <span class="status-indicator" :class="systemStatus"></span>
        <span class="status-text">{{ statusText }}</span>
      </div>
    </nav>
    <router-view />
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import axios from 'axios'

const $route = useRoute()
const systemStatus = ref('unknown')
const statusText = ref('Checking...')

async function checkSystemHealth() {
  try {
    await axios.get('https://vestal-garcia.duckdns.org/api/echo/health', { timeout: 2000 })
    systemStatus.value = 'healthy'
    statusText.value = 'System Online'
  } catch (error) {
    systemStatus.value = 'error'
    statusText.value = 'System Offline'
  }
}

onMounted(() => {
  checkSystemHealth()
  setInterval(checkSystemHealth, 10000)
})
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: 'Roboto Mono', 'SF Mono', Monaco, Consolas, monospace;
  background: #0a0a0f;
  color: #e2e8f0;
}

.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-nav {
  background: #151520;
  border-bottom: 1px solid #2d3748;
  padding: 0 20px;
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.nav-brand {
  font-size: 18px;
  font-weight: 600;
  color: #2a7de1;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.nav-links {
  display: flex;
  gap: 5px;
}

.nav-link {
  padding: 8px 16px;
  color: #94a3b8;
  text-decoration: none;
  border-radius: 4px;
  transition: all 150ms;
  font-size: 14px;
}

.nav-link:hover {
  color: #e2e8f0;
  background: rgba(42, 125, 225, 0.1);
}

.nav-link.active {
  color: #2a7de1;
  background: rgba(42, 125, 225, 0.15);
}

.nav-status {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #475569;
}

.status-indicator.healthy {
  background: #19b37b;
  animation: pulse 2s infinite;
}

.status-indicator.error {
  background: #ef4444;
}

.status-indicator.unknown {
  background: #f59e0b;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.status-text {
  font-size: 12px;
  color: #94a3b8;
  text-transform: uppercase;
}
</style>
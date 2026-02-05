<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router';
</script>

<template>
  <div class="container">
    <!-- Header with Navigation -->
    <div class="header">
      <h1 class="text-mono">ECHO BRAIN</h1>
      <nav class="nav">
        <RouterLink to="/dashboard" class="nav-link">Dashboard</RouterLink>
        <RouterLink to="/ask" class="nav-link">Ask</RouterLink>
        <RouterLink to="/memory" class="nav-link">Memory</RouterLink>
        <RouterLink to="/endpoints" class="nav-link">Endpoints</RouterLink>
        <RouterLink to="/system" class="nav-link">System</RouterLink>
        <RouterLink to="/logs" class="nav-link">Logs</RouterLink>
      </nav>
    </div>

    <!-- Main Content -->
    <RouterView />
  </div>
</template>

<style scoped>
.header {
  border-bottom: 1px solid #21262d;
  padding-bottom: 1rem;
  margin-bottom: 1.5rem;
}

.header h1 {
  font-size: 1.25rem;
  font-weight: 600;
  color: #f0f6fc;
  margin-bottom: 0.5rem;
}

.nav {
  display: flex;
  gap: 1rem;
  margin-top: 1rem;
}

.nav-link {
  padding: 0.5rem 1rem;
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 0.375rem;
  color: #8b949e;
  text-decoration: none;
  font-size: 0.875rem;
  transition: all 0.2s ease;
}

.nav-link:hover {
  background: #21262d;
  color: #f0f6fc;
  border-color: #30363d;
}

.nav-link.router-link-active {
  background: #238636;
  color: #f0f6fc;
  border-color: #238636;
}
</style>
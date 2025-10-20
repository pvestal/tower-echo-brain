<template>
  <div class="space-y-6">
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
      <TowerCard hoverable>
        <h3 class="text-lg font-semibold mb-2 text-tower-text-primary">
          Total Queries
        </h3>
        <p class="text-3xl font-bold text-tower-accent-primary">1,234</p>
      </TowerCard>

      <TowerCard hoverable>
        <h3 class="text-lg font-semibold mb-2 text-tower-text-primary">
          Active Models
        </h3>
        <p class="text-3xl font-bold text-tower-accent-success">5</p>
      </TowerCard>

      <TowerCard hoverable>
        <h3 class="text-lg font-semibold mb-2 text-tower-text-primary">
          Avg Response Time
        </h3>
        <p class="text-3xl font-bold text-tower-accent-warning">2.3s</p>
      </TowerCard>
    </div>

    <TowerCard>
      <template #header>
        <h2 class="text-xl font-bold text-tower-text-primary">System Status</h2>
      </template>
      <div class="space-y-3">
        <div class="status-item">
          <span class="text-tower-text-secondary">Echo Brain Service</span>
          <span class="status-badge success">Running</span>
        </div>
        <div class="status-item">
          <span class="text-tower-text-secondary">Knowledge Base</span>
          <span class="status-badge success">Connected</span>
        </div>
        <div class="status-item">
          <span class="text-tower-text-secondary">Ollama</span>
          <span class="status-badge success">Available</span>
        </div>
      </div>
    </TowerCard>
  </div>
</template>

<script setup>
import { TowerCard } from '@tower/ui-components'
</script>

<style scoped>
.status-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 0;
  border-bottom: 1px solid var(--tower-border);
}

.status-item:last-child {
  border-bottom: none;
}

.status-badge {
  padding: 0.25rem 0.75rem;
  border-radius: 0.25rem;
  font-size: 0.875rem;
  font-weight: 500;
}

.status-badge.success {
  background-color: rgba(16, 185, 129, 0.1);
  color: var(--tower-accent-success);
}
</style>

<template>
  <div class="space-y-6">
    <TowerCard>
      <template #header>
        <h2 class="text-2xl font-bold text-tower-text-primary">Financial Integration</h2>
      </template>
      
      <p class="text-tower-text-secondary mb-4">
        Connect your financial accounts to enable intelligent financial insights.
      </p>

      <TowerButton @click="connectPlaid">
        Connect Bank Account
      </TowerButton>
    </TowerCard>

    <TowerCard v-if="connected">
      <template #header>
        <h3 class="text-lg font-semibold text-tower-text-primary">Connected Accounts</h3>
      </template>
      <p class="text-tower-text-secondary">No accounts connected yet.</p>
    </TowerCard>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const connected = ref(false)

const connectPlaid = () => {
  // Plaid Link integration will be implemented here
  alert('Plaid integration coming soon!')
}
</script>

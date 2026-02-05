<script setup lang="ts">
import { onMounted, onUnmounted } from 'vue';
import { useHealthStore } from '@/stores/health';
import ServiceCard from '@/components/dashboard/ServiceCard.vue';
import ResourceBar from '@/components/dashboard/ResourceBar.vue';

const healthStore = useHealthStore();

onMounted(() => healthStore.startAutoRefresh(5000));
onUnmounted(() => healthStore.stopAutoRefresh());
</script>

<template>
  <div>
    <!-- Status Bar -->
    <div class="text-xs text-dim text-mono mb-2">
      <span v-if="healthStore.overallStatus" style="margin-right: 1rem">
        status: {{ healthStore.overallStatus }}
      </span>
      <span v-if="healthStore.lastFetch">
        last update: {{ healthStore.lastFetch.toLocaleTimeString() }}
      </span>
    </div>

    <!-- Error State -->
    <div v-if="healthStore.error" class="text-xs text-dim text-mono mb-2">
      error: {{ healthStore.error }}
    </div>

    <!-- Resources -->
    <div v-if="healthStore.resources" class="card mb-2">
      <h3>RESOURCES</h3>
      <div class="grid grid-2">
        <ResourceBar
          label="cpu"
          :value="healthStore.resources?.cpu_percent || 0"
          unit="%"
        />
        <ResourceBar
          label="memory"
          :value="(healthStore.resources?.memory_mb || 0) / 1024"
          :max="8"
          unit="gb"
        />
        <ResourceBar
          label="vectors"
          :value="healthStore.resources?.vectors || 0"
          :max="100000"
          unit=""
        />
      </div>

    </div>

    <!-- Services -->
    <div class="mb-2">
      <h3 class="text-mono text-dim text-sm mb-2">SERVICES</h3>
      <div class="grid grid-3">
        <ServiceCard
          v-for="service in healthStore.services"
          :key="service.name"
          :service="service"
        />
      </div>
    </div>

    <!-- Unhealthy Services Alert -->
    <div v-if="healthStore.unhealthyServices.length > 0" class="card">
      <h3>ISSUES</h3>
      <div class="text-xs text-mono text-dim space-y-2">
        <div v-for="svc in healthStore.unhealthyServices" :key="svc.name">
          {{ svc.name }}: {{ svc.status }}
        </div>
      </div>
    </div>

    <!-- Footer Stats -->
    <div class="text-xs text-dim text-mono mt-2" style="padding-top: 1rem; border-top: 1px solid #222;">
      <div v-if="healthStore.health">
        status: {{ healthStore.health.status }}
        | services: {{ healthStore.services.length }}
      </div>
    </div>
  </div>
</template>
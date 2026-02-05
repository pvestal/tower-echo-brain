<script setup lang="ts">
import { onMounted, onUnmounted } from 'vue';
import { useHealthStore } from '@/stores/health';
import ServiceCard from '@/components/dashboard/ServiceCard.vue';
import ResourceBar from '@/components/dashboard/ResourceBar.vue';

const healthStore = useHealthStore();

onMounted(() => healthStore.startAutoRefresh(5000));
onUnmounted(() => healthStore.stopAutoRefresh());

const formatUptime = (seconds: number) => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${minutes}m`;
};
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
          :value="healthStore.resources.cpu_percent || 0"
          unit="%"
        />
        <ResourceBar
          label="memory"
          :value="healthStore.resources.memory_used_gb || 0"
          :max="healthStore.resources.memory_total_gb || 0"
          unit="gb"
        />
        <ResourceBar
          label="disk"
          :value="healthStore.resources.disk_used_gb || 0"
          :max="healthStore.resources.disk_total_gb || 0"
          unit="gb"
        />
        <ResourceBar
          v-if="healthStore.resources.gpu"
          label="gpu"
          :value="healthStore.resources.gpu.memory_used_mb || 0"
          :max="healthStore.resources.gpu.memory_total_mb || 0"
          unit="mb"
        />
      </div>

      <div v-if="healthStore.resources.gpu" class="mt-2 text-xs text-dim text-mono">
        gpu: {{ healthStore.resources.gpu.name }} |
        temp: {{ healthStore.resources.gpu.temperature_c }}°c
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
          {{ svc.name }}: {{ svc.error || svc.status }}
        </div>
      </div>
    </div>

    <!-- Footer Stats -->
    <div class="text-xs text-dim text-mono mt-2" style="padding-top: 1rem; border-top: 1px solid #222;">
      <div v-if="healthStore.health && healthStore.health.uptime_seconds">
        uptime: {{ formatUptime(healthStore.health.uptime_seconds) }}
        <span v-if="healthStore.health.endpoints"> | endpoints: {{ healthStore.health.endpoints.total }}</span>
        | services: {{ healthStore.services.length }}
      </div>
    </div>
  </div>
</template>
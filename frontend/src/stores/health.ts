import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import { healthApi } from '@/api/echoApi';
import type { SystemHealth } from '@/types/echo';

export const useHealthStore = defineStore('health', () => {
  const health = ref<SystemHealth | null>(null);
  const loading = ref(false);
  const error = ref<string | null>(null);
  const lastFetch = ref<Date | null>(null);

  let refreshInterval: number | null = null;

  const overallStatus = computed(() => health.value?.overall_status ?? 'unknown');
  const services = computed(() => health.value?.services ?? []);
  const resources = computed(() => health.value?.resources);
  const isHealthy = computed(() => overallStatus.value === 'healthy');

  const unhealthyServices = computed(() =>
    services.value.filter(s => s.status !== 'healthy')
  );

  async function fetchHealth() {
    loading.value = true;
    error.value = null;
    try {
      const { data } = await healthApi.getFull();
      health.value = data;
      lastFetch.value = new Date();
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to fetch health';
    } finally {
      loading.value = false;
    }
  }

  function startAutoRefresh(intervalMs = 5000) {
    stopAutoRefresh();
    fetchHealth();
    refreshInterval = window.setInterval(fetchHealth, intervalMs);
  }

  function stopAutoRefresh() {
    if (refreshInterval) {
      clearInterval(refreshInterval);
      refreshInterval = null;
    }
  }

  return {
    health,
    loading,
    error,
    lastFetch,
    overallStatus,
    services,
    resources,
    isHealthy,
    unhealthyServices,
    fetchHealth,
    startAutoRefresh,
    stopAutoRefresh,
  };
});
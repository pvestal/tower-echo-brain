import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import { healthApi } from '@/api/echoApi';
import type { EchoHealthResponse } from '@/types/echo';

export const useHealthStore = defineStore('health', () => {
  const health = ref<EchoHealthResponse | null>(null);
  const loading = ref(false);
  const error = ref<string | null>(null);
  const lastFetch = ref<Date | null>(null);

  let refreshInterval: number | null = null;

  const overallStatus = computed(() => health.value?.status ?? 'unknown');
  const services = computed(() => {
    // Convert services object to array format
    if (!health.value?.services) return [];
    if (Array.isArray(health.value.services)) return health.value.services;
    // Convert object format to array with minimal ServiceHealth interface
    return Object.entries(health.value.services).map(([name, healthy]) => ({
      name,
      status: (healthy ? 'healthy' : 'down') as 'healthy' | 'degraded' | 'down',
      latency_ms: 0,
      last_check: new Date().toISOString(),
      details: {}
    }));
  });
  const resources = computed(() => health.value?.metrics);
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
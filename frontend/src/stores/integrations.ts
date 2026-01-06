import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { integrationsApi } from '@/api/integrations'
import type { Integration } from '@/types'

export const useIntegrationsStore = defineStore('integrations', () => {
  const integrations = ref<Integration[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  const connected = computed(() =>
    integrations.value.filter(i => i.status === 'connected')
  )

  async function fetchIntegrations() {
    loading.value = true
    try {
      integrations.value = await integrationsApi.list()
    } catch (e: any) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  async function disconnect(provider: string) {
    try {
      await integrationsApi.disconnect(provider)
      await fetchIntegrations()
    } catch (e: any) {
      error.value = e.message
      throw e
    }
  }

  async function sync(provider: string) {
    try {
      await integrationsApi.sync(provider)
    } catch (e: any) {
      error.value = e.message
      throw e
    }
  }

  return { integrations, connected, loading, error, fetchIntegrations, disconnect, sync }
})
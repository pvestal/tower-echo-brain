import { defineStore } from 'pinia'
import { ref } from 'vue'
import { vaultApi } from '@/api/vault'
import type { VaultKey } from '@/types'

export const useVaultStore = defineStore('vault', () => {
  const keys = ref<VaultKey[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function fetchKeys() {
    loading.value = true
    try {
      keys.value = await vaultApi.listKeys()
    } catch (e: any) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  async function createKey(key: { key_name: string; value: string; service: string }) {
    try {
      await vaultApi.createKey(key)
      await fetchKeys()
    } catch (e: any) {
      error.value = e.message
      throw e
    }
  }

  async function deleteKey(keyName: string) {
    try {
      await vaultApi.deleteKey(keyName)
      keys.value = keys.value.filter(k => k.key_name !== keyName)
    } catch (e: any) {
      error.value = e.message
      throw e
    }
  }

  async function testKey(keyName: string) {
    try {
      return await vaultApi.testKey(keyName)
    } catch (e: any) {
      error.value = e.message
      throw e
    }
  }

  return { keys, loading, error, fetchKeys, createKey, deleteKey, testKey }
})
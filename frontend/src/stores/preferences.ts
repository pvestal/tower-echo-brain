import { defineStore } from 'pinia'
import { ref } from 'vue'
import { preferencesApi } from '@/api/preferences'
import type { Preference } from '@/types'

export const usePreferencesStore = defineStore('preferences', () => {
  const preferences = ref<Preference[]>([])
  const categories = ref<string[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function fetchPreferences(category?: string) {
    loading.value = true
    try {
      const data = await preferencesApi.list(category)
      preferences.value = data.preferences
      categories.value = data.categories
    } catch (e: any) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  async function createPreference(pref: { category: string; key: string; value: any }) {
    try {
      await preferencesApi.create(pref)
      await fetchPreferences()
    } catch (e: any) {
      error.value = e.message
      throw e
    }
  }

  async function updatePreference(id: string, value: any) {
    try {
      await preferencesApi.update(id, { value })
      await fetchPreferences()
    } catch (e: any) {
      error.value = e.message
      throw e
    }
  }

  async function deletePreference(id: string) {
    try {
      await preferencesApi.delete(id)
      preferences.value = preferences.value.filter(p => p.id !== id)
    } catch (e: any) {
      error.value = e.message
      throw e
    }
  }

  return {
    preferences,
    categories,
    loading,
    error,
    fetchPreferences,
    createPreference,
    updatePreference,
    deletePreference
  }
})
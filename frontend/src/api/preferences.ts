import { api } from './client'
import type { Preference } from '@/types'

export const preferencesApi = {
  async list(category?: string) {
    const params = category ? { category } : {}
    const { data } = await api.get('/api/preferences', { params })
    return data as { preferences: Preference[]; categories: string[] }
  },

  async getCategory(category: string) {
    const { data } = await api.get(`/api/preferences/${category}`)
    return data as { category: string; items: Preference[] }
  },

  async create(pref: { category: string; key: string; value: any; metadata?: Record<string, any> }) {
    const { data } = await api.post('/api/preferences', pref)
    return data
  },

  async update(id: string, updates: { value: any; metadata?: Record<string, any> }) {
    const { data } = await api.put(`/api/preferences/${id}`, updates)
    return data
  },

  async delete(id: string) {
    await api.delete(`/api/preferences/${id}`)
  }
}
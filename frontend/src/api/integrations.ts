import { api } from './client'
import type { Integration } from '@/types'

export const integrationsApi = {
  async list() {
    const { data } = await api.get('/api/integrations')
    return data.integrations as Integration[]
  },

  async get(provider: string) {
    const { data } = await api.get(`/api/integrations/${provider}`)
    return data.integration as Integration
  },

  async connect(provider: string) {
    const { data } = await api.post(`/api/integrations/${provider}/connect`)
    return data
  },

  async disconnect(provider: string) {
    await api.delete(`/api/integrations/${provider}`)
  },

  async sync(provider: string) {
    const { data } = await api.post(`/api/integrations/${provider}/sync`)
    return data
  }
}
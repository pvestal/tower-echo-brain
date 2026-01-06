import { api } from './client'
import type { VaultKey } from '@/types'

export const vaultApi = {
  async listKeys() {
    const { data } = await api.get('/api/vault/keys')
    return data.keys as VaultKey[]
  },

  async createKey(key: { key_name: string; value: string; service: string; key_type?: string; description?: string }) {
    const { data } = await api.post('/api/vault/keys', key)
    return data
  },

  async deleteKey(keyName: string) {
    await api.delete(`/api/vault/keys/${keyName}`)
  },

  async testKey(keyName: string) {
    const { data } = await api.post(`/api/vault/keys/${keyName}/test`)
    return data as { valid: boolean; message: string }
  }
}
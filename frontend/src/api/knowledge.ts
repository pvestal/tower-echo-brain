import { api } from './client'
import type { Fact, Subject } from '@/types'

export const knowledgeApi = {
  async listFacts(params?: { subject?: string; search?: string; limit?: number; offset?: number }) {
    const { data } = await api.get('/api/knowledge/facts', { params })
    return data as { facts: Fact[]; total: number }
  },

  async getFact(id: string) {
    const { data } = await api.get(`/api/knowledge/facts/${id}`)
    return data.fact as Fact
  },

  async createFact(fact: { subject: string; predicate: string; object: string; confidence?: number }) {
    const { data } = await api.post('/api/knowledge/facts', fact)
    return data
  },

  async updateFact(id: string, updates: Partial<Fact>) {
    const { data } = await api.put(`/api/knowledge/facts/${id}`, updates)
    return data
  },

  async deleteFact(id: string) {
    await api.delete(`/api/knowledge/facts/${id}`)
  },

  async listSubjects() {
    const { data } = await api.get('/api/knowledge/subjects')
    return data.subjects as Subject[]
  },

  async getFactsAbout(subject: string) {
    const { data } = await api.get(`/api/knowledge/about/${encodeURIComponent(subject)}`)
    return data as { subject: string; facts: Fact[]; count: number }
  }
}
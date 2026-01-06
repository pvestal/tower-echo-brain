import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { knowledgeApi } from '@/api/knowledge'
import type { Fact, Subject } from '@/types'

export const useKnowledgeStore = defineStore('knowledge', () => {
  const facts = ref<Fact[]>([])
  const subjects = ref<Subject[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  const total = ref(0)

  const factsBySubject = computed(() => {
    const grouped: Record<string, Fact[]> = {}
    for (const fact of facts.value) {
      if (!grouped[fact.subject]) {
        grouped[fact.subject] = []
      }
      grouped[fact.subject].push(fact)
    }
    return grouped
  })

  async function fetchFacts(params?: { subject?: string; search?: string }) {
    loading.value = true
    error.value = null
    try {
      const data = await knowledgeApi.listFacts({ ...params, limit: 500 })
      facts.value = data.facts
      total.value = data.total
    } catch (e: any) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  async function fetchSubjects() {
    try {
      subjects.value = await knowledgeApi.listSubjects()
    } catch (e: any) {
      error.value = e.message
    }
  }

  async function createFact(fact: { subject: string; predicate: string; object: string }) {
    try {
      await knowledgeApi.createFact(fact)
      await fetchFacts()
      await fetchSubjects()
    } catch (e: any) {
      error.value = e.message
      throw e
    }
  }

  async function updateFact(id: string, updates: Partial<Fact>) {
    try {
      await knowledgeApi.updateFact(id, updates)
      await fetchFacts()
    } catch (e: any) {
      error.value = e.message
      throw e
    }
  }

  async function deleteFact(id: string) {
    try {
      await knowledgeApi.deleteFact(id)
      facts.value = facts.value.filter(f => f.id !== id)
      await fetchSubjects()
    } catch (e: any) {
      error.value = e.message
      throw e
    }
  }

  return {
    facts,
    subjects,
    loading,
    error,
    total,
    factsBySubject,
    fetchFacts,
    fetchSubjects,
    createFact,
    updateFact,
    deleteFact
  }
})
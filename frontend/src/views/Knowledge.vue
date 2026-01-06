<template>
  <div>
    <div class="flex items-center justify-between mb-8">
      <h1 class="text-3xl font-bold">Knowledge Browser</h1>
      <button @click="showAddModal = true" class="btn btn-primary flex items-center gap-2">
        <Plus class="w-4 h-4" />
        Add Fact
      </button>
    </div>

    <!-- Search & Filter -->
    <div class="flex gap-4 mb-6">
      <div class="flex-1">
        <input
          v-model="searchQuery"
          @input="debouncedSearch"
          type="text"
          placeholder="Search facts..."
          class="input w-full"
        />
      </div>
      <select v-model="selectedSubject" @change="filterBySubject" class="input">
        <option value="">All Subjects</option>
        <option v-for="s in store.subjects" :key="s.name" :value="s.name">
          {{ s.name }} ({{ s.count }})
        </option>
      </select>
    </div>

    <!-- Facts by Subject -->
    <div v-if="store.loading" class="text-center py-8 text-gray-400">
      Loading...
    </div>

    <div v-else class="space-y-6">
      <div
        v-for="(facts, subject) in store.factsBySubject"
        :key="subject"
        class="card"
      >
        <h2 class="text-xl font-semibold text-blue-400 mb-4 flex items-center gap-2">
          <User class="w-5 h-5" />
          {{ subject }}
          <span class="text-sm text-gray-500">({{ facts.length }} facts)</span>
        </h2>

        <div class="space-y-2">
          <div
            v-for="fact in facts"
            :key="fact.id"
            class="flex items-center justify-between py-2 border-b border-gray-700 last:border-0 group"
          >
            <div>
              <span class="text-gray-400">{{ fact.predicate }}</span>
              <span class="text-white ml-2">{{ fact.object }}</span>
            </div>
            <div class="opacity-0 group-hover:opacity-100 transition-opacity flex gap-2">
              <button @click="editFact(fact)" class="text-gray-400 hover:text-blue-400">
                <Pencil class="w-4 h-4" />
              </button>
              <button @click="confirmDelete(fact)" class="text-gray-400 hover:text-red-400">
                <Trash2 class="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>

      <div v-if="Object.keys(store.factsBySubject).length === 0" class="text-center py-12 text-gray-500">
        No facts found. Add some knowledge!
      </div>
    </div>

    <!-- Add/Edit Modal -->
    <div v-if="showAddModal || editingFact" class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div class="card w-full max-w-lg">
        <h3 class="text-xl font-semibold mb-4">
          {{ editingFact ? 'Edit Fact' : 'Add New Fact' }}
        </h3>

        <div class="space-y-4">
          <div>
            <label class="block text-sm text-gray-400 mb-1">Subject</label>
            <input v-model="factForm.subject" type="text" class="input w-full" placeholder="Patrick, Tower server, etc." />
          </div>
          <div>
            <label class="block text-sm text-gray-400 mb-1">Predicate (relationship)</label>
            <input v-model="factForm.predicate" type="text" class="input w-full" placeholder="drives, owns, has, uses, etc." />
          </div>
          <div>
            <label class="block text-sm text-gray-400 mb-1">Object (value)</label>
            <input v-model="factForm.object" type="text" class="input w-full" placeholder="2022 Toyota Tundra 1794 Edition" />
          </div>
        </div>

        <div class="flex justify-end gap-3 mt-6">
          <button @click="closeModal" class="btn btn-secondary">Cancel</button>
          <button @click="saveFact" class="btn btn-primary">
            {{ editingFact ? 'Update' : 'Add' }} Fact
          </button>
        </div>
      </div>
    </div>

    <!-- Delete Confirmation -->
    <div v-if="deletingFact" class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div class="card w-full max-w-md">
        <h3 class="text-xl font-semibold mb-4">Delete Fact?</h3>
        <p class="text-gray-400 mb-4">
          Are you sure you want to delete this fact?
        </p>
        <div class="bg-gray-700 rounded-lg p-3 mb-6">
          <span class="text-blue-400">{{ deletingFact.subject }}</span>
          <span class="text-gray-400 mx-2">{{ deletingFact.predicate }}</span>
          <span class="text-white">{{ deletingFact.object }}</span>
        </div>
        <div class="flex justify-end gap-3">
          <button @click="deletingFact = null" class="btn btn-secondary">Cancel</button>
          <button @click="doDelete" class="btn btn-danger">Delete</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Plus, Pencil, Trash2, User } from 'lucide-vue-next'
import { useKnowledgeStore } from '@/stores/knowledge'
import { useDebounceFn } from '@vueuse/core'
import type { Fact } from '@/types'

const store = useKnowledgeStore()

const searchQuery = ref('')
const selectedSubject = ref('')
const showAddModal = ref(false)
const editingFact = ref<Fact | null>(null)
const deletingFact = ref<Fact | null>(null)

const factForm = ref({
  subject: '',
  predicate: '',
  object: ''
})

const debouncedSearch = useDebounceFn(() => {
  store.fetchFacts({ search: searchQuery.value || undefined })
}, 300)

function filterBySubject() {
  store.fetchFacts({ subject: selectedSubject.value || undefined })
}

function editFact(fact: Fact) {
  editingFact.value = fact
  factForm.value = {
    subject: fact.subject,
    predicate: fact.predicate,
    object: fact.object
  }
}

function confirmDelete(fact: Fact) {
  deletingFact.value = fact
}

async function doDelete() {
  if (deletingFact.value) {
    await store.deleteFact(deletingFact.value.id)
    deletingFact.value = null
  }
}

function closeModal() {
  showAddModal.value = false
  editingFact.value = null
  factForm.value = { subject: '', predicate: '', object: '' }
}

async function saveFact() {
  if (editingFact.value) {
    await store.updateFact(editingFact.value.id, factForm.value)
  } else {
    await store.createFact(factForm.value)
  }
  closeModal()
}

onMounted(() => {
  store.fetchFacts()
  store.fetchSubjects()
})
</script>

<style scoped>
.card {
  @apply bg-gray-800 rounded-xl p-6 shadow-lg;
}

.btn {
  @apply px-4 py-2 rounded-lg font-medium transition-colors;
}

.btn-primary {
  @apply bg-blue-600 hover:bg-blue-700 text-white;
}

.btn-secondary {
  @apply bg-gray-700 hover:bg-gray-600 text-white;
}

.btn-danger {
  @apply bg-red-600 hover:bg-red-700 text-white;
}

.input {
  @apply bg-gray-700 border border-gray-600 rounded-lg px-4 py-2
         focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent;
}
</style>
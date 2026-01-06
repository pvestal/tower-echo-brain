<template>
  <div>
    <div class="flex items-center justify-between mb-8">
      <h1 class="text-3xl font-bold">Preferences</h1>
      <button @click="showAddModal = true" class="btn btn-primary flex items-center gap-2">
        <Plus class="w-4 h-4" />
        Add Preference
      </button>
    </div>

    <!-- Category Tabs -->
    <div class="flex gap-2 mb-6 border-b border-gray-700 pb-4">
      <button
        v-for="cat in ['all', ...store.categories]"
        :key="cat"
        @click="selectedCategory = cat === 'all' ? '' : cat"
        class="px-4 py-2 rounded-lg transition-colors"
        :class="(cat === 'all' && !selectedCategory) || selectedCategory === cat
          ? 'bg-blue-600 text-white'
          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'"
      >
        {{ cat === 'all' ? 'All' : cat }}
      </button>
    </div>

    <!-- Preferences List -->
    <div class="space-y-4">
      <div
        v-for="pref in filteredPreferences"
        :key="pref.id"
        class="card flex items-center justify-between"
      >
        <div>
          <div class="flex items-center gap-2">
            <span class="text-xs px-2 py-1 bg-blue-900 text-blue-300 rounded">{{ pref.category }}</span>
            <span class="font-medium">{{ pref.key }}</span>
          </div>
          <div class="text-gray-400 mt-1">
            <span v-if="Array.isArray(pref.value)">{{ pref.value.join(', ') }}</span>
            <span v-else-if="typeof pref.value === 'object'">{{ JSON.stringify(pref.value) }}</span>
            <span v-else>{{ pref.value }}</span>
          </div>
        </div>
        <div class="flex gap-2">
          <button @click="editPreference(pref)" class="text-gray-400 hover:text-blue-400">
            <Pencil class="w-4 h-4" />
          </button>
          <button @click="confirmDelete(pref)" class="text-gray-400 hover:text-red-400">
            <Trash2 class="w-4 h-4" />
          </button>
        </div>
      </div>

      <div v-if="filteredPreferences.length === 0" class="text-center py-12 text-gray-500">
        No preferences found.
      </div>
    </div>

    <!-- Add/Edit Modal -->
    <div v-if="showAddModal || editingPref" class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div class="card w-full max-w-lg">
        <h3 class="text-xl font-semibold mb-4">
          {{ editingPref ? 'Edit Preference' : 'Add Preference' }}
        </h3>

        <div class="space-y-4">
          <div>
            <label class="block text-sm text-gray-400 mb-1">Category</label>
            <input v-model="prefForm.category" type="text" class="input w-full"
                   placeholder="music, anime, communication, etc."
                   :disabled="!!editingPref" />
          </div>
          <div>
            <label class="block text-sm text-gray-400 mb-1">Key</label>
            <input v-model="prefForm.key" type="text" class="input w-full"
                   placeholder="favorite_genres, preferred_style, etc."
                   :disabled="!!editingPref" />
          </div>
          <div>
            <label class="block text-sm text-gray-400 mb-1">Value (JSON or plain text)</label>
            <textarea v-model="prefForm.valueStr" class="input w-full h-32"
                      placeholder='["rock", "electronic"] or just plain text'></textarea>
          </div>
        </div>

        <div class="flex justify-end gap-3 mt-6">
          <button @click="closeModal" class="btn btn-secondary">Cancel</button>
          <button @click="savePreference" class="btn btn-primary">Save</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Plus, Pencil, Trash2 } from 'lucide-vue-next'
import { usePreferencesStore } from '@/stores/preferences'
import type { Preference } from '@/types'

const store = usePreferencesStore()
const selectedCategory = ref('')
const showAddModal = ref(false)
const editingPref = ref<Preference | null>(null)

const prefForm = ref({
  category: '',
  key: '',
  valueStr: ''
})

const filteredPreferences = computed(() => {
  if (!selectedCategory.value) return store.preferences
  return store.preferences.filter(p => p.category === selectedCategory.value)
})

function editPreference(pref: Preference) {
  editingPref.value = pref
  prefForm.value = {
    category: pref.category,
    key: pref.key,
    valueStr: typeof pref.value === 'string' ? pref.value : JSON.stringify(pref.value, null, 2)
  }
}

function confirmDelete(pref: Preference) {
  if (confirm(`Delete preference "${pref.key}"?`)) {
    store.deletePreference(pref.id)
  }
}

function closeModal() {
  showAddModal.value = false
  editingPref.value = null
  prefForm.value = { category: '', key: '', valueStr: '' }
}

async function savePreference() {
  let value: any
  try {
    value = JSON.parse(prefForm.value.valueStr)
  } catch {
    value = prefForm.value.valueStr
  }

  if (editingPref.value) {
    await store.updatePreference(editingPref.value.id, value)
  } else {
    await store.createPreference({
      category: prefForm.value.category,
      key: prefForm.value.key,
      value
    })
  }
  closeModal()
}

onMounted(() => store.fetchPreferences())
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

.input {
  @apply bg-gray-700 border border-gray-600 rounded-lg px-4 py-2
         focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent;
}
</style>
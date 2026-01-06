<template>
  <div>
    <div class="flex items-center justify-between mb-8">
      <div>
        <h1 class="text-3xl font-bold">Vault</h1>
        <p class="text-gray-400 mt-1">API keys and secrets (values never displayed)</p>
      </div>
      <button @click="showAddModal = true" class="btn btn-primary flex items-center gap-2">
        <Plus class="w-4 h-4" />
        Add Key
      </button>
    </div>

    <!-- Warning Banner -->
    <div class="bg-yellow-900/30 border border-yellow-700 rounded-lg p-4 mb-6 flex items-center gap-3">
      <AlertTriangle class="w-5 h-5 text-yellow-500 flex-shrink-0" />
      <p class="text-yellow-200 text-sm">
        API key values are stored securely and never displayed in the UI. You can test, update, or delete keys.
      </p>
    </div>

    <!-- Keys List -->
    <div class="card">
      <table class="w-full">
        <thead>
          <tr class="text-left text-gray-400 border-b border-gray-700">
            <th class="pb-3">Service</th>
            <th class="pb-3">Key Name</th>
            <th class="pb-3">Status</th>
            <th class="pb-3">Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="key in store.keys" :key="key.key_name" class="border-b border-gray-700 last:border-0">
            <td class="py-4">{{ key.service }}</td>
            <td class="py-4 font-mono text-sm">{{ key.key_name }}</td>
            <td class="py-4">
              <span
                class="flex items-center gap-2"
                :class="key.is_set ? 'text-green-400' : 'text-gray-500'"
              >
                <span class="w-2 h-2 rounded-full" :class="key.is_set ? 'bg-green-400' : 'bg-gray-500'"></span>
                {{ key.is_set ? 'Set' : 'Not set' }}
              </span>
            </td>
            <td class="py-4">
              <div class="flex gap-2">
                <button
                  @click="testKey(key)"
                  class="btn btn-secondary text-sm py-1"
                  :disabled="testing === key.key_name"
                >
                  {{ testing === key.key_name ? 'Testing...' : 'Test' }}
                </button>
                <button @click="editKey(key)" class="text-gray-400 hover:text-blue-400">
                  <Pencil class="w-4 h-4" />
                </button>
                <button @click="confirmDelete(key)" class="text-gray-400 hover:text-red-400">
                  <Trash2 class="w-4 h-4" />
                </button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Test Result Toast -->
    <div
      v-if="testResult"
      class="fixed bottom-4 right-4 p-4 rounded-lg shadow-lg"
      :class="testResult.valid ? 'bg-green-900' : 'bg-red-900'"
    >
      <div class="flex items-center gap-2">
        <CheckCircle v-if="testResult.valid" class="w-5 h-5 text-green-400" />
        <XCircle v-else class="w-5 h-5 text-red-400" />
        <span>{{ testResult.message }}</span>
      </div>
    </div>

    <!-- Add/Edit Modal -->
    <div v-if="showAddModal || editingKey" class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div class="card w-full max-w-lg">
        <h3 class="text-xl font-semibold mb-4">
          {{ editingKey ? 'Update Key' : 'Add New Key' }}
        </h3>

        <div class="space-y-4">
          <div>
            <label class="block text-sm text-gray-400 mb-1">Service</label>
            <input v-model="keyForm.service" type="text" class="input w-full"
                   placeholder="openai, anthropic, plaid, etc."
                   :disabled="!!editingKey" />
          </div>
          <div>
            <label class="block text-sm text-gray-400 mb-1">Key Name</label>
            <input v-model="keyForm.key_name" type="text" class="input w-full"
                   placeholder="api_key, client_id, etc."
                   :disabled="!!editingKey" />
          </div>
          <div>
            <label class="block text-sm text-gray-400 mb-1">Value</label>
            <input v-model="keyForm.value" type="password" class="input w-full"
                   placeholder="sk-..." />
          </div>
        </div>

        <div class="flex justify-end gap-3 mt-6">
          <button @click="closeModal" class="btn btn-secondary">Cancel</button>
          <button @click="saveKey" class="btn btn-primary">Save</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Plus, Pencil, Trash2, AlertTriangle, CheckCircle, XCircle } from 'lucide-vue-next'
import { useVaultStore } from '@/stores/vault'
import type { VaultKey } from '@/types'

const store = useVaultStore()
const showAddModal = ref(false)
const editingKey = ref<VaultKey | null>(null)
const testing = ref<string | null>(null)
const testResult = ref<{ valid: boolean; message: string } | null>(null)

const keyForm = ref({
  service: '',
  key_name: '',
  value: ''
})

async function testKey(key: VaultKey) {
  testing.value = key.key_name
  try {
    testResult.value = await store.testKey(key.key_name)
    setTimeout(() => testResult.value = null, 3000)
  } finally {
    testing.value = null
  }
}

function editKey(key: VaultKey) {
  editingKey.value = key
  keyForm.value = {
    service: key.service,
    key_name: key.key_name,
    value: ''
  }
}

function confirmDelete(key: VaultKey) {
  if (confirm(`Delete key "${key.key_name}"?`)) {
    store.deleteKey(key.key_name)
  }
}

function closeModal() {
  showAddModal.value = false
  editingKey.value = null
  keyForm.value = { service: '', key_name: '', value: '' }
}

async function saveKey() {
  const fullKeyName = editingKey.value
    ? editingKey.value.key_name
    : `${keyForm.value.service}.${keyForm.value.key_name}`

  await store.createKey({
    key_name: fullKeyName,
    value: keyForm.value.value,
    service: keyForm.value.service
  })
  closeModal()
}

onMounted(() => store.fetchKeys())
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
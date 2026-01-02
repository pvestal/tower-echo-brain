<template>
  <div class="space-y-8 p-6">
    <h1 class="text-3xl font-bold text-tower-text-primary mb-8">Tower UI Components Showcase</h1>

    <!-- TowerCard Examples -->
    <section>
      <h2 class="text-2xl font-semibold text-tower-text-primary mb-4">TowerCard</h2>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <TowerCard variant="default">
          <template #header>
            <h3 class="font-semibold">Default Card</h3>
          </template>
          <p>This is a default card with header and body.</p>
        </TowerCard>

        <TowerCard variant="elevated">
          <template #header>
            <h3 class="font-semibold">Elevated Card</h3>
          </template>
          <p>This card has elevation (shadow).</p>
          <template #footer>
            <div class="flex justify-end">
              <TowerButton size="sm">Action</TowerButton>
            </div>
          </template>
        </TowerCard>

        <TowerCard variant="outlined">
          <template #header>
            <h3 class="font-semibold">Outlined Card</h3>
          </template>
          <p>This card has a prominent border.</p>
        </TowerCard>
      </div>
    </section>

    <!-- TowerButton Examples -->
    <section>
      <h2 class="text-2xl font-semibold text-tower-text-primary mb-4">TowerButton</h2>
      <TowerCard>
        <div class="space-y-4">
          <div>
            <h3 class="font-medium mb-2">Variants</h3>
            <div class="flex flex-wrap gap-2">
              <TowerButton variant="primary">Primary</TowerButton>
              <TowerButton variant="secondary">Secondary</TowerButton>
              <TowerButton variant="success">Success</TowerButton>
              <TowerButton variant="warning">Warning</TowerButton>
              <TowerButton variant="danger">Danger</TowerButton>
              <TowerButton variant="ghost">Ghost</TowerButton>
              <TowerButton variant="link">Link</TowerButton>
            </div>
          </div>

          <div>
            <h3 class="font-medium mb-2">Sizes</h3>
            <div class="flex flex-wrap items-center gap-2">
              <TowerButton size="xs">Extra Small</TowerButton>
              <TowerButton size="sm">Small</TowerButton>
              <TowerButton size="md">Medium</TowerButton>
              <TowerButton size="lg">Large</TowerButton>
              <TowerButton size="xl">Extra Large</TowerButton>
            </div>
          </div>

          <div>
            <h3 class="font-medium mb-2">States</h3>
            <div class="flex flex-wrap gap-2">
              <TowerButton :loading="true">Loading</TowerButton>
              <TowerButton disabled>Disabled</TowerButton>
              <TowerButton rounded>Rounded</TowerButton>
              <TowerButton full-width>Full Width</TowerButton>
            </div>
          </div>
        </div>
      </TowerCard>
    </section>

    <!-- TowerInput Examples -->
    <section>
      <h2 class="text-2xl font-semibold text-tower-text-primary mb-4">TowerInput</h2>
      <TowerCard>
        <div class="space-y-4 max-w-md">
          <TowerInput
            v-model="textInput"
            label="Text Input"
            placeholder="Enter some text..."
            helper-text="This is helper text"
          />

          <TowerInput
            v-model="emailInput"
            type="email"
            label="Email"
            placeholder="user@example.com"
            :error="emailError"
            clearable
            required
          />

          <TowerInput
            v-model="numberInput"
            type="number"
            label="Number Input"
            placeholder="0"
            min="0"
            max="100"
            show-character-count
            :maxlength="3"
          >
            <template #prefix>
              <span class="text-tower-text-secondary">$</span>
            </template>
            <template #suffix>
              <span class="text-tower-text-secondary">.00</span>
            </template>
          </TowerInput>

          <TowerInput
            label="Different Sizes"
            placeholder="Small input"
            size="sm"
          />
        </div>
      </TowerCard>
    </section>

    <!-- TowerSelect Examples -->
    <section>
      <h2 class="text-2xl font-semibold text-tower-text-primary mb-4">TowerSelect</h2>
      <TowerCard>
        <div class="space-y-4 max-w-md">
          <TowerSelect
            v-model="selectValue"
            :options="selectOptions"
            label="Select Option"
            placeholder="Choose an option..."
            helper-text="Select one of the available options"
          />

          <TowerSelect
            v-model="userSelect"
            :options="userOptions"
            label="Select User"
            value-key="id"
            label-key="name"
            disabled-key="inactive"
            :error="userSelectError"
          />
        </div>
      </TowerCard>
    </section>

    <!-- TowerTextarea Examples -->
    <section>
      <h2 class="text-2xl font-semibold text-tower-text-primary mb-4">TowerTextarea</h2>
      <TowerCard>
        <div class="space-y-4 max-w-md">
          <TowerTextarea
            v-model="textareaValue"
            label="Description"
            placeholder="Enter a description..."
            helper-text="Provide detailed information"
            :rows="4"
            :maxlength="200"
            show-word-count
          />

          <TowerTextarea
            v-model="autoResizeText"
            label="Auto-resize Textarea"
            placeholder="This textarea will grow as you type..."
            auto-resize
            :min-height="80"
            :max-height="300"
          />
        </div>
      </TowerCard>
    </section>

    <!-- TowerModal Example -->
    <section>
      <h2 class="text-2xl font-semibold text-tower-text-primary mb-4">TowerModal</h2>
      <TowerCard>
        <div class="space-y-4">
          <div class="flex gap-2">
            <TowerButton @click="showSmallModal = true">Small Modal</TowerButton>
            <TowerButton @click="showLargeModal = true" variant="secondary">Large Modal</TowerButton>
          </div>

          <!-- Small Modal -->
          <TowerModal
            v-model:open="showSmallModal"
            title="Small Modal"
            size="sm"
          >
            <p>This is a small modal with a title and close button.</p>
            <template #footer>
              <div class="flex gap-2 justify-end">
                <TowerButton @click="showSmallModal = false" variant="secondary">Cancel</TowerButton>
                <TowerButton @click="showSmallModal = false">Confirm</TowerButton>
              </div>
            </template>
          </TowerModal>

          <!-- Large Modal -->
          <TowerModal
            v-model:open="showLargeModal"
            title="Large Modal Example"
            size="lg"
          >
            <div class="space-y-4">
              <p>This is a larger modal with more content and form elements.</p>
              <TowerInput
                v-model="modalInput"
                label="Input in Modal"
                placeholder="Type something..."
              />
              <TowerTextarea
                v-model="modalTextarea"
                label="Textarea in Modal"
                placeholder="Enter details..."
                :rows="3"
              />
            </div>
            <template #footer>
              <div class="flex gap-2 justify-end">
                <TowerButton @click="showLargeModal = false" variant="ghost">Cancel</TowerButton>
                <TowerButton @click="handleModalSubmit">Save Changes</TowerButton>
              </div>
            </template>
          </TowerModal>
        </div>
      </TowerCard>
    </section>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

// Input values
const textInput = ref('')
const emailInput = ref('')
const numberInput = ref('')
const textareaValue = ref('')
const autoResizeText = ref('')

// Select values
const selectValue = ref('')
const userSelect = ref('')

// Modal values
const modalInput = ref('')
const modalTextarea = ref('')
const showSmallModal = ref(false)
const showLargeModal = ref(false)

// Options for selects
const selectOptions = [
  { value: 'option1', label: 'Option 1' },
  { value: 'option2', label: 'Option 2' },
  { value: 'option3', label: 'Option 3' },
  { value: 'option4', label: 'Option 4 (Disabled)', disabled: true }
]

const userOptions = [
  { id: '1', name: 'John Doe', inactive: false },
  { id: '2', name: 'Jane Smith', inactive: false },
  { id: '3', name: 'Bob Wilson (Inactive)', inactive: true },
  { id: '4', name: 'Alice Johnson', inactive: false }
]

// Validation
const emailError = computed(() => {
  if (emailInput.value && !emailInput.value.includes('@')) {
    return 'Please enter a valid email address'
  }
  return null
})

const userSelectError = computed(() => {
  if (userSelect.value === '3') {
    return 'Cannot select inactive user'
  }
  return null
})

// Modal handlers
const handleModalSubmit = () => {
  console.log('Modal submitted:', { input: modalInput.value, textarea: modalTextarea.value })
  showLargeModal.value = false
  // Reset form
  modalInput.value = ''
  modalTextarea.value = ''
}
</script>

<style scoped>
/* Component-specific styles if needed */
</style>
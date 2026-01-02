<template>
  <div class="tower-select-wrapper">
    <!-- Label -->
    <label
      v-if="label"
      :for="selectId"
      :class="labelClasses"
    >
      {{ label }}
      <span
        v-if="required"
        class="text-tower-status-error ml-1"
        aria-label="required"
      >*</span>
    </label>

    <!-- Select container -->
    <div class="relative">
      <!-- Select button -->
      <button
        :id="selectId"
        type="button"
        :disabled="disabled"
        :aria-expanded="isOpen"
        :aria-haspopup="listbox"
        :aria-labelledby="label ? `${selectId}-label` : null"
        :aria-describedby="errorId"
        :aria-invalid="hasError"
        :class="selectClasses"
        @click="toggle"
        @keydown="handleKeydown"
        @blur="handleBlur"
      >
        <!-- Selected value display -->
        <span class="block truncate">
          {{ displayValue || placeholder }}
        </span>

        <!-- Dropdown arrow -->
        <span class="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3">
          <svg
            class="h-5 w-5 text-tower-text-secondary transition-transform duration-200"
            :class="{ 'rotate-180': isOpen }"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
          </svg>
        </span>
      </button>

      <!-- Options dropdown -->
      <Transition
        name="tower-select-dropdown"
        @after-leave="resetOptionNavigation"
      >
        <div
          v-if="isOpen"
          class="tower-select-dropdown"
          role="listbox"
          :aria-labelledby="selectId"
        >
          <div
            v-for="(option, index) in options"
            :key="option.value || option"
            :class="optionClasses(option, index)"
            role="option"
            :aria-selected="isSelected(option)"
            :aria-disabled="isDisabled(option)"
            @click="selectOption(option)"
            @mouseenter="highlightedIndex = index"
          >
            <!-- Custom option slot -->
            <slot
              name="option"
              :option="option"
              :selected="isSelected(option)"
              :disabled="isDisabled(option)"
            >
              <span class="block truncate">
                {{ getOptionLabel(option) }}
              </span>

              <!-- Selected checkmark -->
              <span
                v-if="isSelected(option)"
                class="absolute inset-y-0 right-0 flex items-center pr-4"
              >
                <svg class="h-4 w-4 text-tower-accent-primary" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                </svg>
              </span>
            </slot>
          </div>

          <!-- No options message -->
          <div
            v-if="!options.length"
            class="px-3 py-2 text-tower-text-secondary text-sm"
          >
            {{ noOptionsText }}
          </div>
        </div>
      </Transition>
    </div>

    <!-- Helper text -->
    <p
      v-if="helperText && !hasError"
      class="mt-1 text-sm text-tower-text-secondary"
    >
      {{ helperText }}
    </p>

    <!-- Error message -->
    <p
      v-if="hasError"
      :id="errorId"
      class="mt-1 text-sm text-tower-status-error"
      role="alert"
      aria-live="polite"
    >
      {{ errorMessage }}
    </p>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'

const props = defineProps({
  /**
   * Selected value (v-model)
   */
  modelValue: {
    type: [String, Number, Object],
    default: null
  },
  /**
   * Options array
   */
  options: {
    type: Array,
    default: () => []
  },
  /**
   * Select label
   */
  label: {
    type: String,
    default: null
  },
  /**
   * Placeholder text
   */
  placeholder: {
    type: String,
    default: 'Select an option'
  },
  /**
   * Helper text
   */
  helperText: {
    type: String,
    default: null
  },
  /**
   * Error message
   */
  error: {
    type: String,
    default: null
  },
  /**
   * Disabled state
   */
  disabled: {
    type: Boolean,
    default: false
  },
  /**
   * Required field
   */
  required: {
    type: Boolean,
    default: false
  },
  /**
   * No options text
   */
  noOptionsText: {
    type: String,
    default: 'No options available'
  },
  /**
   * Option value key (for object options)
   */
  valueKey: {
    type: String,
    default: 'value'
  },
  /**
   * Option label key (for object options)
   */
  labelKey: {
    type: String,
    default: 'label'
  },
  /**
   * Option disabled key (for object options)
   */
  disabledKey: {
    type: String,
    default: 'disabled'
  }
})

const emit = defineEmits(['update:modelValue', 'change', 'open', 'close'])

// Refs
const isOpen = ref(false)
const highlightedIndex = ref(-1)

// Generate unique IDs
const selectId = `tower-select-${Math.random().toString(36).substr(2, 9)}`
const errorId = `tower-select-error-${Math.random().toString(36).substr(2, 9)}`

// Computed properties
const hasError = computed(() => !!props.error)
const errorMessage = computed(() => props.error)

const displayValue = computed(() => {
  if (!props.modelValue) return null

  const selectedOption = props.options.find(option => {
    if (typeof option === 'object') {
      return option[props.valueKey] === props.modelValue
    }
    return option === props.modelValue
  })

  return selectedOption ? getOptionLabel(selectedOption) : props.modelValue
})

const labelClasses = computed(() => [
  'block text-sm font-medium mb-2',
  {
    'text-tower-text-primary': !hasError.value,
    'text-tower-status-error': hasError.value
  }
])

const selectClasses = computed(() => [
  'tower-select',
  'relative w-full py-2 pl-3 pr-10 text-left border rounded-md cursor-pointer',
  'focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-tower-bg-primary',
  'transition-colors duration-200',

  // States
  {
    'bg-tower-bg-primary border-tower-border-primary text-tower-text-primary': !props.disabled,
    'bg-tower-bg-tertiary border-tower-border-primary text-tower-text-secondary cursor-not-allowed': props.disabled,
    'border-tower-status-error focus:ring-tower-status-error': hasError.value && !props.disabled,
    'border-tower-accent-primary focus:ring-tower-accent-primary': !hasError.value && !props.disabled,
    'text-tower-text-secondary': !displayValue.value
  }
])

// Helper functions
const getOptionValue = (option) => {
  return typeof option === 'object' ? option[props.valueKey] : option
}

const getOptionLabel = (option) => {
  return typeof option === 'object' ? option[props.labelKey] : option
}

const isSelected = (option) => {
  return getOptionValue(option) === props.modelValue
}

const isDisabled = (option) => {
  return typeof option === 'object' ? option[props.disabledKey] : false
}

const optionClasses = (option, index) => [
  'relative cursor-pointer select-none py-2 pl-3 pr-9',
  'transition-colors duration-150',
  {
    'bg-tower-bg-hover text-tower-text-primary': index === highlightedIndex.value,
    'bg-tower-accent-primary text-white': isSelected(option),
    'text-tower-text-primary': !isSelected(option) && index !== highlightedIndex.value,
    'opacity-50 cursor-not-allowed': isDisabled(option)
  }
]

// Event handlers
const toggle = () => {
  if (props.disabled) return

  if (isOpen.value) {
    close()
  } else {
    open()
  }
}

const open = () => {
  isOpen.value = true
  highlightedIndex.value = props.options.findIndex(option => isSelected(option))
  emit('open')

  nextTick(() => {
    // Focus management can be added here if needed
  })
}

const close = () => {
  isOpen.value = false
  highlightedIndex.value = -1
  emit('close')
}

const selectOption = (option) => {
  if (isDisabled(option)) return

  const value = getOptionValue(option)
  emit('update:modelValue', value)
  emit('change', value, option)
  close()
}

const handleKeydown = (event) => {
  if (props.disabled) return

  switch (event.key) {
    case 'Enter':
    case ' ':
      event.preventDefault()
      if (isOpen.value) {
        if (highlightedIndex.value >= 0) {
          selectOption(props.options[highlightedIndex.value])
        }
      } else {
        open()
      }
      break

    case 'Escape':
      if (isOpen.value) {
        event.preventDefault()
        close()
      }
      break

    case 'ArrowDown':
      event.preventDefault()
      if (isOpen.value) {
        highlightedIndex.value = Math.min(
          highlightedIndex.value + 1,
          props.options.length - 1
        )
      } else {
        open()
      }
      break

    case 'ArrowUp':
      event.preventDefault()
      if (isOpen.value) {
        highlightedIndex.value = Math.max(highlightedIndex.value - 1, 0)
      } else {
        open()
      }
      break

    case 'Home':
      if (isOpen.value) {
        event.preventDefault()
        highlightedIndex.value = 0
      }
      break

    case 'End':
      if (isOpen.value) {
        event.preventDefault()
        highlightedIndex.value = props.options.length - 1
      }
      break
  }
}

const handleBlur = (event) => {
  // Close dropdown if focus moves outside the component
  setTimeout(() => {
    if (!event.currentTarget.contains(document.activeElement)) {
      close()
    }
  }, 100)
}

const handleClickOutside = (event) => {
  if (isOpen.value && !event.target.closest('.tower-select-wrapper')) {
    close()
  }
}

const resetOptionNavigation = () => {
  highlightedIndex.value = -1
}

// Lifecycle
onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>

<style scoped>
.tower-select {
  /* Custom properties for theme integration */
  --tower-bg-primary: var(--bg-primary, #1a1a1a);
  --tower-bg-tertiary: var(--bg-tertiary, #3a3a3a);
  --tower-bg-hover: var(--bg-hover, #404040);
  --tower-border-primary: var(--border-primary, #3a3a3a);
  --tower-text-primary: var(--text-primary, #e0e0e0);
  --tower-text-secondary: var(--text-secondary, #a0a0a0);
  --tower-accent-primary: var(--accent-primary, #7aa2f7);
  --tower-status-error: var(--status-error, #a05050);
}

.tower-select {
  background-color: var(--tower-bg-primary);
  border-color: var(--tower-border-primary);
  color: var(--tower-text-primary);
}

.tower-select:focus {
  border-color: var(--tower-accent-primary);
  box-shadow: 0 0 0 2px var(--tower-accent-primary);
}

.tower-select-dropdown {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  z-index: 50;
  margin-top: 4px;
  max-height: 200px;
  overflow-y: auto;
  background-color: var(--tower-bg-primary);
  border: 1px solid var(--tower-border-primary);
  border-radius: 0.375rem;
  box-shadow: var(--shadow-lg, 0 4px 8px rgba(0, 0, 0, 0.5));
}

/* Dropdown transition */
.tower-select-dropdown-enter-active,
.tower-select-dropdown-leave-active {
  transition: all 0.2s ease;
}

.tower-select-dropdown-enter-from,
.tower-select-dropdown-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}

/* Custom scrollbar for dropdown */
.tower-select-dropdown::-webkit-scrollbar {
  width: 6px;
}

.tower-select-dropdown::-webkit-scrollbar-track {
  background: var(--tower-bg-tertiary);
}

.tower-select-dropdown::-webkit-scrollbar-thumb {
  background: var(--tower-border-primary);
  border-radius: 3px;
}

.tower-select-dropdown::-webkit-scrollbar-thumb:hover {
  background: var(--tower-text-secondary);
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
  .tower-select,
  .tower-select-dropdown-enter-active,
  .tower-select-dropdown-leave-active {
    transition: none;
  }
}
</style>
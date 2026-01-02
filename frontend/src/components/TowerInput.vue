<template>
  <div class="tower-input-wrapper">
    <!-- Label -->
    <label
      v-if="label"
      :for="inputId"
      :class="labelClasses"
    >
      {{ label }}
      <span
        v-if="required"
        class="text-tower-status-error ml-1"
        aria-label="required"
      >*</span>
    </label>

    <!-- Input container -->
    <div class="relative">
      <!-- Prefix icon -->
      <div
        v-if="$slots.prefix"
        class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"
      >
        <slot name="prefix" />
      </div>

      <!-- Input field -->
      <input
        :id="inputId"
        :type="type"
        :value="modelValue"
        :placeholder="placeholder"
        :disabled="disabled"
        :readonly="readonly"
        :required="required"
        :min="min"
        :max="max"
        :step="step"
        :maxlength="maxlength"
        :pattern="pattern"
        :autocomplete="autocomplete"
        :aria-label="ariaLabel"
        :aria-describedby="errorId"
        :aria-invalid="hasError"
        :class="inputClasses"
        @input="handleInput"
        @change="handleChange"
        @blur="handleBlur"
        @focus="handleFocus"
        @keydown="handleKeydown"
      >

      <!-- Suffix icon -->
      <div
        v-if="$slots.suffix || showClearButton"
        class="absolute inset-y-0 right-0 pr-3 flex items-center"
      >
        <!-- Clear button -->
        <button
          v-if="showClearButton && modelValue && !disabled"
          type="button"
          class="text-tower-text-secondary hover:text-tower-text-primary focus:outline-none focus:text-tower-text-primary"
          aria-label="Clear input"
          @click="clearInput"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        <!-- Custom suffix -->
        <slot v-else name="suffix" />
      </div>
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

    <!-- Character count -->
    <p
      v-if="showCharacterCount && maxlength"
      class="mt-1 text-xs text-tower-text-muted text-right"
    >
      {{ characterCount }}/{{ maxlength }}
    </p>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'

const props = defineProps({
  /**
   * Input value (v-model)
   */
  modelValue: {
    type: [String, Number],
    default: ''
  },
  /**
   * Input type
   */
  type: {
    type: String,
    default: 'text',
    validator: (value) => [
      'text', 'email', 'password', 'number', 'tel', 'url', 'search', 'date', 'time', 'datetime-local'
    ].includes(value)
  },
  /**
   * Input label
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
    default: null
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
   * Input size
   */
  size: {
    type: String,
    default: 'md',
    validator: (value) => ['sm', 'md', 'lg'].includes(value)
  },
  /**
   * Input variant
   */
  variant: {
    type: String,
    default: 'default',
    validator: (value) => ['default', 'filled'].includes(value)
  },
  /**
   * Disabled state
   */
  disabled: {
    type: Boolean,
    default: false
  },
  /**
   * Readonly state
   */
  readonly: {
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
   * Show clear button
   */
  clearable: {
    type: Boolean,
    default: false
  },
  /**
   * Show character count
   */
  showCharacterCount: {
    type: Boolean,
    default: false
  },
  /**
   * HTML input attributes
   */
  min: [String, Number],
  max: [String, Number],
  step: [String, Number],
  maxlength: [String, Number],
  pattern: String,
  autocomplete: String,
  ariaLabel: String
})

const emit = defineEmits(['update:modelValue', 'change', 'blur', 'focus', 'clear', 'keydown'])

// Generate unique IDs
const inputId = `tower-input-${Math.random().toString(36).substr(2, 9)}`
const errorId = `tower-input-error-${Math.random().toString(36).substr(2, 9)}`

// Internal state
const isFocused = ref(false)

// Computed properties
const hasError = computed(() => !!props.error)
const errorMessage = computed(() => props.error)
const showClearButton = computed(() => props.clearable && !props.readonly)
const characterCount = computed(() => String(props.modelValue || '').length)

const labelClasses = computed(() => [
  'block text-sm font-medium mb-2',
  {
    'text-tower-text-primary': !hasError.value,
    'text-tower-status-error': hasError.value
  }
])

const inputClasses = computed(() => [
  'tower-input',
  'block w-full border rounded-md transition-colors duration-200',
  'focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-tower-bg-primary',
  'disabled:cursor-not-allowed disabled:opacity-50',

  // Size variants
  {
    'px-3 py-1.5 text-sm': props.size === 'sm',
    'px-3 py-2 text-sm': props.size === 'md',
    'px-4 py-3 text-base': props.size === 'lg'
  },

  // Style variants
  {
    'bg-tower-bg-primary border-tower-border-primary text-tower-text-primary': props.variant === 'default',
    'bg-tower-bg-tertiary border-tower-border-primary text-tower-text-primary': props.variant === 'filled'
  },

  // States
  {
    'border-tower-status-error focus:ring-tower-status-error focus:border-tower-status-error': hasError.value,
    'border-tower-border-focus focus:ring-tower-accent-primary focus:border-tower-accent-primary': !hasError.value && isFocused.value,
    'hover:border-tower-border-focus': !hasError.value && !props.disabled && !props.readonly
  },

  // Padding adjustments for icons
  {
    'pl-10': !!props.$slots?.prefix,
    'pr-10': !!props.$slots?.suffix || showClearButton.value
  }
])

// Event handlers
const handleInput = (event) => {
  emit('update:modelValue', event.target.value)
}

const handleChange = (event) => {
  emit('change', event.target.value, event)
}

const handleBlur = (event) => {
  isFocused.value = false
  emit('blur', event.target.value, event)
}

const handleFocus = (event) => {
  isFocused.value = true
  emit('focus', event.target.value, event)
}

const handleKeydown = (event) => {
  emit('keydown', event)
}

const clearInput = () => {
  emit('update:modelValue', '')
  emit('clear')
}

// Watch for external value changes
watch(() => props.modelValue, (newValue) => {
  if (document.activeElement?.id !== inputId) {
    // Only update if input is not focused to avoid cursor jumping
  }
})
</script>

<style scoped>
.tower-input {
  /* Custom properties for theme integration */
  --tower-bg-primary: var(--bg-primary, #1a1a1a);
  --tower-bg-tertiary: var(--bg-tertiary, #3a3a3a);
  --tower-border-primary: var(--border-primary, #3a3a3a);
  --tower-border-focus: var(--border-focus, #606060);
  --tower-text-primary: var(--text-primary, #e0e0e0);
  --tower-text-secondary: var(--text-secondary, #a0a0a0);
  --tower-text-muted: var(--text-muted, #707070);
  --tower-accent-primary: var(--accent-primary, #7aa2f7);
  --tower-status-error: var(--status-error, #a05050);
}

.tower-input {
  background-color: var(--tower-bg-primary);
  border-color: var(--tower-border-primary);
  color: var(--tower-text-primary);
}

.tower-input:hover:not(:disabled):not(:read-only) {
  border-color: var(--tower-border-focus);
}

.tower-input:focus {
  border-color: var(--tower-accent-primary);
  box-shadow: 0 0 0 2px var(--tower-accent-primary);
}

.tower-input.border-tower-status-error {
  border-color: var(--tower-status-error);
}

.tower-input.border-tower-status-error:focus {
  border-color: var(--tower-status-error);
  box-shadow: 0 0 0 2px var(--tower-status-error);
}

.tower-input::placeholder {
  color: var(--tower-text-secondary);
}

.tower-input:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.tower-input:read-only {
  background-color: var(--tower-bg-tertiary);
  cursor: default;
}

/* High contrast mode support */
@media (prefers-contrast: high) {
  .tower-input {
    border-width: 2px;
  }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
  .tower-input {
    transition: none;
  }
}

/* Remove number input spinners */
.tower-input[type="number"]::-webkit-outer-spin-button,
.tower-input[type="number"]::-webkit-inner-spin-button {
  -webkit-appearance: none;
  margin: 0;
}

.tower-input[type="number"] {
  -moz-appearance: textfield;
}
</style>
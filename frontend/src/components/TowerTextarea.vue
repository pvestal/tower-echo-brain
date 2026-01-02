<template>
  <div class="tower-textarea-wrapper">
    <!-- Label -->
    <label
      v-if="label"
      :for="textareaId"
      :class="labelClasses"
    >
      {{ label }}
      <span
        v-if="required"
        class="text-tower-status-error ml-1"
        aria-label="required"
      >*</span>
    </label>

    <!-- Textarea container -->
    <div class="relative">
      <textarea
        :id="textareaId"
        ref="textareaRef"
        :value="modelValue"
        :placeholder="placeholder"
        :disabled="disabled"
        :readonly="readonly"
        :required="required"
        :rows="rows"
        :maxlength="maxlength"
        :aria-label="ariaLabel"
        :aria-describedby="errorId"
        :aria-invalid="hasError"
        :class="textareaClasses"
        @input="handleInput"
        @change="handleChange"
        @blur="handleBlur"
        @focus="handleFocus"
        @keydown="handleKeydown"
      />

      <!-- Resize handle (for manual resize) -->
      <div
        v-if="!autoResize && resize !== 'none'"
        class="absolute bottom-1 right-1 w-3 h-3 cursor-se-resize opacity-50"
      >
        <svg fill="currentColor" viewBox="0 0 12 12">
          <path d="M8 8h4v4H8V8zM4 4h4v4H4V4zM0 0h4v4H0V0z" opacity="0.3"/>
          <path d="M8 8h4v4H8V8z"/>
        </svg>
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
    <div
      v-if="showCharacterCount || (maxlength && showCharacterCount !== false)"
      class="mt-1 flex justify-between text-xs text-tower-text-muted"
    >
      <span v-if="showWordCount" class="word-count">
        {{ wordCount }} {{ wordCount === 1 ? 'word' : 'words' }}
      </span>
      <span class="character-count">
        {{ characterCount }}{{ maxlength ? `/${maxlength}` : '' }}
      </span>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, ref, watch } from 'vue'

const props = defineProps({
  /**
   * Textarea value (v-model)
   */
  modelValue: {
    type: String,
    default: ''
  },
  /**
   * Textarea label
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
   * Number of visible rows
   */
  rows: {
    type: [Number, String],
    default: 4
  },
  /**
   * Auto-resize based on content
   */
  autoResize: {
    type: Boolean,
    default: false
  },
  /**
   * Resize behavior
   */
  resize: {
    type: String,
    default: 'vertical',
    validator: (value) => ['none', 'both', 'horizontal', 'vertical'].includes(value)
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
   * Maximum character length
   */
  maxlength: {
    type: [String, Number],
    default: null
  },
  /**
   * Show character count
   */
  showCharacterCount: {
    type: Boolean,
    default: null // null = auto (show if maxlength is set)
  },
  /**
   * Show word count
   */
  showWordCount: {
    type: Boolean,
    default: false
  },
  /**
   * Minimum height for auto-resize
   */
  minHeight: {
    type: [String, Number],
    default: null
  },
  /**
   * Maximum height for auto-resize
   */
  maxHeight: {
    type: [String, Number],
    default: null
  },
  /**
   * Aria label for accessibility
   */
  ariaLabel: {
    type: String,
    default: null
  }
})

const emit = defineEmits(['update:modelValue', 'change', 'blur', 'focus', 'keydown'])

// Refs
const textareaRef = ref(null)
const isFocused = ref(false)

// Generate unique IDs
const textareaId = `tower-textarea-${Math.random().toString(36).substr(2, 9)}`
const errorId = `tower-textarea-error-${Math.random().toString(36).substr(2, 9)}`

// Computed properties
const hasError = computed(() => !!props.error)
const errorMessage = computed(() => props.error)
const characterCount = computed(() => String(props.modelValue || '').length)
const wordCount = computed(() => {
  const text = String(props.modelValue || '').trim()
  if (!text) return 0
  return text.split(/\s+/).length
})

const labelClasses = computed(() => [
  'block text-sm font-medium mb-2',
  {
    'text-tower-text-primary': !hasError.value,
    'text-tower-status-error': hasError.value
  }
])

const textareaClasses = computed(() => [
  'tower-textarea',
  'block w-full border rounded-md transition-colors duration-200',
  'focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-tower-bg-primary',
  'disabled:cursor-not-allowed disabled:opacity-50',
  'placeholder:text-tower-text-secondary',

  // Base styling
  'px-3 py-2 text-sm',

  // Background and colors
  'bg-tower-bg-primary border-tower-border-primary text-tower-text-primary',

  // States
  {
    'border-tower-status-error focus:ring-tower-status-error focus:border-tower-status-error': hasError.value,
    'border-tower-border-focus focus:ring-tower-accent-primary focus:border-tower-accent-primary': !hasError.value && isFocused.value,
    'hover:border-tower-border-focus': !hasError.value && !props.disabled && !props.readonly,
    'bg-tower-bg-tertiary': props.readonly
  },

  // Resize behavior
  {
    'resize-none': props.autoResize || props.resize === 'none',
    'resize': props.resize === 'both' && !props.autoResize,
    'resize-x': props.resize === 'horizontal' && !props.autoResize,
    'resize-y': props.resize === 'vertical' && !props.autoResize
  }
])

// Event handlers
const handleInput = (event) => {
  const value = event.target.value
  emit('update:modelValue', value)

  if (props.autoResize) {
    autoResizeTextarea()
  }
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

// Auto-resize functionality
const autoResizeTextarea = () => {
  if (!textareaRef.value || !props.autoResize) return

  const textarea = textareaRef.value

  // Reset height to auto to get the actual scrollHeight
  textarea.style.height = 'auto'

  // Calculate new height
  let newHeight = textarea.scrollHeight

  // Apply min/max height constraints
  if (props.minHeight) {
    const minHeight = typeof props.minHeight === 'number' ? props.minHeight : parseInt(props.minHeight)
    newHeight = Math.max(newHeight, minHeight)
  }

  if (props.maxHeight) {
    const maxHeight = typeof props.maxHeight === 'number' ? props.maxHeight : parseInt(props.maxHeight)
    newHeight = Math.min(newHeight, maxHeight)
  }

  // Set the new height
  textarea.style.height = `${newHeight}px`
}

// Watch for value changes to trigger auto-resize
watch(() => props.modelValue, () => {
  if (props.autoResize) {
    nextTick(() => {
      autoResizeTextarea()
    })
  }
})

// Initialize auto-resize on mount
onMounted(() => {
  if (props.autoResize) {
    nextTick(() => {
      autoResizeTextarea()
    })
  }
})
</script>

<style scoped>
.tower-textarea {
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

.tower-textarea {
  background-color: var(--tower-bg-primary);
  border-color: var(--tower-border-primary);
  color: var(--tower-text-primary);
  line-height: 1.5;
}

.tower-textarea:hover:not(:disabled):not(:read-only) {
  border-color: var(--tower-border-focus);
}

.tower-textarea:focus {
  border-color: var(--tower-accent-primary);
  box-shadow: 0 0 0 2px var(--tower-accent-primary);
}

.tower-textarea.border-tower-status-error {
  border-color: var(--tower-status-error);
}

.tower-textarea.border-tower-status-error:focus {
  border-color: var(--tower-status-error);
  box-shadow: 0 0 0 2px var(--tower-status-error);
}

.tower-textarea::placeholder {
  color: var(--tower-text-secondary);
}

.tower-textarea:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.tower-textarea:read-only {
  background-color: var(--tower-bg-tertiary);
  cursor: default;
}

/* Hide default resize handle in auto-resize mode */
.tower-textarea.resize-none {
  resize: none;
}

/* Custom scrollbar */
.tower-textarea::-webkit-scrollbar {
  width: 8px;
}

.tower-textarea::-webkit-scrollbar-track {
  background: var(--tower-bg-tertiary);
  border-radius: 4px;
}

.tower-textarea::-webkit-scrollbar-thumb {
  background: var(--tower-border-primary);
  border-radius: 4px;
}

.tower-textarea::-webkit-scrollbar-thumb:hover {
  background: var(--tower-text-secondary);
}

/* Character/word count styling */
.character-count,
.word-count {
  color: var(--tower-text-muted);
}

/* High contrast mode support */
@media (prefers-contrast: high) {
  .tower-textarea {
    border-width: 2px;
  }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
  .tower-textarea {
    transition: none;
  }
}
</style>
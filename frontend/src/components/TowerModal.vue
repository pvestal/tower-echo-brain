<template>
  <Teleport to="body">
    <Transition
      name="tower-modal"
      @enter="onEnter"
      @after-enter="onAfterEnter"
      @leave="onLeave"
      @after-leave="onAfterLeave"
    >
      <div
        v-if="isOpen"
        class="tower-modal-overlay"
        role="dialog"
        aria-modal="true"
        :aria-labelledby="titleId"
        :aria-describedby="contentId"
        @click="handleOverlayClick"
        @keydown.esc="handleEscape"
      >
        <div
          ref="modalRef"
          :class="modalClasses"
          @click.stop
        >
          <!-- Header -->
          <header
            v-if="$slots.header || title || showCloseButton"
            class="tower-modal-header"
          >
            <div class="flex items-center justify-between">
              <!-- Title -->
              <h2
                v-if="title"
                :id="titleId"
                class="text-lg font-semibold text-tower-text-primary"
              >
                {{ title }}
              </h2>

              <!-- Custom header -->
              <slot v-else name="header" />

              <!-- Close button -->
              <button
                v-if="showCloseButton"
                type="button"
                class="tower-modal-close-button"
                aria-label="Close modal"
                @click="close"
              >
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </header>

          <!-- Body -->
          <div
            :id="contentId"
            class="tower-modal-body"
          >
            <slot />
          </div>

          <!-- Footer -->
          <footer
            v-if="$slots.footer"
            class="tower-modal-footer"
          >
            <slot name="footer" />
          </footer>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'

const props = defineProps({
  /**
   * Modal visibility
   */
  open: {
    type: Boolean,
    default: false
  },
  /**
   * Modal title
   */
  title: {
    type: String,
    default: null
  },
  /**
   * Modal size
   */
  size: {
    type: String,
    default: 'md',
    validator: (value) => ['xs', 'sm', 'md', 'lg', 'xl', 'full'].includes(value)
  },
  /**
   * Show close button
   */
  showCloseButton: {
    type: Boolean,
    default: true
  },
  /**
   * Close on overlay click
   */
  closeOnOverlay: {
    type: Boolean,
    default: true
  },
  /**
   * Close on escape key
   */
  closeOnEscape: {
    type: Boolean,
    default: true
  },
  /**
   * Persistent modal (cannot be closed)
   */
  persistent: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['update:open', 'close', 'open', 'after-open', 'after-close'])

// Refs
const modalRef = ref(null)
const previousActiveElement = ref(null)

// Generate unique IDs
const titleId = `tower-modal-title-${Math.random().toString(36).substr(2, 9)}`
const contentId = `tower-modal-content-${Math.random().toString(36).substr(2, 9)}`

// Computed properties
const isOpen = computed({
  get: () => props.open,
  set: (value) => emit('update:open', value)
})

const modalClasses = computed(() => [
  'tower-modal',
  'relative',
  'bg-tower-bg-secondary',
  'border',
  'border-tower-border-primary',
  'rounded-lg',
  'shadow-tower-lg',
  'overflow-hidden',
  'mx-4',
  'my-8',
  'max-h-screen',
  'overflow-y-auto',

  // Size variants
  {
    'max-w-xs': props.size === 'xs',
    'max-w-sm': props.size === 'sm',
    'max-w-md': props.size === 'md',
    'max-w-lg': props.size === 'lg',
    'max-w-4xl': props.size === 'xl',
    'max-w-full h-full mx-0 my-0 rounded-none': props.size === 'full'
  }
])

// Focus management
const focusableElements = computed(() => {
  if (!modalRef.value) return []

  return modalRef.value.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  )
})

const firstFocusableElement = computed(() => focusableElements.value[0])
const lastFocusableElement = computed(() => focusableElements.value[focusableElements.value.length - 1])

// Event handlers
const handleOverlayClick = () => {
  if (props.closeOnOverlay && !props.persistent) {
    close()
  }
}

const handleEscape = () => {
  if (props.closeOnEscape && !props.persistent) {
    close()
  }
}

const handleTabKey = (event) => {
  if (!focusableElements.value.length) return

  if (event.shiftKey) {
    // Shift + Tab (backwards)
    if (document.activeElement === firstFocusableElement.value) {
      event.preventDefault()
      lastFocusableElement.value?.focus()
    }
  } else {
    // Tab (forwards)
    if (document.activeElement === lastFocusableElement.value) {
      event.preventDefault()
      firstFocusableElement.value?.focus()
    }
  }
}

const close = () => {
  if (!props.persistent) {
    isOpen.value = false
    emit('close')
  }
}

// Transition handlers
const onEnter = () => {
  document.body.style.overflow = 'hidden'
  previousActiveElement.value = document.activeElement
  emit('open')
}

const onAfterEnter = async () => {
  await nextTick()

  // Focus the first focusable element or the modal itself
  if (firstFocusableElement.value) {
    firstFocusableElement.value.focus()
  } else if (modalRef.value) {
    modalRef.value.focus()
  }

  emit('after-open')
}

const onLeave = () => {
  document.removeEventListener('keydown', handleGlobalKeydown)
}

const onAfterLeave = () => {
  document.body.style.overflow = ''

  // Restore focus to the previously active element
  if (previousActiveElement.value && typeof previousActiveElement.value.focus === 'function') {
    previousActiveElement.value.focus()
  }

  previousActiveElement.value = null
  emit('after-close')
}

const handleGlobalKeydown = (event) => {
  if (!isOpen.value) return

  if (event.key === 'Escape') {
    handleEscape()
  } else if (event.key === 'Tab') {
    handleTabKey(event)
  }
}

// Watch for open state changes
watch(isOpen, (newValue) => {
  if (newValue) {
    nextTick(() => {
      document.addEventListener('keydown', handleGlobalKeydown)
    })
  } else {
    document.removeEventListener('keydown', handleGlobalKeydown)
  }
})

// Cleanup on unmount
onUnmounted(() => {
  document.removeEventListener('keydown', handleGlobalKeydown)
  document.body.style.overflow = ''
})
</script>

<style scoped>
.tower-modal-overlay {
  /* Custom properties for theme integration */
  --tower-bg-secondary: var(--bg-secondary, #2a2a2a);
  --tower-border-primary: var(--border-primary, #3a3a3a);
  --tower-text-primary: var(--text-primary, #e0e0e0);
  --tower-text-secondary: var(--text-secondary, #a0a0a0);
  --shadow-tower-lg: var(--shadow-lg, 0 4px 8px rgba(0, 0, 0, 0.5));

  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.75);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.tower-modal {
  background-color: var(--tower-bg-secondary);
  border-color: var(--tower-border-primary);
  box-shadow: var(--shadow-tower-lg);
}

.tower-modal-header {
  padding: 1.5rem;
  border-bottom: 1px solid var(--tower-border-primary);
}

.tower-modal-body {
  padding: 1.5rem;
}

.tower-modal-footer {
  padding: 1rem 1.5rem 1.5rem;
  border-top: 1px solid var(--tower-border-primary);
}

.tower-modal-close-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.5rem;
  color: var(--tower-text-secondary);
  background-color: transparent;
  border: none;
  border-radius: 0.375rem;
  cursor: pointer;
  transition: all 0.2s;
}

.tower-modal-close-button:hover {
  color: var(--tower-text-primary);
  background-color: rgba(255, 255, 255, 0.1);
}

.tower-modal-close-button:focus {
  outline: none;
  box-shadow: 0 0 0 2px var(--tower-border-primary);
}

/* Transitions */
.tower-modal-enter-active,
.tower-modal-leave-active {
  transition: opacity 0.3s ease;
}

.tower-modal-enter-active .tower-modal,
.tower-modal-leave-active .tower-modal {
  transition: all 0.3s ease;
}

.tower-modal-enter-from,
.tower-modal-leave-to {
  opacity: 0;
}

.tower-modal-enter-from .tower-modal,
.tower-modal-leave-to .tower-modal {
  transform: scale(0.9) translateY(-20px);
}

/* Responsive design */
@media (max-width: 640px) {
  .tower-modal-overlay {
    align-items: flex-start;
  }

  .tower-modal {
    margin: 1rem;
    max-height: calc(100vh - 2rem);
  }

  .tower-modal-header,
  .tower-modal-body {
    padding: 1rem;
  }

  .tower-modal-footer {
    padding: 0.75rem 1rem 1rem;
  }
}

/* High contrast mode support */
@media (prefers-contrast: high) {
  .tower-modal {
    border-width: 2px;
  }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
  .tower-modal-enter-active,
  .tower-modal-leave-active,
  .tower-modal-enter-active .tower-modal,
  .tower-modal-leave-active .tower-modal {
    transition: none;
  }
}
</style>
<template>
  <button
    :type="type"
    :disabled="disabled || loading"
    :aria-label="ariaLabel"
    :aria-describedby="ariaDescribedby"
    :class="buttonClasses"
    @click="handleClick"
    @keydown.enter="handleClick"
    @keydown.space.prevent="handleClick"
  >
    <!-- Loading spinner -->
    <span
      v-if="loading"
      class="tower-button-spinner"
      aria-hidden="true"
    />

    <!-- Icon slot -->
    <span
      v-if="$slots.icon && !loading"
      class="tower-button-icon"
      :class="{ 'mr-2': $slots.default }"
    >
      <slot name="icon" />
    </span>

    <!-- Button content -->
    <span
      v-if="$slots.default"
      :class="{ 'opacity-0': loading }"
    >
      <slot />
    </span>

    <!-- Screen reader loading text -->
    <span
      v-if="loading"
      class="sr-only"
    >
      Loading...
    </span>
  </button>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  /**
   * Visual variant of the button
   */
  variant: {
    type: String,
    default: 'primary',
    validator: (value) => ['primary', 'secondary', 'success', 'warning', 'danger', 'ghost', 'link'].includes(value)
  },
  /**
   * Button size
   */
  size: {
    type: String,
    default: 'md',
    validator: (value) => ['xs', 'sm', 'md', 'lg', 'xl'].includes(value)
  },
  /**
   * Button type attribute
   */
  type: {
    type: String,
    default: 'button',
    validator: (value) => ['button', 'submit', 'reset'].includes(value)
  },
  /**
   * Disabled state
   */
  disabled: {
    type: Boolean,
    default: false
  },
  /**
   * Loading state
   */
  loading: {
    type: Boolean,
    default: false
  },
  /**
   * Full width button
   */
  fullWidth: {
    type: Boolean,
    default: false
  },
  /**
   * Rounded button
   */
  rounded: {
    type: Boolean,
    default: false
  },
  /**
   * Aria label for accessibility
   */
  ariaLabel: {
    type: String,
    default: null
  },
  /**
   * Aria describedby for accessibility
   */
  ariaDescribedby: {
    type: String,
    default: null
  }
})

const emit = defineEmits(['click'])

const buttonClasses = computed(() => [
  'tower-button',
  'relative',
  'inline-flex',
  'items-center',
  'justify-center',
  'font-medium',
  'transition-all',
  'duration-200',
  'focus:outline-none',
  'focus:ring-2',
  'focus:ring-offset-2',
  'focus:ring-offset-tower-bg-primary',
  'disabled:cursor-not-allowed',
  'disabled:opacity-50',

  // Size variants
  {
    'px-2 py-1 text-xs': props.size === 'xs',
    'px-3 py-1.5 text-sm': props.size === 'sm',
    'px-4 py-2 text-sm': props.size === 'md',
    'px-6 py-3 text-base': props.size === 'lg',
    'px-8 py-4 text-lg': props.size === 'xl'
  },

  // Width variants
  {
    'w-full': props.fullWidth,
    'w-auto': !props.fullWidth
  },

  // Border radius
  {
    'rounded-full': props.rounded,
    'rounded-md': !props.rounded
  },

  // Color variants
  {
    // Primary
    'bg-tower-accent-primary hover:bg-tower-accent-hover focus:ring-tower-accent-primary text-white':
      props.variant === 'primary' && !props.disabled,

    // Secondary
    'bg-tower-bg-tertiary hover:bg-tower-bg-hover focus:ring-tower-bg-hover text-tower-text-primary border border-tower-border-primary':
      props.variant === 'secondary' && !props.disabled,

    // Success
    'bg-tower-status-success hover:bg-green-600 focus:ring-tower-status-success text-white':
      props.variant === 'success' && !props.disabled,

    // Warning
    'bg-tower-status-warning hover:bg-yellow-600 focus:ring-tower-status-warning text-white':
      props.variant === 'warning' && !props.disabled,

    // Danger
    'bg-tower-status-error hover:bg-red-600 focus:ring-tower-status-error text-white':
      props.variant === 'danger' && !props.disabled,

    // Ghost
    'bg-transparent hover:bg-tower-bg-hover focus:ring-tower-bg-hover text-tower-text-primary border border-tower-border-primary':
      props.variant === 'ghost' && !props.disabled,

    // Link
    'bg-transparent hover:bg-tower-bg-hover focus:ring-tower-accent-primary text-tower-accent-primary p-0 h-auto font-normal underline-offset-4 hover:underline':
      props.variant === 'link' && !props.disabled
  }
])

const handleClick = (event) => {
  if (!props.disabled && !props.loading) {
    emit('click', event)
  }
}
</script>

<style scoped>
.tower-button {
  /* Custom properties for theme integration */
  --tower-bg-primary: var(--bg-primary, #1a1a1a);
  --tower-bg-tertiary: var(--bg-tertiary, #3a3a3a);
  --tower-bg-hover: var(--bg-hover, #404040);
  --tower-border-primary: var(--border-primary, #3a3a3a);
  --tower-text-primary: var(--text-primary, #e0e0e0);
  --tower-accent-primary: var(--accent-primary, #7aa2f7);
  --tower-accent-hover: var(--accent-hover, #4a7cf3);
  --tower-status-success: var(--status-success, #50a050);
  --tower-status-warning: var(--status-warning, #a08050);
  --tower-status-error: var(--status-error, #a05050);
}

.tower-button-spinner {
  position: absolute;
  width: 1rem;
  height: 1rem;
  border: 2px solid transparent;
  border-top: 2px solid currentColor;
  border-radius: 50%;
  animation: tower-spin 1s linear infinite;
}

.tower-button-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

@keyframes tower-spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

/* Color variants with CSS custom properties */
.tower-button {
  color: var(--tower-text-primary);
}

/* Focus management */
.tower-button:focus {
  outline: none;
  box-shadow: 0 0 0 2px var(--tower-accent-primary);
}

/* Disabled state */
.tower-button:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

/* High contrast mode support */
@media (prefers-contrast: high) {
  .tower-button {
    border-width: 2px;
  }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
  .tower-button {
    transition: none;
  }

  .tower-button-spinner {
    animation: none;
  }
}
</style>
<template>
  <div
    :class="cardClasses"
    role="region"
    :aria-labelledby="headerSlotId"
    :aria-describedby="bodySlotId"
  >
    <!-- Header slot -->
    <header
      v-if="$slots.header"
      :id="headerSlotId"
      :class="headerClasses"
    >
      <slot name="header" />
    </header>

    <!-- Body content -->
    <div
      :id="bodySlotId"
      :class="bodyClasses"
    >
      <slot />
    </div>

    <!-- Footer slot -->
    <footer
      v-if="$slots.footer"
      :class="footerClasses"
    >
      <slot name="footer" />
    </footer>
  </div>
</template>

<script setup>
import { computed, useSlots } from 'vue'

const props = defineProps({
  /**
   * Visual variant of the card
   */
  variant: {
    type: String,
    default: 'default',
    validator: (value) => ['default', 'elevated', 'outlined'].includes(value)
  },
  /**
   * Remove default padding from body
   */
  noPadding: {
    type: Boolean,
    default: false
  },
  /**
   * Make card take full width
   */
  fullWidth: {
    type: Boolean,
    default: false
  }
})

const slots = useSlots()

// Generate unique IDs for accessibility
const headerSlotId = computed(() => `tower-card-header-${Math.random().toString(36).substr(2, 9)}`)
const bodySlotId = computed(() => `tower-card-body-${Math.random().toString(36).substr(2, 9)}`)

const cardClasses = computed(() => [
  'tower-card',
  'bg-tower-bg-secondary',
  'border',
  'border-tower-border-primary',
  'rounded-lg',
  'overflow-hidden',
  'transition-all',
  'duration-200',
  {
    'shadow-tower-md': props.variant === 'elevated',
    'border-2': props.variant === 'outlined',
    'w-full': props.fullWidth,
    'hover:bg-tower-bg-hover': props.variant === 'elevated',
    'hover:border-tower-border-focus': props.variant === 'outlined'
  }
])

const headerClasses = computed(() => [
  'tower-card-header',
  'px-6',
  'py-4',
  'border-b',
  'border-tower-border-secondary',
  'bg-tower-bg-tertiary'
])

const bodyClasses = computed(() => [
  'tower-card-body',
  {
    'p-6': !props.noPadding,
    'p-0': props.noPadding
  }
])

const footerClasses = computed(() => [
  'tower-card-footer',
  'px-6',
  'py-4',
  'border-t',
  'border-tower-border-secondary',
  'bg-tower-bg-tertiary'
])
</script>

<style scoped>
.tower-card {
  /* Custom properties for theme integration */
  --tower-bg-secondary: var(--bg-secondary, #2a2a2a);
  --tower-bg-tertiary: var(--bg-tertiary, #3a3a3a);
  --tower-bg-hover: var(--bg-hover, #404040);
  --tower-border-primary: var(--border-primary, #3a3a3a);
  --tower-border-secondary: var(--border-secondary, #2a2a2a);
  --tower-border-focus: var(--border-focus, #606060);
  --shadow-tower-md: var(--shadow-md, 0 2px 4px rgba(0, 0, 0, 0.4));
}

.tower-card {
  background-color: var(--tower-bg-secondary);
  border-color: var(--tower-border-primary);
}

.tower-card-header,
.tower-card-footer {
  background-color: var(--tower-bg-tertiary);
  border-color: var(--tower-border-secondary);
}

.tower-card.shadow-tower-md {
  box-shadow: var(--shadow-tower-md);
}

.tower-card:hover.hover\:bg-tower-bg-hover {
  background-color: var(--tower-bg-hover);
}

.tower-card:hover.hover\:border-tower-border-focus {
  border-color: var(--tower-border-focus);
}

/* Focus management */
.tower-card:focus-within {
  outline: 2px solid var(--tower-border-focus);
  outline-offset: 2px;
}

/* Responsive design */
@media (max-width: 640px) {
  .tower-card-header,
  .tower-card-body:not(.p-0),
  .tower-card-footer {
    padding-left: 1rem;
    padding-right: 1rem;
  }
}
</style>
<script setup lang="ts">
const props = defineProps<{
  label: string;
  value: number;
  max?: number;
  unit?: string;
}>();

const percentage = props.max ? ((props.value || 0) / props.max) * 100 : (props.value || 0);
</script>

<template>
  <div class="mb-2">
    <div class="flex justify-between text-xs text-mono mb-1">
      <span class="text-dim">{{ label }}</span>
      <span style="color: #666;">
        {{ value }}{{ unit || '' }}
        <span v-if="max" class="text-dim"> / {{ max }}{{ unit || '' }}</span>
      </span>
    </div>
    <div class="bar">
      <div
        class="bar-fill"
        :style="{ width: `${Math.min(percentage, 100)}%` }"
      />
    </div>
  </div>
</template>
<template>
  <div class="echo-msg-content">
    <!-- Main response text with lightweight markdown -->
    <div class="msg-text" v-html="renderedText"></div>

    <!-- Metadata badges row -->
    <div class="msg-badges" v-if="queryType || confidence != null">
      <span class="badge" :class="confidenceClass" :title="`Confidence: ${((confidence ?? 0) * 100).toFixed(0)}%`">
        {{ confidenceLabel }}
      </span>
      <span v-if="queryType" class="badge badge-type">
        {{ queryTypeLabel }}
      </span>
      <span v-if="executionTimeMs" class="badge badge-timing">
        {{ executionTimeMs }}ms
      </span>
    </div>

    <!-- Sources (collapsible) -->
    <details v-if="sources?.length" class="msg-detail">
      <summary class="msg-detail-summary">
        Sources ({{ sources.length }})
      </summary>
      <div class="msg-detail-body">
        <div v-for="(s, i) in sources" :key="i" class="source-item">
          <span class="source-dot"></span>
          {{ s }}
        </div>
      </div>
    </details>

    <!-- Actions taken (collapsible) -->
    <details v-if="actionsTaken?.length" class="msg-detail">
      <summary class="msg-detail-summary">
        Actions ({{ actionsTaken.length }})
      </summary>
      <div class="msg-detail-body">
        <div v-for="(a, i) in actionsTaken" :key="i" class="action-item">
          <span class="action-indicator" :class="a.success ? 'action-ok' : 'action-fail'">
            {{ a.success ? 'OK' : 'FAIL' }}
          </span>
          <span class="action-label">{{ a.action }}</span>
          <pre v-if="a.result" class="action-result">{{ formatResult(a.result) }}</pre>
        </div>
      </div>
    </details>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  text: string
  queryType?: string
  confidence?: number | null
  sources?: string[]
  actionsTaken?: any[]
  executionTimeMs?: number
}>()

/** Lightweight markdown → HTML (bold, inline code, code blocks, lists) */
const renderedText = computed(() => {
  let html = escapeHtml(props.text)

  // Code blocks (```...```)
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="code-block"><code>$2</code></pre>')
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>')
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  // Italic
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>')
  // Line breaks
  html = html.replace(/\n/g, '<br>')
  // Numbered lists (lines starting with "1. ", "2. " etc.)
  html = html.replace(/(^|\<br\>)(\d+)\.\s/g, '$1<span class="list-num">$2.</span> ')
  // Bullet lists
  html = html.replace(/(^|\<br\>)[-*]\s/g, '$1<span class="list-bullet">&bull;</span> ')

  return html
})

const confidenceClass = computed(() => {
  const c = props.confidence ?? 0
  if (c >= 0.7) return 'badge-high'
  if (c >= 0.4) return 'badge-mid'
  return 'badge-low'
})

const confidenceLabel = computed(() => {
  const c = props.confidence ?? 0
  return `${(c * 100).toFixed(0)}%`
})

const queryTypeLabel = computed(() => {
  const labels: Record<string, string> = {
    'self_introspection': 'Self',
    'system_query': 'System',
    'code_query': 'Code',
    'action_request': 'Action',
    'general_knowledge': 'General',
  }
  return labels[props.queryType ?? ''] ?? props.queryType ?? ''
})

function escapeHtml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

function formatResult(result: any): string {
  if (typeof result === 'string') return result
  try {
    return JSON.stringify(result, null, 2)
  } catch {
    return String(result)
  }
}
</script>

<style scoped>
.echo-msg-content {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.msg-text {
  line-height: 1.5;
  word-wrap: break-word;
}

.msg-text :deep(.code-block) {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 0.75rem;
  overflow-x: auto;
  font-size: 0.8rem;
  margin: 0.5rem 0;
}

.msg-text :deep(.inline-code) {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 3px;
  padding: 0.1rem 0.3rem;
  font-size: 0.85em;
}

.msg-text :deep(.list-num),
.msg-text :deep(.list-bullet) {
  color: #58a6ff;
  margin-right: 0.25rem;
}

/* Badges */
.msg-badges {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 0.65rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

.badge-high { background: #238636; color: #fff; }
.badge-mid  { background: #9e6a03; color: #fff; }
.badge-low  { background: #da3633; color: #fff; }

.badge-type {
  background: #1f6feb;
  color: #fff;
}

.badge-timing {
  background: #30363d;
  color: #8b949e;
}

/* Collapsible details */
.msg-detail {
  border-top: 1px solid #21262d;
  padding-top: 0.4rem;
}

.msg-detail-summary {
  cursor: pointer;
  font-size: 0.7rem;
  color: #8b949e;
  user-select: none;
}

.msg-detail-summary:hover {
  color: #58a6ff;
}

.msg-detail-body {
  margin-top: 0.4rem;
  font-size: 0.75rem;
}

/* Sources */
.source-item {
  display: flex;
  align-items: baseline;
  gap: 0.4rem;
  padding: 0.15rem 0;
  color: #c9d1d9;
}

.source-dot {
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: #58a6ff;
  flex-shrink: 0;
  margin-top: 0.4em;
}

/* Actions */
.action-item {
  padding: 0.25rem 0;
  display: flex;
  align-items: baseline;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.action-indicator {
  padding: 1px 5px;
  border-radius: 3px;
  font-size: 0.6rem;
  font-weight: 700;
}

.action-ok   { background: #238636; color: #fff; }
.action-fail { background: #da3633; color: #fff; }

.action-label {
  color: #c9d1d9;
}

.action-result {
  width: 100%;
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 4px;
  padding: 0.4rem;
  font-size: 0.7rem;
  overflow-x: auto;
  margin-top: 0.25rem;
}
</style>

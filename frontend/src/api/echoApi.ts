import axios from 'axios';
import type { SystemHealth } from '@/types/echo';

// Always use /api/echo for all endpoints
const api = axios.create({
  baseURL: '/api/echo',
  timeout: 60000,
});

// Health (existing)
export const healthApi = {
  getFull: () => api.get<SystemHealth>('/health/'),
  getQuick: () => api.get<{ status: string; services: Record<string, string> }>('/health/quick'),
  getResources: () => api.get('/health/resources'),
  getService: (name: string) => api.get(`/health/services/${name}`),
  getMetrics: () => api.get('/health/metrics'),
};

// Memory
export const memoryApi = {
  search: (query: string, limit = 10) =>
    api.post('/memory/search', { query, limit }),
  ingest: (content: string, metadata?: Record<string, any>) =>
    api.post('/memory/ingest', { content, metadata }),
  status: () => api.get('/memory/status'),
  health: () => api.get('/memory/health'),
};

// Intelligence
export const intelligenceApi = {
  think: (query: string, context?: string) =>
    api.post('/intelligence/think', { query, context }),
  // Note: Streaming not supported in browser axios - would need SSE or WebSockets
  thinkStream: (query: string) =>
    api.post('/intelligence/think', { query }), // Removed stream for now
  compareKnowledge: (topic: string) =>
    api.post('/intelligence/compare-knowledge', { topic }),
  knowledgeMap: () => api.get('/intelligence/knowledge-map'),
  testUnderstanding: (query: string) =>
    api.post('/intelligence/test-understanding', { query }),
  thinkingLog: () => api.get('/intelligence/thinking-log'),
};

// Ask (main interface)
export const askApi = {
  ask: (question: string) => api.post('/ask', { question }),
  askGet: (question: string) => api.get('/ask', { params: { question } }),
  // Note: Streaming not supported in browser axios - use regular endpoint
  stream: (question: string) =>
    api.post('/ask', { question }), // Using regular endpoint instead of stream
};

// Conversations
export const conversationsApi = {
  search: (query: string, limit = 10) =>
    api.post('/conversations/search', { query, limit }),
  health: () => api.get('/conversations/health'),
  test: () => api.get('/conversations/test'),
};

// System
export const systemApi = {
  status: () => api.get('/system/status'),
  ready: () => api.get('/system/ready'),
  alive: () => api.get('/system/alive'),
  metrics: () => api.get('/system/metrics'),
  metricsHistory: () => api.get('/system/metrics/history'),
  diagnostics: () => api.get('/system/diagnostics'),
  diagnosticsDatabase: () => api.get('/system/diagnostics/database'),
  diagnosticsServices: () => api.get('/system/diagnostics/services'),
  logs: () => api.get('/system/status/logs'),
  operationsStatus: () => api.get('/system/operations/status'),
  operationsJobs: () => api.get('/system/operations/jobs'),
  healthDetailed: () => api.get('/system/health/detailed'),
};

// Echo
export const echoApi = {
  health: () => api.get('/health'),
  query: (query: string) => api.post('/query', { query }),
  brain: () => api.get('/brain'),
  status: () => api.get('/status'),
  modelsList: () => api.get('/models/list'),
};

// MCP
export const mcpApi = {
  call: (method: string, params: any) =>
    api.post('/mcp', { jsonrpc: '2.0', method, params, id: Date.now() }),
  searchMemory: (query: string, limit = 5) =>
    api.post('/mcp', {
      jsonrpc: '2.0',
      method: 'tools/call',
      params: { name: 'search_memory', arguments: { query, limit } },
      id: Date.now(),
    }),
  getFacts: (topic: string) =>
    api.post('/mcp', {
      jsonrpc: '2.0',
      method: 'tools/call',
      params: { name: 'get_facts', arguments: { topic } },
      id: Date.now(),
    }),
  health: () => api.get('/mcp/health'),
};

// Self-test
export const selfTestApi = {
  quick: () => api.get('/self-test/quick'),
  full: () => api.get('/self-test/run'),
};

// Moltbook
export const moltbookApi = {
  health: () => api.get('/moltbook/health'),
  status: () => api.get('/moltbook/status'),
  test: () => api.get('/moltbook/test'),
  profile: () => api.get('/moltbook/profile'),
  establish: (params: any) => api.post('/moltbook/establish', params),
  share: (content: string) => api.post('/moltbook/share', { content }),
  establishmentStatus: () => api.get('/moltbook/establishment/status'),
};

// Reasoning
export const reasoningApi = {
  search: (query: string, limit = 10) =>
    api.post('/search', { query, limit }),
  health: () => api.get('/reasoning/health'),
};

// Generic endpoint tester
export const testEndpoint = (method: string, path: string, data?: any) => {
  const config = { method, url: path, data };
  return api.request(config);
};

// Get all available endpoints for testing
export const getAllEndpoints = () => {
  return [
    // Health
    { method: 'GET', path: '/health/', desc: 'Full health status' },
    { method: 'GET', path: '/health/quick', desc: 'Quick health check' },
    { method: 'GET', path: '/health/resources', desc: 'Resource stats' },
    { method: 'GET', path: '/health/metrics', desc: 'Prometheus metrics' },

    // System
    { method: 'GET', path: '/system/status', desc: 'System status' },
    { method: 'GET', path: '/system/ready', desc: 'Readiness check' },
    { method: 'GET', path: '/system/alive', desc: 'Liveness probe' },
    { method: 'GET', path: '/system/metrics', desc: 'System metrics' },
    { method: 'GET', path: '/system/diagnostics', desc: 'Full diagnostics' },
    { method: 'GET', path: '/system/status/logs', desc: 'Recent logs' },

    // Memory
    { method: 'POST', path: '/memory/search', desc: 'Memory search', body: '{"query":"test","limit":5}' },
    { method: 'POST', path: '/memory/ingest', desc: 'Ingest memory', body: '{"content":"test content"}' },
    { method: 'GET', path: '/memory/status', desc: 'Memory status' },
    { method: 'GET', path: '/memory/health', desc: 'Memory health' },

    // Ask
    { method: 'POST', path: '/ask', desc: 'Ask question', body: '{"question":"What is Echo Brain?"}' },
    { method: 'GET', path: '/ask', desc: 'Ask (GET)', params: '?question=test' },

    // Intelligence
    { method: 'POST', path: '/intelligence/think', desc: 'Think', body: '{"query":"test"}' },
    { method: 'GET', path: '/intelligence/knowledge-map', desc: 'Knowledge map' },
    { method: 'GET', path: '/intelligence/thinking-log', desc: 'Thinking log' },

    // Conversations
    { method: 'POST', path: '/conversations/search', desc: 'Search conversations', body: '{"query":"test"}' },
    { method: 'GET', path: '/conversations/health', desc: 'Conversations health' },
    { method: 'GET', path: '/conversations/test', desc: 'Test conversations' },

    // Echo
    { method: 'GET', path: '/health', desc: 'Echo health' },
    { method: 'POST', path: '/query', desc: 'Echo query', body: '{"query":"test"}' },
    { method: 'GET', path: '/brain', desc: 'Echo brain info' },
    { method: 'GET', path: '/status', desc: 'Echo status' },
    { method: 'GET', path: '/models/list', desc: 'Available models' },

    // MCP
    { method: 'GET', path: '/mcp/health', desc: 'MCP health' },
    { method: 'POST', path: '/mcp', desc: 'MCP search', body: '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":1}' },

    // Self-test
    { method: 'GET', path: '/self-test/quick', desc: 'Quick self-test' },
    { method: 'GET', path: '/self-test/run', desc: 'Full self-test' },

    // Moltbook
    { method: 'GET', path: '/moltbook/health', desc: 'Moltbook health' },
    { method: 'GET', path: '/moltbook/status', desc: 'Moltbook status' },
    { method: 'GET', path: '/moltbook/profile', desc: 'Moltbook profile' },

    // Search
    { method: 'GET', path: '/search', desc: 'Search (GET)', params: '?q=test&limit=5' },
    { method: 'POST', path: '/search', desc: 'Search (POST)', body: '{"query":"test","limit":5}' },

    // Root
    { method: 'GET', path: '/', desc: 'Service info' },
    { method: 'GET', path: '/health', desc: 'Root health check' },
  ];
};
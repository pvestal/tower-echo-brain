import axios from 'axios';
import type { EchoHealthResponse } from '@/types/echo';

// Always use /api/echo for all endpoints
const api = axios.create({
  baseURL: '/api/echo',
  timeout: 60000,
});

const workersApi = axios.create({
  baseURL: '/api/workers',
  timeout: 30000,
});

// Health — all calls go through `api` so nginx proxies to /api/echo/*
export const healthApi = {
  getFull: () => api.get<EchoHealthResponse>('/health'),
  getQuick: () => api.get<{ status: string; services: Record<string, string> }>('/health'),
  getResources: () => api.get('/memory/status'),
  getService: (name: string) => api.get(`/intelligence/service/${name}`),
  getMetrics: () => api.get('/memory/status'),
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
// Backend routes: /intelligence/think, /intelligence/map, /intelligence/status, /intelligence/query
export const intelligenceApi = {
  think: (query: string, context?: string) =>
    api.post('/intelligence/think', { query, context }),
  knowledgeMap: () => api.get('/intelligence/map'),
  status: () => api.get('/intelligence/status'),
  query: (query: string) =>
    api.post('/intelligence/query', { query }),
};

// Ask (main interface)
export const askApi = {
  ask: (question: string) => api.post('/ask', { question }),
  askGet: (question: string) => api.get('/ask', { params: { question } }),
};

// Conversations
// Backend route: /conversations/search only
export const conversationsApi = {
  search: (query: string, limit = 10) =>
    api.post('/conversations/search', { query, limit }),
};

// System — each button calls a distinct endpoint
export const systemApi = {
  status: () => api.get('/intelligence/status'),
  ready: () => api.get('/health'),
  alive: () => api.get('/status'),
  metrics: () => api.get('/memory/status'),
  diagnostics: () => api.get('/diagnostic'),
  diagnosticsDatabase: () => api.get('/memory/health'),
  diagnosticsServices: () => api.get('/diagnostic/quick'),
  logs: (params?: { level?: string; service?: string }) =>
    api.get('/system/logs', { params }),
  operationsStatus: () => api.get('/system/dashboard'),
  operationsJobs: () => workersApi.get('/status'),
  healthDetailed: () => api.get('/system/resources'),
};

// Echo
export const echoApi = {
  health: () => api.get('/health'),
  query: (query: string) => api.post('/query', { query }),
  brain: () => api.get('/brain'),
  status: () => api.get('/status'),
  modelsList: () => api.get('/models'),
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
  health: () => api.get('/health'),
};

// Self-test
export const selfTestApi = {
  quick: () => api.get('/self-test/quick'),
  full: () => api.get('/self-test/run'),
};

// Search
export const searchApi = {
  search: (query: string, limit = 10) =>
    api.get('/search', { params: { q: query, limit } }),
};

// Calendar (separate base URL — /api/calendar)
const calendarAxios = axios.create({ baseURL: '/api/calendar', timeout: 30000 });

export const calendarApi = {
  getCalendars: () => calendarAxios.get('/calendars'),
  getMonthEvents: (year: number, month: number) =>
    calendarAxios.get('/events/month', { params: { year, month } }),
  getStatus: () => calendarAxios.get('/status'),
};

// Google Ingest (separate base URL — /api/google/ingest)
const googleIngestAxios = axios.create({ baseURL: '/api/google/ingest', timeout: 120000 });

export const googleIngestApi = {
  ingestCalendar: () => googleIngestAxios.post('/calendar'),
  ingestEmail: (query?: string) =>
    googleIngestAxios.post('/email', null, { params: query ? { query } : undefined }),
  ingestDrive: () => googleIngestAxios.post('/drive'),
  ingestAll: () => googleIngestAxios.post('/all'),
  stats: () => googleIngestAxios.get('/stats'),
};

// Generic endpoint tester
export const testEndpoint = (method: string, path: string, data?: any) => {
  const config = { method, url: path, data };
  return api.request(config);
};

// Get all available endpoints for testing
// These paths are relative to the api baseURL (/api/echo)
export const getAllEndpoints = () => {
  return [
    // Echo Core
    { method: 'GET', path: '/health', desc: 'Echo health' },
    { method: 'GET', path: '/status', desc: 'Echo status' },
    { method: 'GET', path: '/brain', desc: 'Echo brain info' },
    { method: 'GET', path: '/models', desc: 'Available models' },
    { method: 'POST', path: '/query', desc: 'Echo query', body: '{"query":"test"}' },
    { method: 'POST', path: '/ask', desc: 'Ask question', body: '{"question":"What is Echo Brain?"}' },

    // Memory
    { method: 'GET', path: '/memory/status', desc: 'Memory status' },
    { method: 'GET', path: '/memory/health', desc: 'Memory health' },
    { method: 'POST', path: '/memory/search', desc: 'Memory search', body: '{"query":"test","limit":5}' },
    { method: 'POST', path: '/memory/ingest', desc: 'Ingest memory', body: '{"content":"test content"}' },

    // Intelligence
    { method: 'GET', path: '/intelligence/status', desc: 'Intelligence status' },
    { method: 'GET', path: '/intelligence/map', desc: 'Knowledge map' },
    { method: 'POST', path: '/intelligence/think', desc: 'Think', body: '{"query":"test"}' },
    { method: 'POST', path: '/intelligence/query', desc: 'Intelligence query', body: '{"query":"test"}' },

    // Conversations
    { method: 'POST', path: '/conversations/search', desc: 'Search conversations', body: '{"query":"test"}' },

    // MCP
    { method: 'POST', path: '/mcp', desc: 'MCP call', body: '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":1}' },

    // Self-test
    { method: 'GET', path: '/self-test/quick', desc: 'Quick self-test' },
    { method: 'GET', path: '/self-test/run', desc: 'Full self-test' },

    // System
    { method: 'GET', path: '/system/logs', desc: 'System logs' },
    { method: 'GET', path: '/system/resources', desc: 'System resources' },
    { method: 'GET', path: '/system/dashboard', desc: 'System dashboard' },

    // Diagnostic
    { method: 'GET', path: '/diagnostic', desc: 'Full diagnostic' },
    { method: 'GET', path: '/diagnostic/quick', desc: 'Quick diagnostic' },

    // Search
    { method: 'GET', path: '/search?q=test&limit=5', desc: 'Search' },

    // Voice
    { method: 'GET', path: '/voice/status', desc: 'Voice service status' },
    { method: 'GET', path: '/voice/voices', desc: 'Available TTS voices' },
  ];
};

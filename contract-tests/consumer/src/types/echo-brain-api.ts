/**
 * Echo Brain API Contract Types
 * 
 * These types define what the Vue frontend expects from the FastAPI backend.
 * They serve as the source of truth for consumer contract tests.
 * 
 * IMPORTANT: When the frontend needs a new field or endpoint, add it here
 * FIRST, then write the consumer test, then update the backend.
 */

// ─── Health & Status ───────────────────────────────────────────────

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  uptime_seconds: number;
  services: ServiceStatus;
}

export interface ServiceStatus {
  database: ComponentHealth;
  vector_store: ComponentHealth;
  ollama: ComponentHealth;
}

export interface ComponentHealth {
  status: 'up' | 'down' | 'degraded';
  latency_ms: number;
}

// ─── Knowledge Query ───────────────────────────────────────────────

export interface QueryRequest {
  query: string;
  top_k?: number;          // defaults to 5
  min_score?: number;       // similarity threshold 0-1
  source_filter?: string;   // e.g., 'claude_conversations', 'documents'
}

export interface QueryResponse {
  results: QueryResult[];
  query_time_ms: number;
  model_used: string;       // embedding model name
  total_matches: number;
}

export interface QueryResult {
  id: string;
  content: string;
  score: number;            // similarity score 0-1
  source: string;
  metadata: Record<string, string | number | boolean>;
  created_at: string;       // ISO 8601
}

// ─── Memory Management ────────────────────────────────────────────

export interface MemoryEntry {
  id: string;
  content: string;
  category: string;
  source: string;
  created_at: string;
  updated_at: string;
  embedding_model: string;
}

export interface MemoryListResponse {
  memories: MemoryEntry[];
  total: number;
  page: number;
  page_size: number;
}

export interface MemoryCreateRequest {
  content: string;
  category: string;
  source?: string;
  metadata?: Record<string, string | number | boolean>;
}

export interface MemoryCreateResponse {
  id: string;
  status: 'created';
  embedded: boolean;
}

// ─── Ingestion Status ─────────────────────────────────────────────

export interface IngestionStatusResponse {
  running: boolean;
  last_run: string | null;         // ISO 8601 or null if never run
  last_run_status: 'success' | 'failed' | 'partial' | null;
  documents_processed: number;
  documents_failed: number;
  next_scheduled: string | null;   // ISO 8601
}

// ─── Voice ────────────────────────────────────────────────────────

export interface VoiceStatusResponse {
  initialized: boolean;
  stt_available: boolean;
  tts_available: boolean;
}

export interface VoiceVoicesResponse {
  installed: VoiceInfo[];
  suggested: VoiceSuggestion[];
  models_dir: string;
}

export interface VoiceInfo {
  name: string;
  path: string;
  size_mb: number;
  config_exists: boolean;
}

export interface VoiceSuggestion {
  name: string;
  quality: string;
  description: string;
}

// ─── Error Shape ──────────────────────────────────────────────────

export interface ApiError {
  detail: string;
  error_code?: string;
  timestamp: string;
}

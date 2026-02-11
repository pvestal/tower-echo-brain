/**
 * Echo Brain API Client
 * 
 * Encapsulates all HTTP communication with the FastAPI backend.
 * Contract tests verify that the backend still returns data matching
 * what this client expects to parse.
 * 
 * Usage in Vue components:
 *   const client = new EchoBrainClient('http://tower:8100');
 *   const health = await client.getHealth();
 */

import type {
  HealthResponse,
  QueryRequest,
  QueryResponse,
  MemoryListResponse,
  MemoryCreateRequest,
  MemoryCreateResponse,
  IngestionStatusResponse,
  VoiceStatusResponse,
  VoiceVoicesResponse,
  ApiError
} from '../types/echo-brain-api';

export class EchoBrainApiError extends Error {
  constructor(
    public statusCode: number,
    public detail: string,
    public errorCode?: string
  ) {
    super(`[${statusCode}] ${detail}`);
    this.name = 'EchoBrainApiError';
  }
}

export class EchoBrainClient {
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor(baseUrl: string, authToken?: string) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.headers = {
      'Content-Type': 'application/json',
      ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
    };
  }

  // ─── Health ────────────────────────────────────────────────────

  /** Check backend health and component status */
  async getHealth(): Promise<HealthResponse> {
    return this.get<HealthResponse>('/api/v1/health');
  }

  // ─── Knowledge Query ──────────────────────────────────────────

  /** Search the vector store for relevant knowledge */
  async query(request: QueryRequest): Promise<QueryResponse> {
    return this.post<QueryResponse>('/api/v1/query', request);
  }

  // ─── Memory ───────────────────────────────────────────────────

  /** List memories with pagination */
  async listMemories(page = 1, pageSize = 20): Promise<MemoryListResponse> {
    return this.get<MemoryListResponse>(
      `/api/v1/memories?page=${page}&page_size=${pageSize}`
    );
  }

  /** Create a new memory entry */
  async createMemory(request: MemoryCreateRequest): Promise<MemoryCreateResponse> {
    return this.post<MemoryCreateResponse>('/api/v1/memories', request);
  }

  /** Delete a memory by ID */
  async deleteMemory(id: string): Promise<void> {
    await this.delete(`/api/v1/memories/${id}`);
  }

  // ─── Voice ──────────────────────────────────────────────────────

  /** Get voice service status */
  async getVoiceStatus(): Promise<VoiceStatusResponse> {
    return this.get<VoiceStatusResponse>('/api/echo/voice/status');
  }

  /** List available TTS voices */
  async getVoices(): Promise<VoiceVoicesResponse> {
    return this.get<VoiceVoicesResponse>('/api/echo/voice/voices');
  }

  // ─── Ingestion ────────────────────────────────────────────────

  /** Get current ingestion pipeline status */
  async getIngestionStatus(): Promise<IngestionStatusResponse> {
    return this.get<IngestionStatusResponse>('/api/v1/ingestion/status');
  }

  // ─── HTTP Internals ───────────────────────────────────────────

  private async get<T>(path: string): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method: 'GET',
      headers: this.headers
    });
    return this.handleResponse<T>(response);
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(body)
    });
    return this.handleResponse<T>(response);
  }

  private async delete(path: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method: 'DELETE',
      headers: this.headers
    });
    if (!response.ok) {
      await this.throwApiError(response);
    }
  }

  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      await this.throwApiError(response);
    }
    return response.json() as Promise<T>;
  }

  private async throwApiError(response: Response): Promise<never> {
    let detail = 'Unknown error';
    let errorCode: string | undefined;

    try {
      const body: ApiError = await response.json();
      detail = body.detail;
      errorCode = body.error_code;
    } catch {
      detail = response.statusText;
    }

    throw new EchoBrainApiError(response.status, detail, errorCode);
  }
}

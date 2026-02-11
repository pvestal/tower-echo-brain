/**
 * Echo Brain Consumer Contract Tests
 * 
 * These tests define what the Vue frontend EXPECTS from the FastAPI backend.
 * Pact spins up a mock server that returns the expected responses, verifies
 * the client parses them correctly, then generates a contract JSON file.
 * 
 * The provider (FastAPI) side then replays these interactions to prove
 * it still satisfies the contract.
 * 
 * Run: npm test
 */

import { describe, it, beforeAll, afterAll, afterEach, expect } from 'vitest';
import { PactV4, MatchersV3 } from '@pact-foundation/pact';
import path from 'path';
import { EchoBrainClient } from '../src/api/echo-brain-client';

const {
  like,           // matches structure, not exact values
  eachLike,       // array where each element matches the example
  string,         // any string
  integer,        // any integer
  decimal,        // any decimal number
  boolean,        // any boolean
  regex,          // matches a regex pattern
  datetime        // ISO 8601 datetime string
} = MatchersV3;

// ─── Pact Setup ──────────────────────────────────────────────────

const provider = new PactV4({
  consumer: 'EchoBrainFrontend',
  provider: 'EchoBrainAPI',
  dir: path.resolve(__dirname, '../../contracts'),  // output contract here
  logLevel: 'warn'
});

// ─── Health Endpoint ─────────────────────────────────────────────

describe('Echo Brain API Contract', () => {

  describe('GET /api/v1/health', () => {
    it('returns system health with component status', async () => {
      await provider
        .addInteraction()
        .given('the system is running')
        .uponReceiving('a health check request')
        .withRequest('GET', '/api/v1/health')
        .willRespondWith(200, (builder) => {
          builder
            .headers({ 'Content-Type': 'application/json' })
            .jsonBody({
              status: regex('healthy|degraded|unhealthy', 'healthy'),
              version: string('1.2.0'),
              uptime_seconds: integer(86400),
              services: {
                database: {
                  status: regex('up|down|degraded', 'up'),
                  latency_ms: decimal(2.5)
                },
                vector_store: {
                  status: regex('up|down|degraded', 'up'),
                  latency_ms: decimal(5.1)
                },
                ollama: {
                  status: regex('up|down|degraded', 'up'),
                  latency_ms: decimal(12.3)
                }
              }
            });
        })
        .executeTest(async (mockServer) => {
          const client = new EchoBrainClient(mockServer.url);
          const health = await client.getHealth();

          // Verify the client correctly parses the response
          expect(health.status).toBeDefined();
          expect(health.services.database.status).toBeDefined();
          expect(health.services.vector_store.status).toBeDefined();
          expect(health.services.ollama.status).toBeDefined();
          expect(typeof health.uptime_seconds).toBe('number');
        });
    });

    it('returns degraded status when a component is down', async () => {
      await provider
        .addInteraction()
        .given('the vector store is unreachable')
        .uponReceiving('a health check when vector store is down')
        .withRequest('GET', '/api/v1/health')
        .willRespondWith(200, (builder) => {
          builder
            .headers({ 'Content-Type': 'application/json' })
            .jsonBody({
              status: string('degraded'),
              version: string('1.2.0'),
              uptime_seconds: integer(86400),
              services: {
                database: {
                  status: string('up'),
                  latency_ms: decimal(2.5)
                },
                vector_store: {
                  status: string('down'),
                  latency_ms: decimal(0)
                },
                ollama: {
                  status: string('up'),
                  latency_ms: decimal(12.3)
                }
              }
            });
        })
        .executeTest(async (mockServer) => {
          const client = new EchoBrainClient(mockServer.url);
          const health = await client.getHealth();

          expect(health.status).toBe('degraded');
          expect(health.services.vector_store.status).toBe('down');
        });
    });
  });

  // ─── Query Endpoint ──────────────────────────────────────────

  describe('POST /api/v1/query', () => {
    it('returns search results for a knowledge query', async () => {
      await provider
        .addInteraction()
        .given('the vector store has indexed documents')
        .uponReceiving('a knowledge query request')
        .withRequest('POST', '/api/v1/query', (builder) => {
          builder
            .headers({ 'Content-Type': 'application/json' })
            .jsonBody({
              query: string('How is the Victron MultiPlus configured?'),
              top_k: integer(5)
            });
        })
        .willRespondWith(200, (builder) => {
          builder
            .headers({ 'Content-Type': 'application/json' })
            .jsonBody({
              results: eachLike({
                id: string('mem_abc123'),
                content: string('The Victron MultiPlus II is configured for...'),
                score: decimal(0.87),
                source: string('claude_conversations'),
                metadata: like({
                  file: string('conversation_2025-01-15.jsonl'),
                  chunk_index: integer(3)
                }),
                created_at: datetime('yyyy-MM-dd\'T\'HH:mm:ss\'Z\'', '2025-01-15T10:30:00Z')
              }),
              query_time_ms: decimal(45.2),
              model_used: string('nomic-embed-text'),
              total_matches: integer(12)
            });
        })
        .executeTest(async (mockServer) => {
          const client = new EchoBrainClient(mockServer.url);
          const response = await client.query({
            query: 'How is the Victron MultiPlus configured?',
            top_k: 5
          });

          expect(response.results.length).toBeGreaterThan(0);
          expect(response.results[0].score).toBeGreaterThanOrEqual(0);
          expect(response.results[0].score).toBeLessThanOrEqual(1);
          expect(response.model_used).toBeDefined();
          expect(typeof response.query_time_ms).toBe('number');
        });
    });

    it('returns empty results when no matches found', async () => {
      await provider
        .addInteraction()
        .given('the vector store has no matching documents')
        .uponReceiving('a query with no results')
        .withRequest('POST', '/api/v1/query', (builder) => {
          builder
            .headers({ 'Content-Type': 'application/json' })
            .jsonBody({
              query: string('completely unrelated gibberish xyz123'),
              top_k: integer(5),
              min_score: decimal(0.8)
            });
        })
        .willRespondWith(200, (builder) => {
          builder
            .headers({ 'Content-Type': 'application/json' })
            .jsonBody({
              results: [],   // empty array, not eachLike
              query_time_ms: decimal(12.1),
              model_used: string('nomic-embed-text'),
              total_matches: integer(0)
            });
        })
        .executeTest(async (mockServer) => {
          const client = new EchoBrainClient(mockServer.url);
          const response = await client.query({
            query: 'completely unrelated gibberish xyz123',
            top_k: 5,
            min_score: 0.8
          });

          expect(response.results).toEqual([]);
          expect(response.total_matches).toBe(0);
        });
    });
  });

  // ─── Memory Endpoints ─────────────────────────────────────────

  describe('GET /api/v1/memories', () => {
    it('returns paginated memory list', async () => {
      await provider
        .addInteraction()
        .given('memories exist in the database')
        .uponReceiving('a request to list memories')
        .withRequest('GET', '/api/v1/memories', (builder) => {
          builder.query({ page: '1', page_size: '20' });
        })
        .willRespondWith(200, (builder) => {
          builder
            .headers({ 'Content-Type': 'application/json' })
            .jsonBody({
              memories: eachLike({
                id: string('mem_001'),
                content: string('Tower server runs 28 microservices...'),
                category: string('infrastructure'),
                source: string('claude_conversations'),
                created_at: datetime('yyyy-MM-dd\'T\'HH:mm:ss\'Z\'', '2025-01-10T08:00:00Z'),
                updated_at: datetime('yyyy-MM-dd\'T\'HH:mm:ss\'Z\'', '2025-01-10T08:00:00Z'),
                embedding_model: string('nomic-embed-text')
              }),
              total: integer(150),
              page: integer(1),
              page_size: integer(20)
            });
        })
        .executeTest(async (mockServer) => {
          const client = new EchoBrainClient(mockServer.url);
          const response = await client.listMemories(1, 20);

          expect(response.memories.length).toBeGreaterThan(0);
          expect(response.total).toBeGreaterThanOrEqual(response.memories.length);
          expect(response.page).toBe(1);
          expect(response.memories[0].embedding_model).toBeDefined();
        });
    });
  });

  describe('POST /api/v1/memories', () => {
    it('creates a new memory entry', async () => {
      await provider
        .addInteraction()
        .given('the system can accept new memories')
        .uponReceiving('a request to create a memory')
        .withRequest('POST', '/api/v1/memories', (builder) => {
          builder
            .headers({ 'Content-Type': 'application/json' })
            .jsonBody({
              content: string('New configuration note for Qdrant setup'),
              category: string('infrastructure'),
              source: string('manual_entry')
            });
        })
        .willRespondWith(201, (builder) => {
          builder
            .headers({ 'Content-Type': 'application/json' })
            .jsonBody({
              id: string('mem_new_001'),
              status: string('created'),
              embedded: boolean(true)
            });
        })
        .executeTest(async (mockServer) => {
          const client = new EchoBrainClient(mockServer.url);
          const response = await client.createMemory({
            content: 'New configuration note for Qdrant setup',
            category: 'infrastructure',
            source: 'manual_entry'
          });

          expect(response.id).toBeDefined();
          expect(response.status).toBe('created');
          expect(typeof response.embedded).toBe('boolean');
        });
    });
  });

  // ─── Ingestion Status ─────────────────────────────────────────

  describe('GET /api/v1/ingestion/status', () => {
    it('returns current ingestion pipeline status', async () => {
      await provider
        .addInteraction()
        .given('an ingestion has completed previously')
        .uponReceiving('a request for ingestion status')
        .withRequest('GET', '/api/v1/ingestion/status')
        .willRespondWith(200, (builder) => {
          builder
            .headers({ 'Content-Type': 'application/json' })
            .jsonBody({
              running: boolean(false),
              last_run: datetime('yyyy-MM-dd\'T\'HH:mm:ss\'Z\'', '2025-02-10T03:00:00Z'),
              last_run_status: regex('success|failed|partial', 'success'),
              documents_processed: integer(347),
              documents_failed: integer(2),
              next_scheduled: datetime('yyyy-MM-dd\'T\'HH:mm:ss\'Z\'', '2025-02-11T03:00:00Z')
            });
        })
        .executeTest(async (mockServer) => {
          const client = new EchoBrainClient(mockServer.url);
          const status = await client.getIngestionStatus();

          expect(typeof status.running).toBe('boolean');
          expect(status.last_run).toBeDefined();
          expect(typeof status.documents_processed).toBe('number');
          expect(typeof status.documents_failed).toBe('number');
        });
    });

    it('returns null fields when ingestion has never run', async () => {
      await provider
        .addInteraction()
        .given('no ingestion has ever run')
        .uponReceiving('ingestion status when never run')
        .withRequest('GET', '/api/v1/ingestion/status')
        .willRespondWith(200, (builder) => {
          builder
            .headers({ 'Content-Type': 'application/json' })
            .jsonBody({
              running: boolean(false),
              last_run: null,
              last_run_status: null,
              documents_processed: integer(0),
              documents_failed: integer(0),
              next_scheduled: null
            });
        })
        .executeTest(async (mockServer) => {
          const client = new EchoBrainClient(mockServer.url);
          const status = await client.getIngestionStatus();

          expect(status.running).toBe(false);
          expect(status.last_run).toBeNull();
          expect(status.last_run_status).toBeNull();
          expect(status.documents_processed).toBe(0);
        });
    });
  });

  // ─── Voice API ──────────────────────────────────────────────────

  describe('Voice API', () => {
    describe('GET /api/echo/voice/status', () => {
      it('returns voice service health', async () => {
        await provider
          .addInteraction()
          .given('the voice service is running')
          .uponReceiving('a voice status request')
          .withRequest('GET', '/api/echo/voice/status')
          .willRespondWith(200, (builder) => {
            builder
              .headers({ 'Content-Type': 'application/json' })
              .jsonBody({
                initialized: boolean(true),
                stt_available: boolean(true),
                tts_available: boolean(true),
              });
          })
          .executeTest(async (mockServer) => {
            const client = new EchoBrainClient(mockServer.url);
            const status = await client.getVoiceStatus();

            expect(typeof status.initialized).toBe('boolean');
            expect(typeof status.stt_available).toBe('boolean');
            expect(typeof status.tts_available).toBe('boolean');
          });
      });
    });

    describe('GET /api/echo/voice/voices', () => {
      it('returns available TTS voices', async () => {
        await provider
          .addInteraction()
          .given('the voice service is running')
          .uponReceiving('a request to list available voices')
          .withRequest('GET', '/api/echo/voice/voices')
          .willRespondWith(200, (builder) => {
            builder
              .headers({ 'Content-Type': 'application/json' })
              .jsonBody({
                installed: eachLike({
                  name: string('en_US-lessac-medium'),
                  path: string('/opt/tower-echo-brain/models/voice/piper/en_US-lessac-medium.onnx'),
                  size_mb: decimal(75.2),
                  config_exists: boolean(true),
                }),
                suggested: eachLike({
                  name: string('en_US-lessac-high'),
                  quality: string('high'),
                  description: string('High quality US English'),
                }),
                models_dir: string('/opt/tower-echo-brain/models/voice/piper'),
              });
          })
          .executeTest(async (mockServer) => {
            const client = new EchoBrainClient(mockServer.url);
            const voices = await client.getVoices();

            expect(voices.installed.length).toBeGreaterThan(0);
            expect(voices.installed[0].name).toBeDefined();
            expect(voices.suggested.length).toBeGreaterThan(0);
            expect(voices.models_dir).toBeDefined();
          });
      });
    });
  });

  // ─── Error Contract ───────────────────────────────────────────

  describe('Error responses', () => {
    it('returns structured error for invalid query', async () => {
      await provider
        .addInteraction()
        .given('the system is running')
        .uponReceiving('a query request with empty body')
        .withRequest('POST', '/api/v1/query', (builder) => {
          builder
            .headers({ 'Content-Type': 'application/json' })
            .jsonBody({});
        })
        .willRespondWith(422, (builder) => {
          builder
            .headers({ 'Content-Type': 'application/json' })
            .jsonBody({
              detail: string('query field is required'),
              error_code: string('VALIDATION_ERROR'),
              timestamp: datetime('yyyy-MM-dd\'T\'HH:mm:ss\'Z\'', '2025-02-11T12:00:00Z')
            });
        })
        .executeTest(async (mockServer) => {
          const client = new EchoBrainClient(mockServer.url);

          try {
            await client.query({} as any);
            expect.fail('Should have thrown');
          } catch (error: any) {
            expect(error.statusCode).toBe(422);
            expect(error.detail).toBeDefined();
          }
        });
    });
  });
});

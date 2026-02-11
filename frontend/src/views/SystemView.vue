<template>
  <div class="system-view">
    <!-- Echo Brain Core Functions -->
    <div class="card mb-3">
      <h3>Echo Brain Core</h3>
      <div class="grid grid-3 gap-2">
        <button @click="callEndpoint('echoHealth')" class="btn">
          Echo Health
        </button>
        <button @click="callEndpoint('echoBrain')" class="btn">
          Brain Info
        </button>
        <button @click="callEndpoint('echoStatus')" class="btn">
          Echo Status
        </button>
        <button @click="callEndpoint('echoModels')" class="btn">
          List Models
        </button>
        <button @click="queryEcho" class="btn btn-primary">
          Query Echo
        </button>
      </div>
    </div>

    <!-- Intelligence Functions -->
    <div class="card mb-3">
      <h3>Intelligence Engine</h3>
      <div class="mb-3">
        <input v-model="thinkQuery" placeholder="Query for thinking..." class="w-full mb-2" />
        <div class="grid grid-3 gap-2">
          <button @click="think" class="btn btn-primary">
            Think
          </button>
          <button @click="callEndpoint('knowledgeMap')" class="btn">
            Knowledge Map
          </button>
          <button @click="callEndpoint('intelligenceStatus')" class="btn">
            Intelligence Status
          </button>
        </div>
      </div>
    </div>

    <!-- Conversations -->
    <div class="card mb-3">
      <h3>Conversations Database</h3>
      <div class="mb-3">
        <input v-model="conversationQuery" placeholder="Search conversations..." class="w-full mb-2" />
        <button @click="searchConversations" class="btn btn-primary w-full">
          Search Conversations
        </button>
      </div>
    </div>

    <!-- MCP Server -->
    <div class="card mb-3">
      <h3>MCP Server (Model Context Protocol)</h3>
      <div class="mb-3">
        <input v-model="mcpQuery" placeholder="MCP search query..." class="w-full mb-2" />
        <div class="grid grid-2 gap-2">
          <button @click="mcpSearch" class="btn btn-primary">
            MCP Search Memory
          </button>
          <button @click="mcpGetFacts" class="btn btn-primary">
            MCP Get Facts
          </button>
        </div>
      </div>
      <button @click="callEndpoint('mcpHealth')" class="btn w-full">
        MCP Health Check
      </button>
    </div>

    <!-- System Diagnostics -->
    <div class="card mb-3">
      <h3>System Diagnostics</h3>
      <div class="grid grid-3 gap-2">
        <button @click="callEndpoint('systemStatus')" class="btn">
          System Status
        </button>
        <button @click="callEndpoint('systemReady')" class="btn">
          Ready Check
        </button>
        <button @click="callEndpoint('systemAlive')" class="btn">
          Alive Check
        </button>
        <button @click="callEndpoint('systemMetrics')" class="btn">
          Metrics
        </button>
        <button @click="callEndpoint('systemDiagnostics')" class="btn">
          Full Diagnostics
        </button>
        <button @click="callEndpoint('systemDiagnosticsDatabase')" class="btn">
          Database Check
        </button>
        <button @click="callEndpoint('systemDiagnosticsServices')" class="btn">
          Services Check
        </button>
        <button @click="callEndpoint('systemLogs')" class="btn">
          System Logs
        </button>
        <button @click="callEndpoint('healthDetailed')" class="btn">
          Detailed Health
        </button>
      </div>
    </div>

    <!-- Operations -->
    <div class="card mb-3">
      <h3>Operations & Jobs</h3>
      <div class="grid grid-3 gap-2">
        <button @click="callEndpoint('operationsStatus')" class="btn">
          Operations Status
        </button>
        <button @click="callEndpoint('operationsJobs')" class="btn">
          Active Jobs
        </button>
      </div>
    </div>

    <!-- Self Test -->
    <div class="card mb-3">
      <h3>Self-Test Suite</h3>
      <div class="grid grid-2 gap-2">
        <button @click="callEndpoint('selfTestQuick')" class="btn btn-primary">
          Quick Self-Test
        </button>
        <button @click="callEndpoint('selfTestFull')" class="btn btn-primary">
          Full Self-Test
        </button>
      </div>
    </div>

    <!-- Response Display -->
    <div class="card" v-if="lastResponse">
      <h3>Response</h3>
      <div class="mb-2">
        <span class="text-xs text-muted">{{ lastEndpoint }}</span>
        <span v-if="lastDuration" class="text-xs text-muted ml-2">{{ lastDuration }}ms</span>
      </div>
      <pre>{{ formatResponse(lastResponse) }}</pre>
    </div>

    <!-- Error Display -->
    <div class="card" v-if="lastError" style="border-color: #f85149">
      <h3 style="color: #f85149">Error</h3>
      <pre style="color: #f85149">{{ lastError }}</pre>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import {
  echoApi,
  intelligenceApi,
  conversationsApi,
  mcpApi,
  systemApi,
  selfTestApi
} from '@/api/echoApi';

const thinkQuery = ref('');
const conversationQuery = ref('');
const mcpQuery = ref('');
const lastResponse = ref<any>(null);
const lastError = ref<string>('');
const lastEndpoint = ref<string>('');
const lastDuration = ref<number>(0);

const endpoints: Record<string, () => Promise<any>> = {
  // Echo
  echoHealth: echoApi.health,
  echoBrain: echoApi.brain,
  echoStatus: echoApi.status,
  echoModels: echoApi.modelsList,

  // Intelligence
  knowledgeMap: intelligenceApi.knowledgeMap,
  intelligenceStatus: intelligenceApi.status,

  // MCP
  mcpHealth: mcpApi.health,

  // System
  systemStatus: systemApi.status,
  systemReady: systemApi.ready,
  systemAlive: systemApi.alive,
  systemMetrics: systemApi.metrics,
  systemDiagnostics: systemApi.diagnostics,
  systemDiagnosticsDatabase: systemApi.diagnosticsDatabase,
  systemDiagnosticsServices: systemApi.diagnosticsServices,
  systemLogs: systemApi.logs,
  operationsStatus: systemApi.operationsStatus,
  operationsJobs: systemApi.operationsJobs,
  healthDetailed: systemApi.healthDetailed,

  // Self-test
  selfTestQuick: selfTestApi.quick,
  selfTestFull: selfTestApi.full
};

const callEndpoint = async (name: string) => {
  lastResponse.value = null;
  lastError.value = '';
  lastEndpoint.value = name;
  const start = Date.now();

  try {
    const fn = endpoints[name];
    if (!fn) {
      throw new Error(`Endpoint ${name} not found`);
    }
    const response = await fn();
    lastDuration.value = Date.now() - start;
    lastResponse.value = response.data;
  } catch (error: any) {
    lastDuration.value = Date.now() - start;
    lastError.value = error.response?.data?.detail || error.message;
  }
};

const queryEcho = async () => {
  const query = prompt('Enter Echo query:');
  if (!query) return;

  lastResponse.value = null;
  lastError.value = '';
  lastEndpoint.value = 'Echo Query';
  const start = Date.now();

  try {
    const response = await echoApi.query(query);
    lastDuration.value = Date.now() - start;
    lastResponse.value = response.data;
  } catch (error: any) {
    lastDuration.value = Date.now() - start;
    lastError.value = error.response?.data?.detail || error.message;
  }
};

const think = async () => {
  if (!thinkQuery.value) return;

  lastResponse.value = null;
  lastError.value = '';
  lastEndpoint.value = 'Intelligence Think';
  const start = Date.now();

  try {
    const response = await intelligenceApi.think(thinkQuery.value);
    lastDuration.value = Date.now() - start;
    lastResponse.value = response.data;
  } catch (error: any) {
    lastDuration.value = Date.now() - start;
    lastError.value = error.response?.data?.detail || error.message;
  }
};

const searchConversations = async () => {
  if (!conversationQuery.value) return;

  lastResponse.value = null;
  lastError.value = '';
  lastEndpoint.value = 'Search Conversations';
  const start = Date.now();

  try {
    const response = await conversationsApi.search(conversationQuery.value);
    lastDuration.value = Date.now() - start;
    lastResponse.value = response.data;
  } catch (error: any) {
    lastDuration.value = Date.now() - start;
    lastError.value = error.response?.data?.detail || error.message;
  }
};

const mcpSearch = async () => {
  if (!mcpQuery.value) return;

  lastResponse.value = null;
  lastError.value = '';
  lastEndpoint.value = 'MCP Search Memory';
  const start = Date.now();

  try {
    const response = await mcpApi.searchMemory(mcpQuery.value);
    lastDuration.value = Date.now() - start;
    lastResponse.value = response.data;
  } catch (error: any) {
    lastDuration.value = Date.now() - start;
    lastError.value = error.response?.data?.detail || error.message;
  }
};

const mcpGetFacts = async () => {
  if (!mcpQuery.value) return;

  lastResponse.value = null;
  lastError.value = '';
  lastEndpoint.value = 'MCP Get Facts';
  const start = Date.now();

  try {
    const response = await mcpApi.getFacts(mcpQuery.value);
    lastDuration.value = Date.now() - start;
    lastResponse.value = response.data;
  } catch (error: any) {
    lastDuration.value = Date.now() - start;
    lastError.value = error.response?.data?.detail || error.message;
  }
};

const formatResponse = (data: any) => {
  if (typeof data === 'string') return data;
  return JSON.stringify(data, null, 2);
};
</script>

<style scoped>
.ml-2 {
  margin-left: 0.5rem;
}
</style>

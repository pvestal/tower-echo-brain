<script setup lang="ts">
import { ref, onMounted, watch } from 'vue';
import { photosApi } from '@/api/echoApi';

// Tab state
const activeTab = ref<'overview' | 'local' | 'cloud' | 'runs'>('overview');

// Stats
const stats = ref<any>(null);
const statsLoading = ref(false);

// Local browse
const localItems = ref<any[]>([]);
const localTotal = ref(0);
const localPage = ref(1);
const localTotalPages = ref(0);
const localLoading = ref(false);
const localFilter = ref({
  media_type: '',
  source: '',
  year: '',
  sort: 'newest',
  has_description: undefined as boolean | undefined,
  in_qdrant: undefined as boolean | undefined,
});

// Cloud browse
const cloudItems = ref<any[]>([]);
const cloudTotal = ref(0);
const cloudPage = ref(1);
const cloudTotalPages = ref(0);
const cloudLoading = ref(false);
const cloudMatched = ref<boolean | undefined>(undefined);

// Pipeline runs
const runs = ref<any[]>([]);
const runsLoading = ref(false);

// Action state
const actionLoading = ref('');
const actionResult = ref('');

async function loadStats() {
  statsLoading.value = true;
  try {
    const resp = await photosApi.stats();
    stats.value = resp.data;
  } catch (e: any) {
    console.error('Stats error:', e);
  } finally {
    statsLoading.value = false;
  }
}

async function loadLocal() {
  localLoading.value = true;
  try {
    const params: any = { page: localPage.value, per_page: 40, sort: localFilter.value.sort };
    if (localFilter.value.media_type) params.media_type = localFilter.value.media_type;
    if (localFilter.value.source) params.source = localFilter.value.source;
    if (localFilter.value.year) params.year = localFilter.value.year;
    if (localFilter.value.has_description !== undefined) params.has_description = localFilter.value.has_description;
    if (localFilter.value.in_qdrant !== undefined) params.in_qdrant = localFilter.value.in_qdrant;

    const resp = await photosApi.browse(params);
    localItems.value = resp.data.items;
    localTotal.value = resp.data.total;
    localTotalPages.value = resp.data.total_pages;
  } catch (e: any) {
    console.error('Browse error:', e);
  } finally {
    localLoading.value = false;
  }
}

async function loadCloud() {
  cloudLoading.value = true;
  try {
    const params: any = { page: cloudPage.value, per_page: 40 };
    if (cloudMatched.value !== undefined) params.matched = cloudMatched.value;

    const resp = await photosApi.browseCloud(params);
    cloudItems.value = resp.data.items;
    cloudTotal.value = resp.data.total;
    cloudTotalPages.value = resp.data.total_pages;
  } catch (e: any) {
    console.error('Cloud browse error:', e);
  } finally {
    cloudLoading.value = false;
  }
}

async function loadRuns() {
  runsLoading.value = true;
  try {
    const resp = await photosApi.runs(30);
    runs.value = resp.data.runs;
  } catch (e: any) {
    console.error('Runs error:', e);
  } finally {
    runsLoading.value = false;
  }
}

async function runAction(name: string, fn: () => Promise<any>) {
  actionLoading.value = name;
  actionResult.value = '';
  try {
    const resp = await fn();
    actionResult.value = `${name}: ${JSON.stringify(resp.data).slice(0, 300)}`;
    await loadStats();
  } catch (e: any) {
    actionResult.value = `${name} failed: ${e.response?.data?.error || e.message}`;
  } finally {
    actionLoading.value = '';
  }
}

function formatSize(mb: number | null) {
  if (!mb) return '-';
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb.toFixed(1)} MB`;
}

function formatDate(iso: string | null) {
  if (!iso) return '-';
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}

function formatRunType(t: string) {
  const map: Record<string, string> = {
    local_scan: 'Local Scan',
    takeout_scan: 'Takeout Scan',
    cloud_fetch: 'Cloud Fetch',
    dedup_match: 'Dedup Match',
    vision_analysis: 'Vision Analysis',
    face_detection: 'Face Detection',
    qdrant_ingest: 'Qdrant Ingest',
  };
  return map[t] || t;
}

// Watch tab changes
watch(activeTab, (tab) => {
  if (tab === 'local' && localItems.value.length === 0) loadLocal();
  if (tab === 'cloud' && cloudItems.value.length === 0) loadCloud();
  if (tab === 'runs' && runs.value.length === 0) loadRuns();
});

watch([localPage, localFilter], () => { if (activeTab.value === 'local') loadLocal(); }, { deep: true });
watch([cloudPage, cloudMatched], () => { if (activeTab.value === 'cloud') loadCloud(); });

onMounted(() => {
  loadStats();
});
</script>

<template>
  <div class="media-view">
    <h2>Photos & Videos</h2>

    <!-- Tabs -->
    <div class="tabs">
      <button
        v-for="tab in (['overview', 'local', 'cloud', 'runs'] as const)"
        :key="tab"
        :class="['tab', { active: activeTab === tab }]"
        @click="activeTab = tab"
      >
        {{ tab === 'overview' ? 'Overview' : tab === 'local' ? 'Local Media' : tab === 'cloud' ? 'Google Cloud' : 'Pipeline Runs' }}
      </button>
    </div>

    <!-- Action result banner -->
    <div v-if="actionResult" class="banner" @click="actionResult = ''">
      {{ actionResult }}
    </div>

    <!-- OVERVIEW TAB -->
    <div v-if="activeTab === 'overview'" class="tab-content">
      <div v-if="statsLoading" class="loading">Loading stats...</div>
      <div v-else-if="stats" class="overview-grid">
        <!-- Local stats -->
        <div class="stat-card">
          <h3>Local Media</h3>
          <div class="stat-row"><span>Total files</span><strong>{{ stats.local_photos?.toLocaleString() }}</strong></div>
          <div class="stat-row"><span>Photos</span><strong>{{ stats.photos_count?.toLocaleString() }}</strong></div>
          <div class="stat-row"><span>Videos</span><strong>{{ stats.videos_count?.toLocaleString() }}</strong></div>
          <div class="stat-row"><span>From local dirs</span><strong>{{ stats.local_source?.toLocaleString() }}</strong></div>
          <div class="stat-row"><span>From Takeout</span><strong>{{ stats.takeout_source?.toLocaleString() }}</strong></div>
          <div class="stat-row"><span>Total size</span><strong>{{ stats.total_size_gb }} GB</strong></div>
        </div>

        <!-- Sources & Dedup -->
        <div class="stat-card">
          <h3>Sources & Dedup</h3>
          <div class="stat-row"><span>From local dirs</span><strong>{{ stats.local_source?.toLocaleString() }}</strong></div>
          <div class="stat-row"><span>From Takeout (cloud)</span><strong>{{ stats.takeout_source?.toLocaleString() }}</strong></div>
          <div class="stat-row"><span>Cloud API items</span><strong>{{ stats.cloud_photos?.toLocaleString() }}</strong></div>
          <div class="stat-row"><span>Matched cloud↔local</span><strong>{{ stats.cloud_matched?.toLocaleString() }}</strong></div>
          <div class="stat-row"><span>SHA256 dupes</span><strong>{{ stats.sha256_dupes?.toLocaleString() }}</strong></div>
          <div class="stat-row"><span>Unmatched local</span><strong>{{ stats.unmatched_local?.toLocaleString() }}</strong></div>
          <div class="note">Google Photos Library API was deprecated March 2025. Takeout is the cloud source.</div>
        </div>

        <!-- Ingestion stats -->
        <div class="stat-card">
          <h3>Echo Brain Ingestion</h3>
          <div class="stat-row"><span>Vision analyzed</span><strong>{{ stats.analyzed?.toLocaleString() }}</strong></div>
          <div class="stat-row"><span>In Qdrant</span><strong>{{ stats.embedded_in_qdrant?.toLocaleString() }}</strong></div>
          <div class="stat-row">
            <span>Ingestion %</span>
            <strong>{{ stats.local_photos ? ((stats.embedded_in_qdrant / stats.local_photos) * 100).toFixed(1) : 0 }}%</strong>
          </div>
          <div class="stat-row"><span>Faces detected in</span><strong>{{ stats.photos_with_faces?.toLocaleString() }} photos</strong></div>
          <div class="stat-row"><span>Face clusters</span><strong>{{ stats.face_clusters?.toLocaleString() }}</strong></div>
        </div>

        <!-- Actions -->
        <div class="stat-card actions-card">
          <h3>Pipeline Actions</h3>
          <button @click="runAction('Scan Local', () => photosApi.scanLocal())" :disabled="!!actionLoading">
            {{ actionLoading === 'Scan Local' ? 'Scanning...' : 'Scan Local Photos' }}
          </button>
          <button @click="runAction('Scan Takeout', () => photosApi.scanTakeout())" :disabled="!!actionLoading">
            {{ actionLoading === 'Scan Takeout' ? 'Scanning...' : 'Scan Takeout (Cloud)' }}
          </button>
          <button @click="runAction('Dedup Match', () => photosApi.dedupRun())" :disabled="!!actionLoading">
            {{ actionLoading === 'Dedup Match' ? 'Matching...' : 'Run Dedup Matching' }}
          </button>
          <button @click="runAction('Ingest Qdrant', () => photosApi.ingest())" :disabled="!!actionLoading">
            {{ actionLoading === 'Ingest Qdrant' ? 'Ingesting...' : 'Ingest to Qdrant' }}
          </button>
        </div>
      </div>
    </div>

    <!-- LOCAL TAB -->
    <div v-if="activeTab === 'local'" class="tab-content">
      <div class="filters-bar">
        <select v-model="localFilter.media_type">
          <option value="">All types</option>
          <option value="photo">Photos</option>
          <option value="video">Videos</option>
        </select>
        <select v-model="localFilter.source">
          <option value="">All sources</option>
          <option value="local">Local dirs</option>
          <option value="takeout">Takeout</option>
        </select>
        <select v-model="localFilter.sort">
          <option value="newest">Newest first</option>
          <option value="oldest">Oldest first</option>
          <option value="largest">Largest first</option>
        </select>
        <select v-model="localFilter.in_qdrant" @change="(e: any) => localFilter.in_qdrant = e.target.value === '' ? undefined : e.target.value === 'true'">
          <option value="">Qdrant: Any</option>
          <option value="true">In Qdrant</option>
          <option value="false">Not in Qdrant</option>
        </select>
        <select v-model="localFilter.has_description" @change="(e: any) => localFilter.has_description = e.target.value === '' ? undefined : e.target.value === 'true'">
          <option value="">Analysis: Any</option>
          <option value="true">Analyzed</option>
          <option value="false">Not analyzed</option>
        </select>
        <span class="count-label">{{ localTotal.toLocaleString() }} items</span>
      </div>

      <div v-if="localLoading" class="loading">Loading...</div>
      <div v-else class="media-grid">
        <div v-for="item in localItems" :key="item.id" class="media-card">
          <div class="thumb-wrap">
            <img
              v-if="item.media_type === 'photo'"
              :src="item.thumb_url"
              :alt="item.filename"
              loading="lazy"
            />
            <div v-else class="video-placeholder">
              <span class="video-icon">&#9654;</span>
            </div>
            <span class="badge type-badge">{{ item.media_type }}</span>
            <span v-if="item.in_qdrant" class="badge qdrant-badge" title="In Qdrant">Q</span>
            <span v-if="item.face_count" class="badge face-badge" :title="`${item.face_count} faces`">
              {{ item.face_count }}F
            </span>
          </div>
          <div class="media-info">
            <div class="filename" :title="item.filename">{{ item.filename }}</div>
            <div class="meta">
              <span>{{ item.source }}</span>
              <span v-if="item.year">{{ item.year }}</span>
              <span v-if="item.size_mb">{{ formatSize(item.size_mb) }}</span>
            </div>
            <div v-if="item.description" class="description" :title="item.description">
              {{ item.description.slice(0, 80) }}{{ item.description.length > 80 ? '...' : '' }}
            </div>
          </div>
        </div>
      </div>

      <!-- Pagination -->
      <div v-if="localTotalPages > 1" class="pagination">
        <button @click="localPage = Math.max(1, localPage - 1)" :disabled="localPage <= 1">Prev</button>
        <span>Page {{ localPage }} / {{ localTotalPages }}</span>
        <button @click="localPage = Math.min(localTotalPages, localPage + 1)" :disabled="localPage >= localTotalPages">Next</button>
      </div>
    </div>

    <!-- CLOUD TAB -->
    <div v-if="activeTab === 'cloud'" class="tab-content">
      <div class="filters-bar">
        <select v-model="cloudMatched" @change="(e: any) => { cloudMatched = e.target.value === '' ? undefined : e.target.value === 'true'; cloudPage = 1; }">
          <option value="">All cloud items</option>
          <option value="true">Matched to local</option>
          <option value="false">Unmatched (cloud only)</option>
        </select>
        <span class="count-label">{{ cloudTotal.toLocaleString() }} items</span>
      </div>

      <div v-if="cloudLoading" class="loading">Loading...</div>
      <div v-else-if="cloudItems.length === 0" class="empty-state">
        <p v-if="cloudTotal === 0">No cloud photos synced yet. Use "Fetch Google Cloud" on the Overview tab to pull metadata from Google Photos.</p>
        <p v-else>No items match the current filter.</p>
      </div>
      <div v-else class="cloud-list">
        <div v-for="item in cloudItems" :key="item.id" class="cloud-row">
          <div class="cloud-thumb">
            <img v-if="item.local_thumb_url" :src="item.local_thumb_url" loading="lazy" />
            <div v-else class="no-thumb">No local match</div>
          </div>
          <div class="cloud-info">
            <div class="filename">{{ item.filename }}</div>
            <div class="meta">
              <span>{{ item.mime_type }}</span>
              <span v-if="item.width">{{ item.width }}x{{ item.height }}</span>
              <span v-if="item.camera">{{ item.camera }}</span>
              <span>{{ formatDate(item.created_at) }}</span>
            </div>
          </div>
          <div class="cloud-status">
            <span v-if="item.matched" class="matched-tag">
              Matched
            </span>
            <span v-else class="unmatched-tag">Cloud only</span>
          </div>
        </div>
      </div>

      <div v-if="cloudTotalPages > 1" class="pagination">
        <button @click="cloudPage = Math.max(1, cloudPage - 1)" :disabled="cloudPage <= 1">Prev</button>
        <span>Page {{ cloudPage }} / {{ cloudTotalPages }}</span>
        <button @click="cloudPage = Math.min(cloudTotalPages, cloudPage + 1)" :disabled="cloudPage >= cloudTotalPages">Next</button>
      </div>
    </div>

    <!-- RUNS TAB -->
    <div v-if="activeTab === 'runs'" class="tab-content">
      <button class="refresh-btn" @click="loadRuns" :disabled="runsLoading">
        {{ runsLoading ? 'Loading...' : 'Refresh' }}
      </button>
      <table v-if="runs.length" class="runs-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Type</th>
            <th>Started</th>
            <th>Finished</th>
            <th>Processed</th>
            <th>New</th>
            <th>Skipped</th>
            <th>Errors</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="run in runs" :key="run.id">
            <td>{{ run.id }}</td>
            <td>{{ formatRunType(run.run_type) }}</td>
            <td>{{ formatDate(run.started_at) }}</td>
            <td>{{ run.finished_at ? formatDate(run.finished_at) : 'Running...' }}</td>
            <td>{{ run.items_processed ?? '-' }}</td>
            <td>{{ run.items_new ?? '-' }}</td>
            <td>{{ run.items_skipped ?? '-' }}</td>
            <td :class="{ 'text-red': run.items_error > 0 }">{{ run.items_error ?? '-' }}</td>
          </tr>
        </tbody>
      </table>
      <div v-else class="loading">No runs recorded yet.</div>
    </div>
  </div>
</template>

<style scoped>
.media-view { max-width: 1400px; }
h2 { color: #f0f6fc; margin-bottom: 1rem; }

.tabs { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; }
.tab {
  padding: 0.5rem 1rem; background: #161b22; border: 1px solid #21262d;
  border-radius: 0.375rem; color: #8b949e; cursor: pointer; font-size: 0.875rem;
}
.tab:hover { background: #21262d; color: #f0f6fc; }
.tab.active { background: #238636; color: #f0f6fc; border-color: #238636; }

.banner {
  background: #1c2128; border: 1px solid #30363d; border-radius: 6px;
  padding: 0.75rem 1rem; margin-bottom: 1rem; color: #8b949e; font-size: 0.8rem;
  cursor: pointer; white-space: pre-wrap; word-break: break-all;
}

.loading { color: #8b949e; padding: 2rem; text-align: center; }

/* Overview grid */
.overview-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1rem;
}
.stat-card {
  background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 1.25rem;
}
.stat-card h3 { color: #f0f6fc; font-size: 1rem; margin-bottom: 0.75rem; }
.stat-row {
  display: flex; justify-content: space-between; padding: 0.35rem 0;
  border-bottom: 1px solid #21262d; color: #8b949e; font-size: 0.875rem;
}
.stat-row strong { color: #f0f6fc; }
.text-green { color: #3fb950; }
.text-red { color: #f85149; }
.note { color: #484f58; font-size: 0.75rem; margin-top: 0.5rem; font-style: italic; }

.actions-card { display: flex; flex-direction: column; gap: 0.5rem; }
.actions-card button {
  padding: 0.6rem 1rem; background: #21262d; border: 1px solid #30363d;
  border-radius: 6px; color: #f0f6fc; cursor: pointer; font-size: 0.875rem;
}
.actions-card button:hover:not(:disabled) { background: #30363d; }
.actions-card button:disabled { opacity: 0.5; cursor: default; }

/* Filters */
.filters-bar {
  display: flex; gap: 0.75rem; align-items: center; margin-bottom: 1rem; flex-wrap: wrap;
}
.filters-bar select {
  background: #161b22; border: 1px solid #30363d; border-radius: 6px;
  color: #f0f6fc; padding: 0.4rem 0.6rem; font-size: 0.8rem;
}
.count-label { color: #8b949e; font-size: 0.8rem; margin-left: auto; }

/* Media grid */
.media-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 0.75rem;
}
.media-card {
  background: #161b22; border: 1px solid #21262d; border-radius: 8px; overflow: hidden;
}
.thumb-wrap {
  position: relative; width: 100%; aspect-ratio: 1; background: #0d1117; overflow: hidden;
}
.thumb-wrap img { width: 100%; height: 100%; object-fit: cover; }
.video-placeholder {
  width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;
  background: #161b22; color: #8b949e; font-size: 2rem;
}
.badge {
  position: absolute; padding: 2px 6px; border-radius: 4px;
  font-size: 0.7rem; font-weight: 600;
}
.type-badge { top: 6px; left: 6px; background: rgba(0,0,0,0.7); color: #8b949e; }
.qdrant-badge { top: 6px; right: 6px; background: #238636; color: #fff; }
.face-badge { bottom: 6px; right: 6px; background: #1f6feb; color: #fff; }

.media-info { padding: 0.5rem; }
.filename {
  color: #f0f6fc; font-size: 0.8rem; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis;
}
.meta { color: #8b949e; font-size: 0.7rem; display: flex; gap: 0.5rem; margin-top: 0.25rem; }
.description { color: #8b949e; font-size: 0.7rem; margin-top: 0.25rem; line-height: 1.3; }

/* Pagination */
.pagination {
  display: flex; align-items: center; justify-content: center; gap: 1rem;
  margin-top: 1.5rem; padding: 1rem 0;
}
.pagination button {
  padding: 0.4rem 1rem; background: #21262d; border: 1px solid #30363d;
  border-radius: 6px; color: #f0f6fc; cursor: pointer; font-size: 0.8rem;
}
.pagination button:disabled { opacity: 0.4; cursor: default; }
.pagination span { color: #8b949e; font-size: 0.85rem; }

/* Cloud list */
.cloud-list { display: flex; flex-direction: column; gap: 0.5rem; }
.cloud-row {
  display: flex; align-items: center; gap: 1rem; padding: 0.75rem;
  background: #161b22; border: 1px solid #21262d; border-radius: 8px;
}
.cloud-thumb { width: 60px; height: 60px; flex-shrink: 0; border-radius: 6px; overflow: hidden; background: #0d1117; }
.cloud-thumb img { width: 100%; height: 100%; object-fit: cover; }
.no-thumb {
  width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;
  color: #484f58; font-size: 0.6rem; text-align: center;
}
.cloud-info { flex: 1; min-width: 0; }
.cloud-info .filename { color: #f0f6fc; font-size: 0.85rem; }
.cloud-info .meta { color: #8b949e; font-size: 0.75rem; display: flex; gap: 0.75rem; margin-top: 0.2rem; }
.cloud-status { flex-shrink: 0; }
.matched-tag {
  background: #238636; color: #fff; padding: 3px 8px; border-radius: 12px; font-size: 0.7rem;
}
.unmatched-tag {
  background: #da3633; color: #fff; padding: 3px 8px; border-radius: 12px; font-size: 0.7rem;
}
.empty-state { color: #8b949e; text-align: center; padding: 3rem 1rem; }

/* Runs table */
.refresh-btn {
  padding: 0.4rem 1rem; background: #21262d; border: 1px solid #30363d;
  border-radius: 6px; color: #f0f6fc; cursor: pointer; font-size: 0.8rem; margin-bottom: 1rem;
}
.runs-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.runs-table th {
  text-align: left; padding: 0.5rem; color: #8b949e;
  border-bottom: 1px solid #21262d; font-weight: 500;
}
.runs-table td { padding: 0.5rem; color: #f0f6fc; border-bottom: 1px solid #161b22; }
</style>

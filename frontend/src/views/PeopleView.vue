<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { photosApi } from '@/api/echoApi';

interface FaceCluster {
  id: number;
  name: string | null;
  photo_count: number;
  sample_photo_ids: number[];
  created_at: string | null;
}

interface ReviewResponse {
  clusters: FaceCluster[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}

const clusters = ref<FaceCluster[]>([]);
const loading = ref(false);
const error = ref('');
const page = ref(1);
const perPage = ref(20);
const totalPages = ref(0);
const totalClusters = ref(0);
const minPhotos = ref(5);
const unnamedOnly = ref(true);

// Naming state
const editingId = ref<number | null>(null);
const nameInput = ref('');
const saving = ref(false);

// Merge state
const mergeMode = ref(false);
const mergeSelected = ref<number[]>([]);
const mergeName = ref('');

// Stats
const stats = ref<any>(null);

// Contacts matching state
const contactsTab = ref<'clusters' | 'contacts'>('contacts');
const syncing = ref(false);
const syncResult = ref<any>(null);
const contactsStats = ref<any>(null);
const reviewQueue = ref<any[]>([]);

const thumbUrl = (photoId: number) =>
  `/api/photos/thumb/${photoId}?size=280`;

async function loadClusters() {
  loading.value = true;
  error.value = '';
  try {
    const resp = await photosApi.peopleReview(
      page.value,
      perPage.value,
      unnamedOnly.value,
      minPhotos.value,
    );
    const data: ReviewResponse = resp.data;
    clusters.value = data.clusters;
    totalPages.value = data.total_pages;
    totalClusters.value = data.total;
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message;
  } finally {
    loading.value = false;
  }
}

async function loadStats() {
  try {
    const resp = await photosApi.stats();
    stats.value = resp.data;
  } catch { /* ignore */ }
}

function startNaming(cluster: FaceCluster) {
  editingId.value = cluster.id;
  nameInput.value = cluster.name || '';
}

function cancelNaming() {
  editingId.value = null;
  nameInput.value = '';
}

async function saveName(clusterId: number) {
  if (!nameInput.value.trim()) return;
  saving.value = true;
  try {
    await photosApi.namePerson(clusterId, nameInput.value.trim());
    // Update local state
    const c = clusters.value.find((x) => x.id === clusterId);
    if (c) c.name = nameInput.value.trim();
    editingId.value = null;
    nameInput.value = '';
    // If unnamed-only filter, remove from list
    if (unnamedOnly.value) {
      clusters.value = clusters.value.filter((x) => x.id !== clusterId);
      totalClusters.value--;
    }
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message;
  } finally {
    saving.value = false;
  }
}

async function skipCluster(clusterId: number) {
  try {
    await photosApi.skipPerson(clusterId);
    clusters.value = clusters.value.filter((x) => x.id !== clusterId);
    totalClusters.value--;
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message;
  }
}

function toggleMergeSelect(clusterId: number) {
  const idx = mergeSelected.value.indexOf(clusterId);
  if (idx >= 0) {
    mergeSelected.value.splice(idx, 1);
  } else {
    mergeSelected.value.push(clusterId);
  }
}

async function executeMerge() {
  if (mergeSelected.value.length < 2 || !mergeName.value.trim()) return;
  saving.value = true;
  try {
    await photosApi.mergePeople(mergeSelected.value, mergeName.value.trim());
    mergeMode.value = false;
    mergeSelected.value = [];
    mergeName.value = '';
    await loadClusters();
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message;
  } finally {
    saving.value = false;
  }
}

function prevPage() {
  if (page.value > 1) {
    page.value--;
    loadClusters();
  }
}

function nextPage() {
  if (page.value < totalPages.value) {
    page.value++;
    loadClusters();
  }
}

async function syncContacts() {
  syncing.value = true;
  error.value = '';
  syncResult.value = null;
  try {
    const resp = await photosApi.contactsSyncAndMatch();
    syncResult.value = resp.data;
    await loadContactsStats();
    await loadReviewQueue();
    // Refresh clusters too since some may have been auto-named
    await loadClusters();
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message;
  } finally {
    syncing.value = false;
  }
}

async function loadContactsStats() {
  try {
    const resp = await photosApi.contactsStats();
    contactsStats.value = resp.data;
  } catch { /* ignore */ }
}

async function loadReviewQueue() {
  try {
    const resp = await photosApi.contactsReview();
    reviewQueue.value = resp.data.review || [];
  } catch { /* ignore */ }
}

async function confirmMatch(contactId: number, clusterId: number, name: string) {
  try {
    await photosApi.contactsConfirm(contactId, clusterId, name);
    reviewQueue.value = reviewQueue.value.filter((r) => r.contact_id !== contactId);
    await loadContactsStats();
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message;
  }
}

async function rejectMatch(contactId: number) {
  try {
    await photosApi.contactsReject(contactId);
    reviewQueue.value = reviewQueue.value.filter((r) => r.contact_id !== contactId);
    await loadContactsStats();
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message;
  }
}

onMounted(() => {
  loadClusters();
  loadStats();
  loadContactsStats();
  loadReviewQueue();
});
</script>

<template>
  <div class="people-view">
    <!-- Header -->
    <div class="header-row">
      <h2>People &amp; Face Matching</h2>
      <div class="header-stats" v-if="contactsStats">
        <span class="stat">{{ contactsStats.total_contacts || 0 }} contacts</span>
        <span class="stat">{{ contactsStats.auto_named || 0 }} auto-matched</span>
        <span class="stat">{{ contactsStats.needs_review || 0 }} to review</span>
        <span class="stat">{{ contactsStats.named_clusters || 0 }} named clusters</span>
      </div>
    </div>

    <!-- Tabs -->
    <div class="tab-row">
      <button :class="['tab', contactsTab === 'contacts' ? 'tab-active' : '']"
              @click="contactsTab = 'contacts'">
        Contacts Match
      </button>
      <button :class="['tab', contactsTab === 'clusters' ? 'tab-active' : '']"
              @click="contactsTab = 'clusters'">
        Manual Clusters
      </button>
    </div>

    <!-- ===== CONTACTS TAB ===== -->
    <template v-if="contactsTab === 'contacts'">
      <!-- Sync button -->
      <div class="controls">
        <button class="btn btn-primary" @click="syncContacts()" :disabled="syncing">
          {{ syncing ? 'Syncing contacts + matching faces...' : 'Sync Google Contacts & Match' }}
        </button>
        <button class="btn" @click="loadReviewQueue()">Refresh Review Queue</button>
      </div>

      <!-- Sync result -->
      <div v-if="syncResult" class="sync-result">
        <span>Fetched {{ syncResult.contacts_fetched }} contacts</span>
        <span>{{ syncResult.contacts_with_photos }} with photos</span>
        <span>{{ syncResult.embeddings_computed }} new embeddings</span>
        <span class="result-good">{{ syncResult.auto_named }} auto-named</span>
        <span class="result-warn">{{ syncResult.needs_review }} to review</span>
        <span>{{ syncResult.no_match }} no match</span>
      </div>

      <!-- Review queue -->
      <div v-if="reviewQueue.length > 0" class="review-section">
        <h3>Review Matches ({{ reviewQueue.length }})</h3>
        <div class="review-grid">
          <div v-for="item in reviewQueue" :key="item.contact_id" class="review-card">
            <!-- Contact info -->
            <div class="review-contact">
              <img v-if="item.photo_url" :src="item.photo_url" class="contact-photo" />
              <div class="contact-info">
                <span class="contact-name">{{ item.contact_name }}</span>
                <span v-if="item.organization" class="contact-org">{{ item.organization }}</span>
                <span v-if="item.emails?.length" class="contact-email">{{ item.emails[0] }}</span>
                <span class="match-score">{{ (item.match_confidence * 100).toFixed(0) }}% match</span>
              </div>
            </div>
            <!-- Arrow -->
            <div class="review-arrow">→</div>
            <!-- Cluster photos -->
            <div class="review-cluster">
              <div class="photo-row-sm">
                <img v-for="pid in (item.sample_photo_ids || [])" :key="pid"
                     :src="thumbUrl(pid)" class="thumb-sm" loading="lazy" />
              </div>
              <span class="cluster-meta">{{ item.photo_count?.toLocaleString() }} photos</span>
            </div>
            <!-- Actions -->
            <div class="review-actions">
              <button class="btn btn-primary btn-sm"
                      @click="confirmMatch(item.contact_id, item.cluster_id, item.contact_name)">
                Confirm
              </button>
              <button class="btn btn-danger btn-sm"
                      @click="rejectMatch(item.contact_id)">
                Reject
              </button>
            </div>
          </div>
        </div>
      </div>
      <div v-else-if="!syncing" class="empty-state">
        No matches to review. Click "Sync Google Contacts &amp; Match" to start.
      </div>
    </template>

    <!-- ===== CLUSTERS TAB ===== -->
    <template v-if="contactsTab === 'clusters'">
    <!-- Controls -->
    <div class="controls">
      <label class="control-item">
        <input type="checkbox" v-model="unnamedOnly" @change="page = 1; loadClusters();" />
        Unnamed only
      </label>
      <label class="control-item">
        Min photos:
        <input type="number" v-model.number="minPhotos" min="1" max="1000"
               class="input-sm" @change="page = 1; loadClusters();" />
      </label>
      <button
        :class="['btn', mergeMode ? 'btn-active' : '']"
        @click="mergeMode = !mergeMode; mergeSelected = [];"
      >
        {{ mergeMode ? 'Cancel Merge' : 'Merge Mode' }}
      </button>
      <button class="btn" @click="loadClusters()" :disabled="loading">
        Refresh
      </button>
    </div>

    <!-- Merge bar -->
    <div v-if="mergeMode && mergeSelected.length > 0" class="merge-bar">
      <span>{{ mergeSelected.length }} selected</span>
      <input
        v-model="mergeName"
        placeholder="Name for merged person..."
        class="input-name"
        @keyup.enter="executeMerge()"
      />
      <button class="btn btn-primary" @click="executeMerge()"
              :disabled="mergeSelected.length < 2 || !mergeName.trim() || saving">
        Merge
      </button>
    </div>

    <!-- Error -->
    <div v-if="error" class="error-bar">{{ error }}</div>

    <!-- Loading -->
    <div v-if="loading" class="loading">Loading clusters...</div>

    <!-- Cluster grid -->
    <div v-else class="cluster-grid">
      <div
        v-for="cluster in clusters"
        :key="cluster.id"
        :class="['cluster-card', { 'merge-selected': mergeSelected.includes(cluster.id) }]"
        @click="mergeMode ? toggleMergeSelect(cluster.id) : null"
      >
        <!-- Sample photos -->
        <div class="photo-row">
          <img
            v-for="pid in cluster.sample_photo_ids"
            :key="pid"
            :src="thumbUrl(pid)"
            :alt="`Photo ${pid}`"
            class="thumb"
            loading="lazy"
          />
          <div v-if="cluster.sample_photo_ids.length === 0" class="no-photos">
            No samples
          </div>
        </div>

        <!-- Info -->
        <div class="cluster-info">
          <span class="photo-count">{{ cluster.photo_count.toLocaleString() }} photos</span>
          <span class="cluster-id">#{{ cluster.id }}</span>
        </div>

        <!-- Name / Edit -->
        <div class="cluster-actions" v-if="!mergeMode">
          <div v-if="editingId === cluster.id" class="name-edit">
            <input
              v-model="nameInput"
              ref="nameInputEl"
              placeholder="Enter name..."
              class="input-name"
              @keyup.enter="saveName(cluster.id)"
              @keyup.escape="cancelNaming()"
              autofocus
            />
            <button class="btn btn-primary btn-sm" @click="saveName(cluster.id)" :disabled="saving">
              Save
            </button>
            <button class="btn btn-sm" @click="cancelNaming()">
              Cancel
            </button>
          </div>
          <div v-else class="name-display">
            <span v-if="cluster.name" class="named">{{ cluster.name }}</span>
            <div class="action-buttons">
              <button class="btn btn-primary btn-sm" @click.stop="startNaming(cluster)">
                {{ cluster.name ? 'Rename' : 'Name' }}
              </button>
              <button class="btn btn-sm btn-danger" @click.stop="skipCluster(cluster.id)"
                      v-if="!cluster.name">
                Skip
              </button>
            </div>
          </div>
        </div>

        <!-- Merge checkbox indicator -->
        <div v-if="mergeMode" class="merge-indicator">
          <span v-if="mergeSelected.includes(cluster.id)" class="check">Selected</span>
          <span v-else class="uncheck">Click to select</span>
        </div>
      </div>
    </div>

    <!-- Pagination -->
    <div class="pagination" v-if="totalPages > 1">
      <button class="btn btn-sm" @click="prevPage()" :disabled="page <= 1">Prev</button>
      <span class="page-info">Page {{ page }} of {{ totalPages }}</span>
      <button class="btn btn-sm" @click="nextPage()" :disabled="page >= totalPages">Next</button>
    </div>
    </template>
  </div>
</template>

<style scoped>
.people-view {
  max-width: 1200px;
}

.header-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.header-row h2 {
  font-size: 1.1rem;
  color: #f0f6fc;
  margin: 0;
}

.header-stats {
  display: flex;
  gap: 1rem;
}

.stat {
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 0.25rem;
  padding: 0.25rem 0.5rem;
  font-size: 0.75rem;
  color: #8b949e;
}

.controls {
  display: flex;
  gap: 1rem;
  align-items: center;
  margin-bottom: 1rem;
  flex-wrap: wrap;
}

.control-item {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.85rem;
  color: #8b949e;
}

.input-sm {
  width: 60px;
  padding: 0.2rem 0.4rem;
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 0.25rem;
  color: #f0f6fc;
  font-size: 0.8rem;
}

.merge-bar {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.5rem 0.75rem;
  background: #1a1f35;
  border: 1px solid #388bfd;
  border-radius: 0.375rem;
  margin-bottom: 1rem;
  font-size: 0.85rem;
  color: #8b949e;
}

.error-bar {
  padding: 0.5rem 0.75rem;
  background: #2d1a1a;
  border: 1px solid #f85149;
  border-radius: 0.375rem;
  color: #f85149;
  margin-bottom: 1rem;
  font-size: 0.85rem;
}

.loading {
  text-align: center;
  padding: 2rem;
  color: #8b949e;
}

.cluster-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
  gap: 1.25rem;
}

.cluster-card {
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 0.5rem;
  padding: 0.75rem;
  transition: border-color 0.2s;
}

.cluster-card:hover {
  border-color: #30363d;
}

.cluster-card.merge-selected {
  border-color: #388bfd;
  background: #1a1f35;
}

.photo-row {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
  overflow-x: auto;
  min-height: 140px;
  align-items: center;
}

.thumb {
  width: 140px;
  height: 140px;
  object-fit: cover;
  border-radius: 0.5rem;
  border: 1px solid #21262d;
  flex-shrink: 0;
}

.no-photos {
  color: #484f58;
  font-size: 0.8rem;
  text-align: center;
  width: 100%;
}

.cluster-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.photo-count {
  font-size: 0.8rem;
  color: #f0f6fc;
  font-weight: 500;
}

.cluster-id {
  font-size: 0.7rem;
  color: #484f58;
}

.cluster-actions {
  min-height: 2rem;
}

.name-edit {
  display: flex;
  gap: 0.35rem;
  align-items: center;
}

.input-name {
  flex: 1;
  padding: 0.3rem 0.5rem;
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 0.25rem;
  color: #f0f6fc;
  font-size: 0.85rem;
}

.input-name:focus {
  outline: none;
  border-color: #388bfd;
}

.name-display {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.named {
  color: #3fb950;
  font-weight: 500;
  font-size: 0.9rem;
}

.action-buttons {
  display: flex;
  gap: 0.35rem;
}

.btn {
  padding: 0.35rem 0.65rem;
  background: #21262d;
  border: 1px solid #30363d;
  border-radius: 0.25rem;
  color: #c9d1d9;
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.15s;
}

.btn:hover {
  background: #30363d;
  color: #f0f6fc;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-primary {
  background: #238636;
  border-color: #238636;
  color: #fff;
}

.btn-primary:hover {
  background: #2ea043;
}

.btn-danger {
  background: #3d1a1a;
  border-color: #f85149;
  color: #f85149;
}

.btn-danger:hover {
  background: #4d1a1a;
}

.btn-sm {
  padding: 0.2rem 0.45rem;
  font-size: 0.75rem;
}

.btn-active {
  background: #1a1f35;
  border-color: #388bfd;
  color: #388bfd;
}

.merge-indicator {
  text-align: center;
  font-size: 0.75rem;
  padding-top: 0.25rem;
}

.merge-indicator .check {
  color: #388bfd;
  font-weight: 500;
}

.merge-indicator .uncheck {
  color: #484f58;
}

.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 1rem;
  margin-top: 1.5rem;
  padding-top: 1rem;
  border-top: 1px solid #21262d;
}

.page-info {
  font-size: 0.85rem;
  color: #8b949e;
}

/* Tabs */
.tab-row {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
  border-bottom: 1px solid #21262d;
  padding-bottom: 0.5rem;
}

.tab {
  padding: 0.5rem 1rem;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 0.375rem 0.375rem 0 0;
  color: #8b949e;
  cursor: pointer;
  font-size: 0.9rem;
  transition: all 0.15s;
}

.tab:hover {
  color: #f0f6fc;
}

.tab-active {
  color: #f0f6fc;
  border-color: #21262d;
  border-bottom-color: #0d1117;
  background: #161b22;
}

/* Contacts sync */
.sync-result {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  padding: 0.75rem;
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 0.375rem;
  margin-bottom: 1rem;
  font-size: 0.85rem;
  color: #8b949e;
}

.result-good { color: #3fb950; font-weight: 500; }
.result-warn { color: #d29922; font-weight: 500; }

/* Review section */
.review-section h3 {
  font-size: 1rem;
  color: #f0f6fc;
  margin-bottom: 0.75rem;
}

.review-grid {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.review-card {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.75rem;
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 0.5rem;
}

.review-contact {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  min-width: 200px;
}

.contact-photo {
  width: 56px;
  height: 56px;
  border-radius: 50%;
  object-fit: cover;
  border: 2px solid #30363d;
}

.contact-info {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
}

.contact-name {
  color: #f0f6fc;
  font-weight: 500;
  font-size: 0.9rem;
}

.contact-org, .contact-email {
  color: #484f58;
  font-size: 0.75rem;
}

.match-score {
  color: #d29922;
  font-size: 0.75rem;
  font-weight: 500;
}

.review-arrow {
  color: #484f58;
  font-size: 1.5rem;
  flex-shrink: 0;
}

.review-cluster {
  flex: 1;
}

.photo-row-sm {
  display: flex;
  gap: 0.25rem;
  margin-bottom: 0.25rem;
}

.thumb-sm {
  width: 56px;
  height: 56px;
  object-fit: cover;
  border-radius: 0.25rem;
  border: 1px solid #21262d;
}

.cluster-meta {
  font-size: 0.75rem;
  color: #8b949e;
}

.review-actions {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  flex-shrink: 0;
}

.empty-state {
  text-align: center;
  padding: 3rem;
  color: #484f58;
  font-size: 0.9rem;
}
</style>

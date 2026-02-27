<script setup lang="ts">
import { ref, computed, onMounted, watch, reactive } from 'vue';
import { calendarApi } from '@/api/echoApi';

// ─── Types ───────────────────────────────────────────────────────────
interface CalendarMeta {
  id: string;
  summary: string;
  color: string;
}

interface CalendarEvent {
  id: string;
  summary: string;
  start: string;
  end: string;
  location: string;
  calendar_id: string;
  calendar_name: string;
  calendar_color: string;
}

interface ProcessedEvent extends CalendarEvent {
  startDate: Date;
  endDate: Date;
  isAllDay: boolean;
  isMultiDay: boolean;
  spanDays: number;
}

interface EventSegment {
  event: ProcessedEvent;
  lane: number;
  isStart: boolean;
  isEnd: boolean;
  showTitle: boolean;
}

interface GridCell {
  date: Date;
  day: number;
  inMonth: boolean;
  multiDaySegments: EventSegment[];
  timedEvents: ProcessedEvent[];
}

interface WeekRow {
  cells: GridCell[];
  maxLanes: number;
}

// ─── State ───────────────────────────────────────────────────────────
const now = new Date();
const year = ref(now.getFullYear());
const month = ref(now.getMonth() + 1);
const loading = ref(false);
const error = ref('');
const calendars = ref<CalendarMeta[]>([]);
const events = ref<CalendarEvent[]>([]);
const selectedDate = ref<Date | null>(null);
const hiddenCalendars = reactive(new Set<string>());

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTH_NAMES = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
];

// ─── Helpers ─────────────────────────────────────────────────────────
function sameDay(a: Date, b: Date): boolean {
  return a.getFullYear() === b.getFullYear() &&
    a.getMonth() === b.getMonth() &&
    a.getDate() === b.getDate();
}

function daysBetween(a: Date, b: Date): number {
  const msPerDay = 86400000;
  const aStart = new Date(a.getFullYear(), a.getMonth(), a.getDate());
  const bStart = new Date(b.getFullYear(), b.getMonth(), b.getDate());
  return Math.round((bStart.getTime() - aStart.getTime()) / msPerDay);
}

function parseDate(iso: string): Date {
  if (iso.length <= 10) {
    // All-day: "2026-06-23" — parse as local date
    const parts = iso.split('-').map(Number);
    return new Date(parts[0]!, parts[1]! - 1, parts[2]!);
  }
  return new Date(iso);
}

function formatTime(iso: string): string {
  if (!iso || iso.length <= 10) return 'All day';
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return iso;
  }
}

function formatDateLabel(d: Date): string {
  return `${MONTH_NAMES[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
}

// ─── Computed: Derived state ─────────────────────────────────────────
const monthLabel = computed(() => `${MONTH_NAMES[month.value - 1]} ${year.value}`);

const todayDate = computed(() => {
  const d = new Date();
  return new Date(d.getFullYear(), d.getMonth(), d.getDate());
});

const processedEvents = computed<ProcessedEvent[]>(() => {
  return events.value
    .filter(ev => !hiddenCalendars.has(ev.calendar_id))
    .map(ev => {
      const startDate = parseDate(ev.start);
      const isAllDay = ev.start.length <= 10;
      // For all-day events, Google returns exclusive end date — subtract one day
      let endDate = parseDate(ev.end);
      if (isAllDay && ev.end.length <= 10) {
        endDate = new Date(endDate.getFullYear(), endDate.getMonth(), endDate.getDate() - 1);
        if (endDate < startDate) endDate = new Date(startDate);
      }
      const spanDays = daysBetween(startDate, endDate) + 1;
      const isMultiDay = spanDays > 1 || isAllDay;
      return { ...ev, startDate, endDate, isAllDay, isMultiDay, spanDays };
    });
});

// Build the grid cells with leading/trailing days from adjacent months
const gridCells = computed<GridCell[]>(() => {
  const firstDay = new Date(year.value, month.value - 1, 1);
  const startDow = firstDay.getDay(); // 0=Sun
  const daysInMonth = new Date(year.value, month.value, 0).getDate();

  const cells: GridCell[] = [];

  // Leading days from previous month
  for (let i = startDow - 1; i >= 0; i--) {
    const d = new Date(year.value, month.value - 1, -i);
    cells.push({ date: d, day: d.getDate(), inMonth: false, multiDaySegments: [], timedEvents: [] });
  }
  // Days in current month
  for (let d = 1; d <= daysInMonth; d++) {
    const dt = new Date(year.value, month.value - 1, d);
    cells.push({ date: dt, day: d, inMonth: true, multiDaySegments: [], timedEvents: [] });
  }
  // Trailing days
  while (cells.length % 7 !== 0) {
    const last = cells[cells.length - 1]!.date;
    const d = new Date(last.getFullYear(), last.getMonth(), last.getDate() + 1);
    cells.push({ date: d, day: d.getDate(), inMonth: false, multiDaySegments: [], timedEvents: [] });
  }
  return cells;
});

// Core algorithm: build week rows with lane-assigned multi-day segments
const weekRows = computed<WeekRow[]>(() => {
  const cells = gridCells.value;
  const rows: WeekRow[] = [];

  for (let rowStart = 0; rowStart < cells.length; rowStart += 7) {
    const rowCells = cells.slice(rowStart, rowStart + 7).map(c => ({
      ...c,
      multiDaySegments: [] as EventSegment[],
      timedEvents: [] as ProcessedEvent[],
    }));

    const weekStart = rowCells[0]!.date;
    const weekEnd = rowCells[6]!.date;

    // Find multi-day / all-day events overlapping this week
    const multiDayEvts = processedEvents.value.filter(ev => {
      return ev.isMultiDay && ev.startDate <= weekEnd && ev.endDate >= weekStart;
    });

    // Sort by span length descending (longest gets lowest lane), then by start date
    multiDayEvts.sort((a, b) => {
      const spanDiff = b.spanDays - a.spanDays;
      if (spanDiff !== 0) return spanDiff;
      return a.startDate.getTime() - b.startDate.getTime();
    });

    // Greedy lane assignment
    // lanes[lane] = array of events assigned to that lane (track occupied columns)
    const lanes: { event: ProcessedEvent; startCol: number; endCol: number }[][] = [];

    for (const ev of multiDayEvts) {
      // Calculate column range within this row
      const startCol = Math.max(0, daysBetween(weekStart, ev.startDate));
      const endCol = Math.min(6, daysBetween(weekStart, ev.endDate));

      // Find first lane where no existing event overlaps these columns
      let assignedLane = -1;
      for (let l = 0; l < lanes.length; l++) {
        const conflict = lanes[l]!.some(occ =>
          !(endCol < occ.startCol || startCol > occ.endCol)
        );
        if (!conflict) {
          assignedLane = l;
          break;
        }
      }
      if (assignedLane === -1) {
        assignedLane = lanes.length;
        lanes.push([]);
      }
      lanes[assignedLane]!.push({ event: ev, startCol, endCol });

      // Create segments for each cell
      for (let col = startCol; col <= endCol; col++) {
        const isStart = sameDay(rowCells[col]!.date, ev.startDate);
        const isEnd = sameDay(rowCells[col]!.date, ev.endDate);
        rowCells[col]!.multiDaySegments.push({
          event: ev,
          lane: assignedLane,
          isStart,
          isEnd,
          showTitle: col === startCol,
        });
      }
    }

    // Assign single-day timed events to their cells
    for (const ev of processedEvents.value) {
      if (ev.isMultiDay) continue;
      for (const cell of rowCells) {
        if (sameDay(cell.date, ev.startDate)) {
          cell.timedEvents.push(ev);
        }
      }
    }

    rows.push({ cells: rowCells, maxLanes: lanes.length });
  }

  return rows;
});

// Day detail panel events: include multi-day events passing through the day
const selectedDayEvents = computed<ProcessedEvent[]>(() => {
  if (!selectedDate.value) return [];
  const sel = selectedDate.value;
  return processedEvents.value
    .filter(ev => ev.startDate <= sel && ev.endDate >= sel)
    .sort((a, b) => {
      // Multi-day / all-day first, then timed
      if (a.isMultiDay !== b.isMultiDay) return a.isMultiDay ? -1 : 1;
      return a.startDate.getTime() - b.startDate.getTime();
    });
});

// ─── Actions ─────────────────────────────────────────────────────────
function prevMonth() {
  if (month.value === 1) {
    month.value = 12;
    year.value--;
  } else {
    month.value--;
  }
  selectedDate.value = null;
}

function nextMonth() {
  if (month.value === 12) {
    month.value = 1;
    year.value++;
  } else {
    month.value++;
  }
  selectedDate.value = null;
}

function goToday() {
  const now = new Date();
  year.value = now.getFullYear();
  month.value = now.getMonth() + 1;
  selectedDate.value = new Date(now.getFullYear(), now.getMonth(), now.getDate());
}

function selectDay(cell: GridCell) {
  if (selectedDate.value && sameDay(selectedDate.value, cell.date)) {
    selectedDate.value = null;
  } else {
    selectedDate.value = new Date(cell.date);
  }
}

function toggleCalendar(calId: string) {
  if (hiddenCalendars.has(calId)) {
    hiddenCalendars.delete(calId);
  } else {
    hiddenCalendars.add(calId);
  }
}

function isToday(d: Date): boolean {
  return sameDay(d, todayDate.value);
}

function cellMaxItems(_cell: GridCell, maxLanes: number): number {
  // Show up to 3 total visible items; multi-day bars take priority
  return Math.max(0, 3 - maxLanes);
}

async function fetchMonth() {
  loading.value = true;
  error.value = '';
  try {
    const resp = await calendarApi.getMonthEvents(year.value, month.value) as any;
    calendars.value = resp.data.calendars || [];
    events.value = resp.data.events || [];
  } catch (e: any) {
    error.value = e?.response?.data?.detail || e.message || 'Failed to load calendar';
    calendars.value = [];
    events.value = [];
  } finally {
    loading.value = false;
  }
}

onMounted(fetchMonth);
watch([year, month], fetchMonth);
</script>

<template>
  <div class="calendar-view">
    <!-- Calendar Legend (clickable toggles) -->
    <div v-if="calendars.length" class="calendar-legend">
      <button
        v-for="cal in calendars"
        :key="cal.id"
        class="legend-pill"
        :class="{ hidden: hiddenCalendars.has(cal.id) }"
        @click="toggleCalendar(cal.id)"
      >
        <span class="legend-dot" :style="{ background: cal.color }"></span>
        {{ cal.summary }}
      </button>
    </div>

    <!-- Month Navigation -->
    <div class="month-nav">
      <button class="nav-btn" @click="prevMonth">&larr;</button>
      <button class="nav-btn today-btn" @click="goToday">Today</button>
      <h2 class="month-label">{{ monthLabel }}</h2>
      <button class="nav-btn" @click="nextMonth">&rarr;</button>
    </div>

    <!-- Loading / Error -->
    <div v-if="loading" class="loading">Loading calendar...</div>
    <div v-if="error" class="error-msg">{{ error }}</div>

    <!-- Month Grid -->
    <div class="month-grid">
      <!-- Day headers -->
      <div v-for="d in DAYS" :key="d" class="grid-header">{{ d }}</div>

      <!-- Week rows -->
      <template v-for="(row, rowIdx) in weekRows" :key="rowIdx">
        <div
          v-for="(cell, colIdx) in row.cells"
          :key="`${rowIdx}-${colIdx}`"
          class="grid-cell"
          :class="{
            'in-month': cell.inMonth,
            'out-month': !cell.inMonth,
            'today': isToday(cell.date),
            'selected': selectedDate && sameDay(selectedDate, cell.date),
          }"
          @click="selectDay(cell)"
        >
          <span class="day-number" :class="{ 'out-month-num': !cell.inMonth }">
            {{ cell.day }}
          </span>

          <!-- Multi-day bar area -->
          <div
            class="multiday-area"
            :style="{ minHeight: row.maxLanes > 0 ? (row.maxLanes * 20 + 2) + 'px' : '0' }"
          >
            <div
              v-for="seg in cell.multiDaySegments"
              :key="seg.event.id + '-' + seg.lane"
              class="multiday-bar"
              :class="{
                'bar-start': seg.isStart,
                'bar-end': seg.isEnd,
                'bar-only': seg.isStart && seg.isEnd,
              }"
              :style="{
                top: (seg.lane * 20) + 'px',
                backgroundColor: seg.event.calendar_color || '#238636',
              }"
              :title="seg.event.summary"
            >
              <span v-if="seg.showTitle" class="bar-title">{{ seg.event.summary }}</span>
            </div>
          </div>

          <!-- Single-day timed events -->
          <template v-if="cell.timedEvents.length > 0">
            <div
              v-for="ev in cell.timedEvents.slice(0, cellMaxItems(cell, row.maxLanes))"
              :key="ev.id"
              class="timed-event"
              :style="{ borderLeftColor: ev.calendar_color || '#238636' }"
              :title="ev.summary"
            >
              <span class="timed-time">{{ formatTime(ev.start) }}</span>
              <span class="timed-title">{{ ev.summary }}</span>
            </div>
            <div
              v-if="cell.timedEvents.length > cellMaxItems(cell, row.maxLanes) && cellMaxItems(cell, row.maxLanes) >= 0"
              class="more-count"
            >
              +{{ cell.timedEvents.length - cellMaxItems(cell, row.maxLanes) }} more
            </div>
          </template>
        </div>
      </template>
    </div>

    <!-- Day Detail Panel -->
    <transition name="slide">
      <div v-if="selectedDate !== null" class="day-panel">
        <div class="panel-header">
          <h3>{{ formatDateLabel(selectedDate) }}</h3>
          <button class="close-btn" @click="selectedDate = null">&times;</button>
        </div>
        <div v-if="selectedDayEvents.length === 0" class="no-events">
          No events this day.
        </div>
        <div
          v-for="ev in selectedDayEvents"
          :key="ev.id"
          class="event-card"
          :style="{ borderLeftColor: ev.calendar_color || '#238636' }"
        >
          <div class="event-time">
            <template v-if="ev.isMultiDay || ev.isAllDay">
              All day
              <span v-if="ev.isMultiDay && ev.spanDays > 1" class="event-span-badge">
                {{ ev.spanDays }} days
              </span>
            </template>
            <template v-else>
              {{ formatTime(ev.start) }} - {{ formatTime(ev.end) }}
            </template>
          </div>
          <div class="event-title">{{ ev.summary }}</div>
          <div v-if="ev.location" class="event-location">{{ ev.location }}</div>
          <div class="event-cal">{{ ev.calendar_name }}</div>
        </div>
      </div>
    </transition>
  </div>
</template>

<style scoped>
.calendar-view {
  max-width: 1000px;
}

/* ─── Legend ──────────────────────────────────────────────────────── */
.calendar-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 1rem;
}
.legend-pill {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.8rem;
  color: #c9d1d9;
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 999px;
  padding: 0.25rem 0.75rem;
  cursor: pointer;
  transition: opacity 0.15s, background 0.15s;
}
.legend-pill:hover {
  background: #21262d;
}
.legend-pill.hidden {
  opacity: 0.4;
}
.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
}

/* ─── Navigation ─────────────────────────────────────────────────── */
.month-nav {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  margin-bottom: 1rem;
}
.month-label {
  font-size: 1.25rem;
  font-weight: 600;
  color: #f0f6fc;
  min-width: 200px;
  text-align: center;
}
.nav-btn {
  background: #161b22;
  border: 1px solid #30363d;
  color: #f0f6fc;
  padding: 0.4rem 0.75rem;
  border-radius: 0.375rem;
  cursor: pointer;
  font-size: 1rem;
}
.nav-btn:hover {
  background: #21262d;
}
.today-btn {
  color: #3fb950;
  border-color: #238636;
  font-size: 0.85rem;
  font-weight: 600;
}
.today-btn:hover {
  background: #238636;
  color: #fff;
}

/* ─── Grid ───────────────────────────────────────────────────────── */
.loading {
  text-align: center;
  color: #8b949e;
  padding: 2rem;
}
.error-msg {
  text-align: center;
  color: #f85149;
  padding: 1rem;
  background: #1c1012;
  border: 1px solid #f8514930;
  border-radius: 0.375rem;
  margin-bottom: 1rem;
}

.month-grid {
  display: grid;
  grid-template-columns: repeat(7, 1fr);
  gap: 1px;
  background: #21262d;
  border: 1px solid #21262d;
  border-radius: 0.5rem;
  overflow: hidden;
}
.grid-header {
  background: #161b22;
  color: #8b949e;
  padding: 0.5rem;
  text-align: center;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
}
.grid-cell {
  background: #0d1117;
  min-height: 100px;
  padding: 0.35rem;
  cursor: pointer;
  transition: background 0.15s;
  overflow: hidden;
}
.grid-cell.out-month {
  background: #090c10;
}
.grid-cell.in-month:hover,
.grid-cell.out-month:hover {
  background: #161b22;
}
.grid-cell.today {
  border: 2px solid #238636;
}
.grid-cell.selected {
  background: #1c2333;
}

.day-number {
  font-size: 0.8rem;
  color: #c9d1d9;
  font-weight: 500;
  display: block;
  margin-bottom: 2px;
}
.out-month-num {
  color: #484f58;
}
.grid-cell.today .day-number {
  color: #3fb950;
  font-weight: 700;
}

/* ─── Multi-day bars ─────────────────────────────────────────────── */
.multiday-area {
  position: relative;
  width: 100%;
}
.multiday-bar {
  position: absolute;
  left: 0;
  right: 0;
  height: 18px;
  display: flex;
  align-items: center;
  overflow: hidden;
  font-size: 0.7rem;
  color: #fff;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
  cursor: pointer;
}
.multiday-bar.bar-start {
  border-radius: 3px 0 0 3px;
  margin-left: 1px;
}
.multiday-bar.bar-end {
  border-radius: 0 3px 3px 0;
  margin-right: 1px;
}
.multiday-bar.bar-only {
  border-radius: 3px;
  margin-left: 1px;
  margin-right: 1px;
}
.bar-title {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding: 0 4px;
  font-weight: 500;
}

/* ─── Single-day timed events ────────────────────────────────────── */
.timed-event {
  display: flex;
  align-items: baseline;
  gap: 3px;
  border-left: 3px solid #238636;
  padding: 1px 4px;
  margin-top: 2px;
  font-size: 0.65rem;
  color: #c9d1d9;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  border-radius: 0 2px 2px 0;
}
.timed-time {
  color: #8b949e;
  flex-shrink: 0;
}
.timed-title {
  overflow: hidden;
  text-overflow: ellipsis;
}
.more-count {
  font-size: 0.6rem;
  color: #8b949e;
  margin-top: 2px;
  padding-left: 4px;
}

/* ─── Day detail panel ───────────────────────────────────────────── */
.day-panel {
  margin-top: 1rem;
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 0.5rem;
  padding: 1rem;
}
.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
}
.panel-header h3 {
  color: #f0f6fc;
  font-size: 1rem;
  font-weight: 600;
}
.close-btn {
  background: none;
  border: none;
  color: #8b949e;
  font-size: 1.25rem;
  cursor: pointer;
}
.close-btn:hover {
  color: #f0f6fc;
}

.no-events {
  color: #8b949e;
  font-size: 0.85rem;
}
.event-card {
  border-left: 3px solid #238636;
  padding: 0.5rem 0.75rem;
  margin-bottom: 0.5rem;
  background: #0d1117;
  border-radius: 0 0.25rem 0.25rem 0;
}
.event-time {
  font-size: 0.75rem;
  color: #8b949e;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.event-span-badge {
  font-size: 0.65rem;
  background: #21262d;
  color: #8b949e;
  padding: 1px 6px;
  border-radius: 999px;
}
.event-title {
  font-size: 0.9rem;
  color: #f0f6fc;
  font-weight: 500;
}
.event-location {
  font-size: 0.75rem;
  color: #8b949e;
  margin-top: 2px;
}
.event-cal {
  font-size: 0.7rem;
  color: #484f58;
  margin-top: 2px;
}

/* ─── Slide transition ───────────────────────────────────────────── */
.slide-enter-active, .slide-leave-active {
  transition: all 0.2s ease;
}
.slide-enter-from, .slide-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}
</style>

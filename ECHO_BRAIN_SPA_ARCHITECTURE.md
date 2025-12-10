# Echo Brain Vue SPA Architecture

## Clean Separation of Concerns

### 1. Tower Dashboard (Port 8080) - STATUS & MONITORING
**Purpose**: System-wide monitoring and service health
**Location**: `/opt/tower-dashboard/`

#### Echo Brain Status Tab (Keep Simple)
```html
<!-- In Tower Dashboard index.html -->
<div id="echo-brain-tab">
  <!-- ONLY status monitoring -->
  - Service Health: Running/Stopped
  - Resource Usage: CPU/Memory/VRAM
  - Database Size: PostgreSQL/Qdrant stats
  - Error Rate: System errors
  - Queue Status: Task backlog
  - Last Activity: Timestamp
</div>
```

**Key Point**: Dashboard only shows STATUS, not functionality
- No chat interface
- No query input
- No task creation
- Just monitoring metrics

### 2. Echo Brain SPA (Port 8310) - FULL FUNCTIONALITY
**Purpose**: Complete Echo Brain interface and interaction
**Location**: `/opt/tower-echo-brain/frontend/`

#### Proper Vue SPA Structure
```
/opt/tower-echo-brain/frontend/
├── src/
│   ├── router/
│   │   └── index.js          # Vue Router configuration
│   ├── stores/
│   │   ├── conversation.js   # Pinia store for conversations
│   │   ├── metrics.js        # System metrics store
│   │   └── websocket.js      # WebSocket connection store
│   ├── views/
│   │   ├── ConversationView.vue  # Main chat interface
│   │   ├── LearningView.vue      # Learning pipeline control
│   │   ├── TasksView.vue         # Task queue management
│   │   ├── AnalyticsView.vue    # Pattern analytics & insights
│   │   └── SettingsView.vue     # Model selection & config
│   ├── components/
│   │   ├── chat/
│   │   │   ├── MessageList.vue
│   │   │   ├── MessageInput.vue
│   │   │   └── ThoughtStream.vue
│   │   ├── learning/
│   │   │   ├── PipelineStatus.vue
│   │   │   ├── PatternGrid.vue
│   │   │   └── IngestionControl.vue
│   │   └── shared/
│   │       ├── MetricCard.vue
│   │       └── StatusIndicator.vue
│   ├── composables/
│   │   ├── useEchoAPI.js     # API calls composition
│   │   ├── useWebSocket.js   # WebSocket handling
│   │   └── useMetrics.js     # Metrics polling
│   ├── App.vue
│   └── main.js
├── public/
│   └── index.html
├── vite.config.js
└── package.json
```

## Implementation Plan

### Phase 1: Setup Vue Router & Pinia
```javascript
// router/index.js
import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'conversation',
    component: () => import('../views/ConversationView.vue')
  },
  {
    path: '/learning',
    name: 'learning',
    component: () => import('../views/LearningView.vue')
  },
  {
    path: '/tasks',
    name: 'tasks',
    component: () => import('../views/TasksView.vue')
  },
  {
    path: '/analytics',
    name: 'analytics',
    component: () => import('../views/AnalyticsView.vue')
  },
  {
    path: '/settings',
    name: 'settings',
    component: () => import('../views/SettingsView.vue')
  }
]

export default createRouter({
  history: createWebHistory('/echo/'),
  routes
})
```

### Phase 2: Pinia Stores for State Management
```javascript
// stores/conversation.js
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useConversationStore = defineStore('conversation', () => {
  const messages = ref([])
  const conversationId = ref(null)
  const isLoading = ref(false)

  const sendMessage = async (query) => {
    isLoading.value = true
    const response = await fetch('/api/echo/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, conversation_id: conversationId.value })
    })
    const data = await response.json()
    messages.value.push({ role: 'user', content: query })
    messages.value.push({ role: 'assistant', content: data.response })
    isLoading.value = false
  }

  return { messages, conversationId, isLoading, sendMessage }
})
```

### Phase 3: Service Configuration

#### Nginx Configuration
```nginx
# /etc/nginx/sites-available/tower.conf additions

# Echo Brain SPA (separate from dashboard)
location /echo/ {
    proxy_pass http://localhost:8310/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection 'upgrade';
    proxy_set_header Host $host;
    proxy_cache_bypass $http_upgrade;
}

# Echo Brain API (shared by both dashboard and SPA)
location /api/echo/ {
    proxy_pass http://localhost:8309/api/echo/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

#### Systemd Service
```ini
# /etc/systemd/system/echo-brain-spa.service
[Unit]
Description=Echo Brain Vue SPA
After=network.target

[Service]
Type=simple
User=patrick
WorkingDirectory=/opt/tower-echo-brain/frontend
ExecStart=/usr/bin/npm run preview -- --port 8310 --host 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Integration Points

### Tower Dashboard → Echo SPA
```javascript
// In Tower Dashboard index.html
// Add link to Echo Brain full interface
<div class="echo-brain-status">
  <h3>Echo Brain Status</h3>
  <div class="metrics"><!-- status metrics --></div>
  <a href="/echo/" class="btn-primary">
    Open Echo Brain Interface →
  </a>
</div>
```

### Echo SPA → Tower Dashboard
```javascript
// In Echo SPA navigation
<router-link to="/" class="nav-link">Conversation</router-link>
<router-link to="/learning" class="nav-link">Learning</router-link>
<router-link to="/tasks" class="nav-link">Tasks</router-link>
<a href="/" class="nav-link">← Back to Dashboard</a>
```

## API Usage Strategy

### Dashboard (Status Only)
```javascript
// Only polls these endpoints for monitoring
GET /api/echo/health
GET /api/echo/system/metrics
GET /api/echo/db/stats
GET /api/echo/tasks/queue  // Just count, no details
```

### Echo SPA (Full Functionality)
```javascript
// Full API access for interaction
POST /api/echo/query
POST /api/echo/feedback
POST /api/echo/learning/comprehensive-ingestion
GET /api/echo/conversations
GET /api/echo/thoughts/recent
GET /api/echo/learning/pattern-analytics
// ... all other endpoints
```

## Benefits of This Architecture

1. **Clean Separation**
   - Dashboard: Monitoring only
   - Echo SPA: Functionality only
   - No mixing of concerns

2. **Independent Development**
   - Can update Echo SPA without touching dashboard
   - Can add features to Echo without dashboard changes
   - Clear boundaries between systems

3. **Performance**
   - Dashboard stays lightweight (status only)
   - Echo SPA can be as complex as needed
   - Separate build processes

4. **User Experience**
   - Quick status check in dashboard
   - Full interface when needed
   - No cluttered dashboard

5. **Maintainability**
   - Vue SPA follows standard patterns
   - Dashboard remains simple HTML/JS
   - Clear API boundaries

## File Organization

```
/opt/
├── tower-dashboard/              # Simple monitoring dashboard
│   ├── index.html               # Keep as-is with status tab
│   ├── assets/
│   │   ├── dashboard.js         # Status polling only
│   │   └── echo-status.js       # Echo status display (new)
│   └── server.js                # Express server (port 8080)
│
├── tower-echo-brain/
│   ├── src/                     # Backend API (port 8309)
│   │   └── api/                 # All endpoints
│   │
│   └── frontend/                # Vue SPA (port 8310)
│       ├── dist/                # Built SPA
│       └── src/                 # Vue source code
│
└── nginx/
    └── sites-available/
        └── tower.conf           # Routes for both
```

## Commands to Implement

```bash
# 1. Install Vue Router and Pinia
cd /opt/tower-echo-brain/frontend
npm install vue-router@4 pinia

# 2. Build the SPA
npm run build

# 3. Create systemd service
sudo nano /etc/systemd/system/echo-brain-spa.service

# 4. Update nginx
sudo nano /etc/nginx/sites-available/tower.conf
sudo nginx -t
sudo systemctl reload nginx

# 5. Start the SPA
sudo systemctl enable echo-brain-spa
sudo systemctl start echo-brain-spa
```

## Access Points

- **Tower Dashboard**: https://tower.local/ (monitoring all services)
  - Echo Brain Status Tab: Shows health metrics only

- **Echo Brain SPA**: https://tower.local/echo/ (full functionality)
  - Conversation interface
  - Learning pipeline control
  - Task management
  - Analytics dashboard
  - Settings & configuration
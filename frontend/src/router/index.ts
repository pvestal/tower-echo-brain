import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'dashboard',
    component: () => import('@/views/Dashboard.vue')
  },
  {
    path: '/knowledge',
    name: 'knowledge',
    component: () => import('@/views/Knowledge.vue')
  },
  {
    path: '/preferences',
    name: 'preferences',
    component: () => import('@/views/Preferences.vue')
  },
  {
    path: '/vault',
    name: 'vault',
    component: () => import('@/views/Vault.vue')
  },
  {
    path: '/integrations',
    name: 'integrations',
    component: () => import('@/views/Integrations.vue')
  },
  // Legacy routes
  {
    path: '/health',
    name: 'HealthCheck',
    component: () => import('@/views/HealthCheck.vue')
  },
  {
    path: '/chat-test',
    name: 'ChatTest',
    component: () => import('@/views/ChatTest.vue')
  },
  {
    path: '/logs',
    name: 'LogViewer',
    component: () => import('@/views/LogViewer.vue')
  }
]

export const router = createRouter({
  history: createWebHistory('/echo-brain/'),
  routes
})
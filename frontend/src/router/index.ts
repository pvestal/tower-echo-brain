import { createRouter, createWebHistory } from 'vue-router';
import DashboardView from '@/views/DashboardView.vue';

const routes = [
  {
    path: '/',
    redirect: '/dashboard'
  },
  {
    path: '/dashboard',
    name: 'Dashboard',
    component: DashboardView
  },
  {
    path: '/ask',
    name: 'Ask',
    component: () => import('@/views/AskView.vue')
  },
  {
    path: '/memory',
    name: 'Memory',
    component: () => import('@/views/MemoryView.vue')
  },
  {
    path: '/endpoints',
    name: 'Endpoints',
    component: () => import('@/views/EndpointsView.vue')
  },
  {
    path: '/system',
    name: 'System',
    component: () => import('@/views/SystemView.vue')
  },
  {
    path: '/logs',
    name: 'Logs',
    component: () => import('@/views/LogsView.vue')
  },
  {
    path: '/voice',
    name: 'Voice',
    component: () => import('@/views/VoicePanel.vue')
  },
  {
    path: '/voice-simple',
    name: 'VoiceSimple',
    component: () => import('@/views/VoiceSimple.vue')
  },
  {
    path: '/voice-test',
    name: 'VoiceTest',
    component: () => import('@/views/VoiceTest.vue')
  },
  {
    path: '/calendar',
    name: 'Calendar',
    component: () => import('@/views/CalendarView.vue')
  }
];

export const router = createRouter({
  history: createWebHistory('/echo-brain/'),
  routes,
});
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
  }
];

export const router = createRouter({
  history: createWebHistory('/echo-brain/'),
  routes,
});
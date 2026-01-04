import { createRouter, createWebHistory } from 'vue-router'
import HealthCheck from './views/HealthCheck.vue'
import ChatTest from './views/ChatTest.vue'
import LogViewer from './views/LogViewer.vue'

const routes = [
  {
    path: '/',
    name: 'HealthCheck',
    component: HealthCheck
  },
  {
    path: '/chat-test',
    name: 'ChatTest',
    component: ChatTest
  },
  {
    path: '/logs',
    name: 'LogViewer',
    component: LogViewer
  }
]

const router = createRouter({
  history: createWebHistory('/echo-brain/dist/'),
  routes
})

export default router
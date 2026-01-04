import { createApp } from 'vue'
import { createPinia } from 'pinia'
import router from './router'
import App from './App.vue'
import './style.css'

// Import Tower UI Components
import {
  TowerCard,
  TowerButton,
  TowerInput,
  TowerModal,
  TowerSelect,
  TowerTextarea
} from './components'

const app = createApp(App)
const pinia = createPinia()

// Register Tower UI Components globally
app.component('TowerCard', TowerCard)
app.component('TowerButton', TowerButton)
app.component('TowerInput', TowerInput)
app.component('TowerModal', TowerModal)
app.component('TowerSelect', TowerSelect)
app.component('TowerTextarea', TowerTextarea)

app.use(pinia)
app.use(router)
app.mount('#app')

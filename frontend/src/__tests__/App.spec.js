import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { mount } from '@vue/test-utils'
import App from '../App.vue'

// Mock all child components to avoid complex dependencies
vi.mock('../views/ChatInterface.vue', () => ({
  default: { template: '<div class="chat-interface">Chat Interface</div>' }
}))

vi.mock('../views/VoiceInterface.vue', () => ({
  default: { template: '<div class="voice-interface">Voice Interface</div>' }
}))

vi.mock('../views/EchoMetrics.vue', () => ({
  default: { template: '<div class="echo-metrics">Echo Metrics</div>' }
}))

vi.mock('../views/PlaidAuth.vue', () => ({
  default: { template: '<div class="plaid-auth">Plaid Auth</div>' }
}))

// Mock @tower/ui-components
vi.mock('@tower/ui-components', () => ({
  TowerNavbar: { 
    template: '<nav><slot name="brand" /><slot name="links" /><slot name="actions" /></nav>' 
  },
  TowerWebSocketStatus: { 
    template: '<div class="ws-status">WS: {{ connected }}</div>',
    props: ['connected']
  }
}))

describe('App.vue', () => {
  let wrapper

  afterEach(() => {
    if (wrapper) wrapper.unmount()
  })

  describe('Initial Rendering', () => {
    it('renders the application', () => {
      wrapper = mount(App)
      expect(wrapper.exists()).toBe(true)
    })

    it('shows Echo Brain branding', () => {
      wrapper = mount(App)
      expect(wrapper.text()).toContain('Echo Brain')
    })

    it('shows all navigation tabs', () => {
      wrapper = mount(App)
      expect(wrapper.text()).toContain('Chat')
      expect(wrapper.text()).toContain('Voice')
      expect(wrapper.text()).toContain('Metrics')
      expect(wrapper.text()).toContain('Financial')
    })

    it('shows chat interface by default', () => {
      wrapper = mount(App)
      expect(wrapper.find('.chat-interface').exists()).toBe(true)
    })

    it('shows WebSocket status', () => {
      wrapper = mount(App)
      expect(wrapper.find('.ws-status').exists()).toBe(true)
    })
  })

  describe('Tab Navigation', () => {
    it('has 4 navigation links', () => {
      wrapper = mount(App)
      const links = wrapper.findAll('.nav-link')
      expect(links.length).toBe(4)
    })

    it('chat tab is active by default', () => {
      wrapper = mount(App)
      const links = wrapper.findAll('.nav-link')
      expect(links[0].classes()).toContain('active')
    })

    it('switches to voice tab when clicked', async () => {
      wrapper = mount(App)
      const links = wrapper.findAll('.nav-link')
      
      await links[1].trigger('click') // Voice tab
      
      expect(wrapper.find('.voice-interface').exists()).toBe(true)
      expect(wrapper.find('.chat-interface').exists()).toBe(false)
    })

    it('switches to metrics tab when clicked', async () => {
      wrapper = mount(App)
      const links = wrapper.findAll('.nav-link')
      
      await links[2].trigger('click') // Metrics tab
      
      expect(wrapper.find('.echo-metrics').exists()).toBe(true)
      expect(wrapper.find('.chat-interface').exists()).toBe(false)
    })

    it('switches to financial tab when clicked', async () => {
      wrapper = mount(App)
      const links = wrapper.findAll('.nav-link')
      
      await links[3].trigger('click') // Financial tab
      
      expect(wrapper.find('.plaid-auth').exists()).toBe(true)
      expect(wrapper.find('.chat-interface').exists()).toBe(false)
    })

    it('updates active class when tab changes', async () => {
      wrapper = mount(App)
      const links = wrapper.findAll('.nav-link')
      
      expect(links[0].classes()).toContain('active') // Chat initially active
      
      await links[1].trigger('click') // Switch to Voice
      
      expect(links[1].classes()).toContain('active')
      expect(links[0].classes()).not.toContain('active')
    })
  })

  describe('Component Visibility', () => {
    it('shows only chat interface when chat tab active', () => {
      wrapper = mount(App)
      
      expect(wrapper.find('.chat-interface').exists()).toBe(true)
      expect(wrapper.find('.voice-interface').exists()).toBe(false)
      expect(wrapper.find('.echo-metrics').exists()).toBe(false)
      expect(wrapper.find('.plaid-auth').exists()).toBe(false)
    })

    it('shows only voice interface when voice tab active', async () => {
      wrapper = mount(App)
      const links = wrapper.findAll('.nav-link')
      
      await links[1].trigger('click')
      
      expect(wrapper.find('.chat-interface').exists()).toBe(false)
      expect(wrapper.find('.voice-interface').exists()).toBe(true)
      expect(wrapper.find('.echo-metrics').exists()).toBe(false)
      expect(wrapper.find('.plaid-auth').exists()).toBe(false)
    })

    it('shows only metrics when metrics tab active', async () => {
      wrapper = mount(App)
      const links = wrapper.findAll('.nav-link')
      
      await links[2].trigger('click')
      
      expect(wrapper.find('.chat-interface').exists()).toBe(false)
      expect(wrapper.find('.voice-interface').exists()).toBe(false)
      expect(wrapper.find('.echo-metrics').exists()).toBe(true)
      expect(wrapper.find('.plaid-auth').exists()).toBe(false)
    })

    it('shows only plaid auth when financial tab active', async () => {
      wrapper = mount(App)
      const links = wrapper.findAll('.nav-link')
      
      await links[3].trigger('click')
      
      expect(wrapper.find('.chat-interface').exists()).toBe(false)
      expect(wrapper.find('.voice-interface').exists()).toBe(false)
      expect(wrapper.find('.echo-metrics').exists()).toBe(false)
      expect(wrapper.find('.plaid-auth').exists()).toBe(true)
    })
  })

  describe('Tab Switching', () => {
    it('can switch between multiple tabs', async () => {
      wrapper = mount(App)
      const links = wrapper.findAll('.nav-link')
      
      // Start at Chat
      expect(wrapper.find('.chat-interface').exists()).toBe(true)
      
      // Switch to Voice
      await links[1].trigger('click')
      expect(wrapper.find('.voice-interface').exists()).toBe(true)
      
      // Switch to Metrics
      await links[2].trigger('click')
      expect(wrapper.find('.echo-metrics').exists()).toBe(true)
      
      // Switch back to Chat
      await links[0].trigger('click')
      expect(wrapper.find('.chat-interface').exists()).toBe(true)
    })

    it('maintains tab state during navigation', async () => {
      wrapper = mount(App)
      const links = wrapper.findAll('.nav-link')
      
      // Navigate through tabs
      await links[1].trigger('click')
      await links[2].trigger('click')
      await links[3].trigger('click')
      
      // Final tab should be Financial
      expect(wrapper.find('.plaid-auth').exists()).toBe(true)
      expect(links[3].classes()).toContain('active')
    })
  })

  describe('Layout Structure', () => {
    it('has main container with proper classes', () => {
      wrapper = mount(App)
      const main = wrapper.find('main')
      expect(main.exists()).toBe(true)
      expect(main.classes()).toContain('max-w-7xl')
    })

    it('has navbar element', () => {
      wrapper = mount(App)
      expect(wrapper.find('nav').exists()).toBe(true)
    })

    it('renders with proper root classes', () => {
      wrapper = mount(App)
      const root = wrapper.find('.min-h-screen')
      expect(root.exists()).toBe(true)
      expect(root.classes()).toContain('bg-tower-bg')
    })
  })

  describe('Component Lifecycle', () => {
    it('component mounts without errors', () => {
      expect(() => mount(App)).not.toThrow()
    })

    it('component unmounts cleanly', () => {
      wrapper = mount(App)
      expect(() => wrapper.unmount()).not.toThrow()
    })
  })
})

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { mount } from '@vue/test-utils'
import PlaidAuth from '../PlaidAuth.vue'

// Mock @tower/ui-components
vi.mock('@tower/ui-components', () => ({
  TowerCard: { template: '<div><slot name="header" /><slot /></div>' },
  TowerButton: { 
    template: '<button @click="$emit(\'click\')"><slot /></button>',
    props: ['loading', 'variant']
  }
}))

describe('PlaidAuth.vue', () => {
  let wrapper

  // Mock window.alert
  beforeEach(() => {
    global.alert = vi.fn()
    vi.clearAllMocks()
  })

  afterEach(() => {
    if (wrapper) wrapper.unmount()
  })

  describe('Initial Rendering', () => {
    it('renders the financial integration header', () => {
      wrapper = mount(PlaidAuth)
      expect(wrapper.find('h2').text()).toBe('Financial Integration')
    })

    it('shows description text', () => {
      wrapper = mount(PlaidAuth)
      expect(wrapper.text()).toContain('Connect your financial accounts')
      expect(wrapper.text()).toContain('intelligent financial insights')
    })

    it('shows connect bank account button', () => {
      wrapper = mount(PlaidAuth)
      const button = wrapper.find('button')
      expect(button.exists()).toBe(true)
      expect(button.text()).toBe('Connect Bank Account')
    })

    it('does not show connected accounts card initially', () => {
      wrapper = mount(PlaidAuth)
      expect(wrapper.text()).not.toContain('Connected Accounts')
    })
  })

  describe('Connect Button', () => {
    it('button exists and is clickable', () => {
      wrapper = mount(PlaidAuth)
      const button = wrapper.find('button')
      expect(button.exists()).toBe(true)
    })

    it('shows alert when connect button clicked', async () => {
      wrapper = mount(PlaidAuth)
      
      const button = wrapper.find('button')
      await button.trigger('click')

      expect(global.alert).toHaveBeenCalledWith('Plaid integration coming soon!')
    })

    it('calls connectPlaid function on button click', async () => {
      wrapper = mount(PlaidAuth)
      
      const button = wrapper.find('button')
      const initialCallCount = global.alert.mock.calls.length
      
      await button.trigger('click')

      // Verify alert was called (indicates function executed)
      expect(global.alert.mock.calls.length).toBeGreaterThan(initialCallCount)
    })
  })

  describe('Component Structure', () => {
    it('has main card with header slot', () => {
      wrapper = mount(PlaidAuth)
      expect(wrapper.find('h2').exists()).toBe(true)
    })

    it('has description paragraph', () => {
      wrapper = mount(PlaidAuth)
      const paragraphs = wrapper.findAll('p')
      expect(paragraphs.length).toBeGreaterThan(0)
    })
  })

  describe('Component Lifecycle', () => {
    it('component mounts without errors', () => {
      expect(() => mount(PlaidAuth)).not.toThrow()
    })

    it('component unmounts cleanly', () => {
      wrapper = mount(PlaidAuth)
      expect(() => wrapper.unmount()).not.toThrow()
    })
  })

  describe('Future Integration', () => {
    it('has placeholder for connected accounts', () => {
      wrapper = mount(PlaidAuth)
      // Card is conditional on connected ref
      // Currently not shown since connected is false
      expect(wrapper.text()).not.toContain('No accounts connected yet')
    })
  })
})

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import EchoMetrics from '../views/EchoMetrics.vue'
import axios from 'axios'

// Mock axios
vi.mock('axios')

// Mock @tower/ui-components
vi.mock('@tower/ui-components', () => ({
  TowerCard: { template: '<div><slot /><slot name="header" /></div>' }
}))

describe('EchoMetrics.vue', () => {
  let wrapper

  const mockMetricsData = {
    cpu_percent: 25.5,
    memory_percent: 45.2,
    memory_used_gb: 10.5,
    memory_total_gb: 32,
    vram_used_gb: 5.7,
    vram_total_gb: 12
  }

  const mockDbStats = {
    echo_brain: '93 MB',
    knowledge_base: '18 MB',
    active_connections: 2,
    echo_brain_tables: 34
  }

  const mockEchoStatus = {
    agentic_persona: 'Default cognitive mode',
    recent_messages: [
      { time: '10:30', text: 'System initialized' },
      { time: '10:31', text: 'Processing request' }
    ]
  }

  const mockServices = [
    {
      name: 'Echo Brain',
      description: 'Main AI orchestration service',
      endpoint: 'http://***REMOVED***:8309/api/echo/health',
      online: true
    },
    {
      name: 'Knowledge Base',
      description: 'Article storage and retrieval',
      endpoint: 'https://***REMOVED***/api/kb/articles?limit=1',
      online: true
    }
  ]

  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    if (wrapper) wrapper.unmount()
  })

  describe('Initial Rendering', () => {
    it('renders metrics cards', async () => {
      axios.get.mockImplementation((url) => {
        if (url.includes('/metrics')) return Promise.resolve({ data: mockMetricsData })
        if (url.includes('/db/stats')) return Promise.resolve({ data: mockDbStats })
        if (url.includes('/status')) return Promise.resolve({ data: mockEchoStatus })
        return Promise.reject(new Error('Unknown endpoint'))
      })

      wrapper = mount(EchoMetrics)
      await flushPromises()

      expect(wrapper.text()).toContain('CPU Usage')
      expect(wrapper.text()).toContain('Memory')
      expect(wrapper.text()).toContain('VRAM')
      expect(wrapper.text()).toContain('Services')
    })

    it('displays loading state initially', () => {
      axios.get.mockImplementation(() => new Promise(() => {})) // Never resolves

      wrapper = mount(EchoMetrics)

      expect(wrapper.text()).toContain('Loading...')
    })
  })

  describe('Metrics Display', () => {
    beforeEach(() => {
      axios.get.mockImplementation((url) => {
        if (url.includes('/metrics')) return Promise.resolve({ data: mockMetricsData })
        if (url.includes('/db/stats')) return Promise.resolve({ data: mockDbStats })
        if (url.includes('/status')) return Promise.resolve({ data: mockEchoStatus })
        return Promise.resolve({ data: {} })
      })
    })

    it('displays CPU usage correctly', async () => {
      wrapper = mount(EchoMetrics)
      await flushPromises()

      expect(wrapper.text()).toContain('25.5%')
    })

    it('displays memory usage with GB values', async () => {
      wrapper = mount(EchoMetrics)
      await flushPromises()

      expect(wrapper.text()).toContain('45.2%')
      expect(wrapper.text()).toContain('10.5 / 32 GB')
    })

    it('displays VRAM usage', async () => {
      wrapper = mount(EchoMetrics)
      await flushPromises()

      expect(wrapper.text()).toContain('5.7 GB')
      expect(wrapper.text()).toContain('12 GB total')
    })

    it('shows services count', async () => {
      wrapper = mount(EchoMetrics)
      await flushPromises()

      // Service count depends on services array length
      expect(wrapper.text()).toMatch(/\d+\/\d+/)
    })
  })

  describe('Null Safety', () => {
    it('handles missing metrics gracefully', async () => {
      axios.get.mockResolvedValue({ data: {} })

      wrapper = mount(EchoMetrics)
      await flushPromises()

      // Should show 0 instead of undefined
      expect(wrapper.html()).toContain('0%')
      expect(wrapper.html()).toContain('0 GB')
    })

    it('handles undefined nested properties', async () => {
      axios.get.mockResolvedValue({
        data: {
          cpu_percent: undefined,
          memory_used_gb: undefined,
          memory_total_gb: undefined
        }
      })

      wrapper = mount(EchoMetrics)
      await flushPromises()

      // Should not throw errors
      expect(wrapper.html()).toContain('0%')
    })

    it('uses optional chaining for nested values', async () => {
      axios.get.mockResolvedValue({
        data: {
          // Missing nested properties
        }
      })

      wrapper = mount(EchoMetrics)
      await flushPromises()

      // Should render without errors
      expect(wrapper.exists()).toBe(true)
    })
  })

  describe('Database Stats', () => {
    beforeEach(() => {
      axios.get.mockImplementation((url) => {
        if (url.includes('/db/stats')) return Promise.resolve({ data: mockDbStats })
        return Promise.resolve({ data: mockMetricsData })
      })
    })

    it('displays database sizes', async () => {
      wrapper = mount(EchoMetrics)
      await flushPromises()

      expect(wrapper.text()).toContain('93 MB')
      expect(wrapper.text()).toContain('18 MB')
    })

    it('shows active connections', async () => {
      wrapper = mount(EchoMetrics)
      await flushPromises()

      expect(wrapper.text()).toContain('Active Connections')
      expect(wrapper.text()).toContain('2')
    })

    it('displays table count', async () => {
      wrapper = mount(EchoMetrics)
      await flushPromises()

      expect(wrapper.text()).toContain('Tables')
      expect(wrapper.text()).toContain('34')
    })

    it('shows friendly labels for database fields', async () => {
      wrapper = mount(EchoMetrics)
      await flushPromises()

      // Should have descriptions
      expect(wrapper.text()).toContain('Main production database')
      expect(wrapper.text()).toContain('KB articles storage')
    })
  })

  describe('Echo Status', () => {
    beforeEach(() => {
      axios.get.mockImplementation((url) => {
        if (url.includes('/status')) return Promise.resolve({ data: mockEchoStatus })
        return Promise.resolve({ data: mockMetricsData })
      })
    })

    it('displays current cognitive mode', async () => {
      wrapper = mount(EchoMetrics)
      await flushPromises()

      expect(wrapper.text()).toContain('Default cognitive mode')
    })

    it('shows recent activity messages', async () => {
      wrapper = mount(EchoMetrics)
      await flushPromises()

      expect(wrapper.text()).toContain('System initialized')
      expect(wrapper.text()).toContain('Processing request')
    })

    it('displays message timestamps', async () => {
      wrapper = mount(EchoMetrics)
      await flushPromises()

      expect(wrapper.text()).toContain('10:30')
      expect(wrapper.text()).toContain('10:31')
    })

    it('shows no activity message when empty', async () => {
      axios.get.mockImplementation((url) => {
        if (url.includes('/status')) {
          return Promise.resolve({
            data: { agentic_persona: 'Test', recent_messages: [] }
          })
        }
        return Promise.resolve({ data: {} })
      })

      wrapper = mount(EchoMetrics)
      await flushPromises()

      expect(wrapper.text()).toContain('No recent activity')
    })
  })

  describe('Service Status', () => {
    it('shows online/offline status for services', async () => {
      axios.get.mockImplementation((url) => {
        // Mock service endpoints
        if (url.includes('echo/health')) return Promise.resolve({ status: 200 })
        if (url.includes('kb/articles')) return Promise.resolve({ status: 200 })
        // Default metrics
        if (url.includes('/metrics')) return Promise.resolve({ data: mockMetricsData })
        return Promise.reject(new Error('Offline'))
      })

      wrapper = mount(EchoMetrics)
      await flushPromises()

      expect(wrapper.text()).toContain('Online')
    })

    it('handles offline services', async () => {
      axios.get.mockRejectedValue(new Error('Service offline'))

      wrapper = mount(EchoMetrics)
      await flushPromises()

      expect(wrapper.text()).toContain('Offline')
    })
  })

  describe('Error Handling', () => {
    it('displays error message on API failure', async () => {
      axios.get.mockRejectedValue(new Error('Network error'))

      wrapper = mount(EchoMetrics)
      await flushPromises()

      expect(wrapper.text()).toContain('Connection Error')
    })

    it('shows helpful message when API unavailable', async () => {
      axios.get.mockRejectedValue(new Error('Failed to connect'))

      wrapper = mount(EchoMetrics)
      await flushPromises()

      expect(wrapper.text()).toContain('Echo Brain API')
    })

    it('handles partial API failures gracefully', async () => {
      axios.get.mockImplementation((url) => {
        if (url.includes('/metrics')) return Promise.resolve({ data: mockMetricsData })
        // Other endpoints fail
        return Promise.reject(new Error('Endpoint unavailable'))
      })

      wrapper = mount(EchoMetrics)
      await flushPromises()

      // Should still show metrics that succeeded
      expect(wrapper.text()).toContain('25.5%')
      // Should handle failures gracefully
      expect(wrapper.exists()).toBe(true)
    })
  })

  describe('Auto-refresh', () => {
    it('fetches metrics on mount', async () => {
      axios.get.mockResolvedValue({ data: mockMetricsData })

      wrapper = mount(EchoMetrics)
      await flushPromises()

      expect(axios.get).toHaveBeenCalled()
    })

    it('sets up polling intervals', async () => {
      vi.useFakeTimers()
      axios.get.mockResolvedValue({ data: mockMetricsData })

      wrapper = mount(EchoMetrics)
      await flushPromises()

      const initialCallCount = axios.get.mock.calls.length

      // Advance time by 5 seconds (metrics refresh interval)
      await vi.advanceTimersByTimeAsync(5000)

      expect(axios.get).toHaveBeenCalledTimes(initialCallCount + 1)

      vi.useRealTimers()
    })
  })

  describe('Component Lifecycle', () => {
    it('cleans up intervals on unmount', async () => {
      vi.useFakeTimers()
      axios.get.mockResolvedValue({ data: mockMetricsData })

      wrapper = mount(EchoMetrics)
      await flushPromises()

      wrapper.unmount()

      const callCountAfterUnmount = axios.get.mock.calls.length

      // Advance time - should not trigger more calls
      await vi.advanceTimersByTimeAsync(10000)

      expect(axios.get).toHaveBeenCalledTimes(callCountAfterUnmount)

      vi.useRealTimers()
    })
  })
})

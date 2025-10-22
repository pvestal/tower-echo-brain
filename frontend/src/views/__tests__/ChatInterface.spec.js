import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import ChatInterface from '../views/ChatInterface.vue'
import axios from 'axios'

// Mock axios
vi.mock('axios')

// Mock @tower/ui-components
vi.mock('@tower/ui-components', () => ({
  TowerCard: { template: '<div><slot /></div>' },
  TowerButton: { template: '<button @click="$emit(\'click\')"><slot /></button>', props: ['loading'] },
  TowerInput: { template: '<input :value="modelValue" @input="$emit(\'update:modelValue\', $event.target.value)" />', props: ['modelValue', 'placeholder'] }
}))

describe('ChatInterface.vue', () => {
  let wrapper

  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    if (wrapper) wrapper.unmount()
  })

  describe('Basic Rendering', () => {
    it('renders the chat interface', () => {
      wrapper = mount(ChatInterface)
      expect(wrapper.find('h2').text()).toBe('Chat with Echo')
    })

    it('has an input field for messages', () => {
      wrapper = mount(ChatInterface)
      const input = wrapper.find('input')
      expect(input.exists()).toBe(true)
    })

    it('has a send button', () => {
      wrapper = mount(ChatInterface)
      const button = wrapper.find('button')
      expect(button.exists()).toBe(true)
    })
  })

  describe('Message Sending', () => {
    it('sends message with correct payload format', async () => {
      const mockResponse = {
        data: {
          response: 'Test response',
          model_used: 'llama3.2:3b',
          intelligence_level: 'standard'
        },
        status: 200
      }
      axios.post.mockResolvedValueOnce(mockResponse)

      wrapper = mount(ChatInterface)

      // Type a message
      const input = wrapper.find('input')
      await input.setValue('Hello Echo')

      // Click send
      const button = wrapper.find('button')
      await button.trigger('click')
      await flushPromises()

      // Verify API was called with correct payload
      expect(axios.post).toHaveBeenCalledWith(
        'http://192.168.50.135:8309/api/echo/chat',
        {
          query: 'Hello Echo',
          user_id: 'web_user',
          conversation_id: expect.stringMatching(/^web_chat_\d+$/),
          intelligence_level: 'auto',
          context: {}
        },
        {
          timeout: 30000,
          headers: {
            'Content-Type': 'application/json'
          }
        }
      )
    })

    it('displays user message immediately', async () => {
      axios.post.mockResolvedValueOnce({
        data: { response: 'Echo response' }
      })

      wrapper = mount(ChatInterface)

      await wrapper.find('input').setValue('Test message')
      await wrapper.find('button').trigger('click')

      // User message should appear immediately
      const messages = wrapper.findAll('.message')
      expect(messages.length).toBeGreaterThan(0)
      expect(messages[0].text()).toContain('Test message')
    })

    it('displays assistant response after API call', async () => {
      axios.post.mockResolvedValueOnce({
        data: { response: 'Echo response from AI' }
      })

      wrapper = mount(ChatInterface)

      await wrapper.find('input').setValue('Test')
      await wrapper.find('button').trigger('click')
      await flushPromises()

      const messages = wrapper.findAll('.message')
      expect(messages[messages.length - 1].text()).toContain('Echo response from AI')
    })

    it('clears input after sending', async () => {
      axios.post.mockResolvedValueOnce({
        data: { response: 'Response' }
      })

      wrapper = mount(ChatInterface)
      const input = wrapper.find('input')

      await input.setValue('Test message')
      expect(input.element.value).toBe('Test message')

      await wrapper.find('button').trigger('click')
      await flushPromises()

      expect(input.element.value).toBe('')
    })
  })

  describe('Error Handling', () => {
    it('displays error message on API failure', async () => {
      axios.post.mockRejectedValueOnce(new Error('Network error'))

      wrapper = mount(ChatInterface)

      await wrapper.find('input').setValue('Test')
      await wrapper.find('button').trigger('click')
      await flushPromises()

      const errorMessages = wrapper.findAll('.message.error')
      expect(errorMessages.length).toBeGreaterThan(0)
      expect(errorMessages[0].text()).toContain('Error:')
    })

    it('shows retry button on retryable errors', async () => {
      axios.post.mockRejectedValueOnce({
        response: { status: 500 },
        message: 'Server error'
      })

      wrapper = mount(ChatInterface)

      await wrapper.find('input').setValue('Test')
      await wrapper.find('button').trigger('click')
      await flushPromises()

      expect(wrapper.text()).toContain('Retry')
    })
  })

  describe('Retry Logic', () => {
    it('retries on timeout errors with exponential backoff', async () => {
      // First two attempts fail, third succeeds
      axios.post
        .mockRejectedValueOnce({ name: 'AbortError', message: 'Timeout' })
        .mockRejectedValueOnce({ name: 'AbortError', message: 'Timeout' })
        .mockResolvedValueOnce({ data: { response: 'Success after retry' } })

      wrapper = mount(ChatInterface)

      await wrapper.find('input').setValue('Test')
      await wrapper.find('button').trigger('click')

      // First attempt
      await flushPromises()

      // Wait for retry delay (1s)
      await vi.advanceTimersByTimeAsync(1000)
      await flushPromises()

      // Wait for second retry delay (2s)
      await vi.advanceTimersByTimeAsync(2000)
      await flushPromises()

      // Should have made 3 attempts total
      expect(axios.post).toHaveBeenCalledTimes(3)

      // Should show success message
      const messages = wrapper.findAll('.message')
      const lastMessage = messages[messages.length - 1]
      expect(lastMessage.text()).toContain('Success after retry')
    })

    it('stops retrying after max attempts', async () => {
      // All attempts fail
      axios.post.mockRejectedValue({
        name: 'AbortError',
        message: 'Timeout'
      })

      wrapper = mount(ChatInterface)

      await wrapper.find('input').setValue('Test')
      await wrapper.find('button').trigger('click')

      // Initial attempt
      await flushPromises()

      // Retry 1 (1s delay)
      await vi.advanceTimersByTimeAsync(1000)
      await flushPromises()

      // Retry 2 (2s delay)
      await vi.advanceTimersByTimeAsync(2000)
      await flushPromises()

      // Retry 3 (4s delay)
      await vi.advanceTimersByTimeAsync(4000)
      await flushPromises()

      // Should have made exactly 3 attempts (MAX_RETRIES)
      expect(axios.post).toHaveBeenCalledTimes(3)

      // Should show error with retry button
      const errorMessages = wrapper.findAll('.message.error')
      expect(errorMessages.length).toBeGreaterThan(0)
    })

    it('does not retry on non-retryable errors (422)', async () => {
      axios.post.mockRejectedValueOnce({
        response: {
          status: 422,
          data: { detail: 'Validation error' }
        }
      })

      wrapper = mount(ChatInterface)

      await wrapper.find('input').setValue('Test')
      await wrapper.find('button').trigger('click')
      await flushPromises()

      // Should only attempt once (422 is not retryable)
      expect(axios.post).toHaveBeenCalledTimes(1)

      // Should show error without retry button
      const text = wrapper.text()
      expect(text).toContain('Error:')
      expect(text).not.toContain('Retry')
    })

    it('retries on 500 errors', async () => {
      axios.post
        .mockRejectedValueOnce({ response: { status: 500 } })
        .mockResolvedValueOnce({ data: { response: 'Success' } })

      wrapper = mount(ChatInterface)

      await wrapper.find('input').setValue('Test')
      await wrapper.find('button').trigger('click')
      await flushPromises()

      await vi.advanceTimersByTimeAsync(1000)
      await flushPromises()

      expect(axios.post).toHaveBeenCalledTimes(2)
    })
  })

  describe('Connection Error Banner', () => {
    it('shows connection error banner when retrying', async () => {
      axios.post.mockRejectedValue({ name: 'AbortError' })

      wrapper = mount(ChatInterface)

      await wrapper.find('input').setValue('Test')
      await wrapper.find('button').trigger('click')
      await flushPromises()

      expect(wrapper.text()).toContain('Connection issues detected')
    })

    it('hides banner on successful request', async () => {
      axios.post.mockResolvedValueOnce({
        data: { response: 'Success' }
      })

      wrapper = mount(ChatInterface)

      await wrapper.find('input').setValue('Test')
      await wrapper.find('button').trigger('click')
      await flushPromises()

      expect(wrapper.text()).not.toContain('Connection issues detected')
    })
  })

  describe('Loading State', () => {
    it('disables input while loading', async () => {
      axios.post.mockImplementation(() => new Promise(resolve => {
        setTimeout(() => resolve({ data: { response: 'Done' } }), 1000)
      }))

      wrapper = mount(ChatInterface)
      const input = wrapper.find('input')

      await input.setValue('Test')
      await wrapper.find('button').trigger('click')

      // Should be disabled during request
      expect(input.attributes('disabled')).toBeDefined()
    })

    it('shows loading state on send button', async () => {
      axios.post.mockImplementation(() => new Promise(resolve => {
        setTimeout(() => resolve({ data: { response: 'Done' } }), 1000)
      }))

      wrapper = mount(ChatInterface)

      await wrapper.find('input').setValue('Test')
      await wrapper.find('button').trigger('click')

      const button = wrapper.findComponent({ name: 'TowerButton' })
      expect(button.props('loading')).toBe(true)
    })
  })

  describe('Manual Retry', () => {
    it('allows manual retry of failed messages', async () => {
      // First attempt fails
      axios.post.mockRejectedValueOnce({
        response: { status: 500 },
        message: 'Server error'
      })

      wrapper = mount(ChatInterface)

      await wrapper.find('input').setValue('Test message')
      await wrapper.find('button').trigger('click')
      await flushPromises()

      // Should show retry button
      const retryButton = wrapper.find('.retry-button')
      expect(retryButton.exists()).toBe(true)

      // Mock successful retry
      axios.post.mockResolvedValueOnce({
        data: { response: 'Success after manual retry' }
      })

      // Click retry button
      await retryButton.trigger('click')
      await flushPromises()

      // Should have made second call
      expect(axios.post).toHaveBeenCalledTimes(2)

      // Error message should be replaced with success
      const messages = wrapper.findAll('.message')
      const lastMessage = messages[messages.length - 1]
      expect(lastMessage.text()).toContain('Success after manual retry')
    })
  })
})

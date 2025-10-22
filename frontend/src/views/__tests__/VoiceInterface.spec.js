import { describe, it, expect, afterEach } from 'vitest'
import { mount } from '@vue/test-utils'
import VoiceInterface from '../VoiceInterface.vue'
import { nextTick } from 'vue'

// Mock @tower/ui-components  
import { vi } from 'vitest'

vi.mock('@tower/ui-components', () => ({
  TowerCard: { template: '<div><slot name="header" /><slot /></div>' },
  TowerButton: { 
    name: 'TowerButton',
    template: '<button @click="handleClick"><slot /></button>',
    props: ['size', 'variant', 'loading'],
    methods: {
      handleClick(e) {
        this.$emit('click', e)
      }
    }
  }
}))

describe('VoiceInterface.vue', () => {
  let wrapper

  afterEach(() => {
    if (wrapper) wrapper.unmount()
  })

  describe('Initial Rendering', () => {
    it('renders the voice interface', () => {
      wrapper = mount(VoiceInterface)
      expect(wrapper.find('h2').text()).toBe('Voice Interface')
    })

    it('shows start button initially', () => {
      wrapper = mount(VoiceInterface)
      expect(wrapper.find('button').text()).toBe('Start')
    })

    it('shows "Click to start speaking" message', () => {
      wrapper = mount(VoiceInterface)
      expect(wrapper.text()).toContain('Click to start speaking')
    })

    it('does not show transcript initially', () => {
      wrapper = mount(VoiceInterface)
      expect(wrapper.text()).not.toContain('Transcript')
    })
  })

  describe('Recording State', () => {
    it('changes to recording state when start clicked', async () => {
      wrapper = mount(VoiceInterface)
      
      // Directly call the toggleRecording method
      wrapper.vm.toggleRecording()
      await nextTick()

      expect(wrapper.vm.isRecording).toBe(true)
      expect(wrapper.text()).toContain('Listening...')
      expect(wrapper.find('button').text()).toBe('Stop')
    })

    it('toggles back to not recording when stop clicked', async () => {
      wrapper = mount(VoiceInterface)
      
      wrapper.vm.toggleRecording() // Start
      await nextTick()
      wrapper.vm.toggleRecording() // Stop
      await nextTick()

      expect(wrapper.vm.isRecording).toBe(false)
      expect(wrapper.text()).toContain('Click to start speaking')
      expect(wrapper.find('button').text()).toBe('Start')
    })

    it('applies active class to voice indicator when recording', async () => {
      wrapper = mount(VoiceInterface)
      
      const indicator = wrapper.find('.voice-indicator')
      expect(indicator.classes()).not.toContain('active')

      wrapper.vm.toggleRecording()
      await nextTick()
      
      expect(indicator.classes()).toContain('active')
    })

    it('removes active class when recording stops', async () => {
      wrapper = mount(VoiceInterface)
      
      wrapper.vm.toggleRecording() // Start
      await nextTick()
      wrapper.vm.toggleRecording() // Stop
      await nextTick()

      const indicator = wrapper.find('.voice-indicator')
      expect(indicator.classes()).not.toContain('active')
    })
  })

  describe('Transcript Display', () => {
    it('shows transcript after recording stops', async () => {
      wrapper = mount(VoiceInterface)
      
      wrapper.vm.toggleRecording() // Start
      await nextTick()
      wrapper.vm.toggleRecording() // Stop
      await nextTick()

      expect(wrapper.vm.transcript).toBeTruthy()
      expect(wrapper.text()).toContain('Transcript')
      expect(wrapper.text()).toContain('Voice recording not yet implemented')
    })

    it('transcript persists after multiple recordings', async () => {
      wrapper = mount(VoiceInterface)
      
      // First recording
      wrapper.vm.toggleRecording()
      await nextTick()
      wrapper.vm.toggleRecording()
      await nextTick()
      expect(wrapper.text()).toContain('Voice recording not yet implemented')

      // Second recording
      wrapper.vm.toggleRecording()
      await nextTick()
      wrapper.vm.toggleRecording()
      await nextTick()
      expect(wrapper.text()).toContain('Voice recording not yet implemented')
    })
  })

  describe('Button States', () => {
    it('button text changes from Start to Stop', async () => {
      wrapper = mount(VoiceInterface)
      
      const button = wrapper.find('button')
      expect(button.text()).toBe('Start')

      wrapper.vm.toggleRecording()
      await nextTick()
      
      expect(button.text()).toBe('Stop')
    })

    it('button text changes from Stop to Start', async () => {
      wrapper = mount(VoiceInterface)
      
      wrapper.vm.toggleRecording() // Start
      await nextTick()
      wrapper.vm.toggleRecording() // Stop
      await nextTick()

      const button = wrapper.find('button')
      expect(button.text()).toBe('Start')
    })
  })

  describe('Visual States', () => {
    it('shows voice indicator element', () => {
      wrapper = mount(VoiceInterface)
      expect(wrapper.find('.voice-indicator').exists()).toBe(true)
    })

    it('shows indicator wave element', () => {
      wrapper = mount(VoiceInterface)
      expect(wrapper.find('.indicator-wave').exists()).toBe(true)
    })

    it('changes status message based on recording state', async () => {
      wrapper = mount(VoiceInterface)
      
      expect(wrapper.text()).toContain('Click to start speaking')
      
      wrapper.vm.toggleRecording()
      await nextTick()
      
      expect(wrapper.text()).toContain('Listening...')
    })
  })

  describe('Component Lifecycle', () => {
    it('component mounts without errors', () => {
      expect(() => mount(VoiceInterface)).not.toThrow()
    })

    it('component unmounts cleanly', () => {
      wrapper = mount(VoiceInterface)
      expect(() => wrapper.unmount()).not.toThrow()
    })
  })
})

import { config } from '@vue/test-utils'

// Mock @tower/ui-components to avoid import errors
config.global.mocks = {
  $t: (key) => key // Mock translation function if needed
}

// Global test setup
beforeEach(() => {
  // Reset any mocks or state before each test
})

afterEach(() => {
  // Cleanup after each test
})

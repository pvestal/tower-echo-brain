export interface Fact {
  id: string
  subject: string
  predicate: string
  object: string
  confidence: number
  created_at: string
  source?: string
}

export interface Subject {
  name: string
  count: number
}

export interface Preference {
  id: string
  category: string
  key: string
  value: any
  metadata: Record<string, any>
  created_at: string
  updated_at: string
}

export interface VaultKey {
  key_name: string
  service: string
  key_type: string
  is_set: boolean
  value_preview?: string
}

export interface Integration {
  id: string
  provider: string
  display_name: string
  status: 'connected' | 'disconnected' | 'error'
  scopes?: string[]
  last_sync_at?: string
  connected_at?: string
  error_message?: string
}

export interface DashboardStats {
  facts_count: number
  preferences_count: number
  integrations_connected: number
  integrations_total: number
}

export interface ApiResponse<T> {
  data?: T
  error?: string
  status: 'success' | 'error'
}
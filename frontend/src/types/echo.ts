export interface ServiceHealth {
  name: string;
  status: 'healthy' | 'degraded' | 'down';
  latency_ms: number;
  last_check: string;
  details: Record<string, any>;
  error?: string;
}

// Actual API response format
export interface EchoHealthResponse {
  status: string;
  version: string;
  timestamp: string;
  services: Record<string, boolean> | ServiceHealth[];
  metrics: {
    requests: number;
    memory_mb: number;
    cpu_percent: number;
    vectors: number;
    avg_response_ms: number;
    error_rate: string;
  };
}

export interface SystemHealth {
  overall_status: 'healthy' | 'degraded' | 'critical';
  uptime_seconds: number;
  services: ServiceHealth[];
  resources: ResourceStats;
  endpoints: EndpointStats;
  timestamp: string;
}

export interface ResourceStats {
  cpu_percent: number;
  cpu_count: number;
  memory_percent: number;
  memory_used_gb: number;
  memory_total_gb: number;
  memory_available_gb: number;
  disk_percent: number;
  disk_used_gb: number;
  disk_total_gb: number;
  network_sent_mb: number;
  network_recv_mb: number;
  gpu?: {
    type: string;
    name: string;
    utilization_percent: number;
    memory_used_mb: number;
    memory_total_mb: number;
    temperature_c: number;
  };
}

export interface EndpointStats {
  total: number;
  by_router: Record<string, number>;
}
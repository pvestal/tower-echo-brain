# Echo Brain Monitoring & Observability Setup Guide

## Overview

This guide covers the comprehensive monitoring and observability setup for Echo Brain, including metrics collection, log aggregation, alerting, and visualization.

## Architecture

### Monitoring Stack Components

1. **Prometheus** - Metrics collection and storage
2. **Grafana** - Visualization and dashboards
3. **Loki** - Log aggregation
4. **Promtail** - Log shipping
5. **Alertmanager** - Alert routing and notification
6. **Node Exporter** - System metrics
7. **PostgreSQL Exporter** - Database metrics
8. **Redis Exporter** - Cache metrics

### Network Topology

```
Echo Brain (8309) → Prometheus (9090) → Grafana (3000)
       ↓                    ↓
   Logs → Promtail → Loki (3100) → Grafana
       ↓
   Alerts → Alertmanager (9093) → Email/Slack
```

## Installation & Setup

### 1. Prerequisites

Ensure the following are installed on the system:
- Docker and Docker Compose
- Python 3.8+ with required packages
- prometheus-client library for Python

```bash
# Install Python dependencies
cd /opt/tower-echo-brain
source venv/bin/activate
pip install prometheus-client aiohttp psutil
```

### 2. Start Monitoring Stack

```bash
cd /opt/tower-echo-brain
docker-compose -f docker-compose-essential.yml up -d
```

### 3. Configure Echo Brain Metrics

The Prometheus exporter is integrated into Echo Brain at:
- `/opt/tower-echo-brain/src/api/prometheus_exporter.py`

Metrics are exposed at:
- `http://***REMOVED***:8309/api/echo/metrics` (Prometheus format)
- `http://***REMOVED***:8309/api/echo/metrics/json` (JSON format)

## Access Information

### Service URLs

- **Grafana**: http://***REMOVED***:3000 (admin/admin)
- **Prometheus**: http://***REMOVED***:9090
- **Alertmanager**: http://***REMOVED***:9093
- **Loki**: http://***REMOVED***:3100
- **Echo Brain Metrics**: http://***REMOVED***:8309/api/echo/metrics

### Quick Start Commands

```bash
# Start monitoring stack
cd /opt/tower-echo-brain
docker-compose -f docker-compose-essential.yml up -d

# Check service status
docker-compose -f docker-compose-essential.yml ps

# View metrics
curl http://***REMOVED***:8309/api/echo/metrics

# Run health check
python3 scripts/health_monitor.py --once

# View logs
docker logs echo-prometheus
docker logs echo-grafana
```

## Key Features Implemented

1. **Comprehensive Metrics**: CPU, memory, VRAM, database, tasks, requests, errors
2. **Smart Alerting**: Critical alerts for service down, resource exhaustion
3. **Log Aggregation**: Structured logs from Echo Brain and system services
4. **Health Monitoring**: Automated checks with alert generation
5. **Performance Tracking**: Request duration histograms and error rates
6. **Visual Dashboards**: Real-time system overview with key metrics

For detailed configuration and troubleshooting, see the full documentation in this directory.

# TASK: Production Hardening — SearXNG Auto-Start + Health Monitoring

## CONTEXT

This is a quick follow-up to Prompt 01 (Web Search Deploy). It ensures SearXNG survives reboots and is monitored by Echo Brain's existing autonomous health system.

Run this AFTER Prompt 01 is verified working.

## PHASE 1: SYSTEMD SERVICE FOR SEARXNG

```bash
# Create systemd service for SearXNG Docker container
sudo tee /etc/systemd/system/tower-searxng.service << 'EOF'
[Unit]
Description=Tower SearXNG Search Engine
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/tower-echo-brain/docker
ExecStart=/usr/bin/docker compose -f docker-compose.searxng.yml up -d
ExecStop=/usr/bin/docker compose -f docker-compose.searxng.yml down
User=root

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable tower-searxng
sudo systemctl start tower-searxng
sudo systemctl status tower-searxng --no-pager
```

## PHASE 2: ADD TO ECHO BRAIN HEALTH MONITORING

### Step 1: Find the existing health/service monitoring configuration

```bash
# Find where services are registered for monitoring
grep -rn "health\|monitor\|service.*check\|circuit.*breaker" /opt/tower-echo-brain/config/ | head -20
grep -rn "health_endpoint\|restart_command" /opt/tower-echo-brain/config/ | head -10
cat /opt/tower-echo-brain/config/autonomous_services.yaml 2>/dev/null || echo "Not found"
find /opt/tower-echo-brain/config -name "*.yaml" -o -name "*.yml" | xargs grep -l "health\|service" 2>/dev/null
```

### Step 2: Add SearXNG to the monitoring config

Add SearXNG to whatever service monitoring configuration exists:

```yaml
searxng:
  name: "SearXNG Search"
  url: "http://localhost:8888"
  health_endpoint: "http://localhost:8888/healthz"
  restart_command: "docker restart tower-searxng"
  check_interval: 60
  critical: false  # Echo Brain still works without it (just no web search)
```

### Step 3: Add SearXNG to the main health endpoint

Find the health endpoint and add SearXNG status:

```bash
grep -rn "def.*health\|/health" /opt/tower-echo-brain/src/api/ --include="*.py" | head -10
```

Add a SearXNG check so `/health` or `/api/echo/health/detailed` includes:

```json
{
  "searxng": {
    "status": "healthy",
    "url": "http://localhost:8888",
    "response_time_ms": 15
  }
}
```

## PHASE 3: VERIFY

```bash
# Test auto-start
sudo systemctl status tower-searxng --no-pager
docker ps | grep searxng

# Test health monitoring
curl -s http://localhost:8309/health | python3 -c "
import sys, json; d=json.load(sys.stdin)
print(json.dumps(d, indent=2))
" | grep -A5 "searxng"

echo "DONE: SearXNG is production-hardened"
```

## DONE WHEN

- [ ] `tower-searxng.service` enabled and starts on boot
- [ ] SearXNG shows in Echo Brain health endpoint
- [ ] Health monitoring will detect if SearXNG goes down

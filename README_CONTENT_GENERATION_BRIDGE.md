# Content Generation Bridge

## Overview

The Content Generation Bridge is a production-ready system that connects Echo Brain's autonomous agents to ComfyUI workflow execution for anime production. This bridge enables fully autonomous content generation from scene descriptions to final video assets.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Echo Brain    │    │ Content Gen      │    │    ComfyUI      │
│  Autonomous     │───▶│    Bridge        │───▶│   Workflow      │
│    Agents       │    │                  │    │   Execution     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Anime Production│    │ SSOT Tracking    │    │ Generated       │
│   Database      │    │    System        │    │   Assets        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Components

### 1. Content Generation Bridge (`content_generation_bridge.py`)
- **Purpose**: Core bridge connecting Echo Brain agents to ComfyUI
- **Features**:
  - Scene analysis and workflow generation
  - LoRA model selection and integration
  - ComfyUI workflow submission and monitoring
  - Quality validation and asset management
  - SSOT tracking integration

### 2. Autonomous Content Coordinator (`autonomous_content_coordinator.py`)
- **Purpose**: High-level orchestration of autonomous generation workflows
- **Features**:
  - Multi-agent coordination (ReasoningAgent, NarrationAgent, CodingAgent)
  - Project-level generation planning and execution
  - Batch processing and resource management
  - Quality assurance and validation

### 3. Workflow Generator (`workflow_generator.py`)
- **Purpose**: Dynamic ComfyUI workflow generation
- **Features**:
  - FramePack workflow template system
  - Character LoRA integration
  - Image conditioning support
  - Parameter optimization

### 4. Content Generation API (`content_generation_api.py`)
- **Purpose**: REST API service for autonomous generation
- **Features**:
  - Scene and project generation endpoints
  - Job status monitoring
  - Webhook notifications
  - System capabilities reporting

## Installation

### Prerequisites

1. **Echo Brain System**: Running on localhost:8309
2. **ComfyUI**: Running on localhost:8188 with FramePack support
3. **PostgreSQL**: Anime production database configured
4. **Python 3.12+** with required dependencies

### Setup Steps

1. **Install Python Dependencies**:
```bash
cd /opt/tower-echo-brain
pip install -r requirements.txt
```

2. **Install Required Packages**:
```bash
pip install fastapi uvicorn httpx psycopg2-binary pytest pytest-asyncio
```

3. **Configure Database Connection**:
   - Ensure anime_production database is accessible
   - Update database credentials in configuration files

4. **Install System Service**:
```bash
sudo cp tower-content-generation.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tower-content-generation
sudo systemctl start tower-content-generation
```

5. **Verify Installation**:
```bash
curl http://localhost:8313/api/health
```

## Configuration

### Environment Variables

```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_DB=anime_production
POSTGRES_USER=patrick
POSTGRES_PASSWORD=RP78eIrW7cI2jYvL5akt1yurE

# Service URLs
ECHO_BRAIN_URL=http://localhost:8309
COMFYUI_URL=http://localhost:8188

# Generation Settings
MAX_CONCURRENT_JOBS=3
WORKFLOW_TIMEOUT=900
QUALITY_THRESHOLD=0.7

# File Paths
LORA_PATH=/mnt/1TB-storage/models/loras
OUTPUT_PATH=/mnt/1TB-storage/ComfyUI/output
```

### Configuration File (Optional)

Create `/opt/tower-echo-brain/config.json`:

```json
{
  "echo_brain_url": "http://localhost:8309",
  "comfyui_url": "http://localhost:8188",
  "max_concurrent_scenes": 3,
  "quality_threshold": 0.8,
  "workflow_timeout": 900,
  "lora_path": "/mnt/1TB-storage/models/loras",
  "output_path": "/mnt/1TB-storage/ComfyUI/output"
}
```

## API Usage

### Scene Generation

```bash
# Generate a single scene
curl -X POST "http://localhost:8313/api/generate/scene/scene-uuid-123" \
  -H "Content-Type: application/json" \
  -d '{
    "priority": "high",
    "quality_preset": "ultra",
    "notify_webhook": "http://example.com/webhook"
  }'
```

### Project Generation

```bash
# Generate entire project
curl -X POST "http://localhost:8313/api/generate/project/123" \
  -H "Content-Type: application/json" \
  -d '{
    "priority": "normal",
    "quality_preset": "high"
  }'
```

### Status Monitoring

```bash
# Check job status
curl "http://localhost:8313/api/status/job-uuid-456"

# List all jobs
curl "http://localhost:8313/api/jobs?status=running&limit=10"
```

### System Information

```bash
# Health check
curl "http://localhost:8313/api/health"

# System capabilities
curl "http://localhost:8313/api/capabilities"
```

## Command Line Usage

### Scene Generation

```bash
# Generate single scene
cd /opt/tower-echo-brain
python content_generation_bridge.py --scene-id "scene-uuid-123"

# With custom configuration
python content_generation_bridge.py --scene-id "scene-uuid-123" --config config.json
```

### Project Generation

```bash
# Generate entire project
python autonomous_content_coordinator.py --mode project --project-id 123

# Generate single scene with analysis
python autonomous_content_coordinator.py --mode scene --scene-id "scene-uuid-123"
```

## Database Schema Requirements

### Required Tables

The system expects these tables in the `anime_production` database:

1. **scenes**: Scene definitions and metadata
2. **characters**: Character information and LoRA mappings
3. **projects**: Project configurations
4. **generated_assets**: Generated content tracking
5. **ssot_tracking**: Single source of truth tracking

### Key Fields

**scenes table**:
- `id` (UUID): Scene identifier
- `description` (TEXT): Scene description
- `visual_description` (TEXT): Enhanced visual description
- `characters` (TEXT): Character list
- `character_lora_mapping` (JSONB): Character to LoRA mapping
- `frame_count` (INTEGER): Target frame count
- `fps` (INTEGER): Frames per second
- `shot_list` (JSONB): Detailed shot specifications

**characters table**:
- `name` (VARCHAR): Character name
- `description` (TEXT): Character description
- `design_prompt` (TEXT): Generation prompt
- `traits` (JSONB): Character traits and attributes

## Generation Pipeline

### 1. Scene Analysis Phase
- **ReasoningAgent**: Analyzes scene requirements and constraints
- **NarrationAgent**: Enhances visual descriptions for optimal generation
- **CodingAgent**: Generates technical workflow specifications

### 2. Workflow Generation Phase
- Dynamic ComfyUI workflow creation
- Character LoRA integration
- Parameter optimization based on scene complexity
- Quality preset application

### 3. Execution Phase
- ComfyUI workflow submission
- Real-time monitoring and progress tracking
- Error handling and recovery
- Resource management

### 4. Validation Phase
- Generated content quality assessment
- Duration and resolution verification
- Character consistency checking
- SSOT tracking updates

### 5. Finalization Phase
- Asset registration in database
- Metadata and metrics recording
- Webhook notifications
- Cleanup and optimization

## Quality Standards

### Generation Parameters

**High Quality Preset**:
- Resolution: 704x544 (optimized for FramePack)
- Sampling Steps: 28
- CFG Scale: 7.0
- Sampler: dpmpp_2m
- Scheduler: karras
- Frame Count: 120 (5 seconds @ 24fps)

**Ultra Quality Preset**:
- Resolution: 768x768
- Sampling Steps: 35
- CFG Scale: 8.0
- Enhanced LoRA strength
- Extended context windows

### Quality Metrics

- **Technical Quality**: Resolution, frame rate, compression
- **Visual Quality**: Clarity, consistency, artistic merit
- **Character Consistency**: LoRA effectiveness, appearance stability
- **Narrative Quality**: Scene coherence, emotional impact

## Performance Optimization

### Concurrent Processing
- Maximum 3 concurrent scene generations
- Intelligent batch scheduling
- Resource-aware load balancing

### Memory Management
- Dynamic VRAM allocation
- Model offloading strategies
- Garbage collection optimization

### Caching Strategies
- Workflow template caching
- LoRA model preloading
- Generated asset caching

## Monitoring and Logging

### Service Health
```bash
# Check service status
sudo systemctl status tower-content-generation

# View logs
sudo journalctl -u tower-content-generation -f
```

### Performance Metrics
- Generation success rate
- Average processing time
- Resource utilization
- Quality scores

### Error Handling
- Automatic retry mechanisms
- Graceful degradation
- Comprehensive error logging
- Alert notifications

## Testing

### Run Test Suite

```bash
cd /opt/tower-echo-brain
python test_content_generation_bridge.py
```

### Test Categories

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: End-to-end workflow validation
3. **Performance Tests**: Stress testing and benchmarks
4. **Quality Tests**: Generated content validation

### Sample Test Scene

Create a test scene in the database:

```sql
INSERT INTO scenes (id, title, description, visual_description, characters, character_lora_mapping, frame_count, fps)
VALUES (
  'test-scene-001',
  'Test Garden Scene',
  'A peaceful anime garden scene with cherry blossoms',
  'Beautiful moonlit garden with soft pink cherry blossoms, serene atmosphere',
  'Mei',
  '{"Mei": "mei_character_v1"}',
  120,
  24
);
```

Then test generation:

```bash
python autonomous_content_coordinator.py --mode scene --scene-id test-scene-001
```

## Troubleshooting

### Common Issues

1. **Echo Brain Connection Failed**
   - Verify Echo Brain service is running: `curl http://localhost:8309/health`
   - Check network connectivity and firewall settings

2. **ComfyUI Workflow Submission Failed**
   - Verify ComfyUI is accessible: `curl http://localhost:8188/system_stats`
   - Check model files are present and accessible

3. **Database Connection Issues**
   - Verify PostgreSQL service: `sudo systemctl status postgresql`
   - Check database credentials and permissions

4. **LoRA Model Not Found**
   - Verify LoRA files exist in `/mnt/1TB-storage/models/loras/`
   - Check file permissions and naming conventions

5. **Generation Timeout**
   - Increase `WORKFLOW_TIMEOUT` in configuration
   - Check ComfyUI performance and VRAM availability
   - Monitor system resource usage

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python content_generation_bridge.py --scene-id test-scene
```

### Performance Issues

Monitor system resources:

```bash
# GPU usage
nvidia-smi -l 1

# System resources
htop

# Disk I/O
iotop
```

## Integration Examples

### Webhook Integration

```python
# Webhook handler example
from fastapi import FastAPI

app = FastAPI()

@app.post("/generation-webhook")
async def handle_generation_complete(data: dict):
    if data["status"] == "completed":
        # Process successful generation
        job_id = data["job_id"]
        # Update external systems, send notifications, etc.
    else:
        # Handle generation failure
        error = data.get("error", "Unknown error")
        # Log error, retry, or notify administrators

    return {"status": "received"}
```

### Custom Quality Validation

```python
from content_generation_bridge import ContentGenerationBridge

class CustomValidationBridge(ContentGenerationBridge):
    async def _validate_generated_content(self, execution_result):
        # Custom validation logic
        validation = await super()._validate_generated_content(execution_result)

        # Add custom checks
        if validation['passed']:
            # Additional quality metrics
            validation['custom_score'] = self._calculate_custom_quality(execution_result)

        return validation
```

## Contributing

### Development Setup

1. Clone the repository and create development environment
2. Install development dependencies: `pip install -r requirements-dev.txt`
3. Run tests: `python test_content_generation_bridge.py`
4. Follow coding standards and add tests for new features

### Architecture Guidelines

1. **Modularity**: Keep components loosely coupled
2. **Error Handling**: Comprehensive error handling and recovery
3. **Performance**: Optimize for concurrent operations
4. **Monitoring**: Add comprehensive logging and metrics
5. **Testing**: Unit tests for all new functionality

## Support

### Documentation
- Echo Brain documentation: `/opt/tower-echo-brain/docs/`
- ComfyUI documentation: https://github.com/comfyanonymous/ComfyUI
- FramePack documentation: https://github.com/kijai/ComfyUI-FramePackWrapper

### Logs and Diagnostics
- Service logs: `sudo journalctl -u tower-content-generation`
- Application logs: `/var/log/tower-content-generation/`
- Debug logs: Enable with `LOG_LEVEL=DEBUG`

### Performance Monitoring
- System metrics: Use monitoring tools like Grafana + Prometheus
- Generation metrics: Available via API endpoints
- Resource usage: Monitor GPU, CPU, and memory utilization

---

## License

Part of the Tower Echo Brain System
© 2026 Patrick & Claude Code
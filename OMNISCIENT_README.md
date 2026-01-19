# Echo Brain Omniscient Capabilities

## Overview

The Echo Brain Omniscient system integrates Wyze Pan cameras with conversation data extraction to provide comprehensive awareness and continuous learning capabilities. This system enables Echo Brain to:

- **Monitor environments** through live camera feeds
- **Detect and recognize faces** for personalized interactions
- **Analyze scenes** for environmental awareness
- **Extract conversation history** for training and learning
- **Detect behavioral patterns** from integrated data sources
- **Learn continuously** from all data streams

## 🚀 Quick Start

### 1. Setup System
```bash
cd /opt/tower-echo-brain
./scripts/setup_omniscient.sh
```

### 2. Configure Cameras
Edit `/opt/tower-echo-brain/config/cameras.json`:
```json
{
  "cameras": [
    {
      "id": "wyze_living_room",
      "name": "Living Room Camera",
      "ip": "192.168.50.200",
      "rtmp_url": "rtmp://192.168.50.200:1935/live/stream",
      "username": "admin",
      "password": "your_password",
      "enabled": true,
      "location": "Living Room"
    }
  ]
}
```

### 3. Add Known Faces
Place face images in `/opt/tower-echo-brain/data/known_faces/`:
- `patrick.jpg`
- `family_member.jpg`
- etc.

### 4. Start Service
```bash
sudo systemctl start echo-omniscient
sudo systemctl enable echo-omniscient
```

### 5. Activate in Echo Brain
```python
from src.integrations.omniscient_integration import create_omniscient_integration

# In your Echo Brain startup
omniscient = create_omniscient_integration(echo_brain_instance)
await omniscient.initialize()
```

## 📁 File Structure

```
/opt/tower-echo-brain/
├── src/
│   ├── integrations/
│   │   ├── wyze_camera.py              # Camera integration
│   │   └── omniscient_integration.py   # Echo Brain integration
│   ├── training/
│   │   └── conversation_extractor.py   # Conversation data extraction
│   └── omniscient_pipeline.py          # Main orchestration
├── config/
│   ├── cameras.json                    # Camera configurations
│   └── omniscient.json                 # System settings
├── data/
│   ├── known_faces/                    # Face recognition training
│   └── training/                       # Exported training data
├── models/                             # AI models (YOLO, etc.)
├── scripts/
│   ├── setup_omniscient.sh            # Setup script
│   └── activate_omniscient.py          # Activation script
└── requirements_omniscient.txt         # Python dependencies
```

## 🎯 Core Features

### 1. Wyze Camera Integration (`wyze_camera.py`)

**Features:**
- Live RTMP stream processing
- Real-time motion detection
- Facial recognition with known faces
- Object detection using YOLO
- Scene analysis and environmental awareness
- Privacy mode support

**Key Classes:**
- `WyzeCameraIntegration`: Main camera manager
- `DetectionEvent`: Detection event data structure
- `SceneAnalysis`: Scene analysis results
- `CameraInfo`: Camera configuration

### 2. Conversation Data Extraction (`conversation_extractor.py`)

**Features:**
- Extract from Echo Brain interaction logs
- Parse Knowledge Base Q&A content
- Import Claude conversation history
- Quality scoring and filtering
- Deduplication and cleaning
- Export in multiple formats

**Key Classes:**
- `ConversationExtractor`: Main extraction engine
- `ConversationData`: Structured conversation data
- `TrainingDataset`: Complete dataset structure

### 3. Omniscient Pipeline (`omniscient_pipeline.py`)

**Features:**
- Unified data processing from all sources
- Real-time pattern detection
- Behavioral analysis
- Environmental state tracking
- Continuous learning integration
- Echo Brain context provision

**Key Classes:**
- `OmniscientPipeline`: Main orchestrator
- `OmniscientContext`: Comprehensive awareness context
- `BehaviorPattern`: Detected behavioral patterns

### 4. Echo Brain Integration (`omniscient_integration.py`)

**Features:**
- Camera control API endpoints
- Environmental awareness queries
- Behavioral pattern analysis
- Real-time context updates
- Learning event processing

## 🔧 Configuration

### Camera Settings (`config/cameras.json`)
```json
{
  "cameras": [
    {
      "id": "unique_camera_id",
      "name": "Display Name",
      "ip": "192.168.1.100",
      "rtmp_url": "rtmp://192.168.1.100:1935/live/stream",
      "username": "camera_username",
      "password": "camera_password",
      "enabled": true,
      "motion_detection": true,
      "facial_recognition": true,
      "recording": true,
      "location": "Room Name"
    }
  ],
  "global_settings": {
    "max_concurrent_streams": 4,
    "detection_interval_seconds": 30,
    "motion_threshold": 5000
  }
}
```

### System Settings (`config/omniscient.json`)
```json
{
  "camera_settings": {
    "enable_facial_recognition": true,
    "enable_motion_detection": true,
    "analysis_frequency": 30
  },
  "learning_settings": {
    "conversation_window_hours": 24,
    "behavior_detection_threshold": 0.7,
    "real_time_training": true
  },
  "database": {
    "host": "192.168.50.135",
    "user": "patrick",
    "password": "your_password",
    "database": "echo_brain"
  }
}
```

## 📊 API Integration

### Camera Control
```python
# Get camera status
result = await echo_brain.camera_monitoring({
    "action": "get_status"
})

# Get recent events
result = await echo_brain.camera_monitoring({
    "action": "get_recent_events",
    "hours": 24
})

# Enable/disable camera
result = await echo_brain.camera_monitoring({
    "action": "enable_camera",
    "camera_id": "wyze_living_room"
})
```

### Environmental Awareness
```python
# Get environmental state
result = await echo_brain.environmental_awareness({})

# Result includes:
# - occupancy status
# - lighting conditions
# - activity levels
# - location-specific data
```

### Behavioral Analysis
```python
# Get detected patterns
result = await echo_brain.behavioral_analysis({
    "action": "get_patterns"
})

# Get learning insights
result = await echo_brain.behavioral_analysis({
    "action": "get_insights"
})
```

## 🔍 Monitoring & Maintenance

### Service Management
```bash
# Check status
sudo systemctl status echo-omniscient

# View logs
sudo journalctl -u echo-omniscient -f

# Restart service
sudo systemctl restart echo-omniscient
```

### Log Files
- Service logs: `journalctl -u echo-omniscient`
- Application logs: `/opt/tower-echo-brain/logs/omniscient.log`
- Echo Brain logs: `/opt/tower-echo-brain/logs/`

### Data Export
```python
# Export training data
await pipeline.export_training_data("/path/to/export")

# Get conversation dataset
dataset = await extractor.extract_all_conversations(start_date, end_date)
await extractor.export_dataset(dataset, "/path/to/output")
```

## 🛠 Troubleshooting

### Camera Issues
1. **Camera not detected:**
   - Check IP address and network connectivity
   - Verify RTMP URL format
   - Test with `ping camera_ip`

2. **RTMP stream errors:**
   - Ensure camera RTMP is enabled
   - Check firewall settings
   - Verify username/password

3. **Face recognition not working:**
   - Add face images to `/opt/tower-echo-brain/data/known_faces/`
   - Ensure good lighting in images
   - Check face-recognition library installation

### Database Issues
1. **Connection errors:**
   - Verify PostgreSQL is running
   - Check credentials in config
   - Test with `psql -h host -U user -d database`

2. **No conversation data:**
   - Check database table structure
   - Verify interaction_logs table exists
   - Check date range parameters

### Performance Issues
1. **High CPU usage:**
   - Reduce camera analysis frequency
   - Disable unused detection features
   - Limit concurrent streams

2. **Memory issues:**
   - Reduce context memory limit
   - Clear old detection buffers
   - Monitor with `htop` or `free -m`

## 🔒 Security Considerations

### Privacy
- Enable privacy mode for bedrooms/private areas
- Configure recording schedules
- Encrypt face recognition data
- Review access control settings

### Network Security
- Use strong camera passwords
- Enable network encryption
- Restrict camera network access
- Monitor unusual access patterns

### Data Protection
- Regular backup of training data
- Secure storage of face recognition data
- Audit logs for access tracking
- GDPR compliance considerations

## 📈 Performance Optimization

### Camera Optimization
- Adjust detection frequency based on needs
- Use motion detection to trigger other features
- Enable GPU acceleration if available
- Optimize camera resolution and bitrate

### Database Optimization
- Regular database maintenance
- Index frequently queried columns
- Archive old conversation data
- Monitor query performance

### System Optimization
- Use SSD storage for models and data
- Allocate sufficient RAM for processing
- Monitor CPU temperature under load
- Use dedicated GPU for AI processing

## 🚀 Advanced Features

### Custom Learning Callbacks
```python
async def custom_learning_callback(context):
    # Process learning events
    insights = context.learning_insights
    # Custom logic here

pipeline.add_learning_callback(custom_learning_callback)
```

### Pattern Detection Extensions
```python
# Add custom pattern detection
async def detect_custom_patterns(detections, conversations):
    # Custom pattern analysis
    return patterns

# Register with pipeline
pipeline.pattern_detectors.append(detect_custom_patterns)
```

### Integration with Other Systems
```python
# Integrate with Home Assistant
await pipeline.send_to_home_assistant(detection_event)

# Send notifications
await pipeline.send_notification(pattern_discovery)

# Update external databases
await pipeline.sync_with_external_db(training_data)
```

## 📚 Additional Resources

- **Wyze Camera Setup**: https://wyze.com/camera-setup
- **OpenCV Documentation**: https://docs.opencv.org/
- **Face Recognition**: https://face-recognition.readthedocs.io/
- **YOLO Object Detection**: https://github.com/AlexeyAB/darknet

## 🤝 Contributing

To extend the omniscient capabilities:

1. **Add new detection features** in `wyze_camera.py`
2. **Extend conversation extraction** in `conversation_extractor.py`
3. **Add pattern detection** in `omniscient_pipeline.py`
4. **Create new API endpoints** in `omniscient_integration.py`

See the Echo Brain development guidelines for coding standards and testing requirements.

---

**⚠️ Important Notes:**
- Respect privacy and local laws when using camera systems
- Ensure proper network security for camera access
- Regular maintenance and updates are required
- Monitor system resources and performance
- Backup important training data and configurations

The omniscient system transforms Echo Brain into a comprehensive awareness platform, enabling truly intelligent and context-aware interactions based on real-world environmental data and continuous learning from all available sources.
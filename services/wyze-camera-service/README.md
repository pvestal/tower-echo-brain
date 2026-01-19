# Wyze Camera Service

A FastAPI-based microservice for integrating Wyze cameras with the Tower Network Echo Brain system. Provides comprehensive camera control, monitoring, and integration capabilities.

## 🚀 Features

- **Full Wyze Camera Control**: Power, PTZ, motion detection, night vision
- **Real-time Monitoring**: Camera status, events, and health monitoring
- **Modern Web Interface**: Vue.js 3 + Tailwind CSS dashboard
- **RESTful API**: Complete API for integration with Echo Brain
- **Docker Ready**: Containerized deployment with nginx proxy
- **Auto-refresh**: Real-time status updates and notifications

## 📁 Project Structure

```
wyze-camera-service/
├── app/
│   └── main.py              # FastAPI application
├── index.html               # Vue.js frontend dashboard
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Multi-container setup
├── nginx.conf              # Nginx proxy configuration
├── .env.example            # Environment template
└── README.md               # This file
```

## 🔧 Setup

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional)
- Wyze account with API access

### 2. Wyze API Setup

1. Visit [Wyze Developer Console](https://developer-api-console.wyze.com/)
2. Create an API key and get your Key ID
3. Note your Wyze account email and password

### 3. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit with your Wyze credentials
nano .env
```

Required environment variables:
```env
WYZE_EMAIL=your_email@example.com
WYZE_PASSWORD=your_wyze_password
WYZE_KEY_ID=your_wyze_key_id
WYZE_API_KEY=your_wyze_api_key
```

### 4. Installation Options

#### Option A: Docker Compose (Recommended)

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f
```

Services:
- API: http://localhost:8100
- Frontend: http://localhost:8101

#### Option B: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
cd app
uvicorn main:app --host 0.0.0.0 --port 8100 --reload

# Serve frontend (separate terminal)
python -m http.server 8080
```

Services:
- API: http://localhost:8100
- Frontend: http://localhost:8080

## 🎯 API Endpoints

### Health & Status
- `GET /health` - Service health check

### Camera Management
- `GET /cameras` - List all cameras
- `GET /cameras/{mac}` - Get specific camera info

### Camera Controls
- `POST /cameras/power` - Toggle camera power
- `POST /cameras/ptz` - PTZ controls (up, down, left, right, reset)
- `POST /cameras/{mac}/motion` - Toggle motion detection
- `POST /cameras/{mac}/night-vision` - Toggle night vision
- `POST /cameras/{mac}/siren` - Trigger siren
- `DELETE /cameras/{mac}/siren` - Stop siren

### Events
- `GET /cameras/{mac}/events` - Get camera events

### Authentication
- `POST /auth/refresh` - Refresh Wyze authentication

## 🖥️ Web Interface

The Vue.js dashboard provides:

- **Camera Grid**: Visual list of all cameras with status
- **Live Controls**: PTZ controls with directional pad
- **Feature Toggles**: Motion detection, night vision controls
- **Event History**: Recent camera events and thumbnails
- **System Status**: Real-time connection and health monitoring
- **Auto-refresh**: Automatic status updates every 30 seconds

### Interface Features

- **Responsive Design**: Works on desktop and mobile
- **Real-time Updates**: Live status and notification system
- **One-click Controls**: Easy camera management
- **Visual Feedback**: Status indicators and notifications

## 🔌 Echo Brain Integration

### Register with Echo Brain

Add to your Echo Brain capabilities:

```python
# In Echo Brain startup.py
from integrations.wyze_integration import WyzeIntegration

async def register_wyze_capability():
    wyze = WyzeIntegration("http://localhost:8100")
    echo_brain.register_capability("camera_control", wyze.handle_request)
```

### Usage in Echo Brain

```python
# Camera control
result = await echo_brain.camera_control({
    "action": "get_status"
})

# PTZ control
result = await echo_brain.camera_control({
    "action": "ptz",
    "camera_mac": "CAMERA_MAC_ADDRESS",
    "direction": "up"
})

# Toggle features
result = await echo_brain.camera_control({
    "action": "toggle_motion",
    "camera_mac": "CAMERA_MAC_ADDRESS",
    "enabled": True
})
```

## 🐛 Troubleshooting

### Authentication Issues

1. **Invalid credentials**: Verify email/password in .env
2. **API key invalid**: Check Key ID and API key from Wyze console
3. **2FA enabled**: Add TOTP key to environment

```bash
# Test authentication
curl http://localhost:8100/health
```

### Camera Connection Issues

1. **No cameras found**: Ensure cameras are set up in Wyze app
2. **Camera offline**: Check camera power and WiFi connection
3. **API errors**: Check Wyze service status

### Service Issues

```bash
# Check service health
curl http://localhost:8100/health

# View detailed logs
docker-compose logs wyze-camera-service

# Restart service
docker-compose restart wyze-camera-service
```

### Network Issues

1. **CORS errors**: Update CORS_ORIGINS in environment
2. **Port conflicts**: Change ports in docker-compose.yml
3. **Proxy issues**: Check nginx configuration

## 🔒 Security Considerations

### Credentials
- Store Wyze credentials securely
- Use environment variables, not hardcoded values
- Consider using Docker secrets in production

### Network
- Run behind reverse proxy in production
- Use HTTPS for external access
- Limit network access to trusted sources

### API Access
- Monitor API usage and rate limits
- Implement authentication for production use
- Log all camera control actions

## 📊 Monitoring

### Health Checks

The service includes built-in health monitoring:

```bash
# Service health
curl http://localhost:8100/health

# Docker health check
docker ps --format "table {{.Names}}\t{{.Status}}"
```

### Logging

Logs include:
- Authentication events
- Camera control actions
- API requests and responses
- Error conditions

View logs:
```bash
# Docker logs
docker-compose logs -f wyze-camera-service

# Application logs
tail -f logs/wyze-service.log
```

## 🚀 Deployment

### Production Deployment

1. **Environment**: Set production environment variables
2. **Security**: Enable HTTPS and authentication
3. **Monitoring**: Set up log aggregation and alerting
4. **Backup**: Backup configuration and credentials

### Scaling

- Multiple instances can share the same Wyze account
- Consider rate limiting for API calls
- Monitor Wyze API quotas and limits

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

Part of the Tower Network Echo Brain system. See main project license.

## 🔗 Related

- [Echo Brain Main Project](../README.md)
- [Tower Network Dashboard](../../tower-dashboard/)
- [Wyze SDK Documentation](https://github.com/shauntarves/wyze-sdk)

---

**⚠️ Note**: This service requires valid Wyze API credentials and active camera setup in the Wyze mobile app.
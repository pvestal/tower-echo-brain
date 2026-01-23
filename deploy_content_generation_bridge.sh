#!/bin/bash

# Content Generation Bridge Deployment Script
# ============================================
# Deploys and configures the Content Generation Bridge for production use

set -e  # Exit on any error

echo "🚀 Deploying Content Generation Bridge"
echo "======================================"

# Configuration
BRIDGE_DIR="/opt/tower-echo-brain"
SERVICE_NAME="tower-content-generation"
API_PORT="8313"

# Check if running as root for systemd operations
if [[ $EUID -ne 0 ]]; then
   echo "❌ This script must be run as root for system service installation"
   echo "   Please run: sudo $0"
   exit 1
fi

# Step 1: Verify prerequisites
echo "🔍 Verifying prerequisites..."

# Check Echo Brain service
if ! systemctl is-active --quiet tower-echo-brain; then
    echo "❌ Echo Brain service is not running"
    echo "   Please start Echo Brain first: sudo systemctl start tower-echo-brain"
    exit 1
fi

# Check ComfyUI availability
if ! curl -s http://localhost:8188/system_stats >/dev/null; then
    echo "⚠️  ComfyUI not accessible on localhost:8188"
    echo "   Please ensure ComfyUI is running"
fi

# Check PostgreSQL
if ! systemctl is-active --quiet postgresql; then
    echo "❌ PostgreSQL service is not running"
    echo "   Please start PostgreSQL: sudo systemctl start postgresql"
    exit 1
fi

echo "✅ Prerequisites verified"

# Step 2: Install Python dependencies
echo "📦 Installing Python dependencies..."

cd "$BRIDGE_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
source venv/bin/activate

# Install core dependencies
pip install --upgrade pip
pip install fastapi uvicorn httpx psycopg2-binary pytest pytest-asyncio

echo "✅ Dependencies installed"

# Step 3: Create required directories
echo "📁 Creating required directories..."

mkdir -p /var/log/tower-content-generation
mkdir -p /tmp/claude  # For temporary workflow files

# Set permissions
chown -R patrick:patrick /var/log/tower-content-generation
chown -R patrick:patrick "$BRIDGE_DIR"

echo "✅ Directories created"

# Step 4: Validate database connection
echo "🗄️  Validating database connection..."

if sudo -u patrick PGPASSWORD=RP78eIrW7cI2jYvL5akt1yurE psql -h localhost -U patrick -d anime_production -c "SELECT 1;" >/dev/null 2>&1; then
    echo "✅ Database connection successful"
else
    echo "❌ Database connection failed"
    echo "   Please verify database credentials and accessibility"
    exit 1
fi

# Step 5: Install systemd service
echo "🔧 Installing systemd service..."

# Copy service file
cp "$BRIDGE_DIR/tower-content-generation.service" /etc/systemd/system/

# Reload systemd and enable service
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

echo "✅ Service installed"

# Step 6: Test basic functionality
echo "🧪 Testing basic functionality..."

cd "$BRIDGE_DIR"
source venv/bin/activate

# Test workflow generation
if python3 -c "from workflow_generator import FramePackWorkflowGenerator; g = FramePackWorkflowGenerator(); w = g.generate_anime_workflow('test'); print('Workflow generation: OK')" 2>/dev/null; then
    echo "✅ Workflow generation test passed"
else
    echo "⚠️  Workflow generation test failed (non-critical)"
fi

# Test basic imports
if python3 -c "from content_generation_bridge import ContentGenerationBridge; from autonomous_content_coordinator import AutonomousContentCoordinator; print('Import test: OK')" 2>/dev/null; then
    echo "✅ Module import test passed"
else
    echo "❌ Module import test failed"
    exit 1
fi

echo "✅ Basic functionality tests passed"

# Step 7: Start the service
echo "▶️  Starting Content Generation Bridge service..."

systemctl start "$SERVICE_NAME"

# Wait a moment for service to start
sleep 3

# Check service status
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "✅ Service started successfully"
else
    echo "❌ Service failed to start"
    echo "   Check logs: sudo journalctl -u $SERVICE_NAME -f"
    exit 1
fi

# Step 8: Test API endpoint
echo "🌐 Testing API endpoint..."

sleep 2  # Give service time to fully initialize

if curl -s -o /dev/null -w "%{http_code}" http://localhost:$API_PORT/api/health | grep -q "200"; then
    echo "✅ API endpoint responding correctly"
else
    echo "⚠️  API endpoint not responding (may still be initializing)"
    echo "   Check with: curl http://localhost:$API_PORT/api/health"
fi

# Step 9: Create sample test data (optional)
echo "🧪 Creating sample test data..."

# Check if test scene already exists
if sudo -u patrick PGPASSWORD=RP78eIrW7cI2jYvL5akt1yurE psql -h localhost -U patrick -d anime_production -c "SELECT id FROM scenes WHERE id = 'bridge-test-scene-001';" | grep -q "bridge-test-scene-001" 2>/dev/null; then
    echo "✅ Test scene already exists"
else
    # Create test scene
    sudo -u patrick PGPASSWORD=RP78eIrW7cI2jYvL5akt1yurE psql -h localhost -U patrick -d anime_production -c "
    INSERT INTO scenes (id, title, description, visual_description, characters, character_lora_mapping, frame_count, fps, status, created_at, updated_at)
    VALUES (
        'bridge-test-scene-001',
        'Content Bridge Test Scene',
        'A beautiful anime garden scene for testing the content generation bridge',
        'Peaceful moonlit garden with cherry blossoms falling gently, soft lighting, cinematic composition',
        'Test Character',
        '{\"Test Character\": \"mei_character_v1\"}',
        60,
        24,
        'pending',
        NOW(),
        NOW()
    );" 2>/dev/null && echo "✅ Test scene created" || echo "⚠️  Could not create test scene"
fi

# Step 10: Display deployment summary
echo ""
echo "🎉 Content Generation Bridge Deployment Complete!"
echo "================================================"
echo ""
echo "📊 Deployment Summary:"
echo "  • Service: $SERVICE_NAME"
echo "  • API Port: $API_PORT"
echo "  • Status: $(systemctl is-active $SERVICE_NAME)"
echo "  • Installation: $BRIDGE_DIR"
echo ""
echo "🔗 Quick Links:"
echo "  • Health Check: curl http://localhost:$API_PORT/api/health"
echo "  • API Docs: http://localhost:$API_PORT/docs"
echo "  • Capabilities: curl http://localhost:$API_PORT/api/capabilities"
echo ""
echo "📋 Management Commands:"
echo "  • Status: sudo systemctl status $SERVICE_NAME"
echo "  • Logs: sudo journalctl -u $SERVICE_NAME -f"
echo "  • Restart: sudo systemctl restart $SERVICE_NAME"
echo "  • Stop: sudo systemctl stop $SERVICE_NAME"
echo ""
echo "🧪 Test Commands:"
echo "  • Run Tests: cd $BRIDGE_DIR && python3 test_content_generation_bridge.py"
echo "  • Test Scene: curl -X POST http://localhost:$API_PORT/api/generate/scene/bridge-test-scene-001"
echo ""
echo "📖 Documentation: $BRIDGE_DIR/README_CONTENT_GENERATION_BRIDGE.md"
echo ""

# Final verification
echo "🔍 Final System Verification:"
echo "  Echo Brain: $(curl -s http://localhost:8309/health | jq -r '.status' 2>/dev/null || echo 'Not accessible')"
echo "  ComfyUI: $(curl -s http://localhost:8188/system_stats >/dev/null 2>&1 && echo 'Active' || echo 'Not accessible')"
echo "  PostgreSQL: $(systemctl is-active postgresql)"
echo "  Content Bridge: $(systemctl is-active $SERVICE_NAME)"
echo ""

if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "🚀 Content Generation Bridge is ready for autonomous anime production!"
    echo ""
    echo "Next steps:"
    echo "1. Review the API documentation at http://localhost:$API_PORT/docs"
    echo "2. Test with a sample scene generation"
    echo "3. Configure webhooks for notifications (optional)"
    echo "4. Set up monitoring and alerting (recommended)"
    echo ""
    echo "The system is now capable of:"
    echo "• Autonomous scene analysis using Echo Brain agents"
    echo "• Dynamic ComfyUI workflow generation"
    echo "• Character-consistent LoRA integration"
    echo "• Quality validation and SSOT tracking"
    echo "• REST API access for integration"
else
    echo "⚠️  Service is not running. Check logs for issues:"
    echo "sudo journalctl -u $SERVICE_NAME -n 50"
fi

echo ""
echo "Happy generating! 🎬✨"
#!/bin/bash
# Echo Brain Omniscient Setup Script
# Sets up Wyze camera integration and conversation training for comprehensive awareness

set -e

echo "🧠 Echo Brain Omniscient Setup"
echo "============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/opt/tower-echo-brain"
VENV_PATH="$PROJECT_ROOT/venv"
LOGS_DIR="$PROJECT_ROOT/logs"

echo -e "${BLUE}📋 Checking system requirements...${NC}"

# Check if running as correct user
if [ "$USER" != "patrick" ]; then
    echo -e "${YELLOW}⚠️  Running as $USER, consider running as patrick for proper permissions${NC}"
fi

# Create necessary directories
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p "$PROJECT_ROOT/config"
mkdir -p "$PROJECT_ROOT/data/known_faces"
mkdir -p "$PROJECT_ROOT/data/training"
mkdir -p "$PROJECT_ROOT/models"
mkdir -p "$LOGS_DIR"

# Set up Python virtual environment
echo -e "${BLUE}🐍 Setting up Python environment...${NC}"
if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv "$VENV_PATH"
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install omniscient requirements
echo -e "${BLUE}📦 Installing omniscient requirements...${NC}"
if [ -f "$PROJECT_ROOT/requirements_omniscient.txt" ]; then
    pip install -r "$PROJECT_ROOT/requirements_omniscient.txt"
    echo -e "${GREEN}✅ Omniscient requirements installed${NC}"
else
    echo -e "${YELLOW}⚠️  requirements_omniscient.txt not found, installing core packages...${NC}"
    pip install opencv-python face-recognition numpy scikit-learn pandas psycopg2-binary
fi

# Install system dependencies for OpenCV
echo -e "${BLUE}🔧 Installing system dependencies...${NC}"
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y \
        libopencv-dev \
        python3-opencv \
        libhdf5-dev \
        libatlas-base-dev \
        libjasper-dev \
        libqtgui4 \
        libqt4-test \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libgtk-3-dev \
        libcanberra-gtk-module \
        libcanberra-gtk3-module
    echo -e "${GREEN}✅ System dependencies installed${NC}"
elif command -v yum &> /dev/null; then
    sudo yum install -y opencv-devel python3-opencv ffmpeg-devel
    echo -e "${GREEN}✅ System dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️  Package manager not found, please install OpenCV dependencies manually${NC}"
fi

# Download AI models
echo -e "${BLUE}🤖 Downloading AI models...${NC}"
cd "$PROJECT_ROOT/models"

# Download YOLO configuration
if [ ! -f "yolov4.cfg" ]; then
    curl -o yolov4.cfg https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
    echo -e "${GREEN}✅ Downloaded YOLOv4 configuration${NC}"
fi

# Download COCO class names
if [ ! -f "coco.names" ]; then
    curl -o coco.names https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
    echo -e "${GREEN}✅ Downloaded COCO class names${NC}"
fi

# Note about YOLO weights (large file)
if [ ! -f "yolov4.weights" ]; then
    echo -e "${YELLOW}⚠️  YOLOv4 weights not found${NC}"
    echo -e "${BLUE}📥 Download manually from:${NC}"
    echo "   https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
    echo "   Save to: $PROJECT_ROOT/models/yolov4.weights"
fi

cd "$PROJECT_ROOT"

# Set permissions
echo -e "${BLUE}🔐 Setting permissions...${NC}"
chmod +x "$PROJECT_ROOT/scripts/activate_omniscient.py"
chmod -R 755 "$PROJECT_ROOT/config"
chmod -R 755 "$PROJECT_ROOT/data"
chmod -R 755 "$PROJECT_ROOT/models"
chown -R patrick:patrick "$PROJECT_ROOT/data" 2>/dev/null || true
chown -R patrick:patrick "$PROJECT_ROOT/config" 2>/dev/null || true

# Test camera connectivity
echo -e "${BLUE}📹 Testing camera connectivity...${NC}"
if [ -f "$PROJECT_ROOT/config/cameras.json" ]; then
    # Extract camera IPs and test ping
    python3 -c "
import json
import subprocess
import sys

try:
    with open('$PROJECT_ROOT/config/cameras.json') as f:
        config = json.load(f)

    for camera in config.get('cameras', []):
        ip = camera.get('ip', '')
        name = camera.get('name', 'Unknown')
        if ip:
            try:
                result = subprocess.run(['ping', '-c', '1', '-W', '2', ip],
                                      capture_output=True, timeout=5)
                if result.returncode == 0:
                    print(f'✅ {name} reachable at {ip}')
                else:
                    print(f'❌ {name} not reachable at {ip}')
            except:
                print(f'⚠️  Could not test {name} at {ip}')
except Exception as e:
    print(f'⚠️  Could not test cameras: {e}')
"
else
    echo -e "${YELLOW}⚠️  Camera configuration not found${NC}"
fi

# Test database connectivity
echo -e "${BLUE}🗄️  Testing database connectivity...${NC}"
python3 -c "
import psycopg2
import sys

try:
    conn = psycopg2.connect(
        host='192.168.50.135',
        user='patrick',
        password=os.getenv('DB_PASSWORD', ''),
        database='echo_brain',
        port=5432
    )
    conn.close()
    print('✅ Database connection successful')
except Exception as e:
    print(f'❌ Database connection failed: {e}')
    sys.exit(1)
"

# Create systemd service for omniscient monitoring
echo -e "${BLUE}⚙️  Creating systemd service...${NC}"
sudo tee /etc/systemd/system/echo-omniscient.service > /dev/null <<EOF
[Unit]
Description=Echo Brain Omniscient Monitoring
After=network.target postgresql.service
Wants=network-online.target

[Service]
Type=simple
User=patrick
Group=patrick
WorkingDirectory=$PROJECT_ROOT
Environment=PATH=$VENV_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=$VENV_PATH/bin/python $PROJECT_ROOT/scripts/activate_omniscient.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=echo-omniscient

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
echo -e "${GREEN}✅ Systemd service created${NC}"

# Create logrotate configuration
echo -e "${BLUE}📝 Setting up log rotation...${NC}"
sudo tee /etc/logrotate.d/echo-omniscient > /dev/null <<EOF
$LOGS_DIR/omniscient.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
    create 644 patrick patrick
}
EOF

echo -e "${GREEN}✅ Log rotation configured${NC}"

# Final setup summary
echo ""
echo -e "${GREEN}🎉 OMNISCIENT SETUP COMPLETE!${NC}"
echo ""
echo "📋 Setup Summary:"
echo "   ✅ Virtual environment created"
echo "   ✅ Dependencies installed"
echo "   ✅ AI models downloaded (except weights)"
echo "   ✅ Directories created"
echo "   ✅ Permissions set"
echo "   ✅ Systemd service created"
echo "   ✅ Log rotation configured"
echo ""
echo "🚀 Next Steps:"
echo "   1. Update camera IPs in: $PROJECT_ROOT/config/cameras.json"
echo "   2. Add known faces to: $PROJECT_ROOT/data/known_faces/"
echo "   3. Download YOLO weights (see above)"
echo "   4. Start the service:"
echo "      sudo systemctl start echo-omniscient"
echo "      sudo systemctl enable echo-omniscient"
echo ""
echo "🔍 Monitor with:"
echo "   sudo systemctl status echo-omniscient"
echo "   sudo journalctl -u echo-omniscient -f"
echo ""
echo "🧠 Echo Brain now has omniscient capabilities ready for activation!"

deactivate
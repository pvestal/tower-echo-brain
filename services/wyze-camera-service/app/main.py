"""
Wyze Camera Microservice
Tower Network - Echo Brain Integration
"""

import os
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import uvicorn

# Wyze Direct API (SDK is broken with API keys)
from .wyze_direct import WyzeDirectAPI

# RTSP Local Streaming
from .rtsp_stream import rtsp_handler

# Google Auth
from .google_auth import google_auth


# ============================================================================
# CONFIGURATION
# ============================================================================

class Settings(BaseModel):
    service_name: str = "wyze-camera-service"
    service_version: str = "0.1.0"
    wyze_email: str = Field(default_factory=lambda: os.getenv("WYZE_EMAIL", ""))
    wyze_password: str = Field(default_factory=lambda: os.getenv("WYZE_PASSWORD", ""))
    wyze_key_id: str = Field(default_factory=lambda: os.getenv("WYZE_KEY_ID", ""))
    wyze_api_key: str = Field(default_factory=lambda: os.getenv("WYZE_API_KEY", ""))
    wyze_totp_key: Optional[str] = Field(default_factory=lambda: os.getenv("WYZE_TOTP_KEY"))
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]

settings = Settings()


# ============================================================================
# MODELS
# ============================================================================

class CameraInfo(BaseModel):
    mac: str
    nickname: str
    model: str
    is_online: bool
    firmware_ver: Optional[str] = None
    ip: Optional[str] = None
    ssid: Optional[str] = None

class CameraPTZCommand(BaseModel):
    mac: str
    action: str

class CameraToggleCommand(BaseModel):
    mac: str
    enabled: bool

class ServiceHealth(BaseModel):
    status: str
    service: str
    version: str
    wyze_connected: bool
    camera_count: int
    timestamp: datetime

class APIResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# WYZE CLIENT MANAGER
# ============================================================================

# Create global Wyze API instance
wyze_api = WyzeDirectAPI(
    email=settings.wyze_email,
    key_id=settings.wyze_key_id,
    api_key=settings.wyze_api_key
)

# Keep a simple manager for compatibility
class WyzeManager:
    @property
    def is_authenticated(self) -> bool:
        return True  # We have API keys

wyze_manager = WyzeManager()


# ============================================================================
# LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[{settings.service_name}] Starting v{settings.service_version}")

    if settings.wyze_email and settings.wyze_api_key:
        try:
            await wyze_api.authenticate()
            print(f"[{settings.service_name}] Wyze authenticated")
        except Exception as e:
            print(f"[{settings.service_name}] Wyze auth deferred: {e}")

    yield
    print(f"[{settings.service_name}] Shutting down")


# ============================================================================
# APPLICATION
# ============================================================================

app = FastAPI(
    title="Wyze Camera Service",
    description="Tower Network Wyze Camera Integration",
    version=settings.service_version,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


# ============================================================================
# ROUTES - HEALTH
# ============================================================================

@app.get("/health", response_model=ServiceHealth, tags=["Health"])
async def health_check():
    camera_count = 0
    wyze_connected = wyze_manager.is_authenticated

    if wyze_connected:
        try:
            await wyze_api.authenticate()
            devices = await wyze_api.list_devices()
            camera_count = len([d for d in devices if 'CAM' in d.get('product_model', '')])
        except:
            camera_count = 0

    return ServiceHealth(
        status="healthy" if wyze_connected else "degraded",
        service=settings.service_name,
        version=settings.service_version,
        wyze_connected=wyze_connected,
        camera_count=camera_count,
        timestamp=datetime.utcnow()
    )


# ============================================================================
# ROUTES - CAMERAS
# ============================================================================

@app.get("/cameras", response_model=APIResponse, tags=["Cameras"])
async def list_cameras():
    """List cameras using direct Wyze API"""
    try:
        # Authenticate and get devices
        await wyze_api.authenticate()
        devices = await wyze_api.list_devices()

        # Filter cameras and format response
        camera_list = []
        for device in devices:
            # Check for various Wyze camera model identifiers
            model = device.get('product_model', '')
            if 'CAM' in model or 'WYZE' in model or model.startswith('WYZ'):
                params = device.get('device_params', {})
                camera_list.append({
                    "mac": device.get('device_mac'),
                    "nickname": device.get('nickname'),
                    "model": device.get('product_name'),
                    "is_online": params.get('p1', 0) == 1,
                    "firmware_ver": device.get('firmware_ver', '1.0.0'),
                    "ip": params.get('p1301'),
                    "ssid": params.get('p1302')
                })

        return APIResponse(success=True, data={"cameras": camera_list, "count": len(camera_list)})

    except Exception as e:
        return APIResponse(success=False, error=str(e))


@app.get("/cameras/{mac}", response_model=APIResponse, tags=["Cameras"])
async def get_camera(mac: str):
    """Get specific camera by MAC address"""
    try:
        await wyze_api.authenticate()
        devices = await wyze_api.list_devices()

        camera = None
        for device in devices:
            if device.get('device_mac') == mac:
                camera = device
                break

        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")

        params = camera.get('device_params', {})
        return APIResponse(
            success=True,
            data={"camera": {
                "mac": camera.get('device_mac'),
                "nickname": camera.get('nickname'),
                "model": camera.get('product_name'),
                "is_online": params.get('p1', 0) == 1,
                "firmware_ver": camera.get('firmware_ver', '1.0.0'),
                "ip": params.get('p1301'),
                "ssid": params.get('p1302')
            }}
        )

    except Exception as e:
        return APIResponse(success=False, error=str(e))


# ============================================================================
# ROUTES - CONTROLS
# ============================================================================

@app.post("/cameras/power", response_model=APIResponse, tags=["Controls"])
async def toggle_camera_power(cmd: CameraToggleCommand):
    """Toggle camera power on/off"""
    try:
        await wyze_api.authenticate()
        # Power property: p3 (1=on, 0=off)
        result = await wyze_api.control_device(cmd.mac, "p3", 1 if cmd.enabled else 0)
        return APIResponse(success=True, data={"mac": cmd.mac, "power": cmd.enabled})
    except Exception as e:
        return APIResponse(success=False, error=str(e))


@app.post("/cameras/ptz", response_model=APIResponse, tags=["Controls"])
async def camera_ptz(cmd: CameraPTZCommand):
    """Control camera PTZ (pan/tilt/zoom)"""
    try:
        await wyze_api.authenticate()

        # PTZ property mapping for Wyze cameras
        ptz_map = {
            "left": ("p1056", "90"),   # Pan left
            "right": ("p1056", "270"), # Pan right
            "up": ("p1057", "90"),     # Tilt up
            "down": ("p1057", "270"),  # Tilt down
            "reset": ("p1058", "1")    # Reset position
        }

        if cmd.action not in ptz_map:
            return APIResponse(success=False, error=f"Invalid PTZ action: {cmd.action}")

        prop, value = ptz_map[cmd.action]
        result = await wyze_api.control_device(cmd.mac, prop, value)
        return APIResponse(success=True, data={"mac": cmd.mac, "action": cmd.action})

    except Exception as e:
        return APIResponse(success=False, error=str(e))


@app.post("/cameras/{mac}/motion", response_model=APIResponse, tags=["Controls"])
async def toggle_motion_detection(mac: str, enabled: bool = True):
    """Toggle motion detection"""
    try:
        await wyze_api.authenticate()
        # Motion detection property: p1047 (1=enabled, 2=disabled)
        result = await wyze_api.control_device(mac, "p1047", 1 if enabled else 2)
        return APIResponse(success=True, data={"mac": mac, "motion_detection": enabled})
    except Exception as e:
        return APIResponse(success=False, error=str(e))


@app.post("/cameras/{mac}/night-vision", response_model=APIResponse, tags=["Controls"])
async def toggle_night_vision(mac: str, enabled: bool = True):
    """Toggle night vision"""
    try:
        await wyze_api.authenticate()
        # Night vision property: p1048 (1=on, 2=off, 3=auto)
        result = await wyze_api.control_device(mac, "p1048", 1 if enabled else 2)
        return APIResponse(success=True, data={"mac": mac, "night_vision": enabled})
    except Exception as e:
        return APIResponse(success=False, error=str(e))


@app.post("/cameras/{mac}/siren", response_model=APIResponse, tags=["Controls"])
async def trigger_siren(mac: str):
    """Trigger camera siren/alarm"""
    try:
        await wyze_api.authenticate()
        # Siren property: p1049 (1=on)
        result = await wyze_api.control_device(mac, "p1049", 1)
        return APIResponse(success=True, data={"mac": mac, "siren": "triggered"})
    except Exception as e:
        return APIResponse(success=False, error=str(e))


@app.delete("/cameras/{mac}/siren", response_model=APIResponse, tags=["Controls"])
async def stop_siren(mac: str):
    """Stop camera siren/alarm"""
    try:
        await wyze_api.authenticate()
        # Siren property: p1049 (0=off)
        result = await wyze_api.control_device(mac, "p1049", 0)
        return APIResponse(success=True, data={"mac": mac, "siren": "stopped"})
    except Exception as e:
        return APIResponse(success=False, error=str(e))


# ============================================================================
# ROUTES - LOCAL STREAMING
# ============================================================================

@app.get("/cameras/{mac}/stream", tags=["Streaming"])
async def get_camera_stream(mac: str):
    """Get live RTSP stream from camera (local network only)"""
    from fastapi.responses import StreamingResponse

    try:
        stream_generator = await rtsp_handler.get_stream(mac)
        return StreamingResponse(
            stream_generator,
            media_type="multipart/x-mixed-replace;boundary=frame"
        )
    except Exception as e:
        return APIResponse(success=False, error=str(e))


@app.get("/cameras/{mac}/snapshot", tags=["Streaming"])
async def get_camera_snapshot(mac: str):
    """Get single snapshot from camera (local network only)"""
    from fastapi.responses import Response

    try:
        frame = await rtsp_handler.get_frame(mac)
        if frame:
            return Response(content=frame, media_type="image/jpeg")
        else:
            raise HTTPException(status_code=404, detail="Camera not available")
    except Exception as e:
        return APIResponse(success=False, error=str(e))


# ============================================================================
# ROUTES - EVENTS
# ============================================================================

@app.get("/cameras/{mac}/events", response_model=APIResponse, tags=["Events"])
async def get_camera_events(mac: str, limit: int = 20):
    """Get camera events/alerts"""
    try:
        # Events API would require more complex implementation
        # For now return empty list
        return APIResponse(success=True, data={"events": [], "count": 0})
    except Exception as e:
        return APIResponse(success=False, error=str(e))


# ============================================================================
# ROUTES - AUTH
# ============================================================================

@app.get("/auth/google/login", tags=["Auth"])
async def google_login():
    """Initiate Google OAuth login"""
    try:
        auth_url, state = google_auth.get_auth_url()
        return RedirectResponse(url=auth_url)
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.get("/auth/google/callback", response_model=APIResponse, tags=["Auth"])
async def google_callback(request: Request):
    """Handle Google OAuth callback"""
    try:
        result = google_auth.handle_oauth_callback(request)
        return APIResponse(success=True, data=result)
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.get("/auth/status", response_model=APIResponse, tags=["Auth"])
async def auth_status():
    """Get current authentication status"""
    try:
        is_authenticated = google_auth.is_authenticated()
        user = None
        if is_authenticated:
            user = google_auth.get_current_user()

        return APIResponse(success=True, data={
            "authenticated": is_authenticated,
            "user": user,
            "wyze_email": settings.wyze_email
        })
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.post("/auth/logout", response_model=APIResponse, tags=["Auth"])
async def logout():
    """Logout user"""
    try:
        result = google_auth.logout()
        # Also clear Wyze authentication
        wyze_api.access_token = None
        wyze_api.refresh_token = None
        return APIResponse(success=True, data=result)
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.post("/auth/refresh", response_model=APIResponse, tags=["Auth"])
async def refresh_authentication():
    try:
        await wyze_api.authenticate()
        return APIResponse(success=True, data={"message": "Authentication refreshed"})
    except Exception as e:
        return APIResponse(success=False, error=str(e))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8100, reload=True)
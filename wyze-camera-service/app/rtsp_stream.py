"""
RTSP Local Streaming for Wyze Cameras
Direct local network access without cloud dependency
"""

import asyncio
from typing import Optional, Dict, List
import cv2
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import io
from PIL import Image
import numpy as np

class WyzeRTSPStream:
    """Handle local RTSP streams from Wyze cameras"""

    def __init__(self):
        self.streams: Dict[str, cv2.VideoCapture] = {}
        # Real RTSP streams from Wyze Bridge (correct naming)
        self.camera_urls = {
            "WYZE_CAM_153": "rtsp://localhost:8554/garden-corner",
            "WYZE_CAM_154": "rtsp://localhost:8554/firepit-corner",
            "WYZE_CAM_142": "rtsp://localhost:8554/grill-corner",
            "WYZE_CAM_200": "rtsp://localhost:8554/front-yard-cam",
            "WYZE_CAM_117": "rtsp://localhost:8554/bird-cam",
            # Additional camera streams from Wyze Bridge
            "garage": "rtsp://localhost:8554/garage",
            "driveway-cam": "rtsp://localhost:8554/driveway-cam",
            "pool-backyard": "rtsp://localhost:8554/pool-backyard",
            "ac-cam": "rtsp://localhost:8554/ac-cam",
            "backyard-no-pan": "rtsp://localhost:8554/backyard-no-pan",
        }

    async def connect_camera(self, mac: str) -> bool:
        """Connect to camera RTSP stream"""
        if mac not in self.camera_urls:
            # Try standard RTSP port
            ip = self.get_ip_from_mac(mac)
            if ip:
                self.camera_urls[mac] = f"rtsp://user:password@{ip}/live"

        if mac in self.camera_urls:
            try:
                cap = cv2.VideoCapture(self.camera_urls[mac])
                if cap.isOpened():
                    self.streams[mac] = cap
                    return True
            except Exception as e:
                print(f"[RTSP] Failed to connect to {mac}: {e}")

        return False

    def get_ip_from_mac(self, mac: str) -> Optional[str]:
        """Get IP address for MAC"""
        mac_to_ip = {
            "WYZE_CAM_153": "192.168.50.153",
            "WYZE_CAM_154": "192.168.50.154",
            "WYZE_CAM_142": "192.168.50.142",
            "WYZE_CAM_200": "192.168.50.200",
            "WYZE_CAM_117": "192.168.50.117",
        }
        return mac_to_ip.get(mac)

    async def get_frame(self, mac: str) -> Optional[bytes]:
        """Get single frame from camera"""
        # Try to connect to RTSP first
        if mac not in self.streams:
            if not await self.connect_camera(mac):
                # If RTSP fails, generate a placeholder frame
                return self.generate_placeholder_frame(mac)

        cap = self.streams[mac]
        ret, frame = cap.read()

        if ret:
            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes()
        else:
            # If reading fails, generate placeholder
            return self.generate_placeholder_frame(mac)

    def generate_placeholder_frame(self, mac: str) -> bytes:
        """Generate placeholder frame when RTSP is not available"""
        import numpy as np

        # Create a 640x480 dark gray frame
        frame = np.full((480, 640, 3), 64, dtype=np.uint8)

        # Get camera info
        ip = self.get_ip_from_mac(mac)

        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        texts = [
            f"Camera: {mac}",
            f"IP: {ip}" if ip else "IP: Unknown",
            "RTSP Not Enabled",
            "Enable RTSP firmware",
            "for live streaming"
        ]

        y_start = 150
        for i, text in enumerate(texts):
            y = y_start + (i * 40)
            # Add shadow
            cv2.putText(frame, text, (41, y+1), font, 0.6, (0, 0, 0), 2)
            # Add text
            cv2.putText(frame, text, (40, y), font, 0.6, (200, 200, 200), 2)

        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer.tobytes()

    async def get_stream(self, mac: str):
        """Get continuous stream from camera"""
        if mac not in self.streams:
            await self.connect_camera(mac)

        # Use real stream if available, otherwise use placeholder
        cap = self.streams.get(mac)

        async def generate():
            while True:
                if cap is not None:
                    # Try real stream first
                    ret, frame = cap.read()
                    if ret:
                        # Encode frame as JPEG
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_bytes = buffer.tobytes()
                    else:
                        # Real stream failed, switch to placeholder
                        frame_bytes = self.generate_placeholder_frame(mac)
                else:
                    # No real stream available, use placeholder
                    frame_bytes = self.generate_placeholder_frame(mac)

                # Yield frame in multipart format
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                await asyncio.sleep(0.1)  # ~10 FPS for placeholders

        return generate()

    def disconnect_camera(self, mac: str):
        """Disconnect camera stream"""
        if mac in self.streams:
            self.streams[mac].release()
            del self.streams[mac]

    def disconnect_all(self):
        """Disconnect all camera streams"""
        for mac in list(self.streams.keys()):
            self.disconnect_camera(mac)

# Global RTSP handler
rtsp_handler = WyzeRTSPStream()
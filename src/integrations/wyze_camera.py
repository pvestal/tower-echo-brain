#!/usr/bin/env python3
"""
Wyze Camera Integration for Echo Brain Omniscient Capabilities
Provides live video feed processing, motion detection, facial recognition, and scene analysis.
"""

import asyncio
import logging
import cv2
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor
import face_recognition
import requests
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

@dataclass
class CameraInfo:
    """Camera configuration and metadata"""
    camera_id: str
    name: str
    ip_address: str
    username: str
    password: str
    rtmp_url: str
    enabled: bool = True
    motion_detection: bool = True
    facial_recognition: bool = True
    recording: bool = True
    location: str = "Unknown"

@dataclass
class DetectionEvent:
    """Detection event data structure"""
    camera_id: str
    timestamp: datetime
    event_type: str  # motion, face, object, scene_change
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    face_id: Optional[str] = None
    face_name: Optional[str] = None
    object_class: Optional[str] = None
    description: str = ""
    frame_data: Optional[str] = None  # Base64 encoded frame

@dataclass
class SceneAnalysis:
    """Scene analysis results"""
    camera_id: str
    timestamp: datetime
    objects_detected: List[Dict]
    scene_description: str
    lighting_condition: str
    activity_level: str
    safety_assessment: str

class WyzeCameraIntegration:
    """Comprehensive Wyze camera integration for Echo Brain"""

    def __init__(self, echo_brain_instance=None):
        self.echo_brain = echo_brain_instance
        self.cameras: Dict[str, CameraInfo] = {}
        self.active_streams: Dict[str, cv2.VideoCapture] = {}
        self.detection_callbacks: List[Callable] = []
        self.scene_callbacks: List[Callable] = []
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Face recognition setup
        self.known_faces = []
        self.known_face_names = []
        self.load_known_faces()

        # Motion detection setup
        self.background_subtractors = {}
        self.motion_threshold = 5000

        # Object detection setup (using OpenCV DNN)
        self.load_object_detection_model()

        logger.info("✅ Wyze Camera Integration initialized")

    def load_known_faces(self):
        """Load known faces for recognition from Echo Brain database or local files"""
        try:
            # Try to load from Echo Brain's knowledge base
            if self.echo_brain:
                # Implementation would connect to Echo Brain's face database
                pass

            # Fallback to local face database
            faces_dir = "/opt/tower-echo-brain/data/known_faces"
            if os.path.exists(faces_dir):
                for filename in os.listdir(faces_dir):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(faces_dir, filename)
                        image = face_recognition.load_image_file(image_path)
                        encoding = face_recognition.face_encodings(image)
                        if encoding:
                            self.known_faces.append(encoding[0])
                            self.known_face_names.append(filename.split('.')[0])

            logger.info(f"✅ Loaded {len(self.known_faces)} known faces")
        except Exception as e:
            logger.error(f"❌ Error loading known faces: {e}")

    def load_object_detection_model(self):
        """Load YOLO or other object detection model"""
        try:
            # Using OpenCV DNN with pre-trained COCO model
            self.net = cv2.dnn.readNet(
                "/opt/tower-echo-brain/models/yolov4.weights",
                "/opt/tower-echo-brain/models/yolov4.cfg"
            )

            # Load class names
            with open("/opt/tower-echo-brain/models/coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]

            self.output_layers = [self.net.getLayerNames()[i[0] - 1]
                                for i in self.net.getUnconnectedOutLayers()]
            logger.info("✅ Object detection model loaded")
        except Exception as e:
            logger.warning(f"⚠️ Could not load object detection model: {e}")
            self.net = None

    def add_camera(self, camera_info: CameraInfo):
        """Add a Wyze camera to the monitoring system"""
        try:
            self.cameras[camera_info.camera_id] = camera_info

            # Initialize background subtractor for motion detection
            self.background_subtractors[camera_info.camera_id] = cv2.createBackgroundSubtractorMOG2()

            logger.info(f"✅ Added camera: {camera_info.name} ({camera_info.camera_id})")

            # Start monitoring if system is running
            if self.running:
                self.start_camera_stream(camera_info.camera_id)

        except Exception as e:
            logger.error(f"❌ Error adding camera {camera_info.camera_id}: {e}")

    def start_camera_stream(self, camera_id: str):
        """Start monitoring a specific camera stream"""
        try:
            camera = self.cameras.get(camera_id)
            if not camera:
                logger.error(f"❌ Camera {camera_id} not found")
                return

            # Open video stream
            cap = cv2.VideoCapture(camera.rtmp_url)
            if not cap.isOpened():
                logger.error(f"❌ Could not open camera stream for {camera_id}")
                return

            self.active_streams[camera_id] = cap

            # Start processing thread
            self.executor.submit(self._process_camera_stream, camera_id)

            logger.info(f"✅ Started stream for camera {camera_id}")

        except Exception as e:
            logger.error(f"❌ Error starting camera stream {camera_id}: {e}")

    def _process_camera_stream(self, camera_id: str):
        """Process individual camera stream for detection and analysis"""
        camera = self.cameras[camera_id]
        cap = self.active_streams.get(camera_id)

        if not cap:
            return

        last_motion_time = time.time()
        frame_count = 0

        try:
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"⚠️ Could not read frame from camera {camera_id}")
                    time.sleep(1)
                    continue

                frame_count += 1
                current_time = time.time()

                # Process every nth frame to reduce CPU usage
                if frame_count % 5 == 0:
                    # Motion detection
                    if camera.motion_detection:
                        motion_detected = self._detect_motion(camera_id, frame)
                        if motion_detected:
                            last_motion_time = current_time
                            event = DetectionEvent(
                                camera_id=camera_id,
                                timestamp=datetime.now(),
                                event_type="motion",
                                confidence=0.8,
                                description=f"Motion detected in {camera.location}"
                            )
                            self._handle_detection_event(event)

                    # Facial recognition (only if recent motion)
                    if camera.facial_recognition and (current_time - last_motion_time < 30):
                        faces = self._detect_faces(frame)
                        for face_event in faces:
                            face_event.camera_id = camera_id
                            self._handle_detection_event(face_event)

                    # Object detection (every 30 frames)
                    if frame_count % 30 == 0 and self.net:
                        objects = self._detect_objects(frame)
                        if objects:
                            event = DetectionEvent(
                                camera_id=camera_id,
                                timestamp=datetime.now(),
                                event_type="object",
                                confidence=max([obj['confidence'] for obj in objects]),
                                description=f"Objects detected: {', '.join([obj['class'] for obj in objects])}"
                            )
                            self._handle_detection_event(event)

                    # Scene analysis (every 100 frames)
                    if frame_count % 100 == 0:
                        scene = self._analyze_scene(camera_id, frame)
                        self._handle_scene_analysis(scene)

                # Small delay to prevent overwhelming
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"❌ Error processing camera stream {camera_id}: {e}")
        finally:
            if cap:
                cap.release()

    def _detect_motion(self, camera_id: str, frame) -> bool:
        """Detect motion in frame using background subtraction"""
        try:
            bg_subtractor = self.background_subtractors.get(camera_id)
            if not bg_subtractor:
                return False

            # Apply background subtraction
            fg_mask = bg_subtractor.apply(frame)

            # Calculate motion area
            motion_area = cv2.countNonZero(fg_mask)

            return motion_area > self.motion_threshold

        except Exception as e:
            logger.error(f"❌ Error detecting motion: {e}")
            return False

    def _detect_faces(self, frame) -> List[DetectionEvent]:
        """Detect and recognize faces in frame"""
        events = []
        try:
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find faces
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Try to match face
                matches = face_recognition.compare_faces(self.known_faces, face_encoding)
                face_name = "Unknown"
                face_id = "unknown"
                confidence = 0.5

                if True in matches:
                    first_match_index = matches.index(True)
                    face_name = self.known_face_names[first_match_index]
                    face_id = face_name.lower().replace(' ', '_')
                    confidence = 0.9

                event = DetectionEvent(
                    camera_id="",  # Will be set by caller
                    timestamp=datetime.now(),
                    event_type="face",
                    confidence=confidence,
                    bbox=(left, top, right, bottom),
                    face_id=face_id,
                    face_name=face_name,
                    description=f"Face detected: {face_name}"
                )
                events.append(event)

        except Exception as e:
            logger.error(f"❌ Error detecting faces: {e}")

        return events

    def _detect_objects(self, frame) -> List[Dict]:
        """Detect objects using YOLO model"""
        objects = []
        try:
            if not self.net:
                return objects

            height, width = frame.shape[:2]

            # Prepare input blob
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)

            # Parse detections
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        objects.append({
                            'class': self.classes[class_id],
                            'confidence': float(confidence),
                            'bbox': (center_x - w // 2, center_y - h // 2, w, h)
                        })

        except Exception as e:
            logger.error(f"❌ Error detecting objects: {e}")

        return objects

    def _analyze_scene(self, camera_id: str, frame) -> SceneAnalysis:
        """Perform comprehensive scene analysis"""
        try:
            camera = self.cameras[camera_id]

            # Basic scene metrics
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)

            # Lighting condition
            if brightness < 50:
                lighting = "Dark"
            elif brightness < 150:
                lighting = "Normal"
            else:
                lighting = "Bright"

            # Activity level based on edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])

            if edge_density < 0.1:
                activity = "Low"
            elif edge_density < 0.3:
                activity = "Medium"
            else:
                activity = "High"

            # Object detection for scene description
            objects = self._detect_objects(frame)
            object_classes = [obj['class'] for obj in objects]

            # Generate scene description
            if object_classes:
                description = f"Scene contains: {', '.join(set(object_classes))}"
            else:
                description = "Empty scene or no recognizable objects"

            # Safety assessment
            safety = "Safe"
            if "person" in object_classes and lighting == "Dark":
                safety = "Monitor"

            return SceneAnalysis(
                camera_id=camera_id,
                timestamp=datetime.now(),
                objects_detected=objects,
                scene_description=description,
                lighting_condition=lighting,
                activity_level=activity,
                safety_assessment=safety
            )

        except Exception as e:
            logger.error(f"❌ Error analyzing scene: {e}")
            return SceneAnalysis(
                camera_id=camera_id,
                timestamp=datetime.now(),
                objects_detected=[],
                scene_description="Analysis failed",
                lighting_condition="Unknown",
                activity_level="Unknown",
                safety_assessment="Unknown"
            )

    def _handle_detection_event(self, event: DetectionEvent):
        """Handle detection events and integrate with Echo Brain"""
        try:
            # Log event
            logger.info(f"🎯 Detection Event: {event.event_type} - {event.description}")

            # Send to Echo Brain if available
            if self.echo_brain:
                asyncio.create_task(self._send_to_echo_brain(event))

            # Trigger callbacks
            for callback in self.detection_callbacks:
                callback(event)

        except Exception as e:
            logger.error(f"❌ Error handling detection event: {e}")

    def _handle_scene_analysis(self, analysis: SceneAnalysis):
        """Handle scene analysis results"""
        try:
            logger.info(f"🔍 Scene Analysis: {analysis.scene_description}")

            # Send to Echo Brain if available
            if self.echo_brain:
                asyncio.create_task(self._send_scene_to_echo_brain(analysis))

            # Trigger callbacks
            for callback in self.scene_callbacks:
                callback(analysis)

        except Exception as e:
            logger.error(f"❌ Error handling scene analysis: {e}")

    async def _send_to_echo_brain(self, event: DetectionEvent):
        """Send detection event to Echo Brain for learning and context"""
        try:
            # Format for Echo Brain consumption
            brain_data = {
                "type": "camera_detection",
                "timestamp": event.timestamp.isoformat(),
                "camera_id": event.camera_id,
                "event_type": event.event_type,
                "description": event.description,
                "confidence": event.confidence,
                "metadata": asdict(event)
            }

            # Send to Echo Brain's memory system
            if hasattr(self.echo_brain, 'store_context'):
                await self.echo_brain.store_context(
                    f"camera_detection_{event.camera_id}_{event.timestamp}",
                    brain_data,
                    source="wyze_camera"
                )

        except Exception as e:
            logger.error(f"❌ Error sending to Echo Brain: {e}")

    async def _send_scene_to_echo_brain(self, analysis: SceneAnalysis):
        """Send scene analysis to Echo Brain for environmental awareness"""
        try:
            brain_data = {
                "type": "scene_analysis",
                "timestamp": analysis.timestamp.isoformat(),
                "camera_id": analysis.camera_id,
                "scene_description": analysis.scene_description,
                "lighting": analysis.lighting_condition,
                "activity": analysis.activity_level,
                "safety": analysis.safety_assessment,
                "objects": analysis.objects_detected
            }

            if hasattr(self.echo_brain, 'store_context'):
                await self.echo_brain.store_context(
                    f"scene_analysis_{analysis.camera_id}_{analysis.timestamp}",
                    brain_data,
                    source="wyze_camera"
                )

        except Exception as e:
            logger.error(f"❌ Error sending scene analysis to Echo Brain: {e}")

    def start_monitoring(self):
        """Start monitoring all cameras"""
        try:
            self.running = True

            # Start streams for all enabled cameras
            for camera_id, camera in self.cameras.items():
                if camera.enabled:
                    self.start_camera_stream(camera_id)

            logger.info(f"✅ Started monitoring {len(self.cameras)} cameras")

        except Exception as e:
            logger.error(f"❌ Error starting monitoring: {e}")

    def stop_monitoring(self):
        """Stop monitoring all cameras"""
        try:
            self.running = False

            # Close all streams
            for camera_id, cap in self.active_streams.items():
                cap.release()

            self.active_streams.clear()
            self.executor.shutdown(wait=True)

            logger.info("✅ Stopped camera monitoring")

        except Exception as e:
            logger.error(f"❌ Error stopping monitoring: {e}")

    def get_camera_status(self) -> Dict:
        """Get status of all cameras"""
        status = {}
        for camera_id, camera in self.cameras.items():
            status[camera_id] = {
                "name": camera.name,
                "enabled": camera.enabled,
                "streaming": camera_id in self.active_streams,
                "location": camera.location,
                "features": {
                    "motion_detection": camera.motion_detection,
                    "facial_recognition": camera.facial_recognition,
                    "recording": camera.recording
                }
            }
        return status

    def add_detection_callback(self, callback: Callable[[DetectionEvent], None]):
        """Add callback for detection events"""
        self.detection_callbacks.append(callback)

    def add_scene_callback(self, callback: Callable[[SceneAnalysis], None]):
        """Add callback for scene analysis"""
        self.scene_callbacks.append(callback)

    def get_recent_events(self, hours: int = 24) -> List[DetectionEvent]:
        """Get recent detection events from Echo Brain storage"""
        # This would query Echo Brain's memory system for recent camera events
        # Implementation depends on Echo Brain's storage interface
        return []

# Configuration loader
def load_camera_config(config_path: str = "/opt/tower-echo-brain/config/cameras.json") -> List[CameraInfo]:
    """Load camera configuration from JSON file"""
    cameras = []
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        for cam_config in config.get('cameras', []):
            camera = CameraInfo(
                camera_id=cam_config['id'],
                name=cam_config['name'],
                ip_address=cam_config['ip'],
                username=cam_config.get('username', ''),
                password=cam_config.get('password', ''),
                rtmp_url=cam_config['rtmp_url'],
                enabled=cam_config.get('enabled', True),
                motion_detection=cam_config.get('motion_detection', True),
                facial_recognition=cam_config.get('facial_recognition', True),
                recording=cam_config.get('recording', True),
                location=cam_config.get('location', 'Unknown')
            )
            cameras.append(camera)

        logger.info(f"✅ Loaded {len(cameras)} cameras from config")

    except FileNotFoundError:
        logger.warning(f"⚠️ Camera config file not found: {config_path}")
    except Exception as e:
        logger.error(f"❌ Error loading camera config: {e}")

    return cameras

# Integration function for Echo Brain
def create_wyze_integration(echo_brain_instance=None) -> WyzeCameraIntegration:
    """Create and configure Wyze camera integration for Echo Brain"""
    integration = WyzeCameraIntegration(echo_brain_instance)

    # Load cameras from config
    cameras = load_camera_config()
    for camera in cameras:
        integration.add_camera(camera)

    return integration
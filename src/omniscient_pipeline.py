#!/usr/bin/env python3
"""
Omniscient Echo Brain Pipeline
Integrates Wyze camera feeds, conversation data, and real-time learning for comprehensive awareness.
"""

import asyncio
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

# Import our custom integrations
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.wyze_camera import (
    WyzeCameraIntegration, CameraInfo, DetectionEvent, SceneAnalysis,
    create_wyze_integration, load_camera_config
)
from training.conversation_extractor import (
    ConversationExtractor, ConversationData, TrainingDataset,
    create_conversation_extractor
)

logger = logging.getLogger(__name__)

@dataclass
class OmniscientContext:
    """Comprehensive context from all awareness sources"""
    timestamp: datetime
    camera_events: List[DetectionEvent]
    scene_analysis: List[SceneAnalysis]
    recent_conversations: List[ConversationData]
    environmental_state: Dict[str, Any]
    behavioral_patterns: Dict[str, Any]
    learning_insights: Dict[str, Any]

@dataclass
class BehaviorPattern:
    """Detected behavioral pattern from integrated data"""
    pattern_id: str
    pattern_type: str  # routine, anomaly, preference, interaction
    confidence: float
    description: str
    triggers: List[str]
    contexts: List[str]
    frequency: float
    last_observed: datetime

class OmniscientPipeline:
    """Comprehensive awareness and learning pipeline for Echo Brain"""

    def __init__(self, echo_brain_instance=None, config_path: str = "/opt/tower-echo-brain/config"):
        self.echo_brain = echo_brain_instance
        self.config_path = config_path
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=8)

        # Initialize components
        self.camera_integration = None
        self.conversation_extractor = None
        self.context_history: List[OmniscientContext] = []
        self.behavior_patterns: Dict[str, BehaviorPattern] = {}
        self.learning_callbacks: List[Callable] = []

        # Configuration
        self.config = self._load_config()
        self.analysis_window = timedelta(hours=2)  # Context window for analysis

        # Data storage
        self.detection_buffer: List[DetectionEvent] = []
        self.scene_buffer: List[SceneAnalysis] = []
        self.conversation_buffer: List[ConversationData] = []

        logger.info("🧠 Omniscient Pipeline initialized")

    def _load_config(self) -> Dict:
        """Load omniscient pipeline configuration"""
        default_config = {
            "camera_settings": {
                "enable_facial_recognition": True,
                "enable_motion_detection": True,
                "enable_object_detection": True,
                "enable_scene_analysis": True,
                "analysis_frequency": 30  # seconds
            },
            "learning_settings": {
                "conversation_window_hours": 24,
                "behavior_detection_threshold": 0.7,
                "pattern_learning_enabled": True,
                "real_time_training": True,
                "context_memory_limit": 1000
            },
            "integration_settings": {
                "echo_brain_integration": True,
                "auto_context_updates": True,
                "privacy_mode": False,
                "data_retention_days": 90
            },
            "database": {
                "host": "192.168.50.135",
                "user": "patrick",
                "password": "RP78eIrW7cI2jYvL5akt1yurE",
                "database": "echo_brain"
            }
        }

        try:
            config_file = os.path.join(self.config_path, "omniscient.json")
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
        except Exception as e:
            logger.warning(f"⚠️ Could not load config, using defaults: {e}")

        return default_config

    async def initialize(self):
        """Initialize all pipeline components"""
        try:
            logger.info("🔄 Initializing Omniscient Pipeline components...")

            # Initialize camera integration
            await self._initialize_cameras()

            # Initialize conversation extractor
            await self._initialize_conversation_system()

            # Setup data processing pipelines
            await self._setup_data_pipelines()

            # Initialize Echo Brain integration
            if self.echo_brain and self.config["integration_settings"]["echo_brain_integration"]:
                await self._initialize_echo_brain_integration()

            logger.info("✅ Omniscient Pipeline fully initialized")

        except Exception as e:
            logger.error(f"❌ Error initializing Omniscient Pipeline: {e}")
            raise

    async def _initialize_cameras(self):
        """Initialize Wyze camera integration"""
        try:
            self.camera_integration = create_wyze_integration(self.echo_brain)

            # Load camera configurations
            cameras = load_camera_config()
            if not cameras:
                # Create sample camera configuration for testing
                logger.warning("⚠️ No camera config found, creating sample configuration")
                await self._create_sample_camera_config()
                cameras = load_camera_config()

            for camera in cameras:
                self.camera_integration.add_camera(camera)

            # Register callbacks for camera events
            self.camera_integration.add_detection_callback(self._on_detection_event)
            self.camera_integration.add_scene_callback(self._on_scene_analysis)

            logger.info(f"✅ Camera integration initialized with {len(cameras)} cameras")

        except Exception as e:
            logger.error(f"❌ Error initializing cameras: {e}")

    async def _initialize_conversation_system(self):
        """Initialize conversation data extraction system"""
        try:
            db_config = self.config["database"]
            self.conversation_extractor = create_conversation_extractor(db_config)

            logger.info("✅ Conversation extraction system initialized")

        except Exception as e:
            logger.error(f"❌ Error initializing conversation system: {e}")

    async def _setup_data_pipelines(self):
        """Setup data processing pipelines"""
        try:
            # Start background data processing
            asyncio.create_task(self._process_data_continuously())
            asyncio.create_task(self._analyze_patterns_continuously())
            asyncio.create_task(self._update_echo_brain_continuously())

            logger.info("✅ Data pipelines setup complete")

        except Exception as e:
            logger.error(f"❌ Error setting up data pipelines: {e}")

    async def _initialize_echo_brain_integration(self):
        """Initialize integration with Echo Brain's memory and learning systems"""
        try:
            # Register with Echo Brain's consciousness system
            if hasattr(self.echo_brain, 'register_awareness_source'):
                await self.echo_brain.register_awareness_source('omniscient_pipeline', self)

            # Setup context sharing
            if hasattr(self.echo_brain, 'add_context_provider'):
                await self.echo_brain.add_context_provider(self._provide_omniscient_context)

            logger.info("✅ Echo Brain integration established")

        except Exception as e:
            logger.error(f"❌ Error initializing Echo Brain integration: {e}")

    async def start(self):
        """Start the omniscient pipeline"""
        try:
            if self.running:
                logger.warning("⚠️ Pipeline already running")
                return

            self.running = True

            # Start camera monitoring
            if self.camera_integration:
                self.camera_integration.start_monitoring()

            # Start data extraction and learning
            asyncio.create_task(self._continuous_learning_loop())

            logger.info("🚀 Omniscient Pipeline started")

        except Exception as e:
            logger.error(f"❌ Error starting pipeline: {e}")

    async def stop(self):
        """Stop the omniscient pipeline"""
        try:
            self.running = False

            # Stop camera monitoring
            if self.camera_integration:
                self.camera_integration.stop_monitoring()

            # Cleanup
            self.executor.shutdown(wait=True)

            logger.info("🛑 Omniscient Pipeline stopped")

        except Exception as e:
            logger.error(f"❌ Error stopping pipeline: {e}")

    def _on_detection_event(self, event: DetectionEvent):
        """Handle camera detection events"""
        try:
            self.detection_buffer.append(event)

            # Trigger immediate analysis for important events
            if event.event_type == "face" and event.confidence > 0.8:
                asyncio.create_task(self._process_high_priority_event(event))

            # Limit buffer size
            if len(self.detection_buffer) > 1000:
                self.detection_buffer = self.detection_buffer[-500:]

        except Exception as e:
            logger.error(f"❌ Error handling detection event: {e}")

    def _on_scene_analysis(self, analysis: SceneAnalysis):
        """Handle scene analysis results"""
        try:
            self.scene_buffer.append(analysis)

            # Update environmental context
            asyncio.create_task(self._update_environmental_context(analysis))

            # Limit buffer size
            if len(self.scene_buffer) > 500:
                self.scene_buffer = self.scene_buffer[-250:]

        except Exception as e:
            logger.error(f"❌ Error handling scene analysis: {e}")

    async def _process_high_priority_event(self, event: DetectionEvent):
        """Process high-priority events immediately"""
        try:
            # Create immediate context update
            context = await self._generate_current_context()

            # Update Echo Brain immediately
            if self.echo_brain:
                await self._send_context_to_echo_brain(context, priority="high")

            # Check for behavioral patterns
            patterns = await self._analyze_immediate_patterns(event, context)
            for pattern in patterns:
                await self._record_behavior_pattern(pattern)

        except Exception as e:
            logger.error(f"❌ Error processing high-priority event: {e}")

    async def _continuous_learning_loop(self):
        """Main continuous learning loop"""
        while self.running:
            try:
                # Extract new conversation data
                if self.conversation_extractor:
                    await self._extract_recent_conversations()

                # Generate comprehensive context
                context = await self._generate_current_context()
                self.context_history.append(context)

                # Limit context history
                if len(self.context_history) > self.config["learning_settings"]["context_memory_limit"]:
                    self.context_history = self.context_history[-500:]

                # Trigger learning callbacks
                for callback in self.learning_callbacks:
                    try:
                        await callback(context)
                    except Exception as e:
                        logger.warning(f"⚠️ Error in learning callback: {e}")

                await asyncio.sleep(30)  # Process every 30 seconds

            except Exception as e:
                logger.error(f"❌ Error in continuous learning loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _extract_recent_conversations(self):
        """Extract recent conversation data"""
        try:
            window_hours = self.config["learning_settings"]["conversation_window_hours"]
            start_time = datetime.now() - timedelta(hours=window_hours)

            # Extract conversations from the time window
            dataset = await self.conversation_extractor.extract_all_conversations(
                start_date=start_time,
                end_date=datetime.now()
            )

            # Add new conversations to buffer
            for conv in dataset.conversations:
                if conv.timestamp > start_time and conv not in self.conversation_buffer:
                    self.conversation_buffer.append(conv)

            # Limit buffer size
            if len(self.conversation_buffer) > 1000:
                self.conversation_buffer = self.conversation_buffer[-500:]

        except Exception as e:
            logger.error(f"❌ Error extracting recent conversations: {e}")

    async def _generate_current_context(self) -> OmniscientContext:
        """Generate comprehensive current context"""
        try:
            now = datetime.now()
            window_start = now - self.analysis_window

            # Get recent events
            recent_detections = [
                event for event in self.detection_buffer
                if event.timestamp >= window_start
            ]

            recent_scenes = [
                scene for scene in self.scene_buffer
                if scene.timestamp >= window_start
            ]

            recent_conversations = [
                conv for conv in self.conversation_buffer
                if conv.timestamp >= window_start
            ]

            # Analyze environmental state
            environmental_state = await self._analyze_environmental_state(recent_scenes)

            # Detect behavioral patterns
            behavioral_patterns = await self._detect_behavioral_patterns(
                recent_detections, recent_conversations
            )

            # Generate learning insights
            learning_insights = await self._generate_learning_insights(
                recent_detections, recent_scenes, recent_conversations
            )

            return OmniscientContext(
                timestamp=now,
                camera_events=recent_detections,
                scene_analysis=recent_scenes,
                recent_conversations=recent_conversations,
                environmental_state=environmental_state,
                behavioral_patterns=behavioral_patterns,
                learning_insights=learning_insights
            )

        except Exception as e:
            logger.error(f"❌ Error generating current context: {e}")
            return OmniscientContext(
                timestamp=datetime.now(),
                camera_events=[],
                scene_analysis=[],
                recent_conversations=[],
                environmental_state={},
                behavioral_patterns={},
                learning_insights={}
            )

    async def _analyze_environmental_state(self, recent_scenes: List[SceneAnalysis]) -> Dict[str, Any]:
        """Analyze current environmental state from scene data"""
        state = {
            "lighting": "unknown",
            "activity_level": "unknown",
            "occupancy": "unknown",
            "safety_status": "unknown",
            "locations": {}
        }

        try:
            if not recent_scenes:
                return state

            # Aggregate lighting conditions
            lighting_counts = {}
            activity_counts = {}
            safety_counts = {}

            for scene in recent_scenes:
                # Count lighting conditions
                lighting = scene.lighting_condition
                lighting_counts[lighting] = lighting_counts.get(lighting, 0) + 1

                # Count activity levels
                activity = scene.activity_level
                activity_counts[activity] = activity_counts.get(activity, 0) + 1

                # Count safety assessments
                safety = scene.safety_assessment
                safety_counts[safety] = safety_counts.get(safety, 0) + 1

                # Track location-specific data
                camera_id = scene.camera_id
                if camera_id not in state["locations"]:
                    state["locations"][camera_id] = {
                        "last_activity": scene.timestamp.isoformat(),
                        "objects_detected": scene.objects_detected,
                        "description": scene.scene_description
                    }

            # Determine dominant states
            if lighting_counts:
                state["lighting"] = max(lighting_counts, key=lighting_counts.get)
            if activity_counts:
                state["activity_level"] = max(activity_counts, key=activity_counts.get)
            if safety_counts:
                state["safety_status"] = max(safety_counts, key=safety_counts.get)

            # Determine occupancy
            person_detections = sum(1 for scene in recent_scenes
                                  if any("person" in str(obj) for obj in scene.objects_detected))
            state["occupancy"] = "occupied" if person_detections > 0 else "empty"

        except Exception as e:
            logger.error(f"❌ Error analyzing environmental state: {e}")

        return state

    async def _detect_behavioral_patterns(self, detections: List[DetectionEvent],
                                        conversations: List[ConversationData]) -> Dict[str, Any]:
        """Detect behavioral patterns from integrated data"""
        patterns = {
            "routines": [],
            "preferences": [],
            "anomalies": [],
            "interaction_patterns": []
        }

        try:
            # Analyze face detection patterns for routines
            face_events = [d for d in detections if d.event_type == "face"]
            if face_events:
                patterns["routines"] = await self._analyze_routine_patterns(face_events)

            # Analyze conversation patterns for preferences
            if conversations:
                patterns["preferences"] = await self._analyze_preference_patterns(conversations)

            # Detect anomalies
            patterns["anomalies"] = await self._detect_anomalies(detections, conversations)

            # Analyze interaction patterns
            patterns["interaction_patterns"] = await self._analyze_interaction_patterns(
                face_events, conversations
            )

        except Exception as e:
            logger.error(f"❌ Error detecting behavioral patterns: {e}")

        return patterns

    async def _generate_learning_insights(self, detections: List[DetectionEvent],
                                        scenes: List[SceneAnalysis],
                                        conversations: List[ConversationData]) -> Dict[str, Any]:
        """Generate insights for learning and improvement"""
        insights = {
            "data_quality": {},
            "learning_opportunities": [],
            "knowledge_gaps": [],
            "improvement_suggestions": []
        }

        try:
            # Analyze data quality
            insights["data_quality"] = {
                "detection_count": len(detections),
                "scene_count": len(scenes),
                "conversation_count": len(conversations),
                "average_detection_confidence": np.mean([d.confidence for d in detections]) if detections else 0,
                "data_freshness": (datetime.now() - max([d.timestamp for d in detections], default=datetime.now())).seconds if detections else 0
            }

            # Identify learning opportunities
            learning_opportunities = []
            if len(conversations) > 10:
                learning_opportunities.append("Rich conversation data available for training")
            if len(detections) > 50:
                learning_opportunities.append("Extensive detection data for behavior analysis")

            insights["learning_opportunities"] = learning_opportunities

            # Identify knowledge gaps
            gaps = []
            if not any(d.event_type == "face" for d in detections):
                gaps.append("Limited facial recognition data")
            if not conversations:
                gaps.append("No recent conversation data")

            insights["knowledge_gaps"] = gaps

            # Generate improvement suggestions
            suggestions = []
            if insights["data_quality"]["average_detection_confidence"] < 0.7:
                suggestions.append("Improve camera detection models")
            if len(conversations) < 5:
                suggestions.append("Increase conversation data collection")

            insights["improvement_suggestions"] = suggestions

        except Exception as e:
            logger.error(f"❌ Error generating learning insights: {e}")

        return insights

    async def _analyze_routine_patterns(self, face_events: List[DetectionEvent]) -> List[Dict]:
        """Analyze face detection events for routine patterns"""
        routines = []
        try:
            # Group by face and analyze timing patterns
            face_groups = {}
            for event in face_events:
                face_id = event.face_id or "unknown"
                if face_id not in face_groups:
                    face_groups[face_id] = []
                face_groups[face_id].append(event)

            for face_id, events in face_groups.items():
                if len(events) > 5:  # Enough data for pattern analysis
                    # Analyze timing patterns
                    hours = [event.timestamp.hour for event in events]
                    common_hours = [h for h in set(hours) if hours.count(h) >= 2]

                    if common_hours:
                        routines.append({
                            "type": "presence_routine",
                            "face_id": face_id,
                            "common_hours": common_hours,
                            "frequency": len(events),
                            "confidence": min(0.9, len(events) / 20)
                        })

        except Exception as e:
            logger.error(f"❌ Error analyzing routine patterns: {e}")

        return routines

    async def _analyze_preference_patterns(self, conversations: List[ConversationData]) -> List[Dict]:
        """Analyze conversation patterns for user preferences"""
        preferences = []
        try:
            # Extract topics and themes from conversations
            topics = []
            for conv in conversations:
                if conv.topics:
                    topics.extend(conv.topics)

            # Find common topics
            topic_counts = {}
            for topic in topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

            for topic, count in topic_counts.items():
                if count >= 3:  # Topic appears multiple times
                    preferences.append({
                        "type": "topic_preference",
                        "topic": topic,
                        "frequency": count,
                        "confidence": min(0.9, count / len(conversations))
                    })

        except Exception as e:
            logger.error(f"❌ Error analyzing preference patterns: {e}")

        return preferences

    async def _detect_anomalies(self, detections: List[DetectionEvent],
                              conversations: List[ConversationData]) -> List[Dict]:
        """Detect anomalous patterns in the data"""
        anomalies = []
        try:
            # Unusual activity times
            if detections:
                hours = [d.timestamp.hour for d in detections]
                night_activity = [h for h in hours if h < 6 or h > 22]
                if len(night_activity) > len(hours) * 0.3:  # More than 30% night activity
                    anomalies.append({
                        "type": "unusual_activity_time",
                        "description": "High activity during night hours",
                        "confidence": 0.7
                    })

            # Unusual conversation patterns
            if conversations:
                response_lengths = [len(conv.assistant_response.split()) for conv in conversations]
                avg_length = np.mean(response_lengths)
                if avg_length < 5:  # Very short responses
                    anomalies.append({
                        "type": "unusual_conversation_pattern",
                        "description": "Unusually short assistant responses",
                        "confidence": 0.6
                    })

        except Exception as e:
            logger.error(f"❌ Error detecting anomalies: {e}")

        return anomalies

    async def _analyze_interaction_patterns(self, face_events: List[DetectionEvent],
                                          conversations: List[ConversationData]) -> List[Dict]:
        """Analyze patterns between face detections and conversations"""
        patterns = []
        try:
            # Check temporal correlation between face detection and conversations
            if face_events and conversations:
                face_times = [event.timestamp for event in face_events]
                conv_times = [conv.timestamp for conv in conversations]

                # Find conversations shortly after face detection
                correlations = 0
                for conv_time in conv_times:
                    for face_time in face_times:
                        if abs((conv_time - face_time).total_seconds()) < 300:  # Within 5 minutes
                            correlations += 1
                            break

                if correlations > 0:
                    correlation_rate = correlations / len(conversations)
                    patterns.append({
                        "type": "face_conversation_correlation",
                        "correlation_rate": correlation_rate,
                        "description": f"{correlations} conversations correlated with face detection",
                        "confidence": min(0.9, correlation_rate * 2)
                    })

        except Exception as e:
            logger.error(f"❌ Error analyzing interaction patterns: {e}")

        return patterns

    async def _record_behavior_pattern(self, pattern: BehaviorPattern):
        """Record detected behavior pattern"""
        try:
            self.behavior_patterns[pattern.pattern_id] = pattern

            # Send to Echo Brain if available
            if self.echo_brain:
                pattern_data = {
                    "type": "behavior_pattern",
                    "pattern": asdict(pattern)
                }
                if hasattr(self.echo_brain, 'store_context'):
                    await self.echo_brain.store_context(
                        f"behavior_pattern_{pattern.pattern_id}",
                        pattern_data,
                        source="omniscient_pipeline"
                    )

        except Exception as e:
            logger.error(f"❌ Error recording behavior pattern: {e}")

    async def _provide_omniscient_context(self) -> Dict[str, Any]:
        """Provide comprehensive context to Echo Brain"""
        try:
            if not self.context_history:
                return {}

            latest_context = self.context_history[-1]
            return {
                "omniscient_context": {
                    "timestamp": latest_context.timestamp.isoformat(),
                    "environmental_state": latest_context.environmental_state,
                    "behavioral_patterns": latest_context.behavioral_patterns,
                    "learning_insights": latest_context.learning_insights,
                    "recent_activity": {
                        "detections": len(latest_context.camera_events),
                        "scenes": len(latest_context.scene_analysis),
                        "conversations": len(latest_context.recent_conversations)
                    }
                },
                "behavior_patterns": {
                    pattern_id: asdict(pattern)
                    for pattern_id, pattern in self.behavior_patterns.items()
                }
            }

        except Exception as e:
            logger.error(f"❌ Error providing omniscient context: {e}")
            return {}

    async def _create_sample_camera_config(self):
        """Create sample camera configuration for testing"""
        try:
            config_dir = Path(self.config_path)
            config_dir.mkdir(parents=True, exist_ok=True)

            sample_config = {
                "cameras": [
                    {
                        "id": "wyze_living_room",
                        "name": "Living Room Camera",
                        "ip": "192.168.50.200",
                        "rtmp_url": "rtmp://192.168.50.200:1935/live/stream",
                        "username": "admin",
                        "password": "password",
                        "enabled": True,
                        "motion_detection": True,
                        "facial_recognition": True,
                        "recording": True,
                        "location": "Living Room"
                    },
                    {
                        "id": "wyze_kitchen",
                        "name": "Kitchen Camera",
                        "ip": "192.168.50.201",
                        "rtmp_url": "rtmp://192.168.50.201:1935/live/stream",
                        "username": "admin",
                        "password": "password",
                        "enabled": True,
                        "motion_detection": True,
                        "facial_recognition": True,
                        "recording": True,
                        "location": "Kitchen"
                    }
                ]
            }

            with open(config_dir / "cameras.json", 'w') as f:
                json.dump(sample_config, f, indent=2)

            logger.info("✅ Sample camera configuration created")

        except Exception as e:
            logger.error(f"❌ Error creating sample camera config: {e}")

    def add_learning_callback(self, callback: Callable[[OmniscientContext], None]):
        """Add callback for learning events"""
        self.learning_callbacks.append(callback)

    def get_current_awareness(self) -> Dict[str, Any]:
        """Get current awareness state"""
        try:
            if not self.context_history:
                return {"status": "no_data"}

            latest = self.context_history[-1]
            return {
                "timestamp": latest.timestamp.isoformat(),
                "environmental_state": latest.environmental_state,
                "active_cameras": len(set(event.camera_id for event in latest.camera_events)),
                "recent_detections": len(latest.camera_events),
                "behavior_patterns": len(self.behavior_patterns),
                "learning_insights": latest.learning_insights
            }

        except Exception as e:
            logger.error(f"❌ Error getting current awareness: {e}")
            return {"status": "error", "error": str(e)}

# Factory function for Echo Brain integration
async def create_omniscient_pipeline(echo_brain_instance=None, config_path: str = "/opt/tower-echo-brain/config") -> OmniscientPipeline:
    """Create and initialize omniscient pipeline"""
    pipeline = OmniscientPipeline(echo_brain_instance, config_path)
    await pipeline.initialize()
    return pipeline
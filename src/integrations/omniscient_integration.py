#!/usr/bin/env python3
"""
Echo Brain Omniscient Integration
Integrates omniscient capabilities directly into Echo Brain's core systems.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

logger = logging.getLogger(__name__)

class EchoBrainOmniscientIntegration:
    """Integration layer between omniscient pipeline and Echo Brain"""

    def __init__(self, echo_brain_instance):
        self.echo_brain = echo_brain_instance
        self.omniscient_pipeline = None
        self.integration_active = False

    async def initialize(self):
        """Initialize the omniscient integration"""
        try:
            # Import and create omniscient pipeline
            from omniscient_pipeline import create_omniscient_pipeline

            self.omniscient_pipeline = await create_omniscient_pipeline(
                self.echo_brain,
                "/opt/tower-echo-brain/config"
            )

            # Register omniscient capabilities with Echo Brain
            await self._register_capabilities()

            # Setup event handlers
            await self._setup_event_handlers()

            # Start the pipeline
            await self.omniscient_pipeline.start()

            self.integration_active = True
            logger.info("✅ Omniscient integration initialized successfully")

        except Exception as e:
            logger.error(f"❌ Error initializing omniscient integration: {e}")
            raise

    async def _register_capabilities(self):
        """Register omniscient capabilities with Echo Brain"""
        try:
            # Register new API endpoints for camera control
            if hasattr(self.echo_brain, 'register_capability'):
                await self.echo_brain.register_capability(
                    'camera_monitoring',
                    self._handle_camera_requests
                )
                await self.echo_brain.register_capability(
                    'environmental_awareness',
                    self._handle_environment_requests
                )
                await self.echo_brain.register_capability(
                    'behavioral_analysis',
                    self._handle_behavior_requests
                )

            # Enhance Echo Brain's context providers
            if hasattr(self.echo_brain, 'add_context_provider'):
                await self.echo_brain.add_context_provider(
                    'omniscient_context',
                    self._provide_omniscient_context
                )

        except Exception as e:
            logger.error(f"❌ Error registering capabilities: {e}")

    async def _setup_event_handlers(self):
        """Setup event handlers for integration"""
        try:
            # Add learning callback to pipeline
            if self.omniscient_pipeline:
                self.omniscient_pipeline.add_learning_callback(
                    self._on_omniscient_learning
                )

        except Exception as e:
            logger.error(f"❌ Error setting up event handlers: {e}")

    async def _handle_camera_requests(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle camera-related requests from Echo Brain"""
        try:
            action = request.get('action')
            camera_id = request.get('camera_id')

            if action == 'get_status':
                if self.omniscient_pipeline and self.omniscient_pipeline.camera_integration:
                    return {
                        'status': 'success',
                        'data': self.omniscient_pipeline.camera_integration.get_camera_status()
                    }

            elif action == 'get_recent_events':
                hours = request.get('hours', 24)
                if self.omniscient_pipeline:
                    events = []
                    cutoff = datetime.now() - timedelta(hours=hours)

                    for event in self.omniscient_pipeline.detection_buffer:
                        if event.timestamp >= cutoff:
                            events.append({
                                'timestamp': event.timestamp.isoformat(),
                                'type': event.event_type,
                                'camera_id': event.camera_id,
                                'description': event.description,
                                'confidence': event.confidence
                            })

                    return {
                        'status': 'success',
                        'data': {
                            'events': events[-50:],  # Last 50 events
                            'total_count': len(events)
                        }
                    }

            elif action == 'enable_camera':
                if self.omniscient_pipeline and camera_id:
                    camera = self.omniscient_pipeline.camera_integration.cameras.get(camera_id)
                    if camera:
                        camera.enabled = True
                        self.omniscient_pipeline.camera_integration.start_camera_stream(camera_id)
                        return {'status': 'success', 'message': f'Camera {camera_id} enabled'}

            elif action == 'disable_camera':
                if self.omniscient_pipeline and camera_id:
                    camera = self.omniscient_pipeline.camera_integration.cameras.get(camera_id)
                    if camera:
                        camera.enabled = False
                        cap = self.omniscient_pipeline.camera_integration.active_streams.get(camera_id)
                        if cap:
                            cap.release()
                            del self.omniscient_pipeline.camera_integration.active_streams[camera_id]
                        return {'status': 'success', 'message': f'Camera {camera_id} disabled'}

            return {'status': 'error', 'message': 'Unknown or invalid camera action'}

        except Exception as e:
            logger.error(f"❌ Error handling camera request: {e}")
            return {'status': 'error', 'message': str(e)}

    async def _handle_environment_requests(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle environmental awareness requests"""
        try:
            if not self.omniscient_pipeline or not self.omniscient_pipeline.context_history:
                return {'status': 'error', 'message': 'No environmental data available'}

            latest_context = self.omniscient_pipeline.context_history[-1]

            return {
                'status': 'success',
                'data': {
                    'timestamp': latest_context.timestamp.isoformat(),
                    'environmental_state': latest_context.environmental_state,
                    'recent_activity': {
                        'detections': len(latest_context.camera_events),
                        'scenes_analyzed': len(latest_context.scene_analysis)
                    },
                    'locations': latest_context.environmental_state.get('locations', {}),
                    'overall_status': {
                        'occupancy': latest_context.environmental_state.get('occupancy', 'unknown'),
                        'lighting': latest_context.environmental_state.get('lighting', 'unknown'),
                        'activity_level': latest_context.environmental_state.get('activity_level', 'unknown'),
                        'safety_status': latest_context.environmental_state.get('safety_status', 'unknown')
                    }
                }
            }

        except Exception as e:
            logger.error(f"❌ Error handling environment request: {e}")
            return {'status': 'error', 'message': str(e)}

    async def _handle_behavior_requests(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle behavioral analysis requests"""
        try:
            if not self.omniscient_pipeline:
                return {'status': 'error', 'message': 'Behavioral analysis not available'}

            action = request.get('action', 'get_patterns')

            if action == 'get_patterns':
                patterns = []
                for pattern_id, pattern in self.omniscient_pipeline.behavior_patterns.items():
                    patterns.append({
                        'id': pattern_id,
                        'type': pattern.pattern_type,
                        'confidence': pattern.confidence,
                        'description': pattern.description,
                        'frequency': pattern.frequency,
                        'last_observed': pattern.last_observed.isoformat()
                    })

                return {
                    'status': 'success',
                    'data': {
                        'patterns': patterns,
                        'total_count': len(patterns)
                    }
                }

            elif action == 'get_insights':
                if self.omniscient_pipeline.context_history:
                    latest_context = self.omniscient_pipeline.context_history[-1]
                    return {
                        'status': 'success',
                        'data': latest_context.learning_insights
                    }

            return {'status': 'error', 'message': 'Unknown behavior analysis action'}

        except Exception as e:
            logger.error(f"❌ Error handling behavior request: {e}")
            return {'status': 'error', 'message': str(e)}

    async def _provide_omniscient_context(self) -> Dict[str, Any]:
        """Provide omniscient context to Echo Brain"""
        try:
            if not self.omniscient_pipeline:
                return {}

            awareness = self.omniscient_pipeline.get_current_awareness()

            # Enhanced context for Echo Brain
            context = {
                'omniscient': {
                    'active': self.integration_active,
                    'cameras_active': awareness.get('active_cameras', 0),
                    'recent_detections': awareness.get('recent_detections', 0),
                    'behavioral_patterns': awareness.get('behavior_patterns', 0),
                    'environmental_state': awareness.get('environmental_state', {}),
                    'last_update': awareness.get('timestamp', datetime.now().isoformat())
                }
            }

            # Add recent events summary
            if self.omniscient_pipeline.context_history:
                latest = self.omniscient_pipeline.context_history[-1]
                context['recent_activity'] = {
                    'motion_events': len([e for e in latest.camera_events if e.event_type == 'motion']),
                    'face_detections': len([e for e in latest.camera_events if e.event_type == 'face']),
                    'object_detections': len([e for e in latest.camera_events if e.event_type == 'object'])
                }

            return context

        except Exception as e:
            logger.error(f"❌ Error providing omniscient context: {e}")
            return {'error': str(e)}

    async def _on_omniscient_learning(self, context):
        """Handle learning events from omniscient pipeline"""
        try:
            # Send learning insights to Echo Brain
            if hasattr(self.echo_brain, 'process_learning_event'):
                learning_data = {
                    'source': 'omniscient_pipeline',
                    'timestamp': context.timestamp.isoformat(),
                    'insights': context.learning_insights,
                    'behavioral_patterns': context.behavioral_patterns,
                    'environmental_changes': context.environmental_state
                }
                await self.echo_brain.process_learning_event(learning_data)

            # Store important events in Echo Brain's memory
            if hasattr(self.echo_brain, 'store_context'):
                # Store significant detections
                for event in context.camera_events:
                    if event.confidence > 0.8:  # High confidence events
                        await self.echo_brain.store_context(
                            f"high_confidence_detection_{event.camera_id}_{event.timestamp}",
                            {
                                'type': 'camera_detection',
                                'event_type': event.event_type,
                                'description': event.description,
                                'confidence': event.confidence,
                                'timestamp': event.timestamp.isoformat(),
                                'camera_id': event.camera_id
                            },
                            source='omniscient_camera'
                        )

                # Store behavioral pattern discoveries
                for pattern_type, patterns in context.behavioral_patterns.items():
                    if patterns:  # Non-empty pattern list
                        await self.echo_brain.store_context(
                            f"behavioral_pattern_{pattern_type}_{context.timestamp}",
                            {
                                'type': 'behavioral_pattern',
                                'pattern_type': pattern_type,
                                'patterns': patterns,
                                'timestamp': context.timestamp.isoformat()
                            },
                            source='omniscient_behavior'
                        )

        except Exception as e:
            logger.error(f"❌ Error processing omniscient learning: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        try:
            status = {
                'integration_active': self.integration_active,
                'pipeline_running': self.omniscient_pipeline.running if self.omniscient_pipeline else False,
                'cameras_active': 0,
                'last_detection': None,
                'behavior_patterns': 0
            }

            if self.omniscient_pipeline:
                camera_status = self.omniscient_pipeline.camera_integration.get_camera_status() if self.omniscient_pipeline.camera_integration else {}
                status['cameras_active'] = len([c for c in camera_status.values() if c.get('streaming')])

                if self.omniscient_pipeline.detection_buffer:
                    status['last_detection'] = self.omniscient_pipeline.detection_buffer[-1].timestamp.isoformat()

                status['behavior_patterns'] = len(self.omniscient_pipeline.behavior_patterns)

            return status

        except Exception as e:
            logger.error(f"❌ Error getting status: {e}")
            return {'error': str(e)}

    async def shutdown(self):
        """Shutdown omniscient integration"""
        try:
            self.integration_active = False

            if self.omniscient_pipeline:
                await self.omniscient_pipeline.stop()

            logger.info("✅ Omniscient integration shut down successfully")

        except Exception as e:
            logger.error(f"❌ Error shutting down omniscient integration: {e}")

# Factory function for Echo Brain
def create_omniscient_integration(echo_brain_instance):
    """Create omniscient integration for Echo Brain"""
    return EchoBrainOmniscientIntegration(echo_brain_instance)
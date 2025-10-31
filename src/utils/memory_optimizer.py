#!/usr/bin/env python3
"""
Memory Optimization for Long-Running Conversations
Based on expert recommendations from qwen and deepseek

This module implements:
1. Memory leak detection and prevention
2. Conversation context optimization
3. Model memory management
4. Background cleanup processes
5. Resource monitoring and alerts

CRITICAL: Prevent memory bloat in long-running Echo Brain sessions
"""

import gc
import psutil
import logging
import asyncio
import weakref
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict
import tracemalloc
import threading

logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: datetime
    rss_mb: float
    vms_mb: float
    percent: float
    conversation_count: int
    model_cache_size: int
    active_connections: int

class ConversationMemoryManager:
    """Manages memory for individual conversations"""

    def __init__(self, max_context_length: int = 4000, cleanup_interval: int = 300):
        self.max_context_length = max_context_length
        self.cleanup_interval = cleanup_interval
        self.conversation_contexts: Dict[str, List[Dict]] = {}
        self.context_timestamps: Dict[str, datetime] = {}
        self.memory_usage: Dict[str, int] = {}

        # Weak references to avoid circular references
        self.weak_refs: Set[weakref.ref] = set()

        # Start background cleanup
        self._cleanup_task = None
        self.start_background_cleanup()

    def add_conversation_turn(self, conversation_id: str, user_query: str,
                            response: str, model_used: str, metadata: Optional[Dict] = None):
        """Add a conversation turn with memory management"""

        # Initialize conversation if new
        if conversation_id not in self.conversation_contexts:
            self.conversation_contexts[conversation_id] = []
            self.context_timestamps[conversation_id] = datetime.now()
            self.memory_usage[conversation_id] = 0

        # Calculate memory usage for this turn
        turn_data = {
            'timestamp': datetime.now(),
            'user_query': user_query,
            'response': response,
            'model_used': model_used,
            'metadata': metadata or {}
        }

        # Estimate memory usage (rough calculation)
        turn_memory = len(str(turn_data).encode('utf-8'))
        self.memory_usage[conversation_id] += turn_memory

        # Add turn to context
        self.conversation_contexts[conversation_id].append(turn_data)

        # Trim context if too long
        self._trim_conversation_context(conversation_id)

        # Update timestamp
        self.context_timestamps[conversation_id] = datetime.now()

    def _trim_conversation_context(self, conversation_id: str):
        """Trim conversation context to prevent memory bloat"""
        if conversation_id not in self.conversation_contexts:
            return

        context = self.conversation_contexts[conversation_id]

        # If context is too long, keep only recent turns
        if len(context) > self.max_context_length:
            # Keep last N turns
            keep_count = int(self.max_context_length * 0.8)  # 80% of max
            removed_turns = context[:-keep_count]
            self.conversation_contexts[conversation_id] = context[-keep_count:]

            # Update memory usage
            removed_memory = sum(len(str(turn).encode('utf-8')) for turn in removed_turns)
            self.memory_usage[conversation_id] -= removed_memory

            logger.info(
                f"Trimmed conversation {conversation_id}: "
                f"removed {len(removed_turns)} turns, "
                f"freed {removed_memory:,} bytes"
            )

    def cleanup_old_conversations(self, max_age_hours: int = 24):
        """Remove old conversation contexts to free memory"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        removed_conversations = []

        for conv_id, timestamp in list(self.context_timestamps.items()):
            if timestamp < cutoff_time:
                # Remove conversation data
                if conv_id in self.conversation_contexts:
                    memory_freed = self.memory_usage.get(conv_id, 0)
                    del self.conversation_contexts[conv_id]
                    del self.context_timestamps[conv_id]
                    if conv_id in self.memory_usage:
                        del self.memory_usage[conv_id]

                    removed_conversations.append((conv_id, memory_freed))

        if removed_conversations:
            total_freed = sum(memory for _, memory in removed_conversations)
            logger.info(
                f"Cleaned up {len(removed_conversations)} old conversations, "
                f"freed {total_freed:,} bytes"
            )

        return removed_conversations

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        total_conversations = len(self.conversation_contexts)
        total_memory = sum(self.memory_usage.values())
        avg_memory = total_memory / total_conversations if total_conversations > 0 else 0

        return {
            'total_conversations': total_conversations,
            'total_memory_bytes': total_memory,
            'average_memory_per_conversation': avg_memory,
            'largest_conversation': max(self.memory_usage.values()) if self.memory_usage else 0,
            'oldest_conversation': min(self.context_timestamps.values()) if self.context_timestamps else None
        }

    def start_background_cleanup(self):
        """Start background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._background_cleanup_loop())

    async def _background_cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)

                # Clean up old conversations
                self.cleanup_old_conversations()

                # Force garbage collection
                collected = gc.collect()
                if collected > 0:
                    logger.debug(f"Garbage collected {collected} objects")

                # Clean up weak references
                self._cleanup_weak_refs()

            except Exception as e:
                logger.error(f"Error in background cleanup: {e}")

    def _cleanup_weak_refs(self):
        """Clean up dead weak references"""
        dead_refs = [ref for ref in self.weak_refs if ref() is None]
        for ref in dead_refs:
            self.weak_refs.discard(ref)

        if dead_refs:
            logger.debug(f"Cleaned up {len(dead_refs)} dead weak references")

class SystemMemoryMonitor:
    """Monitors system-wide memory usage and alerts"""

    def __init__(self, alert_threshold: float = 85.0, critical_threshold: float = 95.0):
        self.alert_threshold = alert_threshold
        self.critical_threshold = critical_threshold
        self.snapshots: List[MemorySnapshot] = []
        self.max_snapshots = 1000
        self.monitoring_active = False

        # Enable memory tracing
        tracemalloc.start()

    def take_snapshot(self, conversation_count: int = 0, model_cache_size: int = 0,
                     active_connections: int = 0) -> MemorySnapshot:
        """Take a memory usage snapshot"""
        process = psutil.Process()
        memory_info = process.memory_info()

        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            rss_mb=memory_info.rss / 1024 / 1024,  # Convert to MB
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=process.memory_percent(),
            conversation_count=conversation_count,
            model_cache_size=model_cache_size,
            active_connections=active_connections
        )

        # Add to snapshots list
        self.snapshots.append(snapshot)

        # Trim snapshots if too many
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]

        # Check for alerts
        self._check_memory_alerts(snapshot)

        return snapshot

    def _check_memory_alerts(self, snapshot: MemorySnapshot):
        """Check if memory usage requires alerts"""
        if snapshot.percent >= self.critical_threshold:
            logger.critical(
                f"CRITICAL MEMORY USAGE: {snapshot.percent:.1f}% "
                f"({snapshot.rss_mb:.1f} MB RSS)"
            )
            # Trigger emergency cleanup
            self._emergency_memory_cleanup()

        elif snapshot.percent >= self.alert_threshold:
            logger.warning(
                f"HIGH MEMORY USAGE: {snapshot.percent:.1f}% "
                f"({snapshot.rss_mb:.1f} MB RSS)"
            )

    def _emergency_memory_cleanup(self):
        """Emergency memory cleanup procedures"""
        logger.info("Initiating emergency memory cleanup")

        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Emergency GC collected {collected} objects")

        # Clear caches if available
        try:
            # Clear any module-level caches
            if hasattr(gc, 'get_objects'):
                # Find and clear cache-like objects
                for obj in gc.get_objects():
                    if hasattr(obj, 'clear') and hasattr(obj, '__len__'):
                        if isinstance(obj, (dict, list)) and len(obj) > 1000:
                            logger.debug(f"Clearing large cache object: {type(obj)}")
                            obj.clear()

        except Exception as e:
            logger.error(f"Error in emergency cleanup: {e}")

    def get_memory_trend(self, hours: int = 1) -> Dict[str, Any]:
        """Get memory usage trend over specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]

        if not recent_snapshots:
            return {"error": "No recent snapshots available"}

        start_snapshot = recent_snapshots[0]
        end_snapshot = recent_snapshots[-1]

        return {
            "duration_hours": hours,
            "start_memory_mb": start_snapshot.rss_mb,
            "end_memory_mb": end_snapshot.rss_mb,
            "memory_change_mb": end_snapshot.rss_mb - start_snapshot.rss_mb,
            "start_percent": start_snapshot.percent,
            "end_percent": end_snapshot.percent,
            "max_memory_mb": max(s.rss_mb for s in recent_snapshots),
            "avg_memory_mb": sum(s.rss_mb for s in recent_snapshots) / len(recent_snapshots),
            "snapshot_count": len(recent_snapshots)
        }

    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous memory monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            threading.Thread(target=self._monitoring_loop, args=(interval_seconds,), daemon=True).start()

    def _monitoring_loop(self, interval_seconds: int):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                self.take_snapshot()
                import time
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False

class ModelMemoryManager:
    """Manages memory for loaded AI models"""

    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.model_usage_counts: Dict[str, int] = defaultdict(int)
        self.model_last_used: Dict[str, datetime] = {}
        self.max_loaded_models = 3  # Limit concurrent models

    def register_model_usage(self, model_name: str):
        """Register that a model was used"""
        self.model_usage_counts[model_name] += 1
        self.model_last_used[model_name] = datetime.now()

        # Trigger cleanup if too many models loaded
        if len(self.loaded_models) > self.max_loaded_models:
            self._cleanup_unused_models()

    def _cleanup_unused_models(self):
        """Clean up least recently used models"""
        if not self.model_last_used:
            return

        # Sort models by last used time
        sorted_models = sorted(
            self.model_last_used.items(),
            key=lambda x: x[1]
        )

        # Remove oldest models until we're under the limit
        models_to_remove = len(self.loaded_models) - self.max_loaded_models + 1

        for model_name, _ in sorted_models[:models_to_remove]:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                logger.info(f"Unloaded model {model_name} to free memory")

    def get_model_stats(self) -> Dict[str, Any]:
        """Get model memory statistics"""
        return {
            "loaded_models": list(self.loaded_models.keys()),
            "model_count": len(self.loaded_models),
            "usage_counts": dict(self.model_usage_counts),
            "last_used": {k: v.isoformat() for k, v in self.model_last_used.items()}
        }

# Global instances
conversation_memory_manager = ConversationMemoryManager()
system_memory_monitor = SystemMemoryMonitor()
model_memory_manager = ModelMemoryManager()

# Start monitoring
system_memory_monitor.start_monitoring()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "snapshot":
            snapshot = system_memory_monitor.take_snapshot()
            print(f"Memory snapshot: {snapshot.percent:.1f}% ({snapshot.rss_mb:.1f} MB)")

        elif command == "trend":
            hours = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            trend = system_memory_monitor.get_memory_trend(hours)
            print(f"Memory trend over {hours} hour(s):")
            for key, value in trend.items():
                print(f"  {key}: {value}")

        elif command == "cleanup":
            removed = conversation_memory_manager.cleanup_old_conversations()
            print(f"Cleaned up {len(removed)} conversations")

        elif command == "stats":
            conv_stats = conversation_memory_manager.get_memory_stats()
            model_stats = model_memory_manager.get_model_stats()
            print("Conversation stats:", conv_stats)
            print("Model stats:", model_stats)

        else:
            print("Usage: python memory_optimizer.py [snapshot|trend|cleanup|stats] [hours]")
    else:
        print("\n🧠 MEMORY OPTIMIZER")
        print("="*50)
        print("Commands:")
        print("  snapshot        - Take memory snapshot")
        print("  trend [hours]   - Show memory trend")
        print("  cleanup         - Clean up old conversations")
        print("  stats           - Show memory statistics")
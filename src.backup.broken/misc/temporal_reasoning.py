#!/usr/bin/env python3
"""
Temporal Reasoning Module for Echo Brain
Provides temporal consistency validation, paradox detection, and causal chain verification
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import hashlib

class TemporalEventType(Enum):
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"

@dataclass
class TemporalEvent:
    """Represents an event in time with causal relationships"""
    id: str
    timestamp: datetime
    description: str
    event_type: TemporalEventType
    causes: List[str] = None  # IDs of events that caused this
    effects: List[str] = None  # IDs of events this causes
    probability: float = 1.0
    timeline_id: str = "main"
    metadata: Dict = None
    
    def __post_init__(self):
        self.causes = self.causes or []
        self.effects = self.effects or []
        self.metadata = self.metadata or {}

class TemporalReasoning:
    """
    Core temporal logic engine for Echo Brain
    Handles timeline consistency, paradox detection, and causal reasoning
    """
    
    def __init__(self):
        self.timelines: Dict[str, List[TemporalEvent]] = {"main": []}
        self.causal_graph: Dict[str, Dict] = {}
        self.paradoxes: List[Dict] = []
        self.consistency_rules = self._initialize_rules()
        
    def _initialize_rules(self) -> List[callable]:
        """Initialize temporal consistency rules"""
        return [
            self._check_causality_loop,
            self._check_grandfather_paradox,
            self._check_timeline_convergence,
            self._check_event_ordering,
            self._check_probability_consistency
        ]
    
    def add_event(self, event: TemporalEvent) -> bool:
        """Add an event to the timeline with validation"""
        # Validate event doesn't create paradoxes
        if not self.validate_event(event):
            return False
            
        # Add to timeline
        if event.timeline_id not in self.timelines:
            self.timelines[event.timeline_id] = []
        
        self.timelines[event.timeline_id].append(event)
        self.timelines[event.timeline_id].sort(key=lambda e: e.timestamp)
        
        # Update causal graph
        self._update_causal_graph(event)
        
        return True
    
    def validate_event(self, event: TemporalEvent) -> bool:
        """Validate an event against temporal consistency rules"""
        for rule in self.consistency_rules:
            is_valid, message = rule(event)
            if not is_valid:
                self.paradoxes.append({
                    "event_id": event.id,
                    "rule": rule.__name__,
                    "message": message,
                    "timestamp": datetime.now().isoformat()
                })
                return False
        return True
    
    def _check_causality_loop(self, event: TemporalEvent) -> Tuple[bool, str]:
        """Check for causal loops (bootstrap paradox)"""
        visited = set()
        
        def has_loop(event_id: str, path: List[str]) -> bool:
            if event_id in path:
                return True
            if event_id in visited:
                return False
            
            visited.add(event_id)
            
            # Check all effects of this event
            if event_id in self.causal_graph:
                for effect_id in self.causal_graph[event_id].get('effects', []):
                    if has_loop(effect_id, path + [event_id]):
                        return True
            
            return False
        
        # Check if adding this event creates a loop
        for cause_id in event.causes:
            if has_loop(cause_id, [event.id]):
                return False, f"Causal loop detected: Event {event.id} creates a bootstrap paradox"
        
        return True, "No causal loops detected"
    
    def _check_grandfather_paradox(self, event: TemporalEvent) -> Tuple[bool, str]:
        """Check for grandfather paradox (preventing own cause)"""
        # Check if event prevents any of its causes
        for cause_id in event.causes:
            cause_event = self._get_event_by_id(cause_id)
            if cause_event and event.timestamp < cause_event.timestamp:
                if 'prevents' in event.metadata and cause_id in event.metadata['prevents']:
                    return False, f"Grandfather paradox: Event {event.id} prevents its own cause {cause_id}"
        
        return True, "No grandfather paradox detected"
    
    def _check_timeline_convergence(self, event: TemporalEvent) -> Tuple[bool, str]:
        """Check if parallel timelines converge properly"""
        if event.event_type == TemporalEventType.PARALLEL:
            # Check for timeline branching consistency
            timeline_count = len(self.timelines)
            if timeline_count > 10:  # Arbitrary limit for timeline branches
                return False, f"Too many parallel timelines ({timeline_count}), risk of divergence"
        
        return True, "Timeline convergence acceptable"
    
    def _check_event_ordering(self, event: TemporalEvent) -> Tuple[bool, str]:
        """Verify chronological ordering of cause and effect"""
        for cause_id in event.causes:
            cause_event = self._get_event_by_id(cause_id)
            if cause_event and cause_event.timestamp > event.timestamp:
                return False, f"Temporal violation: Effect {event.id} occurs before cause {cause_id}"
        
        return True, "Event ordering is consistent"
    
    def _check_probability_consistency(self, event: TemporalEvent) -> Tuple[bool, str]:
        """Check probability consistency across timeline"""
        if event.probability < 0 or event.probability > 1:
            return False, f"Invalid probability {event.probability} for event {event.id}"
        
        # Check if conditional events have proper dependencies
        if event.event_type == TemporalEventType.CONDITIONAL and event.probability == 1.0:
            return False, f"Conditional event {event.id} cannot have probability 1.0"
        
        return True, "Probability constraints satisfied"
    
    def _update_causal_graph(self, event: TemporalEvent):
        """Update the causal relationship graph"""
        if event.id not in self.causal_graph:
            self.causal_graph[event.id] = {
                'event': event,
                'causes': [],
                'effects': []
            }
        
        # Update causes
        for cause_id in event.causes:
            if cause_id not in self.causal_graph:
                self.causal_graph[cause_id] = {'causes': [], 'effects': []}
            self.causal_graph[cause_id]['effects'].append(event.id)
            self.causal_graph[event.id]['causes'].append(cause_id)
        
        # Update effects
        for effect_id in event.effects:
            if effect_id not in self.causal_graph:
                self.causal_graph[effect_id] = {'causes': [], 'effects': []}
            self.causal_graph[effect_id]['causes'].append(event.id)
            self.causal_graph[event.id]['effects'].append(effect_id)
    
    def _get_event_by_id(self, event_id: str) -> Optional[TemporalEvent]:
        """Retrieve an event by its ID across all timelines"""
        for timeline in self.timelines.values():
            for event in timeline:
                if event.id == event_id:
                    return event
        return None
    
    def detect_paradoxes(self) -> List[Dict]:
        """Detect all paradoxes in current timelines"""
        paradoxes = []
        
        for timeline_id, events in self.timelines.items():
            for event in events:
                is_valid = self.validate_event(event)
                if not is_valid:
                    paradoxes.append({
                        'timeline': timeline_id,
                        'event': event.id,
                        'type': 'consistency_violation'
                    })
        
        return paradoxes + self.paradoxes
    
    def verify_causal_chain(self, start_event_id: str, end_event_id: str) -> Tuple[bool, List[str]]:
        """Verify if a causal chain exists between two events"""
        visited = set()
        path = []
        
        def find_path(current_id: str, target_id: str, current_path: List[str]) -> Optional[List[str]]:
            if current_id == target_id:
                return current_path + [current_id]
            
            if current_id in visited:
                return None
            
            visited.add(current_id)
            
            if current_id in self.causal_graph:
                for effect_id in self.causal_graph[current_id].get('effects', []):
                    result = find_path(effect_id, target_id, current_path + [current_id])
                    if result:
                        return result
            
            return None
        
        path = find_path(start_event_id, end_event_id, [])
        
        if path:
            return True, path
        else:
            return False, []
    
    def maintain_event_sequence(self, events: List[TemporalEvent]) -> List[TemporalEvent]:
        """Maintain proper event sequencing with validation"""
        # Sort by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Validate sequence
        valid_sequence = []
        for event in sorted_events:
            if self.validate_event(event):
                valid_sequence.append(event)
        
        return valid_sequence
    
    def get_timeline_consistency_score(self, timeline_id: str = "main") -> float:
        """Calculate consistency score for a timeline (0-1)"""
        if timeline_id not in self.timelines:
            return 0.0
        
        events = self.timelines[timeline_id]
        if not events:
            return 1.0
        
        valid_events = sum(1 for e in events if self.validate_event(e))
        return valid_events / len(events)
    
    def merge_timelines(self, timeline1_id: str, timeline2_id: str, merge_point: datetime) -> str:
        """Merge two parallel timelines at a specific point"""
        if timeline1_id not in self.timelines or timeline2_id not in self.timelines:
            raise ValueError("Timeline not found")
        
        # Create new merged timeline
        merged_id = f"merged_{timeline1_id}_{timeline2_id}"
        
        # Get events before and after merge point
        t1_before = [e for e in self.timelines[timeline1_id] if e.timestamp < merge_point]
        t2_before = [e for e in self.timelines[timeline2_id] if e.timestamp < merge_point]
        t1_after = [e for e in self.timelines[timeline1_id] if e.timestamp >= merge_point]
        t2_after = [e for e in self.timelines[timeline2_id] if e.timestamp >= merge_point]
        
        # Combine and validate
        merged_events = t1_before + t2_before + t1_after + t2_after
        merged_events = self.maintain_event_sequence(merged_events)
        
        self.timelines[merged_id] = merged_events
        
        return merged_id
    
    def to_json(self) -> str:
        """Serialize temporal state to JSON"""
        state = {
            'timelines': {
                tid: [
                    {
                        'id': e.id,
                        'timestamp': e.timestamp.isoformat(),
                        'description': e.description,
                        'type': e.event_type.value,
                        'causes': e.causes,
                        'effects': e.effects,
                        'probability': e.probability
                    }
                    for e in events
                ]
                for tid, events in self.timelines.items()
            },
            'paradoxes': self.paradoxes,
            'consistency_scores': {
                tid: self.get_timeline_consistency_score(tid)
                for tid in self.timelines.keys()
            }
        }
        return json.dumps(state, indent=2)

# Integration with Echo Brain
class EchoTemporalInterface:
    """Interface for Echo Brain to use temporal reasoning"""
    
    def __init__(self):
        self.temporal_engine = TemporalReasoning()
        
    async def process_temporal_query(self, query: Dict) -> Dict:
        """Process temporal logic queries from Echo"""
        query_type = query.get('type', 'validate')
        
        if query_type == 'validate':
            return await self._validate_temporal_consistency(query)
        elif query_type == 'add_event':
            return await self._add_temporal_event(query)
        elif query_type == 'detect_paradox':
            return await self._detect_paradoxes(query)
        elif query_type == 'verify_causality':
            return await self._verify_causal_chain(query)
        else:
            return {'error': f'Unknown query type: {query_type}'}
    
    async def _validate_temporal_consistency(self, query: Dict) -> Dict:
        """Validate temporal consistency of events"""
        events_data = query.get('events', [])
        
        events = []
        for event_data in events_data:
            event = TemporalEvent(
                id=event_data.get('id', hashlib.md5(str(event_data).encode()).hexdigest()),
                timestamp=datetime.fromisoformat(event_data['timestamp']),
                description=event_data['description'],
                event_type=TemporalEventType(event_data.get('type', 'present')),
                causes=event_data.get('causes', []),
                effects=event_data.get('effects', []),
                probability=event_data.get('probability', 1.0)
            )
            events.append(event)
        
        valid_sequence = self.temporal_engine.maintain_event_sequence(events)
        
        return {
            'valid': len(valid_sequence) == len(events),
            'valid_count': len(valid_sequence),
            'total_count': len(events),
            'consistency_score': len(valid_sequence) / len(events) if events else 1.0,
            'paradoxes': self.temporal_engine.detect_paradoxes()
        }
    
    async def _add_temporal_event(self, query: Dict) -> Dict:
        """Add a new temporal event"""
        event_data = query.get('event', {})
        
        event = TemporalEvent(
            id=event_data.get('id', hashlib.md5(str(event_data).encode()).hexdigest()),
            timestamp=datetime.fromisoformat(event_data['timestamp']),
            description=event_data['description'],
            event_type=TemporalEventType(event_data.get('type', 'present')),
            causes=event_data.get('causes', []),
            effects=event_data.get('effects', []),
            probability=event_data.get('probability', 1.0),
            timeline_id=event_data.get('timeline_id', 'main')
        )
        
        success = self.temporal_engine.add_event(event)
        
        return {
            'success': success,
            'event_id': event.id,
            'timeline_id': event.timeline_id,
            'consistency_score': self.temporal_engine.get_timeline_consistency_score(event.timeline_id)
        }
    
    async def _detect_paradoxes(self, query: Dict) -> Dict:
        """Detect paradoxes in timelines"""
        timeline_id = query.get('timeline_id', 'main')
        
        paradoxes = self.temporal_engine.detect_paradoxes()
        
        return {
            'paradoxes': paradoxes,
            'count': len(paradoxes),
            'timeline_id': timeline_id,
            'consistency_score': self.temporal_engine.get_timeline_consistency_score(timeline_id)
        }
    
    async def _verify_causal_chain(self, query: Dict) -> Dict:
        """Verify causal chain between events"""
        start_id = query.get('start_event_id')
        end_id = query.get('end_event_id')
        
        if not start_id or not end_id:
            return {'error': 'Missing start_event_id or end_event_id'}
        
        exists, path = self.temporal_engine.verify_causal_chain(start_id, end_id)
        
        return {
            'causal_chain_exists': exists,
            'path': path,
            'path_length': len(path) if exists else 0
        }

# Export main components
__all__ = ['TemporalReasoning', 'TemporalEvent', 'TemporalEventType', 'EchoTemporalInterface']

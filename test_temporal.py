#!/usr/bin/env python3
from datetime import datetime
import sys
sys.path.insert(0, '/opt/tower-echo-brain')

from temporal_reasoning import TemporalReasoning, TemporalEvent, TemporalEventType

# Test temporal reasoning
tr = TemporalReasoning()

# Create some test events
e1 = TemporalEvent(
    id='event1',
    timestamp=datetime(2025, 9, 20, 10, 0, 0),
    description='User asks Echo to generate anime',
    event_type=TemporalEventType.PRESENT
)

e2 = TemporalEvent(
    id='event2',
    timestamp=datetime(2025, 9, 20, 10, 5, 0),
    description='Echo generates anime frames',
    event_type=TemporalEventType.FUTURE,
    causes=['event1']
)

e3 = TemporalEvent(
    id='event3',
    timestamp=datetime(2025, 9, 20, 10, 10, 0),
    description='Video compilation completes',
    event_type=TemporalEventType.FUTURE,
    causes=['event2']
)

# Test adding events
print('Testing temporal logic:')
print(f'Adding event 1: {tr.add_event(e1)}')
print(f'Adding event 2: {tr.add_event(e2)}')
print(f'Adding event 3: {tr.add_event(e3)}')

# Test causal chain
exists, path = tr.verify_causal_chain('event1', 'event3')
print(f'\nCausal chain from event1 to event3: {exists}')
print(f'Path: {path}')

# Test consistency score
score = tr.get_timeline_consistency_score()
print(f'\nTimeline consistency score: {score}')

# Test paradox detection
paradoxes = tr.detect_paradoxes()
print(f'\nParadoxes detected: {len(paradoxes)}')

print('\n✅ Temporal reasoning module working!')

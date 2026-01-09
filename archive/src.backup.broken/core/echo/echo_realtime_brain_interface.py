#!/usr/bin/env python3
from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio
from datetime import datetime

class EchoWebSocketManager:
    def __init__(self):
        self.active_connections = []
        self.cognitive_state = {
            'current_model': 'tinyllama:latest',
            'processing': False,
            'last_activity': None,
            'active_providers': []
        }
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Send initial cognitive state
        await websocket.send_text(json.dumps({
            'type': 'cognitive_state',
            'data': self.cognitive_state
        }))
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast_cognitive_update(self, update_data):
        self.cognitive_state.update(update_data)
        self.cognitive_state['last_activity'] = datetime.now().isoformat()
        
        message = json.dumps({
            'type': 'cognitive_update',
            'data': self.cognitive_state,
            'timestamp': datetime.now().isoformat()
        })
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)

# Global manager instance
websocket_manager = EchoWebSocketManager()

async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get('type') == 'ping':
                await websocket.send_text(json.dumps({'type': 'pong'}))
            elif message.get('type') == 'get_state':
                await websocket.send_text(json.dumps({
                    'type': 'cognitive_state',
                    'data': websocket_manager.cognitive_state
                }))
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

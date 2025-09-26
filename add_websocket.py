import asyncio
from fastapi import WebSocket, WebSocketDisconnect
import json
import time
from typing import Set

# This will be added to the main.py file

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            
            # Send acknowledgment
            response = {
                "type": "response",
                "text": f"Echo received: {data}",
                "timestamp": time.time(),
                "hemisphere": "both"
            }
            await websocket.send_json(response)
            
            # Send a thought
            thought = {
                "type": "thought",
                "content": f"Processing: {data}",
                "hemisphere": "left",
                "confidence": 85
            }
            await websocket.send_json(thought)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import requests
import os

app = FastAPI(title="Echo Brain with WebSocket", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket client management
websocket_clients = []

# Core Echo functionality
@app.get("/")
async def root():
    return {"service": "Echo Brain with WebSocket", "version": "2.0.0", "websocket": "/ws/cognitive"}

@app.get("/api/echo/health")
async def health_check():
    return {"status": "healthy", "websocket_clients": len(websocket_clients)}

@app.post("/api/echo/chat")
async def chat_endpoint(request: dict):
    try:
        query = request.get("message", "")
        
        # Use localhost Ollama for chat
        ollama_response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": "tinyllama:latest", "prompt": query, "stream": False},
            timeout=30
        )
        
        if ollama_response.status_code == 200:
            result = ollama_response.json()
            response_text = result.get("response", "No response")
            
            # Broadcast to WebSocket clients
            await broadcast_to_websockets({"type": "chat_response", "query": query, "response": response_text})
            
            return {"response": response_text, "model": "tinyllama:latest"}
        else:
            return {"error": "Model unavailable"}
            
    except Exception as e:
        return {"error": str(e)}

# WebSocket endpoint - CLEAN IMPLEMENTATION
@app.websocket("/ws/cognitive")
async def cognitive_websocket(websocket: WebSocket):
    await websocket.accept()
    websocket_clients.append(websocket)
    
    try:
        await websocket.send_text("✅ Connected to Echo Brain WebSocket!")
        
        while True:
            data = await websocket.receive_text()
            
            # Echo the message back
            await websocket.send_text(f"Echo: {data}")
            
            # Broadcast to other clients
            for client in websocket_clients:
                if client != websocket:
                    try:
                        await client.send_text(f"Broadcast: {data}")
                    except:
                        websocket_clients.remove(client)
                        
    except WebSocketDisconnect:
        if websocket in websocket_clients:
            websocket_clients.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in websocket_clients:
            websocket_clients.remove(websocket)

# Helper function to broadcast to WebSocket clients
async def broadcast_to_websockets(message: dict):
    if websocket_clients:
        for client in websocket_clients[:]:  # Copy list to avoid modification during iteration
            try:
                await client.send_text(str(message))
            except:
                websocket_clients.remove(client)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8309, log_level="info")

import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="WebSocket Diagnostic Service", debug=True)

@app.get("/")
async def root():
    return {"message": "WebSocket Diagnostic Service Running", "websocket_endpoint": "/ws"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info(f"WebSocket connection attempt from {websocket.client}")
    logger.info(f"Headers: {websocket.headers}")
    logger.info(f"Query params: {websocket.query_params}")
    
    try:
        await websocket.accept()
        logger.info("WebSocket connection ACCEPTED successfully")
        
        await websocket.send_text("Connection established! Diagnostic WebSocket working.")
        
        while True:
            try:
                data = await websocket.receive_text()
                logger.info(f"Received: {data}")
                await websocket.send_text(f"Echo: {data}")
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected normally")
                break
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        raise

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8310, log_level="debug")
